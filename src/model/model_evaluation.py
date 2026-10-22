import os
import json
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# Logging Configuration
# ============================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_evaluation_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Load Trained Model
# ============================================================

def load_model(file_path: str):
    """Load the trained model from a pickle file."""

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.debug("Model loaded from %s", file_path)

        return model

    except FileNotFoundError:
        logger.error("Model file not found: %s", file_path)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error while loading model: %s", e
        )
        raise


# ============================================================
# Load Test Data
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load test data from a CSV file."""

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(
                f"The dataset is empty: {file_path}"
            )

        logger.debug("Data loaded from %s", file_path)

        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error while loading data: %s", e
        )
        raise


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the classification model and return metrics."""

    try:
        # Generate predictions
        y_pred = clf.predict(X_test)

        # Ensure labels are represented as NumPy arrays
        y_test = np.asarray(y_test)
        y_pred = np.asarray(y_pred)

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # Precision
        # Macro averaging evaluates both emotion classes.
        # ----------------------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # F1 Score
        # ----------------------------------------------------

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        # ----------------------------------------------------
        # Store Metrics
        # ----------------------------------------------------

        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        if hasattr(clf, "predict_proba") and hasattr(
            clf, "classes_"
        ):
            classes = np.asarray(clf.classes_)

            if len(classes) == 2:
                probabilities = clf.predict_proba(X_test)

                # Use the probability column corresponding
                # to the second class in clf.classes_.
                positive_class = classes[1]

                y_binary = (
                    y_test == positive_class
                ).astype(int)

                if len(np.unique(y_binary)) == 2:
                    auc = roc_auc_score(
                        y_binary,
                        probabilities[:, 1]
                    )

                    metrics_dict["auc"] = float(auc)

                else:
                    logger.warning(
                        "ROC-AUC skipped because the test "
                        "data contains only one target class."
                    )

            else:
                logger.warning(
                    "ROC-AUC skipped because the model does "
                    "not have exactly two classes."
                )

        logger.debug(
            "Model evaluation metrics calculated: %s",
            metrics_dict
        )

        return metrics_dict

    except Exception as e:
        logger.error(
            "Error during model evaluation: %s", e
        )
        raise


# ============================================================
# Save Evaluation Metrics
# ============================================================

def save_metrics(
    metrics: dict,
    file_path: str
) -> None:
    """Save evaluation metrics to a JSON file."""

    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.debug(
            "Metrics saved to %s",
            file_path
        )

    except Exception as e:
        logger.error(
            "Error while saving metrics: %s", e
        )
        raise


# ============================================================
# Main Function
# ============================================================

def main():

    try:
        # Load trained model
        clf = load_model(
            "./models/model.pkl"
        )

        # Load test dataset
        test_data = load_data(
            "./data/processed/test_tfidf.csv"
        )

        # Validate dataset structure
        if test_data.shape[1] < 2:
            raise ValueError(
                "Test dataset must contain at least one "
                "feature column and one target column."
            )

        # Separate features and target
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.debug(
            "Test features shape: %s",
            X_test.shape
        )

        logger.debug(
            "Test target shape: %s",
            y_test.shape
        )

        logger.debug(
            "Target labels: %s",
            np.unique(y_test)
        )

        # Evaluate model
        metrics = evaluate_model(
            clf,
            X_test,
            y_test
        )

        # Save metrics
        save_metrics(
            metrics,
            "reports/metrics.json"
        )

        logger.info(
            "Model evaluation completed successfully."
        )

        logger.info(
            "Final evaluation metrics: %s",
            metrics
        )

    except Exception as e:
        logger.exception(
            "Failed to complete model evaluation: %s",
            e
        )

        # Re-raise the exception so DVC recognizes
        # that the pipeline stage failed.
        raise


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
