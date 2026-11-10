"""
Model Evaluation Module
Credit Card Fraud Detection
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

from config.config import FIGURE_DIR
from src.logger import logger


class ModelEvaluator:

    def __init__(self, model):

        self.model = model

    def evaluate(self, X_test, y_test):

        logger.info("Evaluating model")

        predictions = self.model.predict(X_test)

        probabilities = self.model.predict_proba(X_test)[:, 1]

        metrics = {

            "Accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "Precision":
                precision_score(
                    y_test,
                    predictions
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    predictions
                ),

            "ROC AUC":
                roc_auc_score(
                    y_test,
                    probabilities
                )

        }

        print("\nClassification Report\n")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        self.plot_confusion_matrix(
            y_test,
            predictions
        )

        self.plot_roc(
            y_test,
            probabilities
        )

        self.plot_pr_curve(
            y_test,
            probabilities
        )

        return metrics

    def plot_confusion_matrix(
        self,
        y_test,
        predictions
    ):

        cm = confusion_matrix(
            y_test,
            predictions
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "confusion_matrix.png"
        )

        plt.close()

    def plot_roc(
        self,
        y_test,
        probabilities
    ):

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "roc_curve.png"
        )

        plt.close()

    def plot_pr_curve(
        self,
        y_test,
        probabilities
    ):

        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities
        )

        pr_auc = auc(
            recall,
            precision
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            recall,
            precision,
            label=f"AUC = {pr_auc:.4f}"
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR /
            "precision_recall_curve.png"
        )

        plt.close()

    def feature_importance(
        self,
        feature_names
    ):

        if not hasattr(
            self.model,
            "feature_importances_"
        ):

            logger.warning(
                "Feature importance unavailable."
            )

            return

        importance = self.model.feature_importances_

        sorted_idx = importance.argsort()[::-1]

        plt.figure(figsize=(12, 6))

        plt.bar(
            range(len(sorted_idx)),
            importance[sorted_idx]
        )

        plt.xticks(
            range(len(sorted_idx)),
            [feature_names[i] for i in sorted_idx],
            rotation=90
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR /
            "feature_importance.png"
        )

        plt.close()