"""
Model Training Module
Credit Card Fraud Detection
"""

import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.config import MODEL_FILE, RANDOM_STATE
from src.logger import logger


class ModelTrainer:

    def __init__(self):

        self.models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=RANDOM_STATE
                ),

            "XGBoost":
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=RANDOM_STATE
                ),

            "LightGBM":
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbose=-1
                )
        }

        self.best_model = None
        self.best_name = None
        self.best_score = 0

    def compare_models(self, X_train, y_train):

        logger.info("Comparing models")

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        results = []

        for name, model in self.models.items():

            logger.info(f"Training {name}")

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                scoring="f1",
                cv=cv,
                n_jobs=-1
            )

            mean_score = scores.mean()

            results.append({

                "Model": name,

                "F1 Score": round(mean_score, 4)

            })

            if mean_score > self.best_score:

                self.best_score = mean_score
                self.best_model = model
                self.best_name = name

        return pd.DataFrame(results)

    def tune_random_forest(
        self,
        X_train,
        y_train
    ):

        logger.info("Optimizing Random Forest")

        params = {

            "n_estimators": [100, 200, 300],

            "max_depth": [10, 20, None],

            "min_samples_split": [2, 5],

            "min_samples_leaf": [1, 2]

        }

        search = RandomizedSearchCV(

            RandomForestClassifier(
                random_state=RANDOM_STATE
            ),

            param_distributions=params,

            n_iter=10,

            cv=3,

            scoring="f1",

            n_jobs=-1,

            random_state=RANDOM_STATE

        )

        search.fit(
            X_train,
            y_train
        )

        self.best_model = search.best_estimator_

        logger.info(
            f"Best Parameters : {search.best_params_}"
        )

        return self.best_model

    def fit_best_model(
        self,
        X_train,
        y_train
    ):

        logger.info("Training Best Model")

        self.best_model.fit(
            X_train,
            y_train
        )

    def save_model(self):

        logger.info("Saving model")

        joblib.dump(
            self.best_model,
            MODEL_FILE
        )

        logger.info(
            f"Model saved to {MODEL_FILE}"
        )