"""
Prediction Module
Credit Card Fraud Detection
"""

import joblib
import numpy as np
import pandas as pd

from config.config import (
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_NAMES_FILE
)
from src.logger import logger


class FraudPredictor:
    """
    Load trained model and make predictions.
    """

    def __init__(self):

        logger.info("Loading trained model...")

        self.model = joblib.load(MODEL_FILE)

        self.scaler = joblib.load(SCALER_FILE)

        logger.info("Model loaded successfully.")

    def preprocess(self, data):
        """
        Apply the same preprocessing used during training.
        """

        df = pd.DataFrame(data)

        if "Time" in df.columns:

            df["Hour"] = (df["Time"] // 3600) % 24

            df.drop("Time", axis=1, inplace=True)

        if "Amount" in df.columns:

            df["LogAmount"] = np.log1p(df["Amount"])

        df_scaled = self.scaler.transform(df)

        return df_scaled

    def predict(self, data):
        """
        Predict fraud or normal transaction.
        """

        processed = self.preprocess(data)

        prediction = self.model.predict(processed)[0]

        probability = self.model.predict_proba(processed)[0][1]

        result = {

            "prediction":
                "Fraud"
                if prediction == 1
                else "Normal",

            "fraud_probability":
                round(float(probability), 4),

            "confidence":
                round(
                    max(
                        probability,
                        1 - probability
                    ),
                    4,
                ),
        }

        return result