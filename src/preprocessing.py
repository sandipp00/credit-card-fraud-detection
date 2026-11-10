"""
Data Preprocessing Module
Credit Card Fraud Detection
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config.config import (
    RANDOM_STATE,
    TEST_SIZE,
    PROCESSED_DATA_FILE,
    SCALER_FILE,
)

from src.logger import logger
from src.utils import save_object


class DataPreprocessor:
    """
    Complete preprocessing pipeline.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def clean_data(self):
        """
        Handle missing values and duplicates.
        """

        logger.info("Cleaning dataset...")

        # Remove duplicate rows
        duplicates = self.df.duplicated().sum()

        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows")
            self.df.drop_duplicates(inplace=True)

        # Fill missing numeric values
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].median()
        )

        # Fill missing categorical values
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(
                self.df[col].mode()[0]
            )

        logger.info("Data cleaning completed")

        return self.df

    def split_data(self):
        """
        Split features and target.
        """

        logger.info("Splitting dataset")

        X = self.df.drop("Class", axis=1)
        y = self.df["Class"]

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    def scale_data(self, X_train, X_test):
        """
        Standardize features.
        """

        logger.info("Scaling features")

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        save_object(scaler, SCALER_FILE)

        logger.info("Scaler saved")

        return X_train_scaled, X_test_scaled

    def apply_smote(self, X_train, y_train):
        """
        Balance the training dataset.
        """

        logger.info("Applying SMOTE")

        smote = SMOTE(random_state=RANDOM_STATE)

        X_resampled, y_resampled = smote.fit_resample(
            X_train,
            y_train,
        )

        logger.info(
            f"Balanced dataset shape: {X_resampled.shape}"
        )

        return X_resampled, y_resampled

    def save_processed_dataset(self):
        """
        Save cleaned dataset.
        """

        self.df.to_csv(PROCESSED_DATA_FILE, index=False)

        logger.info(
            f"Processed dataset saved to {PROCESSED_DATA_FILE}"
        )