"""
Feature Engineering Module
Credit Card Fraud Detection
"""

import numpy as np
import pandas as pd

from src.logger import logger


class FeatureEngineer:
    """
    Feature Engineering Pipeline
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def create_time_features(self):
        """
        Create Hour feature from Time column.
        """

        if "Time" in self.df.columns:

            logger.info("Creating Hour feature")

            self.df["Hour"] = (
                self.df["Time"] // 3600
            ) % 24

        return self.df

    def create_amount_features(self):
        """
        Create LogAmount feature.
        """

        if "Amount" in self.df.columns:

            logger.info("Creating LogAmount feature")

            self.df["LogAmount"] = np.log1p(
                self.df["Amount"]
            )

        return self.df

    def remove_unused_columns(self):
        """
        Remove unnecessary columns.
        """

        if "Time" in self.df.columns:

            logger.info("Dropping Time column")

            self.df.drop(
                columns=["Time"],
                inplace=True,
            )

        return self.df

    def feature_summary(self):
        """
        Print dataset information.
        """

        logger.info("Feature Engineering Completed")

        print("\nDataset Shape :", self.df.shape)

        print("\nColumns")

        for column in self.df.columns:
            print(column)

        return self.df

    def transform(self):
        """
        Complete feature engineering pipeline.
        """

        self.create_time_features()

        self.create_amount_features()

        self.remove_unused_columns()

        self.feature_summary()

        return self.df