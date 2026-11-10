"""
Dataset Loader
"""

import pandas as pd
from pathlib import Path
from src.logger import logger


class DataLoader:

    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def load_data(self):
        """
        Load CSV dataset
        """
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.filepath}"
            )

        logger.info("Loading dataset...")

        df = pd.read_csv(self.filepath)

        logger.info(f"Dataset Shape: {df.shape}")

        return df

    @staticmethod
    def save_processed(df, output_path):
        """
        Save processed dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.info(f"Processed data saved to {output_path}")