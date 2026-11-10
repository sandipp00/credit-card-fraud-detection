"""
Main Pipeline
Credit Card Fraud Detection
"""

from config.config import (
    RAW_DATA_FILE,
    FEATURE_NAMES_FILE,
)

from src.logger import logger
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import save_object


def main():

    logger.info("=" * 60)
    logger.info("Credit Card Fraud Detection Pipeline Started")
    logger.info("=" * 60)

    # =====================================================
    # Load Dataset
    # =====================================================

    loader = DataLoader(RAW_DATA_FILE)

    df = loader.load_data()

    # =====================================================
    # Data Cleaning
    # =====================================================

    processor = DataPreprocessor(df)

    df = processor.clean_data()

    # =====================================================
    # Feature Engineering
    # =====================================================

    engineer = FeatureEngineer(df)

    df = engineer.transform()

    # =====================================================
    # Train/Test Split
    # =====================================================

    processor = DataPreprocessor(df)

    X_train, X_test, y_train, y_test = processor.split_data()

    # Save feature names
    save_object(
        list(X_train.columns),
        FEATURE_NAMES_FILE,
    )

    # =====================================================
    # Feature Scaling
    # =====================================================

    X_train_scaled, X_test_scaled = processor.scale_data(
        X_train,
        X_test,
    )

    # =====================================================
    # Handle Class Imbalance
    # =====================================================

    X_train_balanced, y_train_balanced = processor.apply_smote(
        X_train_scaled,
        y_train,
    )

    # =====================================================
    # Model Training
    # =====================================================

    trainer = ModelTrainer()

    comparison = trainer.compare_models(
        X_train_balanced,
        y_train_balanced,
    )

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(comparison)

    print("\nBest Model :", trainer.best_name)
    print("Best F1 Score :", round(trainer.best_score, 4))

    trainer.tune_random_forest(
        X_train_balanced,
        y_train_balanced,
    )

    trainer.fit_best_model(
        X_train_balanced,
        y_train_balanced,
    )

    trainer.save_model()

    # =====================================================
    # Evaluation
    # =====================================================

    evaluator = ModelEvaluator(
        trainer.best_model
    )

    metrics = evaluator.evaluate(
        X_test_scaled,
        y_test,
    )

    evaluator.feature_importance(
        X_train.columns.tolist()
    )

    print("\n")
    print("=" * 70)
    print("FINAL MODEL METRICS")
    print("=" * 70)

    for metric, value in metrics.items():

        print(f"{metric:<15}: {value:.4f}")

    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()