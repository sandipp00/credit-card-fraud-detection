"""
Project Configuration File
Credit Card Fraud Detection
"""

from pathlib import Path

# =====================================================
# Base Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

LOG_DIR = BASE_DIR / "logs"

# =====================================================
# Dataset
# =====================================================

RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"

# =====================================================
# Saved Models
# =====================================================

MODEL_FILE = MODEL_DIR / "best_model.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "feature_names.pkl"

# =====================================================
# Reports
# =====================================================

REPORT_FILE = REPORT_DIR / "report.pdf"

# =====================================================
# Random State
# =====================================================

RANDOM_STATE = 42

# =====================================================
# Test Size
# =====================================================

TEST_SIZE = 0.20

# =====================================================
# Training Parameters
# =====================================================

CV_FOLDS = 5

# =====================================================
# Create Required Directories
# =====================================================

directories = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    LOG_DIR,
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)