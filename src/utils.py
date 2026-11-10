"""
Utility Functions
"""

import joblib
from pathlib import Path


def save_object(obj, filepath):
    """
    Save Python object
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_object(filepath):
    """
    Load Python object
    """
    return joblib.load(filepath)


def print_header(title):
    """
    Pretty Console Header
    """
    print("=" * 60)
    print(title.center(60))
    print("=" * 60)