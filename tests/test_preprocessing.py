import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 200

    df = pd.DataFrame(
        rng.normal(size=(n, 3)),
        columns=["V1", "V2", "V3"],
    )
    df["Time"] = rng.integers(0, 172800, size=n).astype(float)
    df["Amount"] = rng.uniform(0, 500, size=n)
    df["Class"] = [1] * 20 + [0] * (n - 20)

    return df


def test_clean_data_removes_duplicates(raw_df):
    df = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)

    cleaned = DataPreprocessor(df).clean_data()

    assert len(cleaned) == len(raw_df)


def test_clean_data_fills_missing_numeric_values(raw_df):
    raw_df.loc[0, "V1"] = np.nan

    cleaned = DataPreprocessor(raw_df).clean_data()

    assert cleaned["V1"].isna().sum() == 0


def test_feature_engineering_adds_and_drops_columns(raw_df):
    out = FeatureEngineer(raw_df).transform()

    assert "Time" not in out.columns
    assert {"Hour", "LogAmount"}.issubset(out.columns)
    assert out["Hour"].between(0, 23).all()
    assert np.allclose(out["LogAmount"], np.log1p(raw_df["Amount"]))


def test_split_is_stratified_and_sized(raw_df):
    X_train, X_test, y_train, y_test = DataPreprocessor(raw_df).split_data()

    assert "Class" not in X_train.columns
    assert len(X_test) == pytest.approx(0.2 * len(raw_df), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


def test_scale_data_standardises_train_set(raw_df, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.preprocessing.SCALER_FILE", tmp_path / "scaler.pkl"
    )

    X_train, X_test, _, _ = DataPreprocessor(raw_df).split_data()

    X_train_s, X_test_s = DataPreprocessor(raw_df).scale_data(X_train, X_test)

    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-8)
    assert X_test_s.shape == X_test.shape
    assert (tmp_path / "scaler.pkl").exists()


def test_smote_balances_classes(raw_df):
    X = raw_df.drop(columns=["Class"])
    y = raw_df["Class"]

    _, y_res = DataPreprocessor(raw_df).apply_smote(X, y)

    assert y_res.value_counts().nunique() == 1
