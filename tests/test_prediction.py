import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.predict import FraudPredictor

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour", "LogAmount"]


def make_transaction(**overrides):
    data = {f"V{i}": [0.1 * i] for i in range(1, 29)}
    data["Time"] = [10000]
    data["Amount"] = [149.62]
    data.update({k: [v] for k, v in overrides.items()})
    return data


@pytest.fixture
def predictor():
    """FraudPredictor built from a tiny model so no trained files are needed."""
    rng = np.random.default_rng(0)

    X = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    y = np.array([0, 1] * 50)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)

    p = object.__new__(FraudPredictor)
    p.model = model
    p.scaler = scaler
    return p


def test_preprocess_engineers_features(predictor):
    out = predictor.preprocess(make_transaction())

    assert out.shape == (1, len(FEATURES))


def test_predict_returns_expected_keys(predictor):
    result = predictor.predict(make_transaction())

    assert set(result) == {"prediction", "fraud_probability", "confidence"}
    assert result["prediction"] in {"Fraud", "Normal"}


def test_probabilities_are_valid(predictor):
    result = predictor.predict(make_transaction())

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["confidence"] >= 0.5
    assert result["confidence"] == pytest.approx(
        max(result["fraud_probability"], 1 - result["fraud_probability"]),
        abs=1e-3,
    )
