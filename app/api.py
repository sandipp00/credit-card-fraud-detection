"""
FastAPI Backend
Credit Card Fraud Detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import FraudPredictor

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="REST API for predicting fraudulent credit card transactions."
)

predictor = FraudPredictor()


class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/")
def home():

    return {
        "message": "Credit Card Fraud Detection API",
        "status": "Running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    try:

        data = {

            key: [value]

            for key, value in transaction.dict().items()

        }

        result = predictor.predict(data)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )