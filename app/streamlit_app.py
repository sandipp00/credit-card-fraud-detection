"""
Streamlit Dashboard
Credit Card Fraud Detection

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.predict import FraudPredictor

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time"] + V_COLUMNS + ["Amount"]

SAMPLE = {
    "Time": 10000.0,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536346, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
    "Amount": 149.62,
}


@st.cache_resource
def get_predictor():
    return FraudPredictor()


def predict_row(predictor, row):
    return predictor.predict({key: [row[key]] for key in REQUIRED_COLUMNS})


def show_result(result):
    probability = result["fraud_probability"]

    if result["prediction"] == "Fraud":
        st.error(f"Fraud detected - probability {probability:.2%}")
    else:
        st.success(f"Transaction looks normal - fraud probability {probability:.2%}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Model confidence: {result['confidence']:.2%}")


st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Score a single transaction or a whole CSV file with the trained model.")

try:
    predictor = get_predictor()
except (FileNotFoundError, EOFError):
    st.error(
        "Trained model not found. Run `python main.py` first so "
        "`models/best_model.pkl` and `models/scaler.pkl` exist."
    )
    st.stop()

single_tab, batch_tab = st.tabs(["Single transaction", "Batch (CSV)"])

with single_tab:
    if "values" not in st.session_state:
        st.session_state["values"] = {k: 0.0 for k in REQUIRED_COLUMNS}

    if st.button("Load sample transaction"):
        st.session_state["values"] = dict(SAMPLE)

    values = st.session_state["values"]

    col_time, col_amount = st.columns(2)
    time_value = col_time.number_input(
        "Time (seconds)", value=float(values["Time"]), key=f"Time_{values['Time']}"
    )
    amount_value = col_amount.number_input(
        "Amount", value=float(values["Amount"]), key=f"Amount_{values['Amount']}"
    )

    row = {"Time": time_value, "Amount": amount_value}

    with st.expander("PCA features (V1-V28)"):
        columns = st.columns(4)
        for i, name in enumerate(V_COLUMNS):
            row[name] = columns[i % 4].number_input(
                name,
                value=float(values[name]),
                format="%.6f",
                key=f"{name}_{values[name]}",
            )

    if st.button("Predict", type="primary"):
        show_result(predict_row(predictor, row))

with batch_tab:
    st.write(
        "Upload a CSV with the columns `Time`, `V1`-`V28` and `Amount` "
        "(for example, rows from `creditcard.csv`)."
    )

    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            df = df.head(1000)

            results = pd.DataFrame(
                [predict_row(predictor, row) for _, row in df.iterrows()]
            )

            output = pd.concat([df.reset_index(drop=True), results], axis=1)

            fraud_count = int((results["prediction"] == "Fraud").sum())

            st.metric("Flagged as fraud", f"{fraud_count} of {len(results)}")
            st.dataframe(output)
            st.download_button(
                "Download results",
                output.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )
            st.caption("Only the first 1,000 rows are scored.")
