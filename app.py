"""Streamlit interface for Retalix discontinuation predictions."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_regression_tuned_v1.joblib"
THRESHOLD = 0.1

st.set_page_config(
    page_title="Retalix Discontinuation Predictor",
    page_icon="📦",
    layout="centered",
)

st.title("📦 Product Discontinuation Predictor")
st.write("Estimate product discontinuation risk 24 weeks before a range refresh.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the model-training pipeline first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    supplier = st.number_input("Supplier ID", min_value=0, step=1)
    hierarchy1 = st.number_input("Hierarchy Level 1 ID", min_value=0, step=1)
    hierarchy2 = st.number_input("Hierarchy Level 2 ID", min_value=0, step=1)

    sourcing = st.selectbox("Product sourcing", ["DI", "DOM"])
    seasonal = st.checkbox("Seasonal product")
    spring_summer = st.checkbox("Spring/Summer catalogue")

    price = st.number_input("Sale price (£)", min_value=0.0, value=29.99)
    forecast = st.number_input("Forecast sales per week", min_value=0.0, value=50.0)
    actual = st.number_input("Actual sales per week", min_value=0.0, value=40.0)

    submitted = st.form_submit_button("Predict discontinuation risk")

if submitted:
    features = pd.DataFrame([{
        "SalePriceIncVAT": price,
        "ForecastPerWeek": forecast,
        "ActualsPerWeek": actual,
        "ForecastActualGap": forecast - actual,
        "Supplier": supplier,
        "HierarchyLevel1": hierarchy1,
        "HierarchyLevel2": hierarchy2,
        "DIorDOM": sourcing,
        "Seasonal": seasonal,
        "SpringSummer": spring_summer,
    }])

    probability = float(model.predict_proba(features)[0, 1])

    st.metric("Estimated discontinuation probability", f"{probability:.1%}")

    if probability >= THRESHOLD:
        st.warning("Flagged for product review")
    else:
        st.success("Not flagged at the selected threshold")

    st.caption(
        "Research prototype only. Threshold selected on historical "
        "validation data; predictions require operational validation."
    )
