import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = Path(__file__).parent / "diabetes_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# =========================
# TITLE
# =========================

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter patient details below to predict diabetes risk."
)

st.markdown("---")

# =========================
# INPUT FIELDS
# =========================

st.subheader("Patient Information")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=5.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

st.markdown("---")

# =========================
# PREDICTION
# =========================

if st.button("Predict Diabetes", use_container_width=True):

    # Prepare patient data
    data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    try:
        # Predict probability
        probability = model.predict_proba(data)[0][1]

        # Classification threshold
        threshold = 0.60

        st.markdown("---")
        st.subheader("Prediction Result")

        if probability >= threshold:

            st.error("⚠️ High Risk of Diabetes Detected")

        else:

            st.success("✅ Low Risk of Diabetes")

        # Display probability
        st.write(
            f"**Model Estimated Probability: {probability * 100:.2f}%**"
        )

        st.progress(float(probability))

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Machine Learning Based Diabetes Prediction System"
)

st.caption(
    "Educational project only. This prediction is not a medical diagnosis."
)
