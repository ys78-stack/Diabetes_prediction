import streamlit as st
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# TITLE
# =========================
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details below to predict diabetes risk.")

st.markdown("---")

# =========================
# INPUT FIELDS
# =========================
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
    max_value=1000,
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

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Diabetes"):

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes Detected")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write(
        f"**Diabetes Probability:** {probability*100:.2f}%"
    )

    st.progress(float(probability))

st.markdown("---")
st.caption("Machine Learning Based Diabetes Prediction System")