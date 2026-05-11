import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("diabetes_model.h5")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Diabetes Prediction System")

st.write("Enter patient details below:")

# Gender Selection
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# Pregnancy only for females
if gender == "Female":
    pregnancies = st.number_input(
        "Pregnancies (count)",
        min_value=0,
        step=1
    )
else:
    pregnancies = 0

# Other Inputs with Units
glucose = st.number_input(
    "Glucose Level (mg/dL)",
    min_value=0
)

blood_pressure = st.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0
)

skin_thickness = st.number_input(
    "Skin Thickness (mm)",
    min_value=0
)

insulin = st.number_input(
    "Insulin (mu U/ml)",
    min_value=0
)

bmi = st.number_input(
    "BMI (kg/m²)",
    min_value=0.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0
)

age = st.number_input(
    "Age (years)",
    min_value=1
)

# Predict Button
if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)
 
    # Prediction
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    # Result
    if probability > 0.5:
     st.error("Prediction: Diabetes")
    else:
     st.success("Prediction: No Diabetes")

    # Show probability separately
    st.info(f"Probability: {probability:.2f}")