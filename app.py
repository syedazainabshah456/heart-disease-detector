# ============================================
# HEART DISEASE DETECTOR - UI
# AI Project - DUET
# ============================================

import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
# These files are created after running heart_project.py
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page title and description
st.title("Heart Disease Detector")
st.write("Enter the patient details below to predict heart disease risk.")
st.write("---")

# ============================================
# INPUT SECTION
# ============================================

st.subheader("Patient Information")

# Row 1
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    restecg = st.selectbox("Resting ECG Result", 
                           options=[0, 1, 2],
                           format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)

with col2:
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=150)
    slope = st.selectbox("Slope of ST Segment",
                         options=[0, 1, 2],
                         format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    fbs = st.radio("Fasting Blood Sugar above 120?", 
                   options=[0, 1],
                   format_func=lambda x: "No" if x == 0 else "Yes")
    exang = st.radio("Exercise Induced Angina?",
                     options=[0, 1],
                     format_func=lambda x: "No" if x == 0 else "Yes")

st.write("---")

# Chest pain and other details
st.subheader("Additional Details")

col3, col4 = st.columns(2)

with col3:
    cp = st.selectbox("Chest Pain Type",
                      options=[0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "Typical Angina",
                          1: "Atypical Angina",
                          2: "Non-Anginal Pain",
                          3: "Asymptomatic"
                      }[x])
    sex = st.radio("Sex",
                   options=[0, 1],
                   format_func=lambda x: "Female" if x == 0 else "Male")

with col4:
    thal = st.selectbox("Thalassemia",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: {
                            0: "Normal",
                            1: "Fixed Defect",
                            2: "Reversible Defect",
                            3: "Unknown"
                        }[x])

st.write("---")

# ============================================
# PREDICTION SECTION
# ============================================

if st.button("Predict"):

    # Collect all inputs into one array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input same way we scaled training data
    input_scaled = scaler.transform(input_data)

    # Get prediction and confidence
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    st.write("---")
    st.subheader("Result")

    if prediction == 1:
        st.error("Heart Disease Detected")
        st.write(f"Confidence: {probability * 100:.1f}%")
        st.write("Please consult a doctor immediately.")
    else:
        st.success("No Heart Disease Detected")
        st.write(f"Confidence: {probability * 100:.1f}%")
        st.write("Keep maintaining a healthy lifestyle.")