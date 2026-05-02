# ============================================
# HEART DISEASE DETECTOR - UI
# AI Project - DUET
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature columns
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Page title
st.title("Heart Disease Detector")
st.write("Enter the patient details below to predict heart disease risk.")
st.write("---")

# ============================================
# INPUT SECTION
# ============================================

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    restecg = st.selectbox("Resting ECG Result",
                           options=[0, 1, 2],
                           format_func=lambda x: {
                               0: "Normal",
                               1: "ST-T Abnormality",
                               2: "Left Ventricular Hypertrophy"
                           }[x])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)

with col2:
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=150)
    slope = st.selectbox("Slope of ST Segment",
                         options=[0, 1, 2],
                         format_func=lambda x: {
                             0: "Upsloping",
                             1: "Flat",
                             2: "Downsloping"
                         }[x])
    fbs = st.radio("Fasting Blood Sugar above 120?",
                   options=[0, 1],
                   format_func=lambda x: "No" if x == 0 else "Yes")
    exang = st.radio("Exercise Induced Angina?",
                     options=[0, 1],
                     format_func=lambda x: "No" if x == 0 else "Yes")

st.write("---")
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
                        options=[1, 2, 3],
                        format_func=lambda x: {
                            1: "Normal",
                            2: "Fixed Defect",
                            3: "Reversible Defect"
                        }[x])

st.write("---")

# ============================================
# PREDICTION SECTION
# ============================================

if st.button("Predict"):

    # Build input matching training dummy columns
    raw = {}
    for col in feature_columns:
        raw[col] = 0  # sab pehle 0 karo

    # Numeric columns directly assign karo
    if 'age' in raw: raw['age'] = age
    if 'trestbps' in raw: raw['trestbps'] = trestbps
    if 'chol' in raw: raw['chol'] = chol
    if 'thalch' in raw: raw['thalch'] = thalach
    if 'thalach' in raw: raw['thalach'] = thalach
    if 'oldpeak' in raw: raw['oldpeak'] = oldpeak
    if 'ca' in raw: raw['ca'] = ca

    # Sex
    if sex == 1:
        if 'sex_Male' in raw: raw['sex_Male'] = 1
        if 'sex_M' in raw: raw['sex_M'] = 1

    # Chest Pain
    cp_map = {
        0: ['cp_typical angina'],
        1: ['cp_atypical angina'],
        2: ['cp_non-anginal'],
        3: ['cp_asymptomatic']
    }
    for col_name in cp_map.get(cp, []):
        if col_name in raw: raw[col_name] = 1

    # Fasting Blood Sugar
    if fbs == 1:
        if 'fbs_True' in raw: raw['fbs_True'] = 1
        if 'fbs_1' in raw: raw['fbs_1'] = 1

    # Resting ECG
    restecg_map = {
        0: ['restecg_normal'],
        1: ['restecg_st-t abnormality', 'restecg_ST-T abnormality'],
        2: ['restecg_lv hypertrophy', 'restecg_LV hypertrophy']
    }
    for col_name in restecg_map.get(restecg, []):
        if col_name in raw: raw[col_name] = 1

    # Exercise Angina
    if exang == 1:
        if 'exang_True' in raw: raw['exang_True'] = 1
        if 'exang_Yes' in raw: raw['exang_Yes'] = 1

    # Slope
    slope_map = {
        0: ['slope_upsloping'],
        1: ['slope_flat'],
        2: ['slope_downsloping']
    }
    for col_name in slope_map.get(slope, []):
        if col_name in raw: raw[col_name] = 1

    # Thal
    thal_map = {
        1: ['thal_normal'],
        2: ['thal_fixed defect'],
        3: ['thal_reversable defect', 'thal_reversible defect']
    }
    for col_name in thal_map.get(thal, []):
        if col_name in raw: raw[col_name] = 1

    # DataFrame banao aur scale karo
    input_df = pd.DataFrame([raw])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    st.write("---")
    st.subheader("Result")

    if prediction == 1:
        st.error("❤️ Heart Disease Detected")
        st.write(f"Confidence: {probability * 100:.1f}%")
        st.write("Please consult a doctor immediately.")
    else:
        st.success("✅ No Heart Disease Detected")
        st.write(f"Confidence: {probability * 100:.1f}%")
        st.write("Keep maintaining a healthy lifestyle.")