import streamlit as st
import pandas as pd
import joblib
import os
import gdown
from features import FEATURES  # assuming this is your list of features

# File URLs (Google Drive direct links)
MODEL_URL = "https://drive.google.com/uc?id=1Cfop7fwMFQ4oXZ5pB2VvQvAOH2j1wG0w"
SCALER_URL = "https://drive.google.com/uc?id=1yyls9X4vz7C_4bcUWna-sk2PmFl8g55M"
ENCODERS_URL = "https://drive.google.com/uc?id=1mFu7kQR4JEsUwQvJCxmQ9pw1yecMaljG"

# File paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"

# Download missing files
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    gdown.download(SCALER_URL, SCALER_PATH, quiet=False)

if not os.path.exists(ENCODERS_PATH):
    gdown.download(ENCODERS_URL, ENCODERS_PATH, quiet=False)

# Load components
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Page config
st.set_page_config(page_title="Income Prediction", layout="centered")

st.title("💰 Income Prediction App")
st.markdown("Provide your details to predict income category.")

# --- User input ---
user_input = {
    'age': st.slider("Age", 18, 100, 30),
    'education': st.selectbox("Education", encoders['education'].classes_),
    'occupation': st.selectbox("Occupation", encoders['occupation'].classes_),
    'hours-per-week': st.slider("Hours per Week", 1, 100, 40),
    'gender': st.selectbox("Gender", encoders['gender'].classes_)
}

# --- Predict ---
if st.button("Predict Income"):
    input_df = pd.DataFrame([user_input])

    # Encode categoricals
    for col in input_df.columns:
        if input_df[col].dtype == object:
            input_df[col] = encoders[col].transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Income: **{prediction}**")
    st.balloons()
