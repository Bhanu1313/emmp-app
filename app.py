import streamlit as st
import pandas as pd
import joblib
import os
import gdown
from features import FEATURES  # This should be a list of features used in the model

# --- Google Drive File IDs ---
model_id = "1Cfop7fwMFQ4oXZ5pB2VvQvAOH2j1wG0w"
scaler_id = "1yyls9X4vz7C_4bcUWna-sk2PmFl8g55M"
encoders_id = "1mFu7kQR4JEsUwQvJCxmQ9pw1yecMaljG"

# --- Local file paths ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"

# --- Download missing files using gdown ---
if not os.path.exists(MODEL_PATH):
    model_url = f"https://drive.google.com/uc?id={model_id}"
    gdown.download(model_url, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    scaler_url = f"https://drive.google.com/uc?id={scaler_id}"
    gdown.download(scaler_url, SCALER_PATH, quiet=False)

if not os.path.exists(ENCODERS_PATH):
    encoders_url = f"https://drive.google.com/uc?id={encoders_id}"
    gdown.download(encoders_url, ENCODERS_PATH, quiet=False)

# --- Load pre-trained components ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Income Prediction", layout="centered")

st.title("💰 Income Prediction App")
st.markdown("Provide your details below to predict your income category:")

# --- Collect user input ---
user_input = {
    'age': st.slider("Age", 18, 100, 30),
    'education': st.selectbox("Education", encoders['education'].classes_),
    'occupation': st.selectbox("Occupation", encoders['occupation'].classes_),
    'hours-per-week': st.slider("Hours per Week", 1, 100, 40),
    'gender': st.selectbox("Gender", encoders['gender'].classes_)
}

# --- Predict when button is clicked ---
if st.button("Predict Income"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical columns using loaded encoders
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict income
    prediction = model.predict(input_scaled)[0]

    # Show result
    st.success(f"Predicted Income: **{prediction}**")
    st.balloons()
