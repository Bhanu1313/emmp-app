import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download
from features import FEATURES  # This file must contain t

# --- Hugging Face Hub Info ---
REPO_ID = "Bhanu1313/emmp-app"  # Change this to your actual HF repo
# --- Local Paths ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"

# --- Download from Hugging Face Hub if not present ---
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")

if not os.path.exists(SCALER_PATH):
    SCALER_PATH = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl")

if not os.path.exists(ENCODERS_PATH):
    ENCODERS_PATH = hf_hub_download(repo_id=REPO_ID, filename="encoders.pkl")

# --- Load all components ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# --- Streamlit Layout ---
st.set_page_config(page_title="Income Prediction", layout="centered")

st.title("💰 Income Prediction App")
st.markdown("Provide your details below to predict your income category:")

# --- Collect User Input ---
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

    # Encode categorical columns
    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

    # Scale numeric data
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Income: **{prediction}**")
    st.balloons()
