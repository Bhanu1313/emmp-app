import streamlit as st
import pandas as pd
import joblib
import os
import gdown
from features import FEATURES  # List of required features for the model

# File paths and URLs
#MODEL_URL = "https://drive.google.com/uc?id=1T06Ndwy7av1-qY_GQ5ukBjKigUktfIN4&export=download"
MODEL_URL = "https://drive.google.com/uc?id=1Cfop7fwMFQ4oXZ5pB2VvQvAOH2j1wG0w"
SCALER_URL = "https://drive.google.com/uc?id=1yyls9X4vz7C_4bcUWna-sk2PmFl8g55M"
ENCODERS_URL = "https://drive.google.com/uc?id=1mFu7kQR4JEsUwQvJCxmQ9pw1yecMaljG"

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"
# Download the files if they do not exist
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    gdown.download(SCALER_URL, SCALER_PATH, quiet=False)

if not os.path.exists(ENCODERS_PATH):
    gdown.download(ENCODERS_URL, ENCODERS_PATH, quiet=False)


# Download model if not available
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model and other components
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Page Configuration
st.set_page_config(
    page_title="Income Prediction 📈",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
        body {
            background-color: #1a202c;
        }
        .main {
            background-color: #2d3748;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
            max-width: 750px;
            margin: 1rem auto;
            border: 2px solid #4c51bf;
        }
        .header {
            background-color: #4a5568;
            padding: 1rem;
            border-radius: 10px 10px 0 0;
            text-align: center;
            color: #a3bffa;
            font-size: 2em;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #a0aec0;
            font-size: 1.2em;
            margin-bottom: 2rem;
            animation: fadeIn 2s ease-in-out;
        }
        .stButton>button {
            background-color: #4c51bf;
            color: white;
            padding: 0.7em 2em;
            font-size: 16px;
            border-radius: 10px;
            border: none;
            margin: auto;
            display: block;
        }
        .stButton>button:hover {
            background-color: #2a4365;
            transform: scale(1.05);
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Main content wrapper
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>📈 Income Prediction 💰</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill in the details below and click Predict! 🌟</div>", unsafe_allow_html=True)

# --- User Input ---
user_input = {
    'age': st.slider("📅 Age", 18, 100, 30),
    'education': st.selectbox("🎓 Education Level", encoders['education'].classes_),
    'occupation': st.selectbox("💼 Occupation", encoders['occupation'].classes_),
    'hours-per-week': st.slider("⏳ Hours Worked per Week", 1, 100, 40),
    'gender': st.selectbox("👤 Gender", encoders['gender'].classes_)
}

# --- Prediction ---
if st.button("🔍 Predict Income"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for col in input_df.columns:
        if input_df[col].dtype == object:
            input_df[col] = encoders[col].transform(input_df[col])

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Output
    st.markdown("---")
    st.markdown(
        f"<h3 style='color: #90cdf4; text-align: center;'>🎯 Predicted Income Class: <span style='color: #a0aec0'>{prediction}</span></h3>",
        unsafe_allow_html=True
    )
    st.balloons()

# Close wrapper
st.markdown("</div>", unsafe_allow_html=True)
