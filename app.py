import streamlit as st
import pandas as pd
import joblib
from features import FEATURES

# Load model, scaler, encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# Page config
st.set_page_config(
    page_title="Income Prediction 📈",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS with professional dark theme and pastel accents
st.markdown("""
    <style>
        body {
            background-color: #1a202c;
            margin: 0;
            padding: 0;
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
            padding: 1rem 2rem;
            border-radius: 10px 10px 0 0;
            text-align: center;
            margin: -2.5rem -2.5rem 2rem -2.5rem;
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
        .stSlider > div > div > div {
            background-color: #4c51bf;
        }
        .stSlider > div > div > div > div {
            background-color: #90cdf4;
        }
        .stSelectbox > div > div {
            background-color: #2d3748;
            border: 1px solid #4c51bf;
            border-radius: 10px;
            color: #a0aec0;
        }
        .stButton>button {
            background-color: #4c51bf;
            color: white;
            padding: 0.7em 2em;
            font-size: 16px;
            border-radius: 10px;
            transition: all 0.3s ease;
            border: none;
            display: block;
            margin: 0 auto;
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

# Header with title inside the top box
st.markdown("<div class='header'>📈 Income Prediction 💰</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill in the details below and click Predict! 🌟</div>", unsafe_allow_html=True)

# User Inputs
user_input = {}
user_input['age'] = st.slider("📅 Age", min_value=18, max_value=100, value=30)
user_input['education'] = st.selectbox("🎓 Education Level", encoders['education'].classes_)
user_input['occupation'] = st.selectbox("💼 Occupation", encoders['occupation'].classes_)
user_input['hours-per-week'] = st.slider("⏳ Hours Worked per Week", 1, 100, 40)
user_input['gender'] = st.selectbox("👤 Gender", encoders['gender'].classes_)

# Predict button
if st.button("🔍 Predict Income"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = encoders[col].transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Result
    st.markdown("---")
    st.markdown(
        f"<h3 style='color: #90cdf4; text-align: center;'>🎯 Predicted Income Class: <span style='color: #a0aec0'>{prediction}</span></h3>",
        unsafe_allow_html=True
    )

    st.balloons()

# Close main wrapper
st.markdown("</div>", unsafe_allow_html=True)