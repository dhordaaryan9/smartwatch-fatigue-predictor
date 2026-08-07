import streamlit as st
import pandas as pd
import joblib

# Load the trained AI model
model = joblib.load('fatigue_classifier.pkl')

# Front-End UI
st.title("Smartwatch Fatigue Predictor ⌚")
st.write("Enter your health telemetry to predict your current nervous system state.")

# Portfolio Context Note for Recruiters
with st.expander("ℹ️ About this Architecture"):
    st.write("""
    **Simulated Portfolio Interface:** In a production mobile or wearable application, users do not manually input these values. 
    This dashboard serves as a testing interface for the machine learning model. In a real-world architecture:
    1. Raw sensor logs sync silently from the smartwatch via Bluetooth.
    2. A background data pipeline computes the 3-day rolling averages automatically.
    3. A backend API (such as FastAPI) handles the JSON payload and scores it against this Random Forest model in real time.
    """)

st.header("Today's Metrics")
col1, col2 = st.columns(2)
with col1:
    profile = st.selectbox("User Profile", ["Athlete", "Desk Worker", "Shift Worker"])
    sleep = st.slider("Sleep Hours (Last Night)", 0.0, 12.0, 7.0, step=0.5)
    steps = st.number_input("Steps (Today)", min_value=0, max_value=40000, value=8000)
with col2:
    hr = st.slider("Resting Heart Rate (BPM)", 40, 120, 65)
    hrv = st.slider("Heart Rate Variability (HRV)", 10, 120, 50)

st.header("Historical Context (Rolling Averages)")
col3, col4 = st.columns(2)
with col3:
    avg_sleep = st.slider("3-Day Avg Sleep", 0.0, 12.0, 7.0, step=0.5)
    avg_steps = st.number_input("3-Day Avg Steps", min_value=0, max_value=40000, value=8000)
with col4:
    avg_hrv = st.slider("3-Day Avg HRV", 10, 120, 50)

# Prediction Logic
if st.button("Predict Fatigue State"):
    user_data = pd.DataFrame({
        'Steps': [steps],
        'Sleep_Hours': [sleep],
        'Resting_HR': [hr],
        'HRV': [hrv],
        '3_Day_Avg_Sleep': [avg_sleep],
        '3_Day_Avg_HRV': [avg_hrv],
        '3_Day_Avg_Steps': [avg_steps],
        'Profile_Athlete': [1 if profile == 'Athlete' else 0],
        'Profile_Desk Worker': [1 if profile == 'Desk Worker' else 0],
        'Profile_Shift Worker': [1 if profile == 'Shift Worker' else 0]
    })
    
    prediction = model.predict(user_data)[0]
    
    st.subheader("AI Diagnosis:")
    if prediction == "Optimal":
        st.success("🟢 **Optimal** - You are well rested and ready to perform!")
    elif prediction == "Strained":
        st.warning("🟡 **Strained** - You are accumulating fatigue. Consider a light recovery day.")
    else:
        st.error("🔴 **Burnout** - Warning! Your nervous system is exhausted. Rest immediately.")