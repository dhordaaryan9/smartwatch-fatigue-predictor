import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fatigue Predictor",
    page_icon="🔋", 
    layout="centered" 
)

st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F8F9FA !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Style the metric containers */
    div[data-testid="stForm"] {
        background-color: #1A1C23;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #333640;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4B4E55 0%, #2A2C31 100%);
        color: white;
        border: 1px solid #5A5D66;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        border-color: #8C8F99;
        box-shadow: 0 0 10px rgba(140, 143, 153, 0.4);
    }
</style>
""", unsafe_allow_html=True)

try:
    model = joblib.load('fatigue_classifier.pkl')
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.title("🔋 Smartwatch Fatigue Predictor")
st.markdown("### *Real-time IoT Telemetry Analysis*")

st.markdown("""
<div style="background-color: #1A1C23; border-left: 4px solid #4B4E55; border-radius: 8px; padding: 20px; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
    <h4 style="margin-top: 0; color: #A0A4B8; font-family: sans-serif;">ℹ️ Architecture & Data Pipeline Context</h4>
    <p style="font-size: 14.5px; line-height: 1.6; margin-bottom: 0;">
        <b>Simulated Portfolio Interface:</b> In a production wearable application, users do not manually input these values. This dashboard serves as a testing interface for the machine learning model. In a real-world architecture:<br>
        1. Raw sensor logs sync silently from the hardware via Bluetooth.<br>
        2. A background data pipeline computes the 3-day rolling averages automatically.<br>
        3. A backend API handles the JSON payload and scores it against this model in real time.
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("telemetry_form"):
    
    st.subheader("👤 Baseline Profile")
    profile = st.selectbox("Select User Behavioral Profile", ["Athlete", "Desk Worker", "Shift Worker"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Today's Telemetry")
        steps = st.number_input("Steps Today", min_value=0, max_value=40000, value=8000, step=500)
        sleep = st.slider("Sleep (Last Night)", 0.0, 12.0, 7.0, step=0.5)
        hr = st.slider("Resting HR (BPM)", 40, 120, 65)
        hrv = st.slider("HRV (ms)", 10, 120, 50)
        
    with col2:
        st.subheader("📊 3-Day Averages")
        avg_steps = st.number_input("Avg Steps (3-Day)", min_value=0, max_value=40000, value=8000, step=500)
        avg_sleep = st.slider("Avg Sleep (3-Day)", 0.0, 12.0, 7.0, step=0.5)
        st.markdown("<br>", unsafe_allow_html=True) # Spacing alignment
        avg_hrv = st.slider("Avg HRV (3-Day)", 10, 120, 50)

    st.markdown("---")
    submitted = st.form_submit_button("Predict Fatigue State ⚡")

if submitted:
    if not model_loaded:
        st.error("⚠️ **Model Not Found:** Please ensure 'fatigue_classifier.pkl' is in the same directory as this script.")
    else:
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
        
        st.markdown("### 🩺 Diagnostic Output:")
        if prediction == "Optimal":
            st.success("🟢 **SYSTEM STATUS: OPTIMAL** \n\nRecovery metrics are strong. You are well-rested and primed for peak performance.")
        elif prediction == "Strained":
            st.warning("🟡 **SYSTEM STATUS: STRAINED** \n\nFatigue accumulation detected. Telemetry indicates a drop in recovery efficiency. Recommend light activity.")
        else:
            st.error("🔴 **SYSTEM STATUS: BURNOUT** \n\nCritical warning. Nervous system exhaustion detected via HRV and sleep deficits. Mandatory rest required.")

st.markdown("---")
st.subheader("🧪 Validation Scenarios")
st.write("Input these exact parameters into the dashboard above to test the AI's edge-case handling:")

st.markdown("""
| Scenario | Profile | Sleep | Steps | HR | HRV | Avg Sleep (3d) | Avg Steps (3d) | Avg HRV (3d) | Expected Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. The Burnout Athlete** | Athlete | 4.0 | 25,000 | 85 | 20 | 4.5 | 22,000 | 25 | 🔴 **Burnout** |
| **2. The Rested Desk Worker** | Desk Worker | 8.0 | 7,000 | 60 | 65 | 7.5 | 7,500 | 60 | 🟢 **Optimal** |
""")