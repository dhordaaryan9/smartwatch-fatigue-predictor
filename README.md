# Smartwatch Fatigue Predictor ⌚

## Overview
This project is an end-to-end Machine Learning pipeline designed to predict human fatigue states (Optimal, Strained, Burnout) using smartwatch telemetry data. It demonstrates a complete Data Science lifecycle: from synthesizing organic behavioral data to cleaning time-series gaps, engineering historical features, and deploying a production-ready model via a Streamlit web application.

## Key Features & Methodologies

* **Organic Data Synthesis:** Generates a robust dataset simulating realistic user behavior across different profiles (Athlete, Desk Worker, Shift Worker), complete with injected sensor anomalies and daily macro-environmental "wobbles."
* **Advanced Time-Series Imputation:** Handles missing wearable data using group-isolated linear interpolation and backward filling, avoiding naive zero-filling which invalidates health metrics.
* **Feature Engineering:** Computes 3-day rolling averages for Sleep, Steps, and Heart Rate Variability (HRV) to give the algorithm "historical memory," acknowledging that physical and nervous system fatigue is cumulative.
* **Imbalanced Classification:** Trains a `RandomForestClassifier` with balanced class weights to prioritize recall on the minority "Burnout" class, intentionally trading precision to avoid dangerous false-negative health predictions.
* **Interactive Deployment:** Wraps the finalized `.pkl` model in a Streamlit dashboard to simulate a real-time data pipeline for stakeholder evaluation.

## Simulated Production Architecture
While the repository includes a manual input dashboard for testing, the underlying architecture is designed to simulate a real-world automated pipeline:
1. Raw sensor logs sync silently from a wearable device via Bluetooth.
2. A background data pipeline computes rolling averages automatically.
3. A backend API scores the JSON payload against the Random Forest model in real time to issue automated recovery alerts.

## Project Structure
* `generate_telemetry.py`: The data engineering engine that generates `smartwatch_telemetry_v3.csv` with injected missing values and sensor glitches.
* `EDA.ipynb`: The Jupyter Notebook handling data cleaning, EDA, interpolation, feature engineering, and categorical encoding. Outputs the final `model_ready_telemetry.csv`.
* `train_model.py`: The machine learning script that trains the Random Forest algorithm and exports the brain as `fatigue_classifier.pkl`.
* `app.py`: The Streamlit application providing the interactive front-end UI.
* `requirements.txt`: Dependencies required to run the environment.

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/smartwatch-fatigue-predictor.git](https://github.com/your-username/smartwatch-fatigue-predictor.git)
   cd smartwatch-fatigue-predictor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

   ## 🧪 Sample Test Cases
If you want to test the deployed Streamlit app, try entering these values to verify the model's accuracy:

| Scenario | Profile | Today's Metrics | 3-Day Averages | Expected Output |
| :--- | :--- | :--- | :--- | :--- |
| **Severe Burnout** | Athlete | Sleep: 4.0, Steps: 25k, HR: 85, HRV: 20 | Sleep: 4.5, Steps: 22k, HRV: 25 | 🔴 Burnout |
| **Healthy & Optimal** | Desk Worker | Sleep: 8.0, Steps: 7k, HR: 60, HRV: 65 | Sleep: 7.5, Steps: 7.5k, HRV: 60 | 🟢 Optimal |

## Author
**Dhorda Aryan**
* Data Science & AI/ML Project
