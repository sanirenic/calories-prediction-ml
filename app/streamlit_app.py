import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/calories_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Calories Burned Predictor", layout="centered")
st.title("🏋️‍♀️ Calories Burned Prediction App")
st.markdown("Enter your workout and body metrics to estimate how many calories you've burned 🔥")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100)
height = st.number_input("Height (in cm)", min_value=100, max_value=250)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200)
duration = st.number_input("Exercise Duration (in minutes)", min_value=1, max_value=300)
heart_rate = st.number_input("Average Heart Rate", min_value=60, max_value=200)
body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, step=0.1)

if st.button("Predict Calories Burned"):
    # Convert input
    gender_val = 1 if gender == "Male" else 0
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    user_id = 0  # dummy value

    # Match training column order exactly
    input_df = pd.DataFrame([[user_id, gender_val, age, height, weight,
                              duration, heart_rate, body_temp, height_m, bmi]],
        columns=["User_ID", "Gender", "Age", "Height", "Weight",
                 "Duration", "Heart_Rate", "Body_Temp", "Height_m", "BMI"]
    )

    # Scale and predict
    scaled_input = scaler.transform(input_df.values)  # use .values to avoid column name mismatch
    predicted_calories = model.predict(scaled_input)[0]

    st.success(f"🔥 Estimated Calories Burned: **{predicted_calories:.2f} kcal**")
