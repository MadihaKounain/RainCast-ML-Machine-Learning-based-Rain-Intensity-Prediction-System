import streamlit as st
import pandas as pd
import os
from rain_predictor import predict_rain_intensity
from utils.preprocess import log_prediction
from utils.summary import generate_weather_summary

# Streamlit page config
st.set_page_config(page_title="Rain Intensity Predictor", layout="centered")
st.title("🌧️ Rain Intensity Prediction")

st.markdown("Enter current weather parameters below to get a rain intensity prediction.")

# Form for user input
with st.form("rain_form"):
    temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, 25.0)
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 70.0)
    pressure = st.number_input("🔽 Pressure (hPa)", 900.0, 1100.0, 1012.0)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", 0.0, 100.0, 10.0)
    rain_last_hour = st.number_input("🌧️ Rain in Last Hour (mm)", 0.0, 50.0, 0.0)
    time_of_day = st.slider("⏰ Hour of the Day", 0, 23, 12)
    submit = st.form_submit_button("Predict Rain Intensity")

# On submit: run prediction
if submit:
    input_data = {
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "rain_last_hour": rain_last_hour,
        "time_of_day": time_of_day
    }

    # Predict using trained model
    prediction = predict_rain_intensity(input_data)

    # Threshold low predictions to 0 (i.e., no rain)
    if prediction < 0.5:
        prediction = 0.0

    # Log the prediction
    log_prediction(input_data, prediction)

    # Show results
    st.success(f"💧 Predicted Rain Intensity: {prediction:.2f} mm/hr")
    st.info(generate_weather_summary(prediction))

    # Show and download log if available
    if os.path.exists("logs/prediction_log.csv"):
        df = pd.read_csv("logs/prediction_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.subheader("📊 Recent Predictions")
        st.dataframe(df.tail(10), use_container_width=True)

        st.line_chart(df.tail(20).set_index("timestamp")["predicted_rain_intensity"])

        # Download button for CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction Log", csv, "prediction_log.csv", "text/csv")
