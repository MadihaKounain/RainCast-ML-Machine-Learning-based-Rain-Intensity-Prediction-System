# rain_predictor.py
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/rain_model.pkl")

# Predict function
def predict_rain_intensity(features: dict) -> float:
    df = pd.DataFrame([features])
    return model.predict(df)[0]
