# utils/preprocess.py
import pandas as pd
import os
from datetime import datetime

def log_prediction(input_data, prediction, log_path="logs/prediction_log.csv"):
    input_data["predicted_rain_intensity"] = prediction
    input_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([input_data])

    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.isfile(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)
