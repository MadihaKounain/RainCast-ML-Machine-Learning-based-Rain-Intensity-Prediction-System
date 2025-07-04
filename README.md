<pre> # 🌧️ RainCast ML RainCast ML is a machine learning-powered web application designed to predict **rain intensity (in mm/hr)** using real-time weather parameters. The system also interprets the rain category (No Rain, Light, Moderate, Heavy) and offers batch prediction for weather datasets.
## 🚀 Features 
  -Real-time prediction using regression models 
  - Rain category classification 
  - Batch CSV upload for multiple predictions 
  - Prediction log with downloadable history 
  - Alerts for high-intensity rain 
  - Interactive Streamlit interface 
  - Easily extensible to live weather APIs and IoT 
  
## 📊 Input Features 
  - Temperature (°C) 
  - Humidity (%) 
  - Pressure (hPa) 
  - Wind Speed (km/h) 
  - Rain in Last Hour (mm) 
  - Time of Day (Hour) 

  
## 🧠 Machine Learning Models 
  - Linear Regression 
  - Decision Tree 
  - Random Forest (final model with hyperparameter tuning) 
  - Gradient Boosting  

  
## 🖥️ Tech Stack 
  - Python 
  - scikit-learn 
  - pandas, numpy 
  - Streamlit (for the web interface) 
  - joblib (for model persistence) 
  
  ## 📁 Folder Structure 
  ``` raincast_ml/ 
  ├── app.py 
  ├── train_model.py 
  ├── rain_predictor.py 
  ├── data/│
           └── weather_data_with_zero_rain.csv 
  ├── models/ │
              └── rain_model.pkl 
  ├── logs/ │ 
            └── prediction_log.csv 
  └── utils/ 
              ├── preprocess.py 
              └── summary.py 
  
  
  
  ``` ## 📦 Setup Instructions 
  1. **Clone the repository** 
  2. **Install requirements
        ** ``` pip install -r requirements.txt ``` 
  3. **Train the model** 
        ``` python train_model.py ```
  4. **Run the web app** 
        ``` streamlit run app.py ```
  
  ## 
  
  📍 Author 
          Madiha Kounain
          Department of Computer Science and Engineering Project: RainCast ML

</pre>
