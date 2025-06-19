import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv(r"C:\Users\Madiha Kounain\Documents\rain_intensity_prediction\data\weather_data_advanced.csv")
X = data.drop("rain_intensity", axis=1)
y = data["rain_intensity"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

print("\n📊 Model Evaluation:")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name}")
    print(f"  MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"  MSE: {mean_squared_error(y_test, preds):.2f}")
    print(f"  R² Score: {r2_score(y_test, preds):.2f}")

# 🔍 Hyperparameter tuning for Random Forest
print("\n🔍 Tuning Random Forest with GridSearchCV...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
rf_preds = best_model.predict(X_test)

print("\n✅ Best Random Forest Results:")
print(f"  MAE: {mean_absolute_error(y_test, rf_preds):.2f}")
print(f"  MSE: {mean_squared_error(y_test, rf_preds):.2f}")
print(f"  R² Score: {r2_score(y_test, rf_preds):.2f}")
print(f"  Best Parameters: {grid_search.best_params_}")

# Save best Random Forest model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/rain_model.pkl")
print("💾 Model saved to models/rain_model.pkl")
