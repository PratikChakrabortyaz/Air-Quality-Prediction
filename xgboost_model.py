import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

# ============================
# XGBoost Model Definition
# ============================
def train_and_evaluate_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='gpu_hist',  
        gpu_id=0                 
    )

    print("Training XGBoost Model...")
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    end_time = time.time()

    # Evaluate Model
    y_pred = xgb_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Final Model | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")
