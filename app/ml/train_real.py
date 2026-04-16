import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

DATA_PATH = "data/raw/zhvi.csv"
MODEL_PATH = "artifacts/real_model.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH)
    
    df = df.rename(columns={
        "RegionID": "ZipCode",
        "StateName": "State"
    })
    
    date_cols = df.columns[5:]
    df["LatestPrice"] = df[date_cols].iloc[:, -1]
    
    df = df.dropna(subset=["LatestPrice"])
    
    return df[["ZipCode", "State", "LatestPrice"]]

def build_pipeline():
    categorical_features = ["ZipCode", "State"]
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_features)
    ])
    
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline

def train():
    df = load_data()
    
    X = df.drop(columns=["LatestPrice"])
    y = df["LatestPrice"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline()
    
    global_avg = y.mean()
    joblib.dump(global_avg, "artifacts/global_avg.pkl")
    
    print("🚀 Training real-data model...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    
    print(f"✅ Validation RMSE: {rmse:.2f}")
    print(f"✅ Validation R2: {r2:.2f}")
    
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    
    print("Real Model Saved!")
    
if __name__ == "__main__":
    train()