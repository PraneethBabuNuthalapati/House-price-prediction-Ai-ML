import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# -------------------------
# PATHS
# -------------------------
DATA_PATH = "data/raw/train.csv"
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

#LOAD DATA

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def feature_engineering(df):
    df = df.copy()
    
    #Age of House
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    
    #Total Area
    df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    
    return df

#PREPROCESSING

def build_preprocessor(df):
    target = "SalePrice"
    
    X = df.drop(columns=[target])
    y = df[target]
    
    joblib.dump(X.columns.tolist(), "artifacts/columns.pkl")
    
    numerical_features = X.select_dtypes(include=["int64","float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "string"]).columns
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numeric_transformer,numerical_features),
            ("cat",categorical_transformer,categorical_features),
        ]
    )
    
    return preprocessor, X, y

#TRAIN MODEL

def train():
    os.makedirs("artifacts", exist_ok=True)

    df = load_data()
    df = feature_engineering(df)
    
    preprocessor, X, y = build_preprocessor(df)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(random_state=42)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",model)
    ])
    
    param_dist = {
       "model__n_estimators": [200, 300, 400],
       "model__max_depth": [4,6,8],
       "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=1,
        random_state=42
    )
    
    print("🚀 Tuning model...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print(f"🔥 Best Params: {search.best_params_}")
    
    preds = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    
    print(f"✅ Validation RMSE: {rmse:.2f}")
    print(f"✅ R2 Score: {r2:.4f}")

    joblib.dump(best_model, MODEL_PATH)

    print(f"💾 Model saved at: {MODEL_PATH}")



if __name__ == "__main__":
    train()
