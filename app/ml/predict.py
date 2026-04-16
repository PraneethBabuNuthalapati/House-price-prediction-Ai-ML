import joblib
import pandas as pd
import shap

MODEL_PATH = "artifacts/model.pkl"
COLUMNS_PATH = "artifacts/columns.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def create_sample_input():
    data = {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotArea": 8450,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "RoofStyle": "Gable",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "TotalBsmtSF": 856,
        "GrLivArea": 1710,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "GarageCars": 2,
        "GarageArea": 548,
        "YrSold": 2008,
        "SaleCondition": "Normal"
    }

    return pd.DataFrame([data])

def load_columns():
    return joblib.load(COLUMNS_PATH)

def align_input(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = None
            
    df = df[columns]
    
    return df

def feature_engineering(df):
    df = df.copy()
    
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    return df

def explain_prediction(model, df):
    xgb_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    
    X_transformed = preprocessor.transform(df)
    
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    feature_names = preprocessor.get_feature_names_out()
    
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_df)
    
    return shap_values, X_df

def get_top_contributions(shap_values, X_df, top_n=5):
    values = shap_values.values[0]
    features = X_df.columns
    
    contributions = list(zip(features, values))
    
    contributions.sort(key= lambda x: abs(x[1]), reverse=True)
    
    return contributions[:top_n]
    

def predict():
    model = load_model()
    columns = load_columns()
    
    df = create_sample_input()
    df = feature_engineering(df)
    
    df = align_input(df, columns)
    
    prediction = model.predict(df)
    
    print(f"🏠 Predicted Price: ${prediction[0]:,.2f}")
    
    shap_values, X_df = explain_prediction(model, df)
    top_features = get_top_contributions(shap_values, X_df)
    
    print("\n📊 Top contributing factors:")
    
    for feature, value in top_features:
        sign = "+" if value > 0 else "-"
        print(f"{sign} {feature} → {abs(value):.2f}")


if __name__ == "__main__":
    predict()