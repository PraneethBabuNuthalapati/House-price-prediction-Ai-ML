import joblib
import pandas as pd
import shap

PROPERTY_MODEL_PATH = "artifacts/model.pkl"
LOCATION_MODEL_PATH = "artifacts/real_model.pkl"
COLUMNS_PATH = "artifacts/columns.pkl"
GLOBAL_AVG_PATH = "artifacts/global_avg.pkl"

def load_models():
    property_model = joblib.load(PROPERTY_MODEL_PATH)
    location_model = joblib.load(LOCATION_MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    global_avg = joblib.load(GLOBAL_AVG_PATH)
    
    return property_model, location_model, columns, global_avg

def feature_engineering(df):
    df = df.copy()
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    return df

def align_input(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]

def explain_property(model, df):
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

def get_top_features(shap_values, X_df, top_n=5):
    values = shap_values.values[0]
    features = X_df.columns
    
    pairs = list(zip(features, values))
    pairs.sort(key = lambda x: abs(x[1]), reverse=True)
    
    return pairs[:top_n]

def create_property_input():
    return pd.DataFrame([{
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
    }])
    
def create_location_input():
    return pd.DataFrame([{
        "ZipCode": 10001,
        "State": "NY"
    }])
    
def predict():
    property_model, location_model, columns, global_avg = load_models()
    
    prop_df = create_property_input()
    prop_df = feature_engineering(prop_df)
    prop_df = align_input(prop_df, columns)
    
    property_price = property_model.predict(prop_df)[0]
    
    shap_values, X_df = explain_property(property_model, prop_df)
    top_features = get_top_features(shap_values, X_df)
    
    loc_df = create_location_input()
    location_price = location_model.predict(loc_df)[0]
    
    adjustment = location_price - global_avg
    final_price = property_price + adjustment
    
    print(f"\n🏠 Property Price: ${property_price:,.2f}")
    print(f"🌎 Location Avg Price: ${location_price:,.2f}")
    print(f"⚖️ Location Adjustment: ${adjustment:,.2f}")

    print(f"\n🔥 FINAL HYBRID PRICE: ${final_price:,.2f}")

    print("\n📊 WHY (Top Property Factors):")

    for feature, value in top_features:
        sign = "+" if value > 0 else "-"
        print(f"{sign} {feature} → {abs(value):,.2f}")
    
if __name__ == "__main__":
    predict()
    