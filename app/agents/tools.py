from app.services.mapper import map_to_model_input
from app.ml.predict_hybrid import feature_engineering, align_input
from app.ml.shap_explainer import get_shap_values, geta_top_features
from app.services.llm_explainer import generate_explaination

import joblib
import pandas as pd

property_model = joblib.load("artifacts/model.pkl")
location_model = joblib.load("artifacts/real_model.pkl")
columns = joblib.load("artifacts/columns.pkl")
global_avg = joblib.load("artifacts/global_avg.pkl")

def predict_price_tool(req):
    df = map_to_model_input(req)
    df = feature_engineering(df)
    df = align_input(df, columns)
    
    property_price = property_model.predict(df)[0]
    
    loc_df = pd.DataFrame([{
        "ZipCode": req.zip_code,
        "State": req.state
    }])
    
    location_price = location_model.predict(loc_df)[0]
    
    adjustment = location_price = global_avg
    final_price = property_price + adjustment
    
    return {
        "final_price": float(round(final_price , 2)),
        "df": df
    }
    
def explain_tool(df):
    shap_values, feature_names = get_shap_values(df)
    top_features = geta_top_features(shap_values, feature_names)
    
    explaination = generate_explaination(top_features)
    
    return {
        "top_factors": top_features,
        "explaination": explaination
    }