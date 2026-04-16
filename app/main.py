from fastapi import FastAPI
import joblib
import pandas as pd

from app.schemas.request import HouseRequest
from app.services.mapper import map_to_model_input
from app.ml.predict_hybrid import feature_engineering, align_input
from app.ml.shap_explainer import get_shap_values, geta_top_features

app = FastAPI()

# Load the pre-trained model
property_model = joblib.load("artifacts/model.pkl")
location_model = joblib.load("artifacts/real_model.pkl")
columns = joblib.load("artifacts/columns.pkl")
global_avg = joblib.load("artifacts/global_avg.pkl")


@app.post("/predict")
def predict_house(req: HouseRequest):
    df = map_to_model_input(req)
    df = feature_engineering(df)
    df = align_input(df, columns)
    
    property_price = property_model.predict(df)[0]
    shap_values, feature_names = get_shap_values(df)
    top_features = geta_top_features(shap_values, feature_names)
    
    loc_df = pd.DataFrame([{
        "ZipCode": req.zip_code,
        "State": req.state,
    }])
    
    location_price = location_model.predict(loc_df)[0]
    
    adjustment = location_price - global_avg
    final_price = property_price + adjustment
    
    return {
        "final_price": float(round(final_price, 2)),
        "property_price": float(round(property_price, 2)),
        "location_price": float(round(location_price, 2)),
        "adjustment": float(round(adjustment, 2)),
        "top_factors": top_features
    }
