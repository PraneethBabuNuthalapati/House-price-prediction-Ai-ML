from fastapi import FastAPI
import joblib
import pandas as pd

from app.schemas.request import HouseRequest
from app.services.mapper import map_to_model_input
from app.ml.predict_hybrid import feature_engineering, align_input
from app.ml.shap_explainer import get_shap_values, geta_top_features
from app.services.llm_explainer import generate_explaination
# from app.agents.price_agent import run_agent
from app.agents.run_agent import run_agent

app = FastAPI()

# # Load the pre-trained model
# property_model = joblib.load("artifacts/model.pkl")
# location_model = joblib.load("artifacts/real_model.pkl")
# columns = joblib.load("artifacts/columns.pkl")
# global_avg = joblib.load("artifacts/global_avg.pkl")


@app.post("/predict")
def predict_house(req: HouseRequest, query: str = "predict price"):
    result = run_agent(req, query)
    return result