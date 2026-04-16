from pathlib import Path

import joblib
import pandas as pd
import shap

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.pkl"
DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"

def load_model():
    return joblib.load(MODEL_PATH)

def get_sample_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["SalePrice"]).iloc[:100]

def explain_global():
    pipeline = load_model()

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_sample = get_sample_data()

    X_transformed = preprocessor.transform(X_sample)
    
    # Convert sparse → dense
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed_df)

    shap.summary_plot(shap_values, X_transformed_df)
    
def explain_single():
    pipeline = load_model()

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_sample = get_sample_data().iloc[[0]]

    X_transformed = preprocessor.transform(X_sample)
    
    # Convert sparse → dense
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed_df)

    shap.plots.waterfall(shap_values[0])


if __name__ == "__main__":
    explain_global()
    explain_single()
