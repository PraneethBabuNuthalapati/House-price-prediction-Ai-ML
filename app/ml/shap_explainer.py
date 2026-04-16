import shap
import joblib

def get_shap_values(input_df):
    pipeline = joblib.load("artifacts/model.pkl")
    
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    X_transformed = preprocessor.transform(input_df)
    
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    return shap_values[0], feature_names

def geta_top_features(shap_values, feature_names, top_n=5):
    pairs = list(zip(feature_names, shap_values))
    
    pairs = sorted(pairs, key = lambda x: abs(x[1]), reverse=True)
    
    return [
        {"feature": f , "impact": float(v)}
        for f, v in pairs[:top_n]
    ]