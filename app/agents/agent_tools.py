from langchain_core.tools import tool
from app.agents.tools import predict_price_tool, explain_tool

@tool
def predict_price(req_dict: dict):
    """
    Use this tool when the user wants to predict a house price.
    Input: property details like sqft, bedrooms, etc.
    Output: predicted price.
    """
    return predict_price_tool(req_dict)

@tool
def explain_price(df_dict: dict):
    """
    Use this tool when the user asks for explanation or reasoning behind the price.
    Returns SHAP factors and explanation.
    """
    return explain_tool(df_dict)