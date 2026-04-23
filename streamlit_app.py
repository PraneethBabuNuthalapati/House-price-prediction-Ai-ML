import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AI House Price Predictor", layout="wide")

st.title("AI House Price Predictor")
st.write("Hybrid ML + SHAP + LLM + LangGraph")

with st.form("house_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        zip_code = st.number_input("ZIP Code", min_value=0, step=1, value=10001)
        state = st.text_input("State", value="NY")
        sqft = st.number_input("Square Feet", min_value=100, step=50, value=1800)
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=3)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5, value=2.0)
        year_built = st.number_input("Year Built", min_value=1800, step=1, value=2005)
        quality = st.slider("Quality", min_value=1, max_value=10, value=7)
        query = st.text_input("Query", value="breakdown the price")

    submitted = st.form_submit_button("Predict")
    
if submitted:
    payload = {
        "zip_code": int(zip_code),
        "state": state,
        "sqft": int(sqft),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "year_built": int(year_built),
        "quality": int(quality),
    }
    
    try:
        response = requests.post(
            API_URL,
            params={"query": query},
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            st.error(f"API Error {response.status_code}: {response.text}")
        else:
            data = response.json()

            st.subheader("Prediction")
            st.metric("Final Price", f"${data['final_price']:,.2f}")

            if "top_factors" in data:
                st.subheader("Top Factors")
                for factor in data["top_factors"]:
                    feature = factor.get("feature", "")
                    impact = factor.get("impact", 0)
                    st.write(f"**{feature}**: {impact:,.2f}")

            if "explanation" in data:
                st.subheader("Explanation")
                st.write(data["explanation"])

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
            