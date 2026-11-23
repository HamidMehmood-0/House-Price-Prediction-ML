
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.house_price_model import (
    load_data,
    train_model,
    evaluate_model,
    predict_house_price,
)


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

st.title("House Price Prediction")
st.write(
    "Enter the property details below to estimate its selling price."
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    data = get_data()
    return train_model(data)


try:
    data = get_data()
    model, X_test, y_test = get_model()

    st.subheader("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sqft)",
            min_value=1,
            value=1500,
            step=100,
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            value=3,
            step=1,
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            value=2,
            step=1,
        )

        house_age = st.number_input(
            "House Age (years)",
            min_value=0,
            value=10,
            step=1,
        )

    if st.button("Predict House Price", type="primary"):
        predicted_price = predict_house_price(
            model,
            area,
            bedrooms,
            bathrooms,
            house_age,
        )

        st.subheader("Estimated House Price")
        st.success(f"${predicted_price:,.2f}")

    st.divider()

    st.subheader("Model Information")

    metrics = evaluate_model(model, X_test, y_test)

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{metrics['MAE']:,.2f} USD")
    col2.metric("RMSE", f"{metrics['RMSE']:,.2f} USD")
    col3.metric("R² Score", f"{metrics['R2 Score']:.4f}")
   
    

    st.caption(
        "This project uses a small synthetic dataset. "
        "Predictions are for demonstration purposes only."
    )

except FileNotFoundError:
    st.error(
        "Dataset not found. Please add house_prices.csv "
        "to the project's data folder."
    )

except Exception as error:
    st.error(f"Something went wrong: {error}")
