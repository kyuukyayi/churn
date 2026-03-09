# Import streamlit for the web app
import streamlit as st

# Import pandas for handling tabular input data
import pandas as pd

# Import joblib for loading the compressed trained model
import joblib

# Load the trained churn model from the model folder
model = joblib.load("churn.pkl")

# App title
st.title("Customer Churn Prediction")

# User input example
tenure = st.slider("Tenure (months)", 0, 72)

monthly = st.number_input("Monthly Charges")

# Predict button
if st.button("Predict Churn"):
    
    # Example input dataframe
    input_data = pd.DataFrame({
        "tenure":[tenure],
        "MonthlyCharges":[monthly]
    })
    
    # Prediction
    prediction = model.predict(input_data)
    
    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")