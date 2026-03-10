# Import streamlit for building the web app
import streamlit as st

# Import pandas for data handling
import pandas as pd

# Import joblib for loading saved files
import joblib

# Import os for checking whether files exist
import os

# Define file paths for the saved model and training columns
MODEL_PATH = "churn.pkl"
COLUMNS_PATH = "model_columns.pkl"

# Set the title of the app
st.title("☎️Telco Customer Churn Prediction")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload model/churn_model.pkl to your repository.")
    st.stop()

# Check if model columns file exists
if not os.path.exists(COLUMNS_PATH):
    st.error("Model columns file not found. Please upload model/model_columns.pkl to your repository.")
    st.stop()

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load the training column names
model_columns = joblib.load(COLUMNS_PATH)

# Create input widgets for raw customer details
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# Run prediction when the button is clicked
if st.button("Predict Churn"):

    # Create a dataframe from the raw user inputs
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Convert categorical variables into dummy variables
    input_encoded = pd.get_dummies(input_df)

    # Reindex the dataframe so it matches the training columns exactly
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Make probability prediction if supported
    prediction_proba = model.predict_proba(input_encoded)[0][1]

    # Display the result
    if prediction == 1:
        st.error(f"This customer is likely to churn. Probability: {prediction_proba:.2%}")
    else:

        st.success(f"This customer is likely to stay. Probability of churn: {prediction_proba:.2%}")

