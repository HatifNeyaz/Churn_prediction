# ==============================================================================
# Step 4: Interactive UI with Streamlit
# ==============================================================================
#
# This script creates a user-friendly web interface for the churn prediction model.
#
# It provides input fields for a customer's details and sends this data to the
# Flask API backend to get a prediction.
#
# To run this app:
#   1. Make sure the Flask API (`api.py`) is running in a separate terminal.
#   2. Open a new terminal and run the command: streamlit run app.py
#
# ==============================================================================

import streamlit as st
import requests
import json
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- App Title and Description ---
st.title("Telco Customer Churn Predictor 🔮")
st.write(
    "Enter the customer's details below to get a prediction on whether they "
    "are likely to churn. This app communicates with a backend Flask API "
    "that serves a trained PyTorch model."
)

# --- Helper function to get unique values for selectboxes ---
# This mimics the unique values from the training dataset.
def get_unique_values():
    return {
        'gender': ['Male', 'Female'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service'],
        'TechSupport': ['No', 'Yes', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service'],
        'StreamingMovies': ['No', 'Yes', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    }

unique_vals = get_unique_values()

# --- Create Input Fields in Two Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    gender = st.selectbox("Gender", unique_vals['gender'])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    partner = st.selectbox("Partner", unique_vals['Partner'])
    dependents = st.selectbox("Dependents", unique_vals['Dependents'])
    contract = st.selectbox("Contract Type", unique_vals['Contract'])
    payment_method = st.selectbox("Payment Method", unique_vals['PaymentMethod'])
    paperless_billing = st.selectbox("Paperless Billing", unique_vals['PaperlessBilling'])


with col2:
    st.subheader("Service & Usage")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=55.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1200.0)
    internet_service = st.selectbox("Internet Service", unique_vals['InternetService'])
    online_security = st.selectbox("Online Security", unique_vals['OnlineSecurity'])
    tech_support = st.selectbox("Tech Support", unique_vals['TechSupport'])


# --- Collect User Input into a Dictionary ---
input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "Yes",  # Assuming 'Yes' as default for simplicity in this UI
    "MultipleLines": "No", # Assuming 'No' as default
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": "No", # Assuming 'No' as default
    "DeviceProtection": "No", # Assuming 'No' as default
    "TechSupport": tech_support,
    "StreamingTV": "No", # Assuming 'No' as default
    "StreamingMovies": "No", # Assuming 'No' as default
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# --- Prediction Button and Logic ---
if st.button("Predict Churn", type="primary"):
    # API endpoint URL
    API_URL = "http://127.0.0.1:5000/predict"
    
    # Display the input data for verification
    with st.expander("Show Input Data"):
        st.json(input_data)
        
    # Send POST request to the API
    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        
        st.subheader("Prediction Result")
        prediction = result.get("prediction")
        probability = float(result.get("probability", 0))
        
        if prediction == "Churn":
            st.error(f"Prediction: {prediction}", icon="🚨")
        else:
            st.success(f"Prediction: {prediction}", icon="✅")
        
        # Display probability with a progress bar
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
        st.progress(probability)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Make sure the Flask API (`api.py`) is running. Error: {e}")

