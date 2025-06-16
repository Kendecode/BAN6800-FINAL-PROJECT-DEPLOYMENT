import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open("classer.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("📞 Customer Churn Prediction for Online Bookstore")
st.markdown("---")

st.markdown("""
    This application predicts whether a customer is likely to churn based on their
    demographic, usage, and subscription details.
    Fill in the customer details below to get a prediction.
""")

# --- Input Fields for Customer Data ---
st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    customer_id = st.number_input("CustomerID", min_value=0.0, max_value=5000.0, value=100.0, format="%.2f")
    age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)
    gender = st.slider("Gender", min_value=0, max_value=1, value=0, step=1)
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12, step=1)
    usage_frequency = st.slider("Usage Frequency (Times/Month)", min_value=0, max_value=50, value=15, step=1)

with col2:
    support_calls = st.slider("Support Calls (Last 6 Months)", min_value=0, max_value=20, value=2, step=1)
    payment_delay = st.slider("Payment Delay (Days in Last Month)", min_value=0, max_value=60, value=0, step=1)
    # subscription_type = st.selectbox("Subscription Type", ['Basic', 'Premium', 'VIP'])
    subscription_type =  st.slider("Subscription Type", min_value=0, max_value=2, value=0, step=1)
    # contract_length = st.selectbox("Contract Length", ['Monthly', 'Annually', 'Bi-Annually'])
    contract_length = st.slider("Contract Length", min_value=0, max_value=2, value=0, step=1)
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=5000.0, value=100.0, format="%.2f")
    # last_interaction_date = st.date_input("Last Interaction Date", value="today")
    # last_interaction_date = st.text_input("Last Interaction Date", placeholder="dd/mm/yyyy")
    last_interaction_date = st.slider("Last Interaction Date", min_value=0, max_value=100, value=0, step=1)

# --- Prediction Button ---
st.markdown("---")

if st.button("Predict Churn"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([[
        age, gender, tenure, usage_frequency,
        support_calls, payment_delay, subscription_type, contract_length,customer_id,total_spend,last_interaction_date
    ]], columns=[
        'CustomerID','Age', 'Gender', 'Tenure', 'Usage Frequency',
        'Support Calls', 'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'
    ])
    churn_probability = model.predict_proba(input_data)[:, 1][0]
    churn_prediction = model.predict(input_data)[0]
    
    
    st.subheader("Prediction Result:")
    if churn_prediction == 1:
        st.error(f"⚠️ **This customer is LIKELY TO CHURN.**")
    else:
        st.success(f"✅ **This customer is UNLIKELY TO CHURN.**")

    st.markdown(f"**Churn Probability:** `{churn_probability:.2%}`")
    st.markdown("---")
    st.info("""
        * **Likely to Churn:** A higher probability (closer to 100%) suggests a strong risk.
        * **Unlikely to Churn:** A lower probability (closer to 0%) suggests low risk.
        * *Note: This is a simulated model for demonstration purposes. Real-world models require extensive data and rigorous validation.*
    """)