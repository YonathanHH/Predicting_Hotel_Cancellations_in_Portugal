import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Function to load the pickled XGBoost model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Page configuration
st.set_page_config(page_title="Hotel Booking Cancellation Predictor")
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Predict whether a hotel booking will be cancelled using XGBoost")

# Load the final model
try:
    model = load_model('final_model.sav')
    st.success("✅ Model loaded successfully: final_model.sav")
except FileNotFoundError:
    st.error("❌ Error: 'final_model.sav' not found in the current directory!")
    st.stop()

st.subheader("📋 Enter Booking Information")

# Input features based on your unique values
col1, col2 = st.columns(2)

with col1:
    market_segment = st.selectbox(
        "Market Segment",
        ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation']
    )
    
    customer_type = st.selectbox(
        "Customer Type",
        ['Transient-Party', 'Transient', 'Contract', 'Group']
    )
    
    tourist_origin = st.selectbox(
        "Tourist Origin",
        ['International', 'Local']
    )
    
    booking_changes = st.selectbox(
        "Booking Changes",
        ['0', '1', '2+']
    )

with col2:
    previous_cancellations = st.selectbox(
        "Previous Cancellations",
        ['0', '1', '2+']
    )
    
    parking_spaces = st.selectbox(
        "Parking Spaces Required",
        ['0', '1', '2+']
    )
    
    special_requests = st.selectbox(
        "Special Requests",
        ['0', '1+']
    )
    
    waiting_list_days = st.selectbox(
        "Days in Waiting List",
        ['0', '1-30', '31-90', '>90']
    )

# Create input dataframe
user_input = pd.DataFrame({
    'Market Segment': [market_segment],
    'Customer type': [customer_type],
    'Tourist Origin': [tourist_origin],
    'Booking Changes': [booking_changes],
    'Previous Cancellations': [previous_cancellations],
    'Parking Spaces Requirement': [parking_spaces],
    'Special Requests': [special_requests],
    'Waiting List': [waiting_list_days]
})

# Prediction
if st.button("🔮 Predict Cancellation", key="predict_btn"):
    try:
        # Make prediction
        prediction = model.predict(user_input)[0]
        prediction_prob = model.predict_proba(user_input)[0]
        
        st.subheader("📊 Prediction Results")
        
        # Display cancellation probability
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cancellation Probability", f"{prediction_prob[1]:.2%}")
        with col2:
            st.metric("No Cancellation Probability", f"{prediction_prob[0]:.2%}")
        
        # Prediction outcome with visual emphasis
        st.divider()
        if prediction == 1:
            st.warning("⚠️ **Predicted: BOOKING WILL BE CANCELLED**")
        else:
            st.success("✅ **Predicted: BOOKING WILL NOT BE CANCELLED**")
        st.divider()
        
        # Display input summary
        st.subheader("📝 Booking Details Summary")
        st.dataframe(user_input, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("💡 Tip: Make sure your model is compatible with the input features. Check feature names and order match your training data.")