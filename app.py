import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Predict whether a hotel booking will be cancelled using XGBoost — and understand *why* with SHAP explanations.")

try:
    model = load_model('final_model.sav')
except FileNotFoundError:
    st.error("❌ Error: 'final_model.sav' not found in the current directory!")
    st.stop()

st.subheader("📋 Enter Booking Information")

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

if st.button("🔮 Predict Cancellation", key="predict_btn"):
    try:
        prediction = model.predict(user_input)[0]
        prediction_prob = model.predict_proba(user_input)[0]

        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cancellation Probability", f"{prediction_prob[1]:.2%}")
        with col2:
            st.metric("No Cancellation Probability", f"{prediction_prob[0]:.2%}")

        st.divider()
        if prediction == 1:
            st.warning("⚠️ **Predicted: BOOKING WILL BE CANCELLED**")
            st.info("💡 **Recommended Action:** Consider sending a personalised retention offer or flagging this booking for a follow-up call from the customer service team.")
        else:
            st.success("✅ **Predicted: BOOKING WILL NOT BE CANCELLED**")
            st.info("💡 **Recommended Action:** No immediate intervention needed. Standard booking confirmation process applies.")
        st.divider()

        # SHAP explanation
        st.subheader("🔍 Why this prediction? (SHAP Feature Contributions)")
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(user_input)
            fig, ax = plt.subplots(figsize=(8, 3))
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            st.pyplot(fig)
            plt.close()
            st.caption("Bars pushing right (red) increase cancellation probability; bars pushing left (blue) decrease it.")
        except Exception as shap_err:
            st.warning(f"SHAP explanation unavailable for this model type: {shap_err}")

    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("💡 Tip: Make sure your model is compatible with the input features. Check feature names and order match your training data.")

with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Algorithm:** XGBoost (Extreme Gradient Boosting)  
    **Training data:** Hotel booking demand dataset from Portugal  
    **Key metrics:** Accuracy 0.801 | Precision 0.750 | Recall 0.693 | F1 0.720 | ROC-AUC 0.876  
    **Model hyperparameters:** Learning Rate 0.27 | Max Depth 12 | n_estimators 185  
    **Source code & notebook:** [GitHub](https://github.com/YonathanHH/Predicting_Hotel_Cancellations_in_Portugal)
    """)
