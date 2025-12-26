import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# --- Configuration ---
st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="centered")

# --- Load Models ---
@st.cache_resource
def load_artifacts():
    # Load model and scaler
    model_path = 'models/kmeans_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if not os.path.exists(model_path):
        st.error("Model not found! Run the notebook to generate .pkl files.")
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

cluster_descriptions = {
    0: {"name": "Older Standard Customer", "strategy": "Standard promotions."},
    1: {"name": "young Standard Customer", "strategy": "Standard Promotions via Social Media."},
    2: {"name": "Careless Spender", "strategy": "Impulse buy recommendations."},
    3: {"name": "High Income Savers", "strategy": "Loyalty programs & Discounts."},
    4: {"name": "Frugal / Budget", "strategy": "Clearance sales."},
    5: {"name": "Target Customer (Whale)", "strategy": "Tech trends &VIP Exclusive Offers & Luxury Ads."},
}

# --- UI Layout ---
st.title("🛍️ Marketing Segmenter")
st.markdown("Enter customer details below to predict their marketing segment.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    income = st.number_input("Yearly Income (k$)", min_value=10, max_value=200, value=60)

with col3:
    score = st.number_input("Purchase Spending (1-100)", min_value=1, max_value=100, value=50)

# --- Prediction Logic ---
if st.button("Predict Segment", type="primary"):
    if model and scaler:
        # 1. Prepare input
        feature_names = ['age', 'yearly income', 'purchase spending']
        input_df = pd.DataFrame([[age, income, score]], columns=feature_names)
        
        # 2. Scale input 
        scaled_inputs = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_inputs, columns=feature_names)
        # 3. Predict
        cluster_id = model.predict(scaled_df)[0]
        
        # 4. Display Results
        st.divider()
        st.subheader(f"Segment: {cluster_descriptions.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')}")
        
        st.info(f"Recommended Strategy: {cluster_descriptions.get(cluster_id, {}).get('strategy', 'N/A')}")
        
        st.metric(label="Predicted Cluster ID", value=str(cluster_id))