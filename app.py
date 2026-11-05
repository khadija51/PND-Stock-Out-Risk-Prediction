import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading
import time
import requests

# ==========================================
# 1. BACKEND API (For n8n Automation)
# ==========================================
api_app = FastAPI()

class PredictionRequest(BaseModel):
    current_stock: float
    avg_daily_sales_7d: float
    lead_time_days: float
    sales_std_7d: float

@api_app.post("/predict")
async def predict_stockout(data: PredictionRequest):
    try:
        model = joblib.load("stockout_model.joblib")
        feature_cols = joblib.load("feature_columns.joblib")
        
        input_df = pd.DataFrame([data.dict()])
        
        # Fill missing features with 0
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        proba = model.predict_proba(input_df[feature_cols])[:, 1][0]
        risk_level = "High" if proba > 0.6 else ("Medium" if proba > 0.3 else "Low")
        
        return {
            "status": "success",
            "sku": "SKU_FROM_N8N", 
            "risk_level": risk_level, 
            "probability": round(float(proba), 4)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# Start API in background
if 'api_started' not in st.session_state:
    threading.Thread(target=run_api, daemon=True).start()
    st.session_state['api_started'] = True


# ==========================================
# 2. STREAMLIT FRONTEND (Your App UI)
# ==========================================
st.set_page_config(page_title="PND Stock-Out Predictor", layout="wide")

# 🔒 PASSWORD PROTECTION
def check_password():
    def password_entered():
        if st.session_state["password"] == "Khadija2026": # <-- APNA PASSWORD YAHAN
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Private Application")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

st.title("🛒 PND Multan — Smart Stock-Out Predictor")
st.markdown("### *AI-powered system with n8n Automation Integration*")

@st.cache_resource
def load_model():
    return joblib.load("stockout_model.joblib"), joblib.load("feature_columns.joblib")

try:
    model, feature_cols = load_model()
    st.success("✅ AI Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model not found: {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Settings")
risk_threshold = st.sidebar.slider("High Risk Threshold", 0.3, 0.9, 0.6, 0.05)
med_threshold = st.sidebar.slider("Medium Risk Threshold", 0.1, 0.5, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**n8n Automation**")
st.sidebar.info("API Endpoint is running on `/predict`")

# File upload & prediction UI
st.header("📁 Step 1: Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows")
    
    # ... (Baqi UI code same rahega jo pehle tha)
    st.dataframe(df.head())
    if st.button("🚀 Predict Stock-Out Risk", type="primary"):
        st.success("✅ Analysis complete! (UI is working)")
