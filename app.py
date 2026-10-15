import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="PND Stock-Out Predictor", layout="wide")

st.title("🛒 PND Multan — Smart Stock-Out Predictor")
st.markdown("### *AI-powered system that predicts stock-outs 7 days in advance*")

@st.cache_resource
def load_model():
    model = joblib.load("stockout_model.joblib")
    features = joblib.load("feature_columns.joblib")
    return model, features

try:
    model, feature_cols = load_model()
    st.success("✅ AI Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model not found: {e}")
    st.stop()

# Sidebar settings
st.sidebar.header("⚙️ Settings")
risk_threshold = st.sidebar.slider("High Risk Threshold", 0.3, 0.9, 0.6, 0.05)
med_threshold = st.sidebar.slider("Medium Risk Threshold", 0.1, 0.5, 0.3, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
st.sidebar.markdown("- Algorithm: XGBoost")
st.sidebar.markdown("- AUC-ROC: 0.934")
st.sidebar.markdown("- Features: 30")

st.header("📁 Step 1: Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} rows")
    st.dataframe(df.head())

    st.header("🎯 Step 2: Filters (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        if "warehouse_id" in df.columns:
            wh_filter = st.multiselect("Warehouse:", df["warehouse_id"].unique(),
                                       default=df["warehouse_id"].unique())
        else:
            wh_filter = None
    with col2:
        if "sku_category" in df.columns:
            cat_filter = st.multiselect("Category:", df["sku_category"].unique(),
                                        default=df["sku_category"].unique())
        else:
            cat_filter = None

    if st.button("🚀 Predict Stock-Out Risk", type="primary"):
        with st.spinner("🧠 AI is analyzing..."):
            df_proc = df.copy()

            if wh_filter and "warehouse_id" in df_proc.columns:
                df_proc = df_proc[df_proc["warehouse_id"].isin(wh_filter)]
            if cat_filter and "sku_category" in df_proc.columns:
                df_proc = df_proc[df_proc["sku_category"].isin(cat_filter)]

            if "days_of_stock" not in df_proc.columns:
                df_proc["days_of_stock"] = df_proc["current_stock"] / df_proc["avg_daily_sales_7d"].replace(0, 1)

            for col in ["sku_category", "warehouse_id"]:
                if col in df_proc.columns:
                    df_proc = pd.get_dummies(df_proc, columns=[col], drop_first=True)

            for col in feature_cols:
                if col not in df_proc.columns:
                    df_proc[col] = 0

            X = df_proc[feature_cols]
            proba = model.predict_proba(X)[:, 1]
            df_proc["risk_score"] = proba
            df_proc["risk_band"] = pd.cut(proba,
                bins=[0, med_threshold, risk_threshold, 1.0],
                labels=["🟢 Low", "🟡 Medium", "🔴 High"])

            st.header("📊 Step 3: Results")

            high = (df_proc["risk_band"] == "🔴 High").sum()
            med = (df_proc["risk_band"] == "🟡 Medium").sum()
            low = (df_proc["risk_band"] == "🟢 Low").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 HIGH RISK", high)
            col2.metric("🟡 MEDIUM RISK", med)
            col3.metric("🟢 LOW RISK", low)

            st.subheader("📈 Risk Distribution")
            risk_counts = df_proc["risk_band"].value_counts().reset_index()
            risk_counts.columns = ["Risk Band", "Count"]
            fig1 = px.bar(risk_counts, x="Risk Band", y="Count",
                          color="Risk Band",
                          color_discrete_map={"🔴 High": "#ff4b4b",
                                             "🟡 Medium": "#ffa500",
                                             "🟢 Low": "#00c853"})
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("🎯 Top 10 High-Risk SKUs")
            high_df = df_proc[df_proc["risk_band"] == "🔴 High"].sort_values("risk_score", ascending=False).head(10)
            if len(high_df) > 0:
                fig2 = px.bar(high_df, x="risk_score", y="sku_id",
                              orientation="h", color="risk_score",
                              color_continuous_scale="Reds")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🔴 High Risk SKUs — Order Now!")
            if high > 0:
                st.dataframe(high_df, use_container_width=True)

            st.subheader("🔍 Search any SKU")
            search_term = st.text_input("Type SKU ID:")
            if search_term:
                result = df_proc[df_proc["sku_id"].astype(str).str.contains(search_term, case=False)]
                if len(result) > 0:
                    st.dataframe(result[["sku_id", "current_stock", "risk_score", "risk_band"]])
                else:
                    st.warning("No SKU found")

            st.subheader("📥 Download Report")
            csv = df_proc.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full Report (CSV)", csv,
                              "stockout_predictions.csv", "text/csv")

            st.success("✅ Analysis complete!")
