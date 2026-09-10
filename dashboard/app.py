import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

st.set_page_config(page_title="Supply Chain Demand Intelligence", layout="wide")
st.title("Supply Chain Demand Intelligence Platform")
st.caption("M5 Walmart Real Data · LightGBM Walk-Forward Forecast · Inventory Optimization · SHAP Attribution")

OUTPUTS = Path("data/outputs")
IMAGES = Path("images")

tab1, tab2, tab3, tab4 = st.tabs(["Sales Analytics", "Demand Forecast", "Inventory", "Model Performance"])

with tab1:
    st.subheader("Sales Analytics — Real M5 Walmart Data")
    plots = ["eda_daily_sales_trend", "eda_sales_by_category", "eda_sales_by_store",
             "eda_sales_by_dow", "eda_monthly_sales", "eda_event_impact",
             "eda_snap_impact", "eda_price_elasticity", "eda_correlation_matrix"]
    for plot in plots:
        path = OUTPUTS / f"{plot}.html"
        if path.exists():
            st.components.v1.html(path.read_text(), height=450, scrolling=False)

with tab2:
    st.subheader("Demand Forecast — LightGBM Walk-Forward Backtest")
    results_path = OUTPUTS / "final_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        metrics = results.get("model_metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LightGBM MAPE", f"{metrics.get('mape', 'N/A')}%")
        col2.metric("SARIMAX MAPE", f"{results.get('sarimax_mape', 'N/A')}%")
        col3.metric("MAPE Improvement", f"{results.get('mape_improvement', 'N/A')}pp")
        col4.metric("Validation MAE", f"{metrics.get('mae', 'N/A')}")

    for plot in ["model_actual_vs_predicted", "model_shap_importance"]:
        path = OUTPUTS / f"{plot}.html"
        if path.exists():
            st.components.v1.html(path.read_text(), height=500, scrolling=False)

with tab3:
    st.subheader("Inventory Optimization")
    inv_path = OUTPUTS / "inventory_metrics.csv"
    if inv_path.exists():
        inv_df = pd.read_csv(inv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Safety Stock", f"{inv_df['safety_stock'].mean():.1f} units")
        col2.metric("Avg Reorder Point", f"{inv_df['reorder_point'].mean():.1f} units")
        col3.metric("Avg EOQ", f"{inv_df['eoq'].mean():.1f} units")

        st.dataframe(inv_df.nlargest(20, "total_sales"), use_container_width=True)

    for plot in ["inventory_safety_stock", "inventory_risk_by_store"]:
        path = OUTPUTS / f"{plot}.html"
        if path.exists():
            st.components.v1.html(path.read_text(), height=450, scrolling=False)

with tab4:
    st.subheader("Model Performance Summary")
    results_path = OUTPUTS / "final_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        st.json(results)
    else:
        st.info("Run `python main.py` to generate results.")
