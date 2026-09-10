import json
import mlflow
from pathlib import Path

from pipeline.ingest import load_pipeline
from eda.eda import run_eda
from models.forecast import run_models
from models.inventory import run_inventory_optimization

Path("data/outputs").mkdir(parents=True, exist_ok=True)
Path("images").mkdir(exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns.db")

print("=== Supply Chain Demand Intelligence Platform ===")
print("Real M5 Walmart Data · LightGBM Walk-Forward · SARIMAX Baseline · Inventory Optimization\n")

print("[1/4] Loading and engineering features...")
df = load_pipeline()
print(f"Dataset: {len(df):,} rows, {df.shape[1]} features")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Items: {df['item_id'].nunique():,} | Stores: {df['store_id'].nunique()}")

print("\n[2/4] Running EDA...")
run_eda(df)

print("\n[3/4] Training models and walk-forward backtest...")
model_results = run_models(df)

print("\n[4/4] Inventory optimization...")
inventory_df = run_inventory_optimization(df, model_results)

results = {
    "dataset": {
        "total_rows": len(df),
        "unique_items": int(df["item_id"].nunique()),
        "unique_stores": int(df["store_id"].nunique()),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "zero_sales_pct": round(float((df["sales"] == 0).mean() * 100), 2),
    },
    "model_metrics": model_results["metrics"],
    "sarimax_mape": model_results["sarimax_mape"],
    "mape_improvement": model_results["mape_improvement"],
    "top_shap_features": dict(list(model_results["shap_importance"].items())[:10]),
    "inventory": {
        "avg_safety_stock": round(float(inventory_df["safety_stock"].mean()), 2),
        "avg_reorder_point": round(float(inventory_df["reorder_point"].mean()), 2),
        "avg_eoq": round(float(inventory_df["eoq"].mean()), 2),
        "high_risk_items": int((inventory_df["risk_tier"] == "High Risk").sum()),
    },
}

with open("data/outputs/final_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("=== FINAL NUMBERS ===")
print("=" * 60)
print(f"Dataset: {results['dataset']['total_rows']:,} rows | {results['dataset']['unique_items']} items | {results['dataset']['unique_stores']} stores")
print(f"Zero sales rate: {results['dataset']['zero_sales_pct']}%")
print(f"LightGBM MAPE: {results['model_metrics']['mape']}%")
print(f"LightGBM RMSE: {results['model_metrics']['rmse']}")
print(f"LightGBM MAE: {results['model_metrics']['mae']}")
print(f"SARIMAX baseline MAPE: {results['sarimax_mape']}%")
print(f"MAPE improvement over SARIMAX: {results['mape_improvement']}pp")
print(f"Avg safety stock: {results['inventory']['avg_safety_stock']} units")
print(f"Avg reorder point: {results['inventory']['avg_reorder_point']} units")
print(f"Avg EOQ: {results['inventory']['avg_eoq']} units")
print(f"High risk items: {results['inventory']['high_risk_items']}")
print("\nRun dashboard: streamlit run dashboard/app.py")
