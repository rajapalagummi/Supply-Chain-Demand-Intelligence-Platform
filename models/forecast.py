import pandas as pd
import numpy as np
import mlflow
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

OUTPUTS = Path("data/outputs")
IMAGES = Path("images")
OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(exist_ok=True)

FEATURE_COLS = [
    "sell_price", "has_event", "snap", "dayofweek", "dayofmonth",
    "weekofyear", "month", "quarter", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    "price_momentum", "zero_sales_flag",
]
CAT_COLS = ["cat_id", "dept_id", "store_id", "state_id"]
FORECAST_HORIZON = 28
VALIDATION_DAYS = 28


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save(fig, name):
    fig.write_html(str(OUTPUTS / f"{name}.html"))
    try:
        fig.write_image(str(IMAGES / f"{name}.png"), width=1200, height=600)
    except Exception:
        pass


def encode_categoricals(df):
    df = df.copy()
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def train_lgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return model


def walk_forward_backtest(df):
    print("\n=== Walk-Forward Backtest ===")

    df, encoders = encode_categoricals(df)
    feature_cols = FEATURE_COLS + [c + "_enc" for c in CAT_COLS if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    cutoff = df["date"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_df = df[df["date"] <= cutoff].dropna(subset=feature_cols + ["sales"])
    val_df = df[df["date"] > cutoff].dropna(subset=feature_cols + ["sales"])

    X_train = train_df[feature_cols].values
    y_train = train_df["sales"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["sales"].values

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    model = train_lgbm(X_train, y_train, X_val, y_val)
    preds = np.maximum(0, model.predict(X_val))

    val_mape = mape(y_val, preds)
    val_rmse = rmse(y_val, preds)
    val_mae = mean_absolute_error(y_val, preds)

    print(f"Validation MAPE: {val_mape:.2f}%")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")

    return model, feature_cols, val_df, preds, {
        "mape": round(val_mape, 4),
        "rmse": round(val_rmse, 4),
        "mae": round(val_mae, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_iteration": model.best_iteration_,
    }


def sarimax_baseline(df, n_items=20):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    print("\n=== SARIMAX Baseline ===")
    results = []
    items = df.groupby(["store_id", "item_id"])["sales"].sum().nlargest(n_items).index

    for store_id, item_id in items:
        ts = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].set_index("date")["sales"]
        ts = ts.resample("D").sum()
        if len(ts) < 60:
            continue
        cutoff = ts.index.max() - pd.Timedelta(days=VALIDATION_DAYS)
        train = ts[ts.index <= cutoff]
        val = ts[ts.index > cutoff]
        try:
            model = SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,7),
                           enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            pred = fit.forecast(steps=len(val))
            pred = np.maximum(0, pred.values)
            item_mape = mape(val.values, pred)
            if not np.isnan(item_mape):
                results.append(item_mape)
        except Exception:
            continue

    avg_mape = np.mean(results) if results else np.nan
    print(f"SARIMAX avg MAPE across {len(results)} items: {avg_mape:.2f}%")
    return round(avg_mape, 4)


def compute_shap(model, X, feature_cols, n_samples=500):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X[:n_samples])
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance = dict(sorted(
        zip(feature_cols, [round(float(v), 4) for v in mean_abs]),
        key=lambda x: x[1], reverse=True
    ))
    return importance


def plot_results(val_df, preds, shap_importance, metrics):
    val_df = val_df.copy()
    val_df["predicted"] = preds

    store_actual = val_df.groupby("date")["sales"].sum().reset_index()
    store_pred = val_df.groupby("date")["predicted"].sum().reset_index()
    merged = store_actual.merge(store_pred, on="date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["sales"],
                             name="Actual", line=dict(color="#2ecc71", width=2)))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["predicted"],
                             name="Predicted", line=dict(color="#e74c3c", width=2, dash="dash")))
    fig.update_layout(title=f"Walk-Forward Backtest: Actual vs Predicted — MAPE {metrics['mape']:.2f}%",
                      xaxis_title="Date", yaxis_title="Total Units Sold")
    save(fig, "model_actual_vs_predicted")

    shap_df = pd.DataFrame(list(shap_importance.items())[:15],
                            columns=["Feature", "SHAP"])
    fig = px.bar(shap_df, x="SHAP", y="Feature", orientation="h",
                 title="Top 15 SHAP Feature Importances (LightGBM Demand Forecast)",
                 labels={"SHAP": "Mean |SHAP Value|", "Feature": "Feature"},
                 color="SHAP", color_continuous_scale="Blues")
    save(fig, "model_shap_importance")

    metrics_df = pd.DataFrame([
        {"Metric": "MAPE (%)", "Value": metrics["mape"]},
        {"Metric": "RMSE", "Value": metrics["rmse"]},
        {"Metric": "MAE", "Value": metrics["mae"]},
    ])
    fig = px.bar(metrics_df, x="Metric", y="Value",
                 title="Model Evaluation Metrics",
                 color="Value", color_continuous_scale="Greens",
                 text="Value")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    save(fig, "model_metrics")


def run_models(df):
    mlflow.set_experiment("supply_chain_demand_forecast")

    with mlflow.start_run(run_name="lgbm_walk_forward"):
        model, feature_cols, val_df, preds, metrics = walk_forward_backtest(df)
        sarimax_mape = sarimax_baseline(df)

        X_val = val_df[feature_cols].values
        shap_importance = compute_shap(model, X_val, feature_cols)

        mlflow.log_metric("val_mape", metrics["mape"])
        mlflow.log_metric("val_rmse", metrics["rmse"])
        mlflow.log_metric("val_mae", metrics["mae"])
        mlflow.log_metric("sarimax_baseline_mape", sarimax_mape if sarimax_mape else 0)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("best_iteration", metrics["best_iteration"])

        mape_improvement = round(sarimax_mape - metrics["mape"], 2) if sarimax_mape else None

        plot_results(val_df, preds, shap_importance, metrics)

        print(f"\nSARIMAX baseline MAPE: {sarimax_mape:.2f}%")
        print(f"LightGBM MAPE: {metrics['mape']:.2f}%")
        if mape_improvement:
            print(f"MAPE improvement over SARIMAX: {mape_improvement:.2f}pp")

    return {
        "model": model,
        "metrics": metrics,
        "sarimax_mape": sarimax_mape,
        "mape_improvement": mape_improvement,
        "shap_importance": shap_importance,
        "feature_cols": feature_cols,
    }
