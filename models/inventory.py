import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

OUTPUTS = Path("data/outputs")
IMAGES = Path("images")


def save(fig, name):
    fig.write_html(str(OUTPUTS / f"{name}.html"))
    try:
        fig.write_image(str(IMAGES / f"{name}.png"), width=1200, height=600)
    except Exception:
        pass


def compute_safety_stock(demand_series, lead_time_days=7, service_level=0.95):
    from scipy import stats
    z = stats.norm.ppf(service_level)
    std_demand = demand_series.std()
    safety_stock = z * std_demand * np.sqrt(lead_time_days)
    return max(0, round(safety_stock, 2))


def compute_reorder_point(demand_series, lead_time_days=7, service_level=0.95):
    avg_demand = demand_series.mean()
    safety_stock = compute_safety_stock(demand_series, lead_time_days, service_level)
    return round(avg_demand * lead_time_days + safety_stock, 2)


def compute_eoq(annual_demand, order_cost=50, holding_cost_pct=0.25, unit_price=10):
    holding_cost = holding_cost_pct * unit_price
    if holding_cost <= 0 or annual_demand <= 0:
        return 0
    return round(np.sqrt(2 * annual_demand * order_cost / holding_cost), 2)


def run_inventory_optimization(df, model_results):
    print("\n=== Inventory Optimization ===")

    item_stats = df.groupby(["store_id", "item_id"]).agg(
        avg_daily_demand=("sales", "mean"),
        std_daily_demand=("sales", "std"),
        total_sales=("sales", "sum"),
        zero_rate=("sales", lambda x: (x == 0).mean()),
        avg_price=("sell_price", "mean"),
    ).reset_index()

    item_stats["safety_stock"] = item_stats.apply(
        lambda r: compute_safety_stock(
            df[(df["store_id"] == r["store_id"]) & (df["item_id"] == r["item_id"])]["sales"]
        ), axis=1
    )
    item_stats["reorder_point"] = item_stats.apply(
        lambda r: compute_reorder_point(
            df[(df["store_id"] == r["store_id"]) & (df["item_id"] == r["item_id"])]["sales"]
        ), axis=1
    )
    item_stats["annual_demand"] = item_stats["avg_daily_demand"] * 365
    item_stats["eoq"] = item_stats.apply(
        lambda r: compute_eoq(r["annual_demand"], unit_price=max(r["avg_price"], 1)), axis=1
    )

    item_stats["risk_tier"] = pd.qcut(
        item_stats["zero_rate"], q=3, labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    print(f"\nInventory metrics for {len(item_stats):,} item-store combinations:")
    print(f"  Avg safety stock: {item_stats['safety_stock'].mean():.2f} units")
    print(f"  Avg reorder point: {item_stats['reorder_point'].mean():.2f} units")
    print(f"  Avg EOQ: {item_stats['eoq'].mean():.2f} units")
    print(f"  High risk items (>67% zero rate): {(item_stats['risk_tier'] == 'High Risk').sum()}")

    fig = px.scatter(item_stats, x="avg_daily_demand", y="safety_stock",
                     color="risk_tier", size="total_sales",
                     title="Safety Stock vs Average Daily Demand by Risk Tier",
                     labels={"avg_daily_demand": "Avg Daily Demand", "safety_stock": "Safety Stock (units)"})
    save(fig, "inventory_safety_stock")

    fig = px.histogram(item_stats, x="reorder_point", color="risk_tier",
                       title="Reorder Point Distribution by Risk Tier",
                       nbins=30, barmode="overlay", opacity=0.7,
                       labels={"reorder_point": "Reorder Point (units)"})
    save(fig, "inventory_reorder_distribution")

    store_risk = item_stats.groupby(["store_id", "risk_tier"]).size().reset_index(name="count")
    fig = px.bar(store_risk, x="store_id", y="count", color="risk_tier",
                 title="Inventory Risk Tier Distribution by Store",
                 labels={"store_id": "Store", "count": "Number of Items"},
                 barmode="stack")
    save(fig, "inventory_risk_by_store")

    top_eoq = item_stats.nlargest(20, "eoq")
    fig = px.bar(top_eoq, x="item_id", y="eoq",
                 title="Top 20 Items by Economic Order Quantity",
                 labels={"item_id": "Item", "eoq": "EOQ (units)"},
                 color="eoq", color_continuous_scale="Blues")
    fig.update_xaxes(tickangle=45)
    save(fig, "inventory_top_eoq")

    item_stats.to_csv(str(OUTPUTS / "inventory_metrics.csv"), index=False)
    print("Inventory metrics saved")

    return item_stats
