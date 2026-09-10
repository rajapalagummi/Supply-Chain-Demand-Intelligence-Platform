import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

IMAGES = Path("images")
OUTPUTS = Path("data/outputs")
IMAGES.mkdir(exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.write_html(str(OUTPUTS / f"{name}.html"))
    try:
        fig.write_image(str(IMAGES / f"{name}.png"), width=1200, height=600)
    except Exception:
        pass


def run_eda(df):
    print("\n=== EDA ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Items: {df['item_id'].nunique():,}")
    print(f"Stores: {df['store_id'].nunique()}")
    print(f"Categories: {df['cat_id'].unique()}")
    print(f"\nNumerical summary:")
    print(df[["sales", "sell_price", "lag_7", "rolling_mean_7"]].describe().round(3).to_string())
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print(f"\nZero sales: {(df['sales']==0).mean()*100:.1f}%")
    print(f"Sales skewness: {df['sales'].skew():.3f}")
    print(f"Sales kurtosis: {df['sales'].kurtosis():.3f}")

    for col in ["sales", "sell_price", "lag_7", "rolling_mean_7", "rolling_mean_28"]:
        s = df[col].dropna()
        print(f"\n{col}: mean={s.mean():.3f}, median={s.median():.3f}, std={s.std():.3f}, "
              f"min={s.min():.3f}, max={s.max():.3f}, skew={s.skew():.3f}")

    daily = df.groupby("date")["sales"].sum().reset_index()
    fig = px.line(daily, x="date", y="sales",
                  title="Total Daily Sales — M5 Real Walmart Data (3 States, 3 Stores)",
                  labels={"date": "Date", "sales": "Total Units Sold"})
    save(fig, "eda_daily_sales_trend")

    cat_daily = df.groupby(["date", "cat_id"])["sales"].sum().reset_index()
    fig = px.line(cat_daily, x="date", y="sales", color="cat_id",
                  title="Daily Sales by Category",
                  labels={"date": "Date", "sales": "Units Sold", "cat_id": "Category"})
    save(fig, "eda_sales_by_category")

    store_daily = df.groupby(["date", "store_id"])["sales"].sum().reset_index()
    fig = px.line(store_daily, x="date", y="sales", color="store_id",
                  title="Daily Sales by Store",
                  labels={"date": "Date", "sales": "Units Sold", "store_id": "Store"})
    save(fig, "eda_sales_by_store")

    fig = px.histogram(df[df["sales"] > 0], x="sales",
                       title="Sales Distribution (Excluding Zeros) — Log Scale",
                       nbins=50, log_y=True,
                       labels={"sales": "Units Sold", "count": "Frequency"})
    save(fig, "eda_sales_distribution")

    dow = df.groupby("dayofweek")["sales"].mean().reset_index()
    dow["day_name"] = dow["dayofweek"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
    fig = px.bar(dow, x="day_name", y="sales",
                 title="Average Sales by Day of Week",
                 labels={"day_name": "Day", "sales": "Avg Units Sold"})
    save(fig, "eda_sales_by_dow")

    monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
    monthly["year_month"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    fig = px.bar(monthly, x="year_month", y="sales",
                 title="Monthly Total Sales (2011-2016)",
                 labels={"year_month": "Month", "sales": "Total Units Sold"})
    fig.update_xaxes(tickangle=45)
    save(fig, "eda_monthly_sales")

    event_sales = df.groupby("has_event")["sales"].mean().reset_index()
    event_sales["label"] = event_sales["has_event"].map({0: "No Event", 1: "Event Day"})
    fig = px.bar(event_sales, x="label", y="sales",
                 title="Average Sales: Event vs Non-Event Days",
                 labels={"label": "Day Type", "sales": "Avg Units Sold"},
                 color="sales", color_continuous_scale="Blues")
    save(fig, "eda_event_impact")

    snap_sales = df.groupby("snap")["sales"].mean().reset_index()
    snap_sales["label"] = snap_sales["snap"].map({0: "No SNAP", 1: "SNAP Day"})
    fig = px.bar(snap_sales, x="label", y="sales",
                 title="Average Sales: SNAP vs Non-SNAP Days",
                 labels={"label": "Day Type", "sales": "Avg Units Sold"},
                 color="sales", color_continuous_scale="Greens")
    save(fig, "eda_snap_impact")

    price_sales = df[df["sell_price"].notna()].copy()
    price_sales["price_bin"] = pd.qcut(price_sales["sell_price"], q=10, labels=False)
    price_agg = price_sales.groupby("price_bin")["sales"].mean().reset_index()
    fig = px.scatter(price_agg, x="price_bin", y="sales",
                     title="Price Decile vs Average Sales (Price Elasticity)",
                     labels={"price_bin": "Price Decile (0=Cheapest)", "sales": "Avg Units Sold"},
                     trendline="ols")
    save(fig, "eda_price_elasticity")

    num_cols = ["sales", "sell_price", "lag_7", "lag_14", "rolling_mean_7",
                "rolling_mean_28", "dayofweek", "month"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r", text_auto=".2f")
    save(fig, "eda_correlation_matrix")

    zero_by_cat = df.groupby("cat_id").apply(lambda x: (x["sales"]==0).mean()*100).reset_index()
    zero_by_cat.columns = ["cat_id", "zero_pct"]
    fig = px.bar(zero_by_cat, x="cat_id", y="zero_pct",
                 title="Zero Sales Rate by Category (%)",
                 labels={"cat_id": "Category", "zero_pct": "Zero Sales %"},
                 color="zero_pct", color_continuous_scale="Reds")
    save(fig, "eda_zero_sales_by_category")

    top_items = df.groupby("item_id")["sales"].sum().nlargest(20).reset_index()
    fig = px.bar(top_items, x="item_id", y="sales",
                 title="Top 20 Items by Total Sales",
                 labels={"item_id": "Item", "sales": "Total Units Sold"})
    fig.update_xaxes(tickangle=45)
    save(fig, "eda_top_items")

    print("\nEDA complete — all visualizations saved")
    return df
