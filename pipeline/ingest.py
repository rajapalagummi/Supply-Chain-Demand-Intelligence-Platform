import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

SAMPLE_STORES = ["CA_1", "TX_1", "WI_1"]
SAMPLE_ITEMS_PER_STORE = 100
FORECAST_HORIZON = 28


def load_raw():
    sales = pd.read_csv(RAW / "sales_train_validation.csv")
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(RAW / "sell_prices.csv")
    return sales, calendar, prices


def melt_sales(sales, calendar, prices):
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    subset = sales[sales["store_id"].isin(SAMPLE_STORES)]
    sampled = []
    for store, group in subset.groupby("store_id"):
        sampled.append(group.sample(min(SAMPLE_ITEMS_PER_STORE, len(group)), random_state=42))
    items_per_store = pd.concat(sampled).reset_index(drop=True)

    melted = items_per_store[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    melted = melted.merge(calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
                                     "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]],
                          on="d", how="left")
    melted = melted.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    melted["date"] = pd.to_datetime(melted["date"])
    melted = melted.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return melted


def add_features(df):
    df = df.copy()
    df["sell_price"] = df["sell_price"].fillna(df.groupby("item_id")["sell_price"].transform("median"))
    df["has_event"] = df["event_name_1"].notna().astype(int)
    df["snap"] = df.apply(
        lambda r: r["snap_CA"] if r["state_id"] == "CA"
        else r["snap_TX"] if r["state_id"] == "TX"
        else r["snap_WI"], axis=1
    )
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales"].shift(lag)

    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby(["store_id", "item_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).std())
        )

    df["price_momentum"] = df.groupby(["store_id", "item_id"])["sell_price"].pct_change()
    df["zero_sales_flag"] = (df["sales"] == 0).astype(int)

    df = df.dropna(subset=["lag_28"]).reset_index(drop=True)
    return df


def load_pipeline():
    cache = PROCESSED / "features.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"Features loaded from cache: {len(df):,} rows")
        return df

    print("Building feature pipeline from raw data...")
    sales, calendar, prices = load_raw()
    print(f"Raw data: {len(sales):,} items x {len([c for c in sales.columns if c.startswith('d_')])} days")

    df = melt_sales(sales, calendar, prices)
    print(f"Melted: {len(df):,} rows")

    df = add_features(df)
    print(f"Features added: {len(df):,} rows, {df.shape[1]} columns")

    df.to_parquet(cache, index=False)
    print(f"Saved to {cache}")
    return df
