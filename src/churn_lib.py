"""
churn_lib.py
============
библиотека для проекта прогнозирования оттока (Olist E-commerce).

  * load_raw()           -- загрузка сырых данных: PostgreSQL (если доступен) или CSV (fallback)
  * load_from_postgres() -- чтение витрин напрямую из PostgreSQL
  * build_rfm_features() -- расчёт RFM-признаков и метки оттока (target)
  * split_data()         -- корректный train/test split
  * evaluate()           -- метрики + бизнес-стоимость
  * find_optimal_threshold() -- подбор порога по чистой прибыли
"""


from __future__ import annotations
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "k1_dataset")
DATABASE_URL = os.getenv("DATABASE_URL", "")


def _get_engine(database_url: str):
    from sqlalchemy import create_engine
    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect():
        pass
    return engine


def load_from_postgres(database_url: str | None = None) -> Dict[str, pd.DataFrame]:
    url = database_url or DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL не задан — PostgreSQL недоступен.")
    engine = _get_engine(url)
    tables = {
        "orders": "orders",
        "customers": "customers",
        "items": "order_items",
        "payments": "order_payments",
        "reviews": "order_reviews",
        "products": "products",
    }
    raw = {key: pd.read_sql(f"SELECT * FROM {tbl}", engine) for key, tbl in tables.items()}
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        if c in raw["orders"].columns:
            raw["orders"][c] = pd.to_datetime(raw["orders"][c], errors="coerce")
    for c in ["price", "freight_value"]:
        raw["items"][c] = pd.to_numeric(raw["items"][c], errors="coerce")
    for c in ["payment_value", "payment_installments"]:
        raw["payments"][c] = pd.to_numeric(raw["payments"][c], errors="coerce")
    raw["reviews"]["review_score"] = pd.to_numeric(raw["reviews"]["review_score"], errors="coerce")
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            translation = pd.read_sql(
                text("SELECT * FROM product_category_translation"), conn)
        translation.columns = [c.replace("\ufeff", "").strip() for c in translation.columns]
        raw["products"] = raw["products"].merge(
            translation, on="product_category_name", how="left")
    except Exception:
        pass

    return raw


def load_raw(data_dir: str = DATA_DIR, database_url: str | None = None,
             prefer_db: bool = True) -> Dict[str, pd.DataFrame]:
    url = database_url if database_url is not None else DATABASE_URL
    if prefer_db and url:
        try:
            raw = load_from_postgres(url)
            print(f"✅ Данные загружены из PostgreSQL: {url}")
            return raw
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  PostgreSQL недоступен ({e}); fallback на CSV из {data_dir}")
    
    def rd(name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(data_dir, name))

    orders = rd("olist_orders_dataset.csv")
    customers = rd("olist_customers_dataset.csv")
    items = rd("olist_order_items_dataset.csv")
    payments = rd("olist_order_payments_dataset.csv")
    reviews = rd("olist_order_reviews_dataset.csv")
    products = rd("olist_products_dataset.csv")
    translation = rd("product_category_name_translation.csv")
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    products = products.merge(translation, on="product_category_name", how="left")
    print(f"📄 Данные загружены из CSV: {data_dir}")
    return {
        "orders": orders,
        "customers": customers,
        "items": items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
    }


def build_rfm_features(
    raw: Dict[str, pd.DataFrame],
    snapshot_date: str | pd.Timestamp | None = None,
    churn_days: int = 180,
) -> pd.DataFrame:
    """
    Собирает клиентскую витрину (one row per customer_unique_id).

    Логика target:
        snapshot = максимальная дата покупки в датасете
        recency  = snapshot - last_purchase
        churn    = 1, если recency > churn_days  (клиент "уснул")
        churn    = 0 иначе
    """
    orders = raw["orders"]
    customers = raw["customers"]
    items = raw["items"]
    payments = raw["payments"]
    reviews = raw["reviews"]
    orders_ok = orders[orders["order_status"].isin(["delivered", "shipped", "invoiced", "processing"])].copy()
    items_agg = (
        items.groupby("order_id")
        .agg(n_items=("order_item_id", "count"),
             order_value=("price", "sum"),
             freight=("freight_value", "sum"))
        .reset_index()
    )
    pay_agg = (
        payments.groupby("order_id")
        .agg(payment_value=("payment_value", "sum"),
             n_installments=("payment_installments", "max"),
             payment_type=("payment_type", "first"))
        .reset_index()
    )
    rev_agg = reviews.groupby("order_id").agg(review_score=("review_score", "mean")).reset_index()
    o = (
        orders_ok.merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")
        .merge(items_agg, on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(rev_agg, on="order_id", how="left")
    )
    snapshot = pd.Timestamp(snapshot_date) if snapshot_date is not None else o["order_purchase_timestamp"].max()
    grp = o.groupby("customer_unique_id")
    cust = pd.DataFrame({
        "last_purchase": grp["order_purchase_timestamp"].max(),
        "first_purchase": grp["order_purchase_timestamp"].min(),
        "frequency": grp["order_id"].nunique(),
        "monetary": grp["payment_value"].sum(),
        "avg_order_value": grp["payment_value"].mean(),
        "total_items": grp["n_items"].sum(),
        "avg_freight": grp["freight"].mean(),
        "avg_review": grp["review_score"].mean(),
        "max_installments": grp["n_installments"].max(),
        "state": grp["customer_state"].first(),
    }).reset_index()
    cust["recency"] = (snapshot - cust["last_purchase"]).dt.days
    cust["tenure"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days
    cust["avg_days_between_orders"] = np.where(
        cust["frequency"] > 1, cust["tenure"] / (cust["frequency"] - 1), np.nan
    )
    cust["freight_ratio"] = cust["avg_freight"] / (cust["avg_order_value"] + 1e-6)
    cust["items_per_order"] = cust["total_items"] / cust["frequency"]
    cust["is_repeat"] = (cust["frequency"] > 1).astype(int)
    cust["churn"] = (cust["recency"] > churn_days).astype(int)
    cust = cust.dropna(subset=["recency", "monetary", "frequency"]).reset_index(drop=True)
    return cust


FEATURE_COLS = [
    "frequency", "monetary", "avg_order_value", "total_items",
    "avg_freight", "avg_review", "max_installments", "tenure",
    "avg_days_between_orders", "freight_ratio", "items_per_order", "is_repeat",
]


def split_data(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    from sklearn.model_selection import train_test_split
    X = df[FEATURE_COLS].copy()
    y = df["churn"].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def evaluate(model, X, y, threshold: float = 0.5,
             cost_offer: float = 5.0, clv: float = 120.0) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    saved = tp * (clv - cost_offer)
    wasted = fp * cost_offer
    lost = fn * clv
    baseline_lost = int(y.sum()) * clv
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "net_profit": saved - wasted - lost,
        "uplift_vs_do_nothing": (saved - wasted),
        "saved_clv": saved,
        "wasted_offer_cost": wasted,
    }


def find_optimal_threshold(model, X, y, cost_offer: float = 5.0, clv: float = 120.0):
    proba = model.predict_proba(X)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_p = 0.5, -np.inf
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        profit = tp * (clv - cost_offer) - fp * cost_offer - fn * clv
        if profit > best_p:
            best_p, best_t = profit, float(t)
    return best_t, best_p
