from __future__ import annotations
import os
import subprocess
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT, "k1_dataset")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:55432/olist")
TABLES = {
    "olist_customers_dataset.csv": "customers",
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "product_category_translation",
}


def _run_ddl_via_psql(ddl_path: str) -> bool:
    url = make_url(DATABASE_URL)
    env = dict(os.environ, PGPASSWORD=url.password or "")
    cmd = [
        "psql",
        "-h", url.host or "localhost",
        "-p", str(url.port or 5432),
        "-U", url.username or "postgres",
        "-d", url.database or "postgres",
        "-v", "ON_ERROR_STOP=1",
        "-f", ddl_path,
    ]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print("  [warn] psql вернул ошибку:\n", res.stderr[-800:])
            return False
        return True
    except FileNotFoundError:
        return False


def main() -> None:
    engine = create_engine(DATABASE_URL)
    print(f"Подключение к: {DATABASE_URL}")
    for csv_name, table in TABLES.items():
        path = os.path.join(DATA_DIR, csv_name)
        if not os.path.exists(path):
            print(f"  [skip] нет файла {csv_name}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10000)
        print(f"  [ok] {table:32s} <- {csv_name}  ({len(df):,} строк)".replace(",", " "))
    sql_path = os.path.join(HERE, "churn_datamart.sql")
    ok = _run_ddl_via_psql(sql_path)
    if ok:
        print("  [ok] созданы витрины v_orders_enriched, v_customer_features (via psql)")
    else:
        with engine.begin() as conn:
            conn.exec_driver_sql(open(sql_path, encoding="utf-8").read())
        print("  [ok] созданы витрины (via SQLAlchemy)")
    with engine.connect() as conn:
        n_orders = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
        n_cust = conn.execute(text("SELECT COUNT(*) FROM v_customer_features")).scalar()
        churn = conn.execute(text("SELECT AVG(churn) FROM v_customer_features")).scalar()
    print(f"\n✅ Готово. Заказов: {n_orders:,} | клиентов в витрине: {n_cust:,} "
          f"| churn rate: {float(churn):.1%}".replace(",", " "))


if __name__ == "__main__":
    main()
