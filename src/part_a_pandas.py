"""Part A - pandas cross-check of Q2 and exploratory statistics of completed order_amount."""
import sqlite3
import pandas as pd
import numpy as np

c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(
    p[["product_id", "category"]], on="product_id")

pt = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
pd.set_option("display.width", 200)
print(pt.round(2).to_string())
pandas_total = round(pt.loc["All", "All"], 2)
sql_total = pd.read_sql_query(
    "SELECT ROUND(SUM(order_amount),2) t FROM orders WHERE status='completed'", sqlite3.connect("data/instamart.db")).t[0]
print("pivot grand total:", pandas_total, "| SQL total:", sql_total, "| equal:", abs(pandas_total - sql_total) < 0.005)

g = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), n=("order_amount", "count"))
print("\nper-city sum / mean / count:\n", g.round(2).to_string())

x = done["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} median {x.median():.2f} skew {x.skew():.2f} Q1 {q1:.2f} Q3 {q3:.2f} IQR {iqr:.2f}")
print(f"lower fence {lo:.2f} (outliers below: {(x < lo).sum()}) | upper fence {hi:.2f} (outliers above: {(x > hi).sum()})")
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\ncorrelation matrix:\n", corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
top = pairs.abs().idxmax()
print("largest |r| pair:", top, round(pairs[top], 3))
print("order_amount vs quantity:", round(corr.loc["order_amount", "quantity"], 3))
