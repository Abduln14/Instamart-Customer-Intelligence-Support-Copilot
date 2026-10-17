"""Validate Part A reference checkpoints. Prints exactly ALL CHECKPOINTS PASSED when everything holds."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
done = o[o.status == "completed"]
canc = o[o.status == "cancelled"]

checks = {
    "orders rows == 12309": len(o) == 12309,
    "one row per order_id": o.order_id.is_unique,
    "completed == 11110": len(done) == 11110,
    "cancelled == 1199": len(canc) == 1199,
    "no pending rows": (o.status == "pending").sum() == 0,
    "no missing order_amount": o.order_amount.isna().sum() == 0,
    "no missing delivery (completed)": done.delivery_minutes.isna().sum() == 0,
    "1199 missing delivery (cancelled)": canc.delivery_minutes.isna().sum() == 1199,
    "completed revenue == 2998668.80": round(done.order_amount.sum(), 2) == 2998668.80,
    "no negative delivery": (o.delivery_minutes.dropna() < 0).sum() == 0,
    "6 cities": c.city.nunique() == 6,
    "city counts": c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages, ages 18-60": c.age.isna().sum() == 0 and c.age.min() >= 18 and c.age.max() <= 60,
    "40 products": len(p) == 40,
    "order_ts stored as text YYYY-MM-DD HH:MM:SS": o.order_ts.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all(),
}
failed = [k for k, v in checks.items() if not v]
for k in failed:
    print("FAIL " + k)
assert not failed, f"failed: {failed}"
print("ALL CHECKPOINTS PASSED")
