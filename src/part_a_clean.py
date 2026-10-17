"""Part A - audit the raw exports, clean them with explicit rules, store CSV + SQLite."""
import os
import sqlite3
import pandas as pd
import numpy as np

RAW, CLEAN = "data/raw", "data/clean"
os.makedirs(CLEAN, exist_ok=True)


def describe(name, df):
    print(f"\n=== {name}: shape={df.shape} ===")
    t = pd.DataFrame({"dtype": df.dtypes.astype(str),
                      "missing": df.isna().sum(),
                      "missing_pct": (df.isna().mean() * 100).round(2)})
    print(t.to_string())


# --------------------------------------------------------------------- AUDIT
cust_raw = pd.read_csv(f"{RAW}/customers.csv")
prod_raw = pd.read_csv(f"{RAW}/products.csv")
ord_raw = pd.read_csv(f"{RAW}/orders.csv")
tick_raw = pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False)
# string view of orders so that "N/A", "error" and blanks stay visible
ord_str = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)

print("######## AUDIT ########")
for n, d in [("customers", cust_raw), ("products", prod_raw), ("orders", ord_raw), ("tickets", tick_raw)]:
    describe(n, d)
print("\n(tickets read with keep_default_na=False, so empty texts are '' and not counted as missing; "
      f"empty ticket texts = {(tick_raw['text'].str.strip() == '').sum()})")

amt_num = pd.to_numeric(ord_str["order_amount"], errors="coerce")
deliv_num = pd.to_numeric(ord_str["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(cust_raw["age"], errors="coerce")
cust_str = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)
audit = {
    "raw orders.csv rows": len(ord_str),
    "exact duplicate rows": int(ord_str.duplicated().sum()),
    "rows with status pending": int((ord_str["status"] == "pending").sum()),
    "distinct order_id": int(ord_str["order_id"].nunique()),
    "non-numeric order_amount entries": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((deliv_num < 0).sum()),
    "blank delivery_minutes": int((ord_str["delivery_minutes"] == "").sum()),
    "customers: blank age": int((cust_str["age"] == "").sum()),
    "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "customers: distinct raw city spellings": int(cust_str["city"].nunique()),
    "customers: distinct non-blank gender spellings": int(cust_str.loc[cust_str["gender"] != "", "gender"].nunique()),
    "customers: blank gender": int((cust_str["gender"] == "").sum()),
}
print("\n--- audit figures ---")
for k, v in audit.items():
    print(f"{k}: {v}")

# ------------------------------------------------------------ CLEAN CUSTOMERS
print("\n######## CLEAN CUSTOMERS ########")
c = cust_raw.copy()
c["city"] = c["city"].str.strip().str.title()
print("(a) cities after strip+title:", c["city"].nunique())
c["gender"] = c["gender"].fillna("").astype(str).str.strip().str.lower()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female", "": "unknown"}
c["gender"] = c["gender"].map(gmap)
assert c["gender"].notna().all()
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan
n_missing_age = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])
print(f"(c) impossible ages set to missing: {n_bad_age}; total missing ages filled with city median: {n_missing_age}")
print("cities:", c["city"].value_counts().to_dict())
print("genders:", c["gender"].value_counts().to_dict())
print("missing ages:", int(c["age"].isna().sum()), "| age range:", c["age"].min(), "-", c["age"].max())

# ---------------------------------------------------------------- CLEAN ORDERS
print("\n######## CLEAN ORDERS ########")
o = ord_raw.copy()
log = []
o["order_ts"] = pd.to_datetime(o["order_ts"])
log.append(("1. order_ts -> datetime", len(o), "all rows converted"))
n0 = len(o)
o = o.drop_duplicates()
log.append(("2. drop exact duplicate rows", n0 - len(o), f"{n0} -> {len(o)} rows"))
n0 = len(o)
o = o.sort_values(["order_id", "order_ts"], kind="stable").drop_duplicates("order_id", keep="last")
log.append(("3. keep latest order_ts per order_id", n0 - len(o), f"{n0} -> {len(o)} rows"))
price = prod_raw.set_index("product_id")["price_inr"]
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
miss_amt = o["order_amount"].isna()
filled = (o["product_id"].map(price) * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(filled)
log.append(("4. unparseable order_amount refilled from price*qty*(1-disc)", int(miss_amt.sum()), "values filled"))
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = np.nan
log.append(("5a. negative delivery_minutes -> missing", int(neg.sum()), "values set to NaN"))
o = o.merge(prod_raw[["product_id", "category"]], on="product_id", how="left")
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].median()
need = comp & o["delivery_minutes"].isna()
cat_median = o["category"].map(med)
o.loc[need, "delivery_minutes"] = cat_median[need]
log.append(("5b. completed orders: missing delivery_minutes -> category median", int(need.sum()), "values filled"))
print("category medians used:", med.to_dict())
o = o.drop(columns="category").sort_values("order_id").reset_index(drop=True)
for step, n, note in log:
    print(f"{step}: {n} ({note})")

print("\nrows:", len(o), "| completed:", int((o.status == 'completed').sum()),
      "| cancelled:", int((o.status == 'cancelled').sum()), "| pending:", int((o.status == 'pending').sum()))
print("missing order_amount:", int(o.order_amount.isna().sum()))
print("missing delivery (completed):", int(o[o.status == 'completed'].delivery_minutes.isna().sum()),
      "| (cancelled):", int(o[o.status == 'cancelled'].delivery_minutes.isna().sum()))
print("completed order_amount sum: INR", round(o.loc[o.status == 'completed', 'order_amount'].sum(), 2))

# ------------------------------------------------------------------------ STORE
cust_out = c.copy(); cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
ord_out = o.copy(); ord_out["order_ts"] = ord_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
ord_out["delivery_minutes"] = ord_out["delivery_minutes"].astype("float64")
cust_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod_raw.to_csv(f"{CLEAN}/products.csv", index=False)
ord_out.to_csv(f"{CLEAN}/orders.csv", index=False)

db = "data/instamart.db"
if os.path.exists(db):
    os.remove(db)
con = sqlite3.connect(db)
cust_out.to_sql("customers", con, index=False)
prod_raw.to_sql("products", con, index=False)
ord_out.to_sql("orders", con, index=False)
con.commit(); con.close()
print("\nwrote data/clean/*.csv and", db)
