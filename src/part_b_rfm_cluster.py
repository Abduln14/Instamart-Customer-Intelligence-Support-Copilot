"""Part B tasks 1-3: RFM table, K-means (K=2..8), segment profile + names, PCA, customer_segments.csv."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)
SNAP = pd.Timestamp("2025-10-01 00:00:00")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = o[o.status == "completed"]

# ---- task 1: RFM
g = done.groupby("customer_id")
rfm = pd.DataFrame({"recency": (SNAP - g["order_ts"].max()).dt.days,
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum().round(2)}).reset_index()
print("customers:", len(rfm))
print(f"recency median {rfm.recency.median()} min {rfm.recency.min()} max {rfm.recency.max()}")
print(f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()}")
print(f"monetary mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}")

# ---- task 2: K-means sweep
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
tab = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
print("\n", tab.round(3).to_string(index=False))
FINAL_K = 4
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---- task 3: profile, names, premium
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\nsegment profile (K=4):\n", prof.to_string())
premium_id = prof["monetary"].idxmax()
others = prof.drop(index=premium_id)
loyal_id = others["frequency"].idxmax()                      # most frequent, recent, mid-spend
rest = others.drop(index=loyal_id)
lapsing_id = rest["recency"].idxmax()                        # longest since last order
casual_id = [i for i in rest.index if i != lapsing_id][0]
names = {premium_id: "Premium High-Spenders", loyal_id: "Loyal Regulars",
         casual_id: "Occasional Buyers", lapsing_id: "Lapsing / At-Risk"}
rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == premium_id
print("\nnames:", {int(k): v for k, v in names.items()})
print("premium segment id:", premium_id, "size:", int(prof.loc[premium_id, "size"]))

pca = PCA(n_components=2).fit(X)
print("\nPCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", pca.explained_variance_ratio_.sum().round(3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)
print("wrote outputs/customer_segments.csv", rfm.shape)
