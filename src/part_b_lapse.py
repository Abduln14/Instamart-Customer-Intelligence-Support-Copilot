"""Part B tasks 4-8: lapse cohort/label/features, leak-free preprocessing, tuning, evaluation,
class-balance experiment, interpretation, persistence and retention list."""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.base import clone

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"]).merge(p[["product_id", "category"]], on="product_id")

# ---------------------------------------------------------- task 4: cohort, label, features
before = o[o.order_ts < CUTOFF]                         # ONLY orders before the cutoff feed the features
done_b = before[before.status == "completed"]
cohort_ids = done_b.customer_id.unique()
sept = o[(o.order_ts >= CUTOFF) & (o.order_ts < END) & (o.status == "completed")]
label = pd.Series(~pd.Index(cohort_ids).isin(sept.customer_id.unique()), index=cohort_ids).astype(int).rename("lapsed")

g = done_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g.order_ts.max()).dt.days,
    "frequency": g.order_id.nunique(),
    "monetary": g.order_amount.sum(),
    "avg_discount_pct": g.discount_pct.mean(),
    "avg_delivery_minutes": g.delivery_minutes.mean(),
    "fav_category": g.category.agg(lambda s: s.mode().sort_values().iloc[0]),
})
n_all = before[before.status.isin(["completed", "cancelled"])].groupby("customer_id").order_id.nunique()
n_can = before[before.status == "cancelled"].groupby("customer_id").order_id.nunique()
feat["cancel_rate"] = (n_can.reindex(feat.index).fillna(0) / n_all.reindex(feat.index)).astype(float)
cu = c.set_index("customer_id").loc[feat.index]
feat["city"], feat["gender"], feat["age"] = cu.city, cu.gender, cu.age
feat["tenure_days"] = (CUTOFF - cu.signup_date).dt.days
NUM = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate", "avg_delivery_minutes", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], label.loc[feat.index]
print(f"cohort: {len(X)} | lapsed: {int(y.sum())} ({y.mean()*100:.2f}%)")
assert X.isna().sum().sum() == 0
print("missing values in any feature column:", int(X.isna().sum().sum()))
assert (done_b.order_ts < CUTOFF).all() and (before.order_ts < CUTOFF).all()

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train", len(Xtr), "test", len(Xte), "| train lapsed", int(ytr.sum()), "not lapsed", int((ytr == 0).sum()))
base_pred = np.zeros(len(yte), dtype=int)
print(f"baseline (always not lapsed): accuracy {accuracy_score(yte, base_pred)*100:.1f}% | "
      f"lapsed F1 {f1_score(yte, base_pred, zero_division=0):.3f}")

# ---------------------------------------------------------- task 5: leak-free preprocessing
def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
pp = prep().fit(Xtr)                                   # fitted on training rows only
scaler = pp.named_transformers_["num"]
print(f"scaler learned mean of frequency (train only): {scaler.mean_[NUM.index('frequency')]:.3f} | "
      f"full-cohort mean of frequency: {X.frequency.mean():.3f}")
print("one-hot columns per variable (k-1):", {k: len(v) - 1 for k, v in zip(CAT, pp.named_transformers_['cat'].categories_)}
      , "-> dropped first level of each")

# ---------------------------------------------------------- task 6: tuning (preprocessing refit inside every CV fold)
lr_pipe = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_pipe = Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\nLR best params", lr_gs.best_params_, "CV F1", round(lr_gs.best_score_, 4))
print("RF best params", rf_gs.best_params_, "CV F1", round(rf_gs.best_score_, 4))

def evaluate(name, model, Xe=Xte, ye=yte):
    pr = model.predict(Xe)
    cm = confusion_matrix(ye, pr)
    r = dict(model=name, precision=precision_score(ye, pr, zero_division=0), recall=recall_score(ye, pr, zero_division=0),
             f1=f1_score(ye, pr, zero_division=0), accuracy=accuracy_score(ye, pr))
    print(f"{name}: confusion [[TN FP],[FN TP]] = {cm.tolist()} | precision {r['precision']:.3f} recall {r['recall']:.3f} F1 {r['f1']:.3f} acc {r['accuracy']:.3f}")
    return r
print("\n--- test-set evaluation (touched only after tuning) ---")
r_lr = evaluate("LogReg", lr_gs.best_estimator_)
r_rf = evaluate("RandomForest", rf_gs.best_estimator_)
best_is_rf = rf_gs.best_score_ >= lr_gs.best_score_      # selection by cross-validated F1, not by the test set
final = rf_gs.best_estimator_ if best_is_rf else lr_gs.best_estimator_
final_name = "RandomForest" if best_is_rf else "LogReg"
final_f1 = r_rf["f1"] if best_is_rf else r_lr["f1"]
print("selected model:", final_name, "(chosen by higher CV F1) | test F1", round(final_f1, 3))
assert final_f1 >= 0.55

# ---------------------------------------------------------- task 7: class-balance experiment (train only)
rng = np.random.RandomState(42)
lap_idx = np.where(ytr.values == 1)[0]
need = int((ytr == 0).sum() - (ytr == 1).sum())
extra = rng.choice(lap_idx, size=need, replace=True)
Xbal = pd.concat([Xtr, Xtr.iloc[extra]]); ybal = pd.concat([ytr, ytr.iloc[extra]])
print(f"\nbalanced train counts: not lapsed {int((ybal==0).sum())}, lapsed {int((ybal==1).sum())}")
lr_bal = clone(lr_gs.best_estimator_).fit(Xbal, ybal)
print("--- LR on original vs balanced training set, original test set ---")
r_unb = evaluate("LogReg (original)", lr_gs.best_estimator_)
r_bal = evaluate("LogReg (oversampled)", lr_bal)
print("change: precision %+.3f recall %+.3f F1 %+.3f" % (r_bal["precision"] - r_unb["precision"],
      r_bal["recall"] - r_unb["recall"], r_bal["f1"] - r_unb["f1"]))

# ---------------------------------------------------------- task 8: interpretation, persistence, retention list
names = [n.split("__", 1)[1] for n in rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\ntop-5 random-forest feature importances:\n", imp.head(5).round(4).to_string())
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=names)
print("\nLR coefficients (standardised features):")
for f in ["recency", "frequency", "monetary", "cancel_rate", "avg_discount_pct", "tenure_days"]:
    print(f"  {f}: {coef[f]:+.3f}")

joblib.dump(final, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(re.predict(Xte), final.predict(Xte)) and np.allclose(re.predict_proba(Xte), final.predict_proba(Xte))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
te = Xte.copy(); te["lapse_probability"] = final.predict_proba(Xte)[:, 1]; te["actually_lapsed"] = yte
te["is_premium"] = seg.is_premium.reindex(te.index).fillna(False).astype(bool)
prem = te[te.is_premium].sort_values("lapse_probability", ascending=False)
top = prem.head(10).rename_axis("customer_id").reset_index()[["customer_id", "lapse_probability", "actually_lapsed"]]
top["lapse_probability"] = top.lapse_probability.round(4)
top.to_csv("outputs/retention_list.csv", index=False)
print(f"premium customers in test set: {len(prem)} | retention list rows: {len(top)} | truly lapsed: {int(top.actually_lapsed.sum())}")
print(top.to_string(index=False))
