# src/train_baseline.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import lightgbm as lgb
import joblib

# -------- CONFIG --------
TRAIN_CSV = "data/train.csv"
TEST_CSV  = "data/test.csv"
MODEL_OUT = "models/lgbm.pkl"
RANDOM_SEED = 42
TOP_K = 20  # for Precision@K

# -------- LOAD ----------
os.makedirs("models", exist_ok=True)
print("Loading train/test...")
train = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
test  = pd.read_csv(TEST_CSV , parse_dates=["timestamp"])

# drop columns that are identifiers / not features
drop_cols = ["transaction_id","timestamp","sender_id","receiver_id","message","fraud_type"]
for c in drop_cols:
    if c in train.columns:
        train = train.drop(columns=[c])
    if c in test.columns:
        test = test.drop(columns=[c])

# ensure label present
if "is_fraud" not in train.columns:
    raise ValueError("is_fraud column missing in train.csv")

# -------- FEATURES & TARGET ----------
y_train = train["is_fraud"].astype(int)
X_train = train.drop(columns=["is_fraud"])

y_test = test["is_fraud"].astype(int)
X_test = test.drop(columns=["is_fraud"])

# keep feature list
features = X_train.columns.tolist()
print("Num features:", len(features))

# optional: simple imputation for safety
X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

# -------- LIGHTGBM TRAINING ----------
dtrain = lgb.Dataset(X_train, label=y_train)

params = {
    "objective": "binary",
    "metric": "None",
    "boosting": "gbdt",
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

print("Training LightGBM...")
bst = lgb.train(
    params,
    dtrain,
    num_boost_round=300,   # reduced since no early stopping
    verbose_eval=50
)


# -------- PREDICTIONS & METRICS ----------
print("Predicting on test set...")
probs = bst.predict(X_test, num_iteration=bst.best_iteration)

# PR-AUC (Average precision)
pr_auc = average_precision_score(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

def precision_at_k(y_true, y_score, k):
    # pick top-k by score
    idx = np.argsort(y_score)[::-1][:k]
    return y_true.iloc[idx].sum() / float(k)

prec_at_k = precision_at_k(y_test.reset_index(drop=True), pd.Series(probs), TOP_K)

print(f"PR-AUC (avg precision): {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Precision@{TOP_K}: {prec_at_k:.4f}")

# -------- SAVE MODEL ----------
joblib.dump({"model": bst, "features": features}, MODEL_OUT)
print("Saved model to", MODEL_OUT)

# -------- OPTIONAL: show top features (simple importance) ----------
imp = bst.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": features, "gain": imp}).sort_values("gain", ascending=False).head(20)
print("\nTop features by gain:\n", imp_df.to_string(index=False))
