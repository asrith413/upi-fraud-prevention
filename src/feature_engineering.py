# src/feature_engineering.py
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta

# ---------- CONFIG ----------
INPATH = "data/upi_synthetic.csv"
OUTPATH = "data/upi_features.csv"
TRAIN_OUT = "data/train.csv"
TEST_OUT = "data/test.csv"
# time split: last N days for test
TEST_DAYS = 3

# ---------- LOAD ----------
print("Loading data...")
df = pd.read_csv(INPATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ---------- BASIC CLEAN ----------
# ensure types
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
df['device_change'] = df['device_change'].fillna(0).astype(int)
if 'is_fraud' not in df.columns:
    df['is_fraud'] = 0
df['is_fraud'] = df['is_fraud'].astype(int)

# ---------- TIME FEATURES ----------
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Mon
df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)

# ---------- TEXT / MESSAGE FLAGS (quick wins) ----------
df['message'] = df['message'].fillna("").astype(str)
scam_keywords = ["refund", "mistake", "otp", "support", "verification", "screenshot", "accident", "forward funds"]
df['msg_contains_scam_kw'] = df['message'].str.lower().apply(
    lambda t: int(any(k in t for k in scam_keywords))
)

# small TF-IDF or model is optional (stretch)

# ---------- FIRST-TIME RECEIVER FLAG (sender -> receiver) ----------
df['first_time_receiver'] = (~df.duplicated(subset=['sender_id', 'receiver_id'])).astype(int)

# ---------- ROLLING / VELOCITY FEATURES (fast, per-sender & per-receiver)
# We'll compute counts & sums in last 1h and 24h windows using rolling on time-index per user.

# helper to compute time-windowed aggregates per key
def time_window_agg(df, group_key, time_col, window_seconds, agg_col='amount', prefix=""):
    out = []
    df = df.sort_values(time_col)
    # operate per group to keep memory OK
    for k, g in df.groupby(group_key):
        times = g[time_col].values
        amounts = g[agg_col].values
        counts = []
        sums = []
        # two-pointer sliding window
        j = 0
        n = len(times)
        for i in range(n):
            t_i = times[i]
            # increment j until within window
            while j < n and (t_i - times[j]).astype('timedelta64[s]').astype(int) > window_seconds:
                j += 1
            cnt = i - j  # exclude current -> previous trans count in window
            s = amounts[j:i].sum() if i > j else 0.0
            counts.append(cnt)
            sums.append(s)
        tmp = g.copy()
        tmp[prefix + f"{group_key}_cnt_{window_seconds}s"] = counts
        tmp[prefix + f"{group_key}_sum_{window_seconds}s"] = sums
        out.append(tmp[[prefix + f"{group_key}_cnt_{window_seconds}s", prefix + f"{group_key}_sum_{window_seconds}s"]])
    # concat in original order
    out_df = pd.concat(out).sort_index()
    return out_df

print("Computing per-sender 1h and 24h velocity features (may take a moment)...")
# due to performance, compute simpler aggregates using groupby+rolling on resampled times per user
# We'll compute counts in last 1h and 24h using a fast method: for each user, use timestamps as ints and sliding window
def add_window_features(df, key, seconds_list=[3600, 86400], amount_col='amount'):
    res_cols = {}
    grouped = df.groupby(key).indices  # dict: key -> indices
    for sec in seconds_list:
        cnt_col = f"{key}_cnt_{sec}s"
        sum_col = f"{key}_sum_{sec}s"
        res_cols[cnt_col] = np.zeros(len(df), dtype=int)
        res_cols[sum_col] = np.zeros(len(df), dtype=float)

    for entity, idxs in grouped.items():
        idxs = np.array(idxs)
        times = df.loc[idxs, 'timestamp'].values.astype('datetime64[s]').astype(int)
        amounts = df.loc[idxs, amount_col].values
        for sec in seconds_list:
            cnts = np.zeros(len(idxs), dtype=int)
            sums = np.zeros(len(idxs), dtype=float)
            j = 0
            for i in range(len(idxs)):
                t_i = times[i]
                while j < i and (t_i - times[j]) > sec:
                    j += 1
                cnts[i] = i - j
                sums[i] = amounts[j:i].sum() if i > j else 0.0
            res_cols[f"{key}_cnt_{sec}s"][idxs] = cnts
            res_cols[f"{key}_sum_{sec}s"][idxs] = sums
    for k, v in res_cols.items():
        df[k] = v
    return df

df = add_window_features(df, 'sender_id', seconds_list=[3600, 86400])
df = add_window_features(df, 'receiver_id', seconds_list=[3600, 86400])

# Derived velocity features
df['sender_amount_ratio_24h'] = (df['amount'] / (df['sender_id_sum_86400s'] + 1e-9)).replace([np.inf, -np.inf], 0)
df['sender_txn_rate_1h'] = df['sender_id_cnt_3600s']
df['receiver_inflow_1h'] = df['receiver_id_sum_3600s']

# ---------- SMALL-VERIFICATION-BEFORE-LARGE (0.01 then big transfer) ----------
# mark if the same sender received a tiny incoming (0.01) within prior 10 minutes to the current outgoing large txn
# We'll create a helper: for each sender, look at past incoming amounts from others (simulate: check if receiver is current sender)
# For synthetic dataset, assume transactions include both directions; implement simple heuristic.

print("Computing small-verification-before-large flag...")
df = df.sort_values('timestamp').reset_index(drop=True)
# flag tiny incoming to this sender within last 15 minutes
tiny_thresh = 0.05
window_minutes = 15
df['is_incoming_tiny_recent'] = 0
# build quick mapping of transactions by receiver to speed up
recv_groups = df.groupby('receiver_id').indices
for idx, row in df.iterrows():
    s = row['sender_id']
    t = row['timestamp']
    # incoming to sender means rows where receiver_id == this sender
    if s in recv_groups:
        idxs = recv_groups[s]
        # restrict to those with timestamp < current t and within window
        # linear scan backwards (small cost for dataset size used here)
        recent = [i for i in idxs if df.at[i, 'timestamp'] < t and (t - df.at[i, 'timestamp']).total_seconds() <= window_minutes*60]
        tiny_found = any(df.at[i, 'amount'] <= tiny_thresh for i in recent)
        df.at[idx, 'is_incoming_tiny_recent'] = int(tiny_found)

# create rule: tiny incoming recently AND amount is large relative to sender history
df['small_verif_before_large'] = ((df['is_incoming_tiny_recent'] == 1) & (df['amount'] > (df['sender_id_sum_86400s']*0.5 + 1))).astype(int)

# ---------- DEVICE CHANGE, IP, GEO (quick) ----------
# device_change already present; create a feature combining device_change + new_receiver
df['risky_device_newpayee'] = ((df['device_change'] == 1) & (df['first_time_receiver'] == 1)).astype(int)

# ---------- GRAPH FEATURES (node-level) ----------
print("Building transaction graph and computing pagerank / degrees...")
G = nx.DiGraph()
edges = list(zip(df['sender_id'], df['receiver_id'], df['amount']))
G.add_weighted_edges_from(edges)

# compute in/out degree
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
pagerank = nx.pagerank(G, alpha=0.85)

df['receiver_in_degree'] = df['receiver_id'].map(in_deg).fillna(0).astype(int)
df['sender_out_degree'] = df['sender_id'].map(out_deg).fillna(0).astype(int)
df['receiver_pagerank'] = df['receiver_id'].map(pagerank).fillna(0.0)

# detect hubs: many unique senders in short window (we already have receiver_id_cnt_3600s)
df['receiver_unique_senders_1h'] = df['receiver_id_cnt_3600s']  # proxy

# ---------- COLLUSION SCORE (simple heuristic) ----------
# if a receiver receives many small txns from many unique senders and then forwards out quickly -> high collusion risk
df['collusion_score'] = ((df['receiver_in_degree'] > 5).astype(int) * (df['receiver_inflow_1h'] > df['receiver_id_sum_86400s'] * 0.4).astype(int))

# ---------- EXPLAINABILITY CANDIDATE FEATURES (keep small set for SHAP later) ----------
feature_cols = [
    'amount', 'hour', 'dayofweek', 'is_night',
    'msg_contains_scam_kw',
    'first_time_receiver',
    'sender_id_cnt_3600s', 'sender_id_cnt_86400s', 'sender_id_sum_3600s','sender_id_sum_86400s',
    'receiver_id_cnt_3600s','receiver_id_cnt_86400s','receiver_id_sum_3600s','receiver_id_sum_86400s',
    'sender_amount_ratio_24h','sender_txn_rate_1h','receiver_inflow_1h',
    'small_verif_before_large','device_change','risky_device_newpayee',
    'receiver_in_degree','sender_out_degree','receiver_pagerank','receiver_unique_senders_1h',
    'collusion_score'
]

# ensure columns exist
feature_cols = [c for c in feature_cols if c in df.columns]
print("Feature columns prepared:", feature_cols)

# ---------- SAVE FEATURES ----------
df.to_csv(OUTPATH, index=False)
print("Saved features to", OUTPATH)

# ---------- TIME SPLIT: last TEST_DAYS for test ----------
max_time = df['timestamp'].max()
cutoff = max_time - pd.Timedelta(days=TEST_DAYS)
train = df[df['timestamp'] < cutoff].copy()
test = df[df['timestamp'] >= cutoff].copy()
print("Train size:", len(train), "Test size:", len(test))

train.to_csv(TRAIN_OUT, index=False)
test.to_csv(TEST_OUT, index=False)
print("Train/test saved:", TRAIN_OUT, TEST_OUT)
