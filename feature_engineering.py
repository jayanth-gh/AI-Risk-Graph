"""
Feature engineering for the Razorpay AI Risk Manager (Track 02) baseline model.

Key design decision (validated against the actual dataset before writing this):
Raw "accounts per device/IP/payment method" counted over the WHOLE dataset is
NOT discriminative here -- normal traffic shows more raw device reuse than
fraud, because there are only 5,000 devices for 100,000 transactions.
The signal only appears once you window it in time. This script builds
TIME-WINDOWED shared-resource features (trailing 48h), which is what actually
separates fraud from normal in this data.
"""

import pandas as pd
import numpy as np

DATA_DIR = "ai_risk_manager_dataset"  # adjust to your path
WINDOW_HOURS = 48

def load_data():
    tx = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    promotions = pd.read_csv(f"{DATA_DIR}/promotions.csv")
    payment_methods = pd.read_csv(f"{DATA_DIR}/payment_methods.csv")
    merchants = pd.read_csv(f"{DATA_DIR}/merchants.csv")

    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.merge(accounts, on="account_id", how="left")
    tx = tx.merge(promotions, on="promotion_id", how="left")
    tx = tx.merge(payment_methods, on="payment_method_id", how="left")
    tx = tx.merge(merchants, on="merchant_id", how="left")
    return tx


def add_windowed_sharing_feature(tx, entity_col, window_hours, new_col):
    """
    For each transaction, count the number of DISTINCT accounts that used the
    same entity (device/ip/payment_method/promotion) within the trailing
    `window_hours` window (inclusive of the current transaction).

    This only looks backward in time, so it's safe to use in a real-time
    scoring pipeline (no future leakage).
    """
    tx = tx.sort_values("timestamp").reset_index(drop=True)
    result = np.zeros(len(tx), dtype=int)

    for entity_id, group in tx.groupby(entity_col):
        idx = group.index.to_numpy()
        times = group["timestamp"].to_numpy()
        accts = group["account_id"].to_numpy()

        # sliding window via two pointers (already time-sorted within group)
        left = 0
        seen = {}
        window = np.timedelta64(window_hours, "h")
        for right in range(len(idx)):
            while times[right] - times[left] > window:
                acc = accts[left]
                seen[acc] -= 1
                if seen[acc] == 0:
                    del seen[acc]
                left += 1
            acc = accts[right]
            seen[acc] = seen.get(acc, 0) + 1
            result[idx[right]] = len(seen)

    tx[new_col] = result
    return tx


def add_behavioral_features(tx):
    tx["hour_of_day"] = tx["timestamp"].dt.hour
    tx["day_of_week"] = tx["timestamp"].dt.dayofweek

    # transactions per account so far (trailing, cumulative — no future leak)
    tx = tx.sort_values("timestamp")
    tx["account_tx_seq"] = tx.groupby("account_id").cumcount() + 1

    # transactions per account within trailing 48h (burst behavior)
    tx = add_windowed_sharing_feature(
        tx.assign(_acc_as_entity=tx["account_id"]),
        "_acc_as_entity", WINDOW_HOURS, "account_tx_count_48h"
    ).drop(columns="_acc_as_entity")
    return tx


def build_feature_table():
    tx = load_data()
    tx = add_behavioral_features(tx)

    tx = add_windowed_sharing_feature(tx, "device_id", WINDOW_HOURS, "accounts_per_device_48h")
    tx = add_windowed_sharing_feature(tx, "ip_id", WINDOW_HOURS, "accounts_per_ip_48h")
    tx = add_windowed_sharing_feature(tx, "payment_method_id", WINDOW_HOURS, "accounts_per_paymethod_48h")
    tx = add_windowed_sharing_feature(tx, "promotion_id", WINDOW_HOURS, "accounts_per_promo_48h")

    tx["is_success"] = (tx["transaction_status"] == "success").astype(int)

    feature_cols = [
        "amount", "account_age_days", "discount_pct",
        "hour_of_day", "day_of_week", "account_tx_seq",
        "account_tx_count_48h",
        "accounts_per_device_48h", "accounts_per_ip_48h",
        "accounts_per_paymethod_48h", "accounts_per_promo_48h",
        "is_success",
    ]

    X = tx[feature_cols]
    y = tx["is_fraud"]
    meta = tx[["transaction_id", "account_id", "fraud_cluster_id", "timestamp"]]

    return X, y, meta, tx


def cluster_aware_split(meta, test_size=0.15, val_size=0.15, seed=42):
    """
    Splits so that all rows belonging to the same fraud_cluster_id land in the
    same split -- prevents the model from seeing part of a coordinated ring in
    training and the rest in test.
    """
    rng = np.random.RandomState(seed)

    fraud_clusters = meta.loc[meta.fraud_cluster_id != -1, "fraud_cluster_id"].unique().copy()
    rng.shuffle(fraud_clusters)
    n_test = int(len(fraud_clusters) * test_size)
    n_val = int(len(fraud_clusters) * val_size)
    test_clusters = set(fraud_clusters[:n_test])
    val_clusters = set(fraud_clusters[n_test:n_test + n_val])

    normal_idx = meta.index[meta.fraud_cluster_id == -1].to_numpy().copy()
    rng.shuffle(normal_idx)
    n_norm_test = int(len(normal_idx) * test_size)
    n_norm_val = int(len(normal_idx) * val_size)
    normal_test_idx = set(normal_idx[:n_norm_test])
    normal_val_idx = set(normal_idx[n_norm_test:n_norm_test + n_norm_val])

    split = pd.Series("train", index=meta.index)
    split[meta.fraud_cluster_id.isin(test_clusters)] = "test"
    split[meta.fraud_cluster_id.isin(val_clusters)] = "val"
    split.loc[list(normal_test_idx)] = "test"
    split.loc[list(normal_val_idx)] = "val"

    return split


if __name__ == "__main__":
    X, y, meta, tx = build_feature_table()
    split = cluster_aware_split(meta)

    print("Split sizes:\n", split.value_counts())
    print("\nFraud rate by split:")
    for s in ["train", "val", "test"]:
        mask = split == s
        print(f"  {s}: {y[mask].mean():.4f} ({mask.sum()} rows)")

    print("\nFeature summary (fraud vs normal), windowed features:")
    for col in ["accounts_per_device_48h", "accounts_per_ip_48h",
                "accounts_per_paymethod_48h", "accounts_per_promo_48h"]:
        print(f"  {col}: fraud median={X.loc[y==1, col].median()}, "
              f"normal median={X.loc[y==0, col].median()}")

    out = tx.copy()
    out["split"] = split
    for col in X.columns:
        out[col] = X[col]
    out.to_csv("features_with_split.csv", index=False)
    print("\nSaved features_with_split.csv")
