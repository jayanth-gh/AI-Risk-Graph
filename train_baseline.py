

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- run: pip3 install xgboost --break-system-packages")
    print("Continuing with Random Forest only.\n")

# ---- Synthetic cost assumptions (state these explicitly in your writeup) ----
FALSE_POSITIVE_COST = 100    # cost of wrongly blocking/flagging a legit customer
MISSED_FRAUD_COST = 500      # cost of letting a fraudulent transaction through

FEATURE_COLS = [
    "amount", "account_age_days", "discount_pct",
    "hour_of_day", "day_of_week", "account_tx_seq",
    "account_tx_count_48h",
    "accounts_per_device_48h", "accounts_per_ip_48h",
    "accounts_per_paymethod_48h", "accounts_per_promo_48h",
    "is_success",
]


def load_splits(path="features_with_split.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLS + ["is_fraud", "split"])

    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]

    X_train, y_train = train[FEATURE_COLS], train["is_fraud"]
    X_val, y_val = val[FEATURE_COLS], val["is_fraud"]
    X_test, y_test = test[FEATURE_COLS], test["is_fraud"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def tune_threshold(y_true, y_prob):
    """
    Sweep thresholds on the VALIDATION set only, pick the one that minimizes
    total synthetic cost (FP_cost * false_positives + FN_cost * false_negatives).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_threshold, best_cost = 0.5, float("inf")

    for t in np.linspace(0.01, 0.99, 99):
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * FALSE_POSITIVE_COST + fn * MISSED_FRAUD_COST
        if cost < best_cost:
            best_cost, best_threshold = cost, t

    return best_threshold, best_cost


def evaluate(name, model, X_val, y_val, X_test, y_test):
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_cost = tune_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_preds = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_prob)
    total_cost = fp * FALSE_POSITIVE_COST + fn * MISSED_FRAUD_COST

    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")
    print(f"Threshold selected on val (min cost): {threshold:.3f}")
    print(f"ROC-AUC (test):  {auc:.4f}")
    print(f"Precision (test): {precision:.4f}")
    print(f"Recall (test):    {recall:.4f}")
    print(f"F1 (test):        {f1:.4f}")
    print(f"\nConfusion matrix (test):")
    print(f"                 Predicted Normal   Predicted Fraud")
    print(f"  Actual Normal       {tn:>8}           {fp:>8}")
    print(f"  Actual Fraud        {fn:>8}           {tp:>8}")
    print(f"\nFalse positive cost: {fp} x Rs.{FALSE_POSITIVE_COST} = Rs.{fp*FALSE_POSITIVE_COST}")
    print(f"Missed fraud cost:   {fn} x Rs.{MISSED_FRAUD_COST} = Rs.{fn*MISSED_FRAUD_COST}")
    print(f"TOTAL COST:          Rs.{total_cost}")

    return {
        "model": name, "threshold": threshold, "auc": auc,
        "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total_cost": total_cost,
    }


def feature_importance(name, model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop features ({name}):")
    print(imp.head(6).to_string())


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits()

    print(f"Train: {len(X_train)} rows ({y_train.mean():.4f} fraud rate)")
    print(f"Val:   {len(X_val)} rows ({y_val.mean():.4f} fraud rate)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean():.4f} fraud rate)")

    results = []

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_val, y_val, X_test, y_test))
    feature_importance("Random Forest", rf, FEATURE_COLS)

    if HAS_XGB:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="logloss", n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate("XGBoost", xgb, X_val, y_val, X_test, y_test))
        feature_importance("XGBoost", xgb, FEATURE_COLS)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = pd.DataFrame(results)[["model", "precision", "recall", "f1", "auc", "total_cost"]]
    print(summary.to_string(index=False))
    summary.to_csv("baseline_results.csv", index=False)
    print("\nSaved baseline_results.csv")
