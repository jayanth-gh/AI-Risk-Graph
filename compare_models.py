

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

FALSE_POSITIVE_COST = 100
MISSED_FRAUD_COST = 500

MODEL_A_FEATURES = [
    "amount", "account_age_days", "discount_pct",
    "hour_of_day", "day_of_week", "account_tx_seq",
    "account_tx_count_48h",
    "accounts_per_device_48h", "accounts_per_ip_48h",
    "accounts_per_paymethod_48h", "accounts_per_promo_48h",
    "is_success",
]

MODEL_B_FEATURES = MODEL_A_FEATURES + ["ring_size_48h"]


def load_splits(path="features_with_graph.csv", feature_cols=None):
    df = pd.read_csv(path)
    df = df.dropna(subset=feature_cols + ["is_fraud", "split"])

    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]

    return (
        (train[feature_cols], train["is_fraud"]),
        (val[feature_cols], val["is_fraud"]),
        (test[feature_cols], test["is_fraud"]),
    )


def tune_threshold(y_true, y_prob):
    best_threshold, best_cost = 0.5, float("inf")
    for t in np.linspace(0.01, 0.99, 99):
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * FALSE_POSITIVE_COST + fn * MISSED_FRAUD_COST
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold


def evaluate(label, model, X_val, y_val, X_test, y_test):
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_preds = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_prob)
    total_cost = fp * FALSE_POSITIVE_COST + fn * MISSED_FRAUD_COST

    return {
        "label": label, "threshold": threshold, "auc": auc,
        "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total_cost": total_cost,
    }


def run_model(name, feature_cols, algo="rf"):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits(feature_cols=feature_cols)

    if algo == "rf":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    else:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="logloss", n_jobs=-1
        )

    model.fit(X_train, y_train)
    result = evaluate(name, model, X_val, y_val, X_test, y_test)

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        result["top_features"] = imp.head(5)

    return result


def print_result(r):
    print(f"\n{'-'*60}")
    print(f"{r['label']}")
    print(f"{'-'*60}")
    print(f"Threshold: {r['threshold']:.3f}")
    print(f"Precision: {r['precision']:.4f}   Recall: {r['recall']:.4f}   "
          f"F1: {r['f1']:.4f}   AUC: {r['auc']:.4f}")
    print(f"Confusion matrix -- TN:{r['tn']}  FP:{r['fp']}  FN:{r['fn']}  TP:{r['tp']}")
    print(f"Total cost: Rs.{r['total_cost']} "
          f"(FP cost Rs.{r['fp']*FALSE_POSITIVE_COST} + FN cost Rs.{r['fn']*MISSED_FRAUD_COST})")
    if "top_features" in r:
        print("Top features:")
        print(r["top_features"].to_string())


if __name__ == "__main__":
    algos = ["rf"] + (["xgb"] if HAS_XGB else [])
    all_results = []

    for algo in algos:
        algo_name = "Random Forest" if algo == "rf" else "XGBoost"

        result_a = run_model(f"Model A ({algo_name}) -- behavioral + windowed counts",
                              MODEL_A_FEATURES, algo)
        result_b = run_model(f"Model B ({algo_name}) -- + graph ring_size_48h",
                              MODEL_B_FEATURES, algo)

        print_result(result_a)
        print_result(result_b)

        delta_recall = result_b["recall"] - result_a["recall"]
        delta_precision = result_b["precision"] - result_a["precision"]
        delta_cost = result_b["total_cost"] - result_a["total_cost"]

        print(f"\n>>> {algo_name}: adding the graph feature changed "
              f"precision by {delta_precision:+.4f}, recall by {delta_recall:+.4f}, "
              f"cost by Rs.{delta_cost:+d}")

        all_results.append(result_a)
        all_results.append(result_b)

    summary = pd.DataFrame(all_results)[
        ["label", "precision", "recall", "f1", "auc", "total_cost"]
    ]
    print(f"\n{'='*60}")
    print("FULL COMPARISON")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    summary.to_csv("model_a_vs_b_results.csv", index=False)
    print("\nSaved model_a_vs_b_results.csv")
