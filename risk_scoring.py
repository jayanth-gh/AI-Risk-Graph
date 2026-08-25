

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_B_FEATURES = [
    "amount", "account_age_days", "discount_pct",
    "hour_of_day", "day_of_week", "account_tx_seq",
    "account_tx_count_48h",
    "accounts_per_device_48h", "accounts_per_ip_48h",
    "accounts_per_paymethod_48h", "accounts_per_promo_48h",
    "is_success", "ring_size_48h",
]


def load_and_train(path="features_with_graph.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=MODEL_B_FEATURES + ["is_fraud", "split"])

    train = df[df.split == "train"]
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(train[MODEL_B_FEATURES], train["is_fraud"])
    return model, df


def risk_level(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


def build_evidence(row):
    """
    Turns this transaction's own feature values into plain-language reasons.
    Only includes a line if the value is actually notable (thresholds below
    are simple and meant to be tuned once you see real evidence output).
    """
    reasons = []

    if row["ring_size_48h"] >= 5:
        reasons.append(
            f"Account is part of a {int(row['ring_size_48h'])}-account cluster "
            f"linked by shared devices, IPs, or payment methods within 48 hours"
        )
    if row["accounts_per_ip_48h"] >= 5:
        reasons.append(
            f"{int(row['accounts_per_ip_48h'])} accounts used the same IP "
            f"within a 48-hour window"
        )
    if row["accounts_per_device_48h"] >= 5:
        reasons.append(
            f"{int(row['accounts_per_device_48h'])} accounts used the same device "
            f"within a 48-hour window"
        )
    if row["accounts_per_paymethod_48h"] >= 5:
        reasons.append(
            f"{int(row['accounts_per_paymethod_48h'])} accounts used the same "
            f"payment method within a 48-hour window"
        )
    if row["accounts_per_promo_48h"] >= 10:
        reasons.append(
            f"{int(row['accounts_per_promo_48h'])} accounts targeted the same "
            f"promotion within a 48-hour window"
        )
    if row["account_tx_count_48h"] >= 5:
        reasons.append(
            f"This account made {int(row['account_tx_count_48h'])} transactions "
            f"within 48 hours (burst activity)"
        )
    if row["account_age_days"] <= 3:
        reasons.append(
            f"Account is only {int(row['account_age_days'])} day(s) old"
        )

    if not reasons:
        reasons.append("No significant coordinated-abuse indicators found")

    return reasons


def score_transaction(model, row):
    X = row[MODEL_B_FEATURES].to_frame().T
    prob = model.predict_proba(X)[0, 1]
    score = int(round(prob * 100))
    return {
        "transaction_id": int(row["transaction_id"]),
        "account_id": int(row["account_id"]),
        "risk_score": score,
        "risk_level": risk_level(score),
        "actual_label": "FRAUD" if row["is_fraud"] == 1 else "NORMAL",
        "reasons": build_evidence(row),
    }


def print_report(result):
    print(f"\nTransaction: {result['transaction_id']}  (account {result['account_id']})")
    print(f"Risk Score: {result['risk_score']}/100")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Ground truth: {result['actual_label']}")
    print("Reasons:")
    for r in result["reasons"]:
        print(f"  - {r}")


if __name__ == "__main__":
    model, df = load_and_train()
    test = df[df.split == "test"]

    # show a handful of HIGH-risk examples and a couple of clean examples
    print("=" * 60)
    print("SAMPLE HIGH-RISK TRANSACTIONS (from test set)")
    print("=" * 60)
    scored = test.apply(lambda r: score_transaction(model, r), axis=1)
    high_risk = [s for s in scored if s["risk_level"] == "HIGH"][:5]
    for s in high_risk:
        print_report(s)

    print("\n" + "=" * 60)
    print("SAMPLE LOW-RISK TRANSACTIONS (from test set)")
    print("=" * 60)
    low_risk = [s for s in scored if s["risk_level"] == "LOW"][:3]
    for s in low_risk:
        print_report(s)

    # save full scored output as the "risk_results" table your design doc mentions
    out_rows = []
    for s in scored:
        out_rows.append({
            "transaction_id": s["transaction_id"],
            "account_id": s["account_id"],
            "risk_score": s["risk_score"],
            "risk_level": s["risk_level"],
            "actual_label": s["actual_label"],
            "reasons": " | ".join(s["reasons"]),
        })
    pd.DataFrame(out_rows).to_csv("risk_results_scored.csv", index=False)
    print(f"\nSaved risk_results_scored.csv ({len(out_rows)} rows)")
