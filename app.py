

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestClassifier

from risk_scoring import (
    MODEL_B_FEATURES, load_and_train, score_transaction, risk_level
)
from ai_investigator import get_investigation_summary

app = FastAPI(title="AI Risk Manager -- Coordinated Abuse Detector")

model = None
data = None


@app.on_event("startup")
def startup():
    global model, data
    model, data = load_and_train()
    print(f"Model trained. {len(data)} transactions loaded.")


@app.get("/")
def root():
    return {
        "service": "AI Risk Manager",
        "endpoints": {
            "/transaction/{transaction_id}": "Get risk score + evidence for a transaction",
            "/transaction/{transaction_id}/investigate": "Same, plus an AI-generated investigation summary",
            "/sample/high-risk": "List a few known high-risk transactions to try",
            "/sample/low-risk": "List a few known low-risk transactions to try",
        },
    }


@app.get("/transaction/{transaction_id}")
def get_risk(transaction_id: int):
    row = data[data.transaction_id == transaction_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    result = score_transaction(model, row.iloc[0])
    return result


@app.get("/transaction/{transaction_id}/investigate")
def investigate(transaction_id: int):
    """
    Same as /transaction/{id}, plus a natural-language investigation summary
    generated from the evidence. The summary generator only explains the
    evidence already computed here -- it never makes its own fraud decision.
    """
    row = data[data.transaction_id == transaction_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    result = score_transaction(model, row.iloc[0])
    investigation = get_investigation_summary(result)
    result["investigation_summary"] = investigation["summary"]
    result["summary_source"] = investigation["source"]
    return result


@app.get("/sample/high-risk")
def sample_high_risk(n: int = 5):
    test = data[data.split == "test"]
    fraud_sample = test[test.is_fraud == 1].head(n)
    return [score_transaction(model, r) for _, r in fraud_sample.iterrows()]


@app.get("/sample/low-risk")
def sample_low_risk(n: int = 5):
    test = data[data.split == "test"]
    normal_sample = test[test.is_fraud == 0].head(n)
    return [score_transaction(model, r) for _, r in normal_sample.iterrows()]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
