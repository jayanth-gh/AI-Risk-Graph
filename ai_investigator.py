"""
AI Investigator for the Razorpay AI Risk Manager (Track 02) -- Gemini version.

Same purpose as bedrock_investigator.py (turn structured risk evidence into a
natural-language investigation summary), calling the Gemini API directly
instead of going through AWS Bedrock.

Setup:
  1. Get a free API key from https://aistudio.google.com/apikey
  2. On EC2: export GEMINI_API_KEY="your-key-here"
     (don't hardcode it or commit it to git)
  3. pip3 install google-genai

If the key isn't set or the call fails, this falls back to the same
templated summary as the Bedrock version, so the demo still works either way.
"""

import json
import os
from google import genai

MODEL = "gemini-2.5-flash"  # swap to "gemini-3.7-flash" if you want the newest model


def build_prompt(evidence: dict) -> str:
    return f"""You are a fraud investigation assistant. You are given structured
evidence from a deterministic risk-scoring system for a coordinated
promotional-abuse detection system. Write a 2-3 sentence plain-English
investigation summary explaining WHY this transaction was flagged, based
ONLY on the evidence given. Do not invent any facts not present below.
Do not make a final fraud/not-fraud verdict -- just explain the evidence.

Evidence:
{json.dumps(evidence, indent=2)}

Investigation summary:"""


def fallback_summary(evidence: dict) -> str:
    reasons = evidence.get("reasons", [])
    if not reasons or reasons == ["No significant coordinated-abuse indicators found"]:
        return "No coordinated-abuse indicators were found for this transaction."
    joined = "; ".join(reasons)
    return (
        f"This transaction was flagged with a risk score of "
        f"{evidence.get('risk_score')}/100 ({evidence.get('risk_level')} risk). "
        f"Evidence: {joined}."
    )


def get_investigation_summary(evidence: dict) -> dict:
    """
    Returns {"summary": str, "source": "gemini" | "fallback"}
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set; using fallback summary.")
        return {"summary": fallback_summary(evidence), "source": "fallback"}

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=build_prompt(evidence),
        )
        summary = response.text.strip()
        return {"summary": summary, "source": "gemini"}
    except Exception as e:
        print(f"Gemini API call failed ({e}); using fallback summary.")
        return {"summary": fallback_summary(evidence), "source": "fallback"}


if __name__ == "__main__":
    from risk_scoring import load_and_train, score_transaction

    model, data = load_and_train()
    test = data[data.split == "test"]
    fraud_sample = test[test.is_fraud == 1].iloc[0]

    evidence = score_transaction(model, fraud_sample)
    print("Structured evidence:")
    print(json.dumps(evidence, indent=2))

    result = get_investigation_summary(evidence)
    print(f"\nInvestigation summary (source: {result['source']}):")
    print(result["summary"])
