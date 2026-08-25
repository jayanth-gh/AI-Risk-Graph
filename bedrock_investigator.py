"""
Bedrock AI Investigator for the Razorpay AI Risk Manager (Track 02).

Takes the structured evidence the risk engine ALREADY computed (risk_scoring.py)
and asks a Bedrock model to write a natural-language investigation summary.
Bedrock only explains evidence here -- it never independently decides
fraud/not-fraud. That decision stays with the deterministic risk engine.

Requires:
  1. Model access enabled in the Bedrock console for the model you use below
     (Bedrock console -> Model access -> Request/Enable, one-time per account,
     usually instant for text models)
  2. The EC2 instance's IAM role needs a policy allowing bedrock:InvokeModel
     (attach to the same role you already use for Systems Manager)
  3. pip3 install boto3

If Bedrock isn't reachable (access not yet granted, network issue, etc.) this
falls back to a plain templated summary built from the same evidence, so the
rest of the demo still works.
"""

import json
import boto3
from botocore.exceptions import ClientError

# Anthropic Claude Haiku via Bedrock -- cheap, fast, good for structured summarization.
# Swap for "amazon.titan-text-express-v1" if Claude access isn't approved yet.
MODEL_ID = "amazon.nova-micro-v1:0"  # confirmed available in eu-north-1 (Stockholm)
REGION = "eu-north-1"  # Stockholm -- match your Bedrock console region exactly


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
    Returns {"summary": str, "source": "bedrock" | "fallback"}
    """
    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)
        body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": build_prompt(evidence)}]}
            ],
            "inferenceConfig": {"maxTokens": 200, "temperature": 0.3},
        })
        response = client.invoke_model(modelId=MODEL_ID, body=body)
        result = json.loads(response["body"].read())
        summary = result["output"]["message"]["content"][0]["text"].strip()
        return {"summary": summary, "source": "bedrock"}
    except (ClientError, Exception) as e:
        print(f"Bedrock call failed ({e}); using fallback summary.")
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
