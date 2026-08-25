"""
Streamlit dashboard for the Razorpay AI Risk Manager (Track 02).

Calls the already-running FastAPI backend (app.py) rather than retraining
the model itself -- keeps memory usage low on the t3.micro instance, since
the model only needs to live in one process.

Run:  streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
Then open http://<your-ec2-public-ip>:8501 in a browser.

Requires app.py to already be running (e.g. in another terminal / tmux pane)
on the URL set in API_BASE_URL below.
"""

import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"  # FastAPI backend, same instance

st.set_page_config(page_title="AI Risk Manager", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Risk Manager -- Coordinated Abuse Detector")
st.caption("Razorpay Buildathon Track 02 -- detects promotional/payment abuse rings using ML + graph features")

# ---- Model A vs Model B comparison (static results from your evaluation) ----
with st.expander("📊 Model A vs Model B -- why the graph feature matters", expanded=True):
    comparison_data = pd.DataFrame([
        {"Model": "Model A (Random Forest)", "Precision": 0.927, "Recall": 0.999, "Total Cost (Rs.)": 11900},
        {"Model": "Model B (Random Forest)", "Precision": 0.994, "Recall": 1.000, "Total Cost (Rs.)": 800},
        {"Model": "Model A (XGBoost)", "Precision": 0.951, "Recall": 0.997, "Total Cost (Rs.)": 9400},
        {"Model": "Model B (XGBoost)", "Precision": 0.998, "Recall": 1.000, "Total Cost (Rs.)": 300},
    ])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(comparison_data, hide_index=True, use_container_width=True)
    with col2:
        st.metric("Cost reduction (RF)", "93%", "-Rs.11,100")
        st.metric("Cost reduction (XGBoost)", "97%", "-Rs.9,100")
    st.caption(
        "Model A: behavioral features + windowed shared-resource counts. "
        "Model B: + graph ring-size feature (accounts linked by 2+ independently "
        "shared resources within 48h). Same train/val/test split for both."
    )

st.divider()

# ---- Live transaction lookup ----
st.subheader("🔍 Investigate a transaction")

col_a, col_b = st.columns([1, 1])
with col_a:
    st.write("**Try a sample:**")
    sample_col1, sample_col2 = st.columns(2)
    high_risk_clicked = sample_col1.button("Load HIGH-risk example", use_container_width=True)
    low_risk_clicked = sample_col2.button("Load LOW-risk example", use_container_width=True)

with col_b:
    manual_id = st.number_input("Or enter a transaction ID directly", min_value=0, step=1, value=0)
    lookup_clicked = st.button("Look up", use_container_width=True)


def fetch_and_render(transaction_id):
    try:
        with st.spinner(f"Scoring transaction {transaction_id}..."):
            resp = requests.get(
                f"{API_BASE_URL}/transaction/{transaction_id}/investigate", timeout=15
            )
        if resp.status_code == 404:
            st.error(f"Transaction {transaction_id} not found.")
            return
        resp.raise_for_status()
        result = resp.json()
        render_result(result)
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the risk engine API at {API_BASE_URL}. "
                  f"Is app.py running? ({e})")


def render_result(result):
    score = result["risk_score"]
    level = result["risk_level"]

    color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{score}/100")
    col2.metric("Risk Level", f"{color} {level}")
    col3.metric("Ground Truth", result.get("actual_label", "N/A"))

    st.progress(score / 100)

    st.write("**Reasons:**")
    for reason in result["reasons"]:
        st.write(f"- {reason}")

    if "investigation_summary" in result:
        st.write("**AI Investigation Summary:**")
        source = result.get("summary_source", "unknown")
        badge = "🤖 Gemini-generated" if source == "gemini" else "📋 Template-based (AI unavailable)"
        st.info(f"{result['investigation_summary']}\n\n*{badge}*")


if high_risk_clicked:
    try:
        samples = requests.get(f"{API_BASE_URL}/sample/high-risk?n=1", timeout=15).json()
        if samples:
            render_result(samples[0])
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the risk engine API. ({e})")

if low_risk_clicked:
    try:
        samples = requests.get(f"{API_BASE_URL}/sample/low-risk?n=1", timeout=15).json()
        if samples:
            render_result(samples[0])
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the risk engine API. ({e})")

if lookup_clicked and manual_id:
    fetch_and_render(int(manual_id))
