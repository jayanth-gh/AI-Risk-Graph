

import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"  # FastAPI backend, same instance

st.set_page_config(page_title="AI Risk Manager", page_icon="🛡️", layout="wide")

st.title("AI Risk Manager")

st.divider()

# ---- Live transaction lookup ----
st.subheader("Investigate a transaction")

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
