"""
Evaluation dashboard -- visualizes results produced by eval/evaluate.py.

Run:
    streamlit run eval/dashboard.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 RAG Evaluation Dashboard")
st.caption("Retrieval relevance, answer quality, confidence calibration, and latency for the Enterprise Knowledge Assistant.")

results_path = Path(config.EVAL_RESULTS_PATH)
if not results_path.exists():
    st.warning(
        "No evaluation results found yet. Run the evaluation harness first:\n\n"
        "`python -m eval.evaluate`"
    )
    st.stop()

with open(results_path) as f:
    data = json.load(f)

summary = data["summary"]
results = data["results"]
df = pd.DataFrame(results)

# ---------------- Top-line metrics ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Questions evaluated", summary["num_questions"])
c2.metric("Retrieval hit rate", f"{summary['retrieval_hit_rate']*100:.0f}%")
c3.metric("Mean Reciprocal Rank", summary["mean_reciprocal_rank"])
coverage = summary.get("avg_answer_keyword_coverage")
c4.metric("Answer keyword coverage", f"{coverage*100:.0f}%" if coverage is not None else "N/A")
c5.metric("Avg latency", f"{summary['avg_latency_sec']}s")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Confidence Distribution")
    conf_df = pd.DataFrame(
        list(summary["confidence_distribution"].items()), columns=["Confidence", "Count"]
    )
    st.bar_chart(conf_df.set_index("Confidence"))

with col2:
    st.subheader("Latency per Question")
    st.bar_chart(df.set_index("id")["latency_sec"])

st.divider()
st.subheader("Per-Question Results")

def hit_icon(v):
    return "✅" if v else "❌"

display_df = df.copy()
display_df["retrieval_hit"] = display_df["retrieval_hit"].apply(hit_icon)
display_df["keyword_coverage"] = display_df["keyword_coverage"].apply(
    lambda v: f"{v*100:.0f}%" if v is not None else "N/A"
)
st.dataframe(
    display_df[["id", "question", "retrieval_hit", "keyword_coverage", "confidence", "latency_sec"]],
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Inspect a Question")
selected_id = st.selectbox("Select question", df["id"].tolist())
row = df[df["id"] == selected_id].iloc[0]

st.markdown(f"**Question:** {row['question']}")
st.markdown(f"**Answer:**\n\n{row['answer']}")
st.markdown(f"**Expected source:** `{row['expected_source']}`  |  **Retrieved sources:** `{row['retrieved_sources']}`")
st.markdown(f"**Confidence:** {row['confidence']} ({row['confidence_score']})")
st.markdown(f"**Matched keywords:** {row['matched_keywords']} / expected {row['expected_keywords']}")
