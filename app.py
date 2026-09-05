"""
Enterprise Knowledge Assistant -- Streamlit chatbot UI.

Run:
    streamlit run app.py
"""
import time

import streamlit as st

import config
from rag.vector_store import VectorStore
from rag.retriever import retrieve_and_rerank
from rag.generator import generate_answer

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="wide")

CONFIDENCE_COLORS = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


@st.cache_resource
def get_store():
    return VectorStore()


def build_where_clause(departments, sensitivities):
    conditions = []
    if departments:
        conditions.append({"department": {"$in": departments}})
    if sensitivities:
        conditions.append({"sensitivity": {"$in": sensitivities}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def main():
    st.title("🧠 Enterprise Knowledge Assistant")
    st.caption("Ask questions in plain English. Answers are grounded in your organization's documents, with citations.")

    store = get_store()
    doc_summary = store.get_all_documents_summary()

    # ---------------- Sidebar: filters + index status ----------------
    with st.sidebar:
        if config.RAG_MODE == "free":
            st.info(f"**FREE mode**\n\nChat: `{config.OPENROUTER_CHAT_MODEL}` (OpenRouter)\n\nEmbeddings: `{config.LOCAL_EMBEDDING_MODEL}` (local)")
        else:
            provider_label = "Azure OpenAI" if config.PROVIDER == "azure_openai" else "OpenAI"
            st.success(f" **PAID mode**\n\nChat: `{config.CHAT_MODEL if config.PROVIDER != 'azure_openai' else config.AZURE_CHAT_DEPLOYMENT}` ({provider_label})")
        st.caption("Switch modes by setting RAG_MODE=free or RAG_MODE=paid in .env, then restart the app.")
        st.divider()
        st.header("⚙️ Filters")
        depts = sorted({d["department"] for d in doc_summary if d["department"]})
        sens = sorted({d["sensitivity"] for d in doc_summary if d["sensitivity"]})

        selected_depts = st.multiselect("Department", options=depts or config.DEPARTMENTS)
        selected_sens = st.multiselect("Sensitivity", options=sens or config.SENSITIVITY_LEVELS)

        st.divider()
        st.header("📚 Knowledge Base")
        st.metric("Indexed chunks", store.count())
        st.metric("Indexed documents", len(doc_summary))
        with st.expander("View indexed documents"):
            for d in doc_summary:
                st.write(f"**{d['title']}** — {d['department']} / {d['sensitivity']} (v{d['version']})")

        st.divider()
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    if store.count() == 0:
        st.warning(
            "No documents are indexed yet. Run the ingestion pipeline first:\n\n"
            "`python -m ingestion.ingest`\n\n"
            "(Sample HR/IT/Operations documents are provided in `data/sample_docs/`.)"
        )
        return

    # ---------------- Chat state ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                render_citations(msg["citations"], msg.get("confidence"), msg.get("confidence_score"))

    question = st.chat_input("Ask about HR policy, IT support, onboarding, or any indexed document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant documents and generating answer..."):
                where = build_where_clause(selected_depts, selected_sens)
                t0 = time.time()
                chunks = retrieve_and_rerank(question, store, where=where)

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]
                result = generate_answer(question, chunks, chat_history=history)
                latency = round(time.time() - t0, 2)

            st.markdown(result["answer"])
            render_citations(result["citations"], result["confidence"], result["confidence_score"])
            st.caption(f"⏱️ {latency}s · {len(chunks)} sources retrieved")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "confidence_score": result["confidence_score"],
        })


def render_citations(citations, confidence, confidence_score):
    if confidence:
        icon = CONFIDENCE_COLORS.get(confidence, "⚪")
        st.caption(f"{icon} **Confidence: {confidence}** ({confidence_score:.2f})")
    if citations:
        with st.expander(f"📎 Sources ({len(citations)})"):
            for c in citations:
                st.markdown(
                    f"**[{c['tag']}]** {c['source']} — {c['department']}, "
                    f"page {c['page']}, v{c['version']}  \n"
                    f"relevance score: `{c['score']}`"
                )


if __name__ == "__main__":
    main()
