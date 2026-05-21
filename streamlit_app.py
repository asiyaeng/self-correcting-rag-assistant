"""
streamlit_app.py
Bonus: Minimal Streamlit frontend for interactive Q&A.

Run with:
    streamlit run streamlit_app.py

Make sure the FastAPI server is running on port 8000 first.
"""
import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="RAG Documentation Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.caption("Powered by LangGraph + FastAPI")

    st.divider()
    st.subheader("📚 Indexed Documents")
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        if resp.ok:
            data = resp.json()
            st.metric("Documents", data["total_documents"])
            st.metric("Chunks", data["total_chunks"])
            with st.expander("View documents"):
                for doc in data["documents"]:
                    st.text(f"• {doc['title']} ({doc['chunk_count']} chunks)")
        else:
            st.warning("API not reachable")
    except Exception:
        st.error("⚠️ API server not running.\nStart it with:\n`uvicorn app.main:app --reload`")

    st.divider()
    st.subheader("➕ Ingest New Document")
    url_input = st.text_input("URL", placeholder="https://example.com/docs")
    if st.button("Ingest URL") and url_input:
        with st.spinner("Ingesting..."):
            try:
                r = requests.post(f"{API_BASE}/ingest", json={"urls": [url_input]}, timeout=60)
                if r.ok:
                    d = r.json()
                    st.success(f"✅ {d['chunks_created']} chunks added")
                else:
                    st.error(r.json().get("detail", "Ingestion failed"))
            except Exception as e:
                st.error(str(e))

    uploaded = st.file_uploader("Upload file", type=["md", "txt", "html", "pdf"])
    if uploaded and st.button("Ingest File"):
        with st.spinner("Ingesting..."):
            try:
                r = requests.post(
                    f"{API_BASE}/ingest/file",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                    timeout=60,
                )
                if r.ok:
                    d = r.json()
                    st.success(f"✅ {d['chunks_created']} chunks added")
                else:
                    st.error(r.json().get("detail", "Ingestion failed"))
            except Exception as e:
                st.error(str(e))

    st.divider()
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ── Chat UI ────────────────────────────────────────────────────────────────────
st.title("💬 Technical Documentation Assistant")
st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

# Replay messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    score = src.get("relevance_score", 0)
                    st.markdown(f"**{src['title']}** — score: `{score:.2f}`")
                    st.caption(src.get("snippet", ""))
                    st.divider()
        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(4)
            cols[0].caption(f"Type: `{m.get('query_type', '?')}`")
            cols[1].caption(f"Retries: `{m.get('retries', 0)}`")
            cols[2].caption(f"Time: `{m.get('processing_time_ms', 0):.0f}ms`")
            hc = m.get("hallucination_check_passed")
            cols[3].caption(f"Grounded: `{'✅' if hc else '⚠️' if hc is False else 'n/a'}`")


# Chat input
if question := st.chat_input("Ask about LangGraph, FastAPI, RAG..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "top_k": top_k,
                    },
                    timeout=120,
                )
                if r.ok:
                    data = r.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 {len(sources)} source(s)"):
                            for src in sources:
                                score = src.get("relevance_score", 0)
                                st.markdown(f"**{src['title']}** — score: `{score:.2f}`")
                                st.caption(src.get("snippet", ""))
                                st.divider()

                    meta = {
                        "query_type": data.get("query_type"),
                        "retries": data.get("retries", 0),
                        "processing_time_ms": data.get("processing_time_ms", 0),
                        "hallucination_check_passed": data.get("hallucination_check_passed"),
                    }
                    cols = st.columns(4)
                    cols[0].caption(f"Type: `{meta['query_type']}`")
                    cols[1].caption(f"Retries: `{meta['retries']}`")
                    cols[2].caption(f"Time: `{meta['processing_time_ms']:.0f}ms`")
                    hc = meta["hallucination_check_passed"]
                    cols[3].caption(f"Grounded: `{'✅' if hc else '⚠️' if hc is False else 'n/a'}`")

                    # Feedback
                    fb_col1, fb_col2 = st.columns(2)
                    if fb_col1.button("👍", key=f"up_{len(st.session_state.messages)}"):
                        requests.post(f"{API_BASE}/feedback", json={
                            "question": question, "answer": answer, "rating": "thumbs_up",
                            "session_id": st.session_state.session_id,
                        }, timeout=5)
                        st.toast("Thanks for the feedback!")
                    if fb_col2.button("👎", key=f"down_{len(st.session_state.messages)}"):
                        requests.post(f"{API_BASE}/feedback", json={
                            "question": question, "answer": answer, "rating": "thumbs_down",
                            "session_id": st.session_state.session_id,
                        }, timeout=5)
                        st.toast("Thanks for the feedback!")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "meta": meta,
                    })
                else:
                    err = r.json().get("detail", "Unknown error")
                    st.error(f"API Error: {err}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is `uvicorn app.main:app --reload` running?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
