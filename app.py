"""Streamlit frontend. Talks to FastAPI; it does not load RAG models directly."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="YouTube RAG", page_icon="🎥", layout="wide")
st.title("🎥 YouTube RAG Assistant")
st.caption("Ingest videos, ask questions, and inspect timestamped evidence.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_URL)
    if st.button("Check API health"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            response.raise_for_status()
            st.success("API is reachable")
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

st.header("1. Ingest a YouTube video")
with st.form("ingest_form"):
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    refresh = st.checkbox("Refresh if already ingested", value=False)
    ingest_submitted = st.form_submit_button("Ingest video")
if ingest_submitted:
    if not video_url.strip():
        st.warning("Enter a YouTube URL first.")
    else:
        try:
            with st.spinner("Fetching transcript, chunking, embedding, and storing..."):
                response = requests.post(
                    f"{API_URL}/ingest",
                    json={"video_url": video_url.strip(), "refresh": refresh},
                    timeout=300,
                )
                response.raise_for_status()
            st.success("Ingestion request completed")
            st.json(response.json())
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", str(exc))
            st.error(f"Ingestion failed: {detail}")

st.divider()
st.header("2. Ask questions across ingested videos")
question = st.text_input("Your question", max_chars=200, placeholder="What does the speaker say about customer acquisition?")
user_id = st.text_input("Development user ID", value="streamlit-dev-user", help="Demo only; production should derive this from authentication.")
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "user_id": user_id},
                timeout=120,
            )
            response.raise_for_status()
        payload = response.json()
        st.subheader("Answer")
        st.write(payload["answer"])
        st.subheader("Sources")
        for index, source in enumerate(payload.get("sources", []), start=1):
            with st.expander(f"Source {index} · {source.get('video_id') or 'YouTube'}"):
                if source.get("url"):
                    st.markdown(f"[Open at timestamp]({source['url']})")
                st.write(source.get("text", ""))
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", str(exc))
        st.error(f"Question failed: {detail}")

st.divider()
st.header("3. Knowledge graph")
col1, col2 = st.columns([2, 1])
with col1:
    mention = st.text_input("Check a possible entity spelling", placeholder="Alex Shomzi")
with col2:
    entity_type = st.selectbox("Entity type", ["PERSON", "ORG", "PRODUCT", "LOCATION"])
if st.button("Resolve entity", disabled=not mention.strip()):
    try:
        response = requests.post(
            f"{API_URL}/entities/resolve",
            json={"mention": mention, "entity_type": entity_type},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") == "candidate_match":
            st.info("Candidate match found. This is a suggestion, not an automatic correction.")
        elif result.get("status") == "needs_review":
            st.warning("Possible match; manual verification required.")
        else:
            st.warning("No sufficiently strong graph match found.")
        st.json(result)
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", str(exc))
        st.error(f"Entity resolution failed: {detail}")

try:
    summary = requests.get(f"{API_URL}/graph/summary", timeout=5)
    if summary.ok:
        st.caption(f"Knowledge graph: {summary.json().get('entities', 0)} nodes · {summary.json().get('relationships', 0)} relationships")
except requests.RequestException:
    pass
