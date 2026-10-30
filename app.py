"""Streamlit frontend; all RAG/ingestion work is performed by FastAPI."""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="YouTube RAG", page_icon="🎥", layout="wide")

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
try:
    API_KEY = st.secrets.get("RAG_API_KEY", os.getenv("RAG_API_KEY", ""))
except Exception:
    API_KEY = os.getenv("RAG_API_KEY", "")


def api_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "Missing RAG_API_KEY. Set it in Streamlit Cloud → App settings → Secrets "
            "or in your local environment."
        )
    return {"X-API-Key": API_KEY}


def api_post(path: str, **kwargs):
    return requests.post(f"{API_URL}{path}", headers=api_headers(), **kwargs)


def api_get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", headers=api_headers(), **kwargs)


st.title("🎥 YouTube RAG Assistant")
st.caption("Ingest videos, ask questions, and inspect timestamped evidence.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_URL)
    st.caption("Authenticated backend connection")
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
    submitted = st.form_submit_button("Ingest video")
if submitted:
    if not video_url.strip():
        st.warning("Enter a YouTube URL first.")
    else:
        try:
            with st.spinner("Fetching transcript, chunking, embedding, and storing..."):
                response = api_post(
                    "/ingest",
                    json={"video_url": video_url.strip(), "refresh": refresh},
                    timeout=300,
                )
                response.raise_for_status()
            st.success("Ingestion completed")
            st.json(response.json())
        except RuntimeError as exc:
            st.error(str(exc))
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", str(exc))
            st.error(f"Ingestion failed: {detail}")

st.divider()
st.header("2. Ask questions across ingested videos")
question = st.text_input("Your question", max_chars=200)
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            response = api_post(
                "/ask",
                json={"question": question},
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
    except RuntimeError as exc:
        st.error(str(exc))
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", str(exc))
        st.error(f"Question failed: {detail}")

st.divider()
st.header("3. Knowledge graph")
st.caption(
    "Add canonical entities only after independent verification. "
    "Fuzzy matching suggests candidates; it does not prove identity."
)
with st.expander("Register a reviewed entity"):
    with st.form("entity_registration"):
        canonical = st.text_input("Canonical name", placeholder="Alex Hormozi")
        aliases_text = st.text_input("Known aliases (comma-separated)", placeholder="Alex Hormozy")
        evidence = st.text_area(
            "Verification evidence",
            placeholder="Official video metadata or a human-reviewed source...",
        )
        source_video_id = st.text_input("Related video ID (optional)")
        verified = st.checkbox("I independently reviewed this entity and evidence")
        register = st.form_submit_button("Register entity")
    if register:
        if not canonical.strip() or not evidence.strip():
            st.warning("Canonical name and verification evidence are required.")
        else:
            try:
                aliases = [value.strip() for value in aliases_text.split(",") if value.strip()]
                response = api_post(
                    "/entities/register",
                    json={
                        "canonical_name": canonical.strip(),
                        "aliases": aliases,
                        "evidence": evidence.strip(),
                        "source_video_id": source_video_id.strip() or None,
                        "verified": verified,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                st.success("Entity registered")
                st.json(response.json())
            except RuntimeError as exc:
                st.error(str(exc))
            except requests.RequestException as exc:
                detail = getattr(getattr(exc, "response", None), "text", str(exc))
                st.error(f"Registration failed: {detail}")

with st.form("entity_resolution"):
    mention = st.text_input("Possible transcript spelling", placeholder="Alex Shomzi")
    entity_type = st.selectbox("Entity type", ["PERSON", "ORG", "PRODUCT", "LOCATION"])
    resolve = st.form_submit_button("Resolve entity")
if resolve:
    if not mention.strip():
        st.warning("Enter a name to check.")
    else:
        try:
            response = api_post(
                "/entities/resolve",
                json={"mention": mention.strip(), "entity_type": entity_type},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "candidate_match":
                st.info("Strong string candidate. Check the evidence before using it.")
            elif result.get("status") == "needs_review":
                st.warning("Possible match; manual verification required.")
            else:
                st.warning("No sufficiently strong graph match found.")
            st.json(result)
        except RuntimeError as exc:
            st.error(str(exc))
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", str(exc))
            st.error(f"Entity resolution failed: {detail}")

try:
    response = api_get("/graph/summary", timeout=5)
    if response.ok:
        stats = response.json()
        st.caption(
            f"Knowledge graph: {stats.get('entities', 0)} nodes · "
            f"{stats.get('relationships', 0)} relationships"
        )
except (requests.RequestException, RuntimeError):
    pass
