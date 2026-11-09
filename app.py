"""Streamlit frontend; all RAG/ingestion work is performed by FastAPI."""
from __future__ import annotations

import os
import uuid
import time
import requests
import streamlit as st

st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎥", layout="wide")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:16]

# Backend configuration
try:
    API_URL = (
        st.secrets.get("RAG_API_URL")
        or os.getenv("RAG_API_URL")
        or "http://localhost:8000"
    ).rstrip("/")

    API_KEY = (
        st.secrets.get("RAG_API_KEY")
        or os.getenv("RAG_API_KEY")
        or ""
    ).strip()

except Exception:
    API_URL = os.getenv(
        "RAG_API_URL", "http://localhost:8000"
    ).rstrip("/")

    API_KEY = os.getenv("RAG_API_KEY", "").strip()



def api_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "Missing RAG_API_KEY. Set it in Streamlit Cloud → App settings → Secrets "
            "or in your local environment."
        )
    return {
        "X-API-Key": API_KEY,
        "X-Session-ID": st.session_state["session_id"],
    }


def api_post(path: str, **kwargs):
    return requests.post(f"{API_URL}{path}", headers=api_headers(), **kwargs)


def api_get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", headers=api_headers(), **kwargs)


st.title("🎥 YouTube RAG Assistant")
st.caption("Ingest videos, ask questions, and inspect timestamped evidence.")

with st.sidebar:
    st.subheader("Backend Configuration")
    st.code(API_URL)
    st.caption(f"Session ID: `{st.session_state['session_id']}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Liveness"):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                response.raise_for_status()
                st.success("API is live")
            except Exception as exc:
                st.error(f"Unavailable: {exc}")
    with col2:
        if st.button("Readiness"):
            try:
                response = requests.get(f"{API_URL}/ready", timeout=5)
                if response.ok:
                    st.success("System ready")
                    st.json(response.json())
                else:
                    st.warning("Not fully ready")
                    st.json(response.json())
            except Exception as exc:
                st.error(f"Readiness check failed: {exc}")

st.header("1. Ingest a YouTube video")
with st.form("ingest_form"):
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    refresh = st.checkbox("Refresh if already ingested", value=False)
    submitted = st.form_submit_button("Ingest video")

if submitted:
    clean_url = video_url.strip()
    if not clean_url:
        st.warning("Enter a valid YouTube URL first.")
    elif "youtube.com" not in clean_url and "youtu.be" not in clean_url:
        st.error("Please provide a valid YouTube domain (youtube.com or youtu.be).")
    else:
        try:
            with st.spinner("Queueing transcript extraction on your local worker..."):
                response = api_post(
                    "/ingest/jobs",
                    json={"video_url": clean_url, "refresh": refresh},
                    timeout=15,
                )
                if response.status_code == 429:
                    st.warning("Rate limit exceeded. Please wait before trying again.")
                elif response.status_code == 401:
                    st.error("Authentication failed. Check your RAG_API_KEY.")
                else:
                    response.raise_for_status()
                    job = response.json()
                    job_id = job["job_id"]
                    st.info(f"Ingestion job queued: {job_id}")
                    progress = st.progress(0)
                    status_box = st.empty()
                    started = time.monotonic()
                    terminal = {"completed", "failed"}
                    while time.monotonic() - started < 1800:
                        status_response = api_get(f"/ingest/jobs/{job_id}", timeout=15)
                        status_response.raise_for_status()
                        job = status_response.json()
                        state = job.get("status", "queued")
                        status_box.info(f"Job status: {state} — {job.get('message', '')}")
                        if state in terminal:
                            progress.progress(100 if state == "completed" else 0)
                            if state == "completed":
                                st.success("Transcript extracted locally and ingested into the cloud RAG pipeline.")
                                if job.get("result"):
                                    st.json(job["result"])
                            else:
                                st.error(f"Ingestion failed: {job.get('message', 'Unknown worker error')}")
                            break
                        progress.progress(min(95, int((time.monotonic() - started) / 1800 * 95)))
                        time.sleep(3)
                    else:
                        st.warning("Job is still running. You can check its status using job ID: " + job_id)
        except RuntimeError as exc:
            st.error(str(exc))
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", str(exc))
            try:
                if exc.response is not None:
                    payload = exc.response.json()
                    if "detail" in payload:
                        detail = payload["detail"]
            except Exception:
                pass
            st.error(f"Ingestion failed: {detail}")

st.divider()
st.header("2. Ask questions across ingested videos")
question = st.text_input("Your question", max_chars=200, placeholder="e.g. What is the return on focus?")
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            response = api_post(
                "/ask",
                json={"question": question.strip()},
                timeout=120,
            )
            if response.status_code == 429:
                st.warning("Rate limit exceeded. Please wait a moment before sending another question.")
            elif response.status_code == 400:
                detail = response.json().get("detail", "Invalid question.")
                st.warning(f"Request rejected: {detail}")
            elif response.status_code == 503:
                st.error("Backend service is initializing or rate limiter is temporarily unavailable.")
            else:
                response.raise_for_status()
                payload = response.json()
                st.subheader("Answer")
                st.write(payload["answer"])
                st.subheader("Sources")
                sources = payload.get("sources", [])
                if not sources:
                    st.caption("No sources matched.")
                for index, source in enumerate(sources, start=1):
                    label = f"Source {index} · Video {source.get('video_id') or 'YouTube'}"
                    if source.get("start_seconds") is not None:
                        label += f" @ {int(source['start_seconds'])}s"
                    with st.expander(label):
                        if source.get("url"):
                            st.markdown(f"[Open on YouTube at timestamp]({source['url']})")
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
                aliases = [val.strip() for val in aliases_text.split(",") if val.strip()]
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
                st.success("Entity registered successfully")
                st.json(response.json())
            except RuntimeError as exc:
                st.error(str(exc))
            except requests.RequestException as exc:
                detail = getattr(getattr(exc, "response", None), "text", str(exc))
                st.error(f"Registration failed: {detail}")

with st.form("entity_resolution"):
    mention = st.text_input("Possible transcript spelling", placeholder="Alex Hormozy")
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
            f"Knowledge graph: {stats.get('entities', 0)} entities · "
            f"{stats.get('relationships', 0)} relationships"
        )
except (requests.RequestException, RuntimeError):
    pass
