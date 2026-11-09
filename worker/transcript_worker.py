"""Run on the trusted local machine; cookies never leave this process."""
from __future__ import annotations

import logging
import os
import socket
import time
import requests

from config import settings
from infrastructure.ingestion_queue import (
    read_jobs, acknowledge, retry_or_fail, set_job_status,
)
from worker.extractor import extract_transcript

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("transcript-worker")


def run() -> None:
    token = os.getenv("TRANSCRIPT_WORKER_TOKEN", "").strip()
    cookies = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
    api_url = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
    if not token or not cookies:
        raise RuntimeError("Set TRANSCRIPT_WORKER_TOKEN and YOUTUBE_COOKIES_FILE in local .env.")
    session = requests.Session()
    session.headers.update({"X-Transcript-Worker-Token": token})
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    logger.info("Worker started; consumer=%s api=%s", consumer, api_url)
    while True:
        batches = read_jobs(consumer, block_ms=5000)
        if not batches:
            continue
        for _stream, messages in batches:
            for message_id, fields in messages:
                job_id = fields["job_id"]
                try:
                    set_job_status(job_id, "extracting", "Extracting captions on the local worker.")
                    segments = extract_transcript(
                        fields["video_url"], cookies,
                        language=os.getenv("TRANSCRIPT_LANGUAGE", "en"),
                    )
                    set_job_status(job_id, "uploading", "Transcript extracted; uploading to cloud ingestion.")
                    response = session.post(
                        f"{api_url}/worker/transcripts",
                        json={
                            "job_id": job_id,
                            "video_url": fields["video_url"],
                            "refresh": fields.get("refresh", "false").lower() == "true",
                            "segments": segments,
                        },
                        timeout=(10, 180),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    set_job_status(job_id, "completed", payload.get("message", "Video ingested."), result=payload)
                    acknowledge(message_id)
                    logger.info("Job %s completed", job_id)
                except Exception as exc:
                    logger.exception("Job %s failed", job_id)
                    retry_or_fail(message_id, fields, f"{type(exc).__name__}: {exc}")
        time.sleep(0.2)


if __name__ == "__main__":
    run()
