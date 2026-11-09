"""Extract YouTube captions locally using the user's authenticated browser cookies."""
from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL
from ingestion.yt_client import extract_video_id


def _timestamp(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0


def parse_vtt(path: str | Path) -> list[dict]:
    """Parse WebVTT cues, remove markup and collapse adjacent duplicate cues."""
    content = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        match = re.match(r"([\d:.]+)\s+-->", lines[timing_index])
        if not match:
            continue
        raw = " ".join(lines[timing_index + 1:])
        raw = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", " ", raw)
        raw = re.sub(r"</?c(?:\.[^>]*)?>", "", raw)
        raw = re.sub(r"<[^>]+>", "", raw)
        text = re.sub(r"\s+", " ", html.unescape(raw)).strip()
        if not text:
            continue
        start = _timestamp(match.group(1))
        # YouTube auto-captions often repeat cumulative text across adjacent cues.
        if segments and text == segments[-1]["text"]:
            continue
        segments.append({"text": text, "start": start, "duration": 0.0})
    if not segments:
        raise RuntimeError("yt-dlp produced no usable subtitle cues.")
    return segments


def extract_transcript(video_url: str, cookies_file: str, language: str = "en") -> list[dict]:
    extract_video_id(video_url)
    cookie_path = Path(cookies_file).expanduser().resolve()
    if not cookie_path.is_file():
        raise FileNotFoundError(f"Cookie file not found: {cookie_path}")
    with tempfile.TemporaryDirectory(prefix="ytchatbot-") as temp_dir:
        options = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [language, f"{language}.*"],
            "subtitlesformat": "vtt",
            "cookiefile": str(cookie_path),
            "outtmpl": str(Path(temp_dir) / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id")
        candidates = sorted(Path(temp_dir).glob(f"{video_id}*.vtt"))
        if not candidates:
            raise RuntimeError("No English subtitle file was produced for this video.")
        # Prefer the original track when yt-dlp emits both translated and original variants.
        preferred = next((p for p in candidates if "orig" in p.name), candidates[0])
        return parse_vtt(preferred)
