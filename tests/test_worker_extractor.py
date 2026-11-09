from pathlib import Path

from worker.extractor import parse_vtt


def test_parse_vtt_strips_youtube_markup_and_preserves_timestamps(tmp_path):
    path = tmp_path / "sample.en.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:07.750 --> 00:00:09.000 align:start position:0%\n"
        "what<00:00:08.760><c> is</c> up <c>YouTube</c>\n\n"
        "00:00:09.000 --> 00:00:11.000\n"
        "we are officially 11\n",
        encoding="utf-8",
    )

    segments = parse_vtt(path)

    assert segments[0]["start"] == 7.75
    assert segments[0]["text"] == "what is up YouTube"
    assert segments[1]["start"] == 9.0
    assert segments[1]["text"] == "we are officially 11"


def test_parse_vtt_rejects_empty_caption_file(tmp_path):
    path = tmp_path / "empty.vtt"
    path.write_text("WEBVTT\n\n", encoding="utf-8")

    try:
        parse_vtt(path)
    except RuntimeError as exc:
        assert "no usable subtitle cues" in str(exc)
    else:
        raise AssertionError("Expected empty VTT to fail")
