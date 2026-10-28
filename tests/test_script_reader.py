"""Đọc kịch bản: dán văn bản, nhận mp3 + SRT (không có video)."""
from __future__ import annotations

import pytest

from app.extensions import db
from app.models import Job, JobKind, JobStatus, TranscriptSegment
from core import script_reader
from core.pipeline import DubbingConfig, DubbingPipeline
from core.script_reader import MAX_SENTENCE_CHARS, split_script
from core.transcriber import Segment
from core.tts_engine import TTSEngine
from tests.test_chatterbox import _set_english_sample


# ── Tách câu ─────────────────────────────────────────────────
def test_split_sentences_and_paragraphs():
    text = 'Xin chào. Hôm nay trời đẹp! Bạn khoẻ không?\n\n  Đoạn hai "trích dẫn." Hết…  Thật\n'
    assert split_script(text) == [
        (0, "Xin chào."), (0, "Hôm nay trời đẹp!"), (0, "Bạn khoẻ không?"),
        (1, 'Đoạn hai "trích dẫn."'), (1, "Hết…"), (1, "Thật"),
    ]


def test_numbers_and_decimals_stay_in_one_sentence():
    assert split_script("Giá tăng 3.5% trong năm 2024.") == [(0, "Giá tăng 3.5% trong năm 2024.")]


def test_long_sentence_split_at_commas_then_spaces():
    clause = "một vế câu khá dài để thử chia nhỏ"
    long_sentence = ", ".join([clause] * 20) + "."
    pieces = [s for _, s in split_script(long_sentence)]
    assert len(pieces) > 1 and all(len(p) <= MAX_SENTENCE_CHARS for p in pieces)
    assert " ".join(pieces).replace(" ", "") == long_sentence.replace(" ", "")

    no_punctuation = " ".join(["chữ"] * 200)
    pieces = [s for _, s in split_script(no_punctuation)]
    assert all(len(p) <= MAX_SENTENCE_CHARS for p in pieces)
    assert " ".join(pieces) == no_punctuation


def test_blank_script_has_no_sentences():
    assert split_script("  \n\n ") == []


# ── Ghép thành một file ──────────────────────────────────────
def _tone(path, seconds):
    from pydub.generators import Sine

    Sine(440).to_audio_segment(duration=int(seconds * 1000)).export(str(path), format="mp3")
    return path


def test_assemble_pauses_longer_between_paragraphs(tmp_path):
    pytest.importorskip("pydub")
    pairs = [
        (Segment(0, 0, "a", "a"), _tone(tmp_path / "a.mp3", 1.0)),
        (Segment(0, 0, "b", "b"), _tone(tmp_path / "b.mp3", 1.0)),
        (Segment(1, 1, "c", "c"), _tone(tmp_path / "c.mp3", 1.0)),
    ]
    timings = script_reader.assemble(pairs, tmp_path / "out.mp3")
    sentence_gap = timings[1][0] - timings[0][1]
    paragraph_gap = timings[2][0] - timings[1][1]
    assert sentence_gap == pytest.approx(script_reader.SENTENCE_PAUSE_SEC, abs=0.05)
    assert paragraph_gap == pytest.approx(script_reader.PARAGRAPH_PAUSE_SEC, abs=0.05)
    assert (tmp_path / "out.mp3").stat().st_size > 0


def test_pipeline_writes_mp3_and_srt(monkeypatch, tmp_path):
    pytest.importorskip("pydub")

    def fake_synthesize_all(self, segments):
        return [(seg, _tone(tmp_path / f"{i}.mp3", 0.5)) for i, seg in enumerate(segments) if seg.text != "!!"]

    monkeypatch.setattr(TTSEngine, "synthesize_all", fake_synthesize_all)
    segments = [Segment(0, 0, "One.", "One."), Segment(0, 0, "!!", "!!"), Segment(1, 1, "Two.", "Two.")]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = DubbingPipeline(DubbingConfig(tts_engine="edge-tts", target_language="en", output_dir=out_dir)) \
        .synthesize_script(segments)

    assert result.success, result.error
    assert result.output_video.suffix == ".mp3" and result.output_video.exists()
    # Câu không đọc được không có mốc; phụ đề chỉ có câu đã đọc.
    assert result.play_timings[1] is None and result.play_timings[2][0] > result.play_timings[0][1]
    srt = result.srt_path.read_text(encoding="utf-8")
    assert "One." in srt and "Two." in srt and "!!" not in srt


# ── API ──────────────────────────────────────────────────────
def _post_script(client, monkeypatch, **form):
    import app.api as api_mod

    captured: dict = {}
    monkeypatch.setattr(api_mod, "start_script_job",
                        lambda flask_app, job_id, config: captured.update(job_id=job_id, config=config))
    data = {"script": "Xin chào các bạn. Đây là kịch bản.\nĐoạn thứ hai.", "target_language": "vi",
            "tts_engine": "vieneu", "tts_voice": "female", **form}
    return client.post("/api/script", data=data), captured


def test_create_script_job_saves_sentences(app, as_user, monkeypatch):
    response, captured = _post_script(as_user, monkeypatch)
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    assert captured["job_id"] == job_id and captured["config"].tts_engine == "vieneu"
    with app.app_context():
        job = db.session.get(Job, job_id)
        assert job.kind == JobKind.SCRIPT and job.status == JobStatus.QUEUED
        assert job.source_filename == "Xin chào các bạn. Đây là kịch bản."
        assert job.download_video_name == "Xin chào các bạn. Đây là kịch bản (giọng đọc).mp3"
        rows = TranscriptSegment.query.filter_by(job_id=job_id).order_by(TranscriptSegment.idx).all()
        assert [(r.start_sec, r.text_target) for r in rows] == [
            (0, "Xin chào các bạn."), (0, "Đây là kịch bản."), (1, "Đoạn thứ hai."),
        ]


def test_script_rejects_empty_too_long_and_wrong_engine(app, as_user, monkeypatch):
    from config.settings import SCRIPT_MAX_CHARS

    assert _post_script(as_user, monkeypatch, script="   ")[0].status_code == 400
    assert _post_script(as_user, monkeypatch, script="a" * (SCRIPT_MAX_CHARS + 1))[0].status_code == 400
    # Chatterbox chỉ đọc tiếng Anh; kịch bản không dịch nên ngôn ngữ phải khớp.
    assert _post_script(as_user, monkeypatch, tts_engine="chatterbox")[0].status_code == 400
    # Tiếng Nhật chỉ có ở lồng tiếng video (dịch sang) — đọc kịch bản rơi về tiếng Việt.
    response, captured = _post_script(as_user, monkeypatch, target_language="ja")
    assert response.status_code == 202 and captured["config"].target_language == "vi"


def test_script_clone_needs_sample_in_that_language(app, as_user, user, monkeypatch):
    form = {"script": "Hello there.", "target_language": "en", "tts_engine": "chatterbox", "tts_voice": "mine"}
    assert _post_script(as_user, monkeypatch, **form)[0].status_code == 400
    path = _set_english_sample(app, user)
    response, captured = _post_script(as_user, monkeypatch, **form)
    assert response.status_code == 202 and captured["config"].tts_ref_audio == str(path)


def test_run_script_job_marks_done_with_audio(app, as_user, monkeypatch, tmp_path):
    from app.jobs import run_script_job
    from core.pipeline import DubbingResult

    response, captured = _post_script(as_user, monkeypatch)
    job_id = response.get_json()["job_id"]
    audio, srt = tmp_path / "x_script.mp3", tmp_path / "x_script.srt"
    audio.write_bytes(b"mp3")
    srt.write_bytes(b"1\n")

    def fake(self, segments, progress_cb=None):
        return DubbingResult(output_video=audio, srt_path=srt, success=True, segments=segments,
                             play_timings=[(0.0, 1.0), (1.35, 2.0), (2.9, 4.0)], timings={"tts": 3.0})

    monkeypatch.setattr(DubbingPipeline, "synthesize_script", fake)
    run_script_job(app, job_id, captured["config"])
    with app.app_context():
        job = db.session.get(Job, job_id)
        assert job.status == JobStatus.DONE and job.video_url == "/media/output/x_script.mp3"
        assert job.duration_sec == 4.0 and job.file_size == 5
        assert job.segments.all()[1].play_start_sec == 1.35
    page = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert "<audio" in page and "Tải âm thanh" in page


def test_script_page_in_sidebar(as_user):
    html = as_user.get("/doc-kich-ban").get_data(as_text=True)
    assert 'setMode("script")' in html and 'name="script"' in html
    assert 'href="/doc-kich-ban"' in as_user.get("/").get_data(as_text=True)
    assert 'setMode("video")' in as_user.get("/").get_data(as_text=True)


def test_script_page_needs_login(client):
    assert as_login_redirect(client.get("/doc-kich-ban"))


def as_login_redirect(response) -> bool:
    return response.status_code in (301, 302) and "/login" in response.headers["Location"]
