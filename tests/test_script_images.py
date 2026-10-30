"""Ảnh minh hoạ cho "Đọc kịch bản": lập cảnh, prompt, ghép slideshow."""
from __future__ import annotations

import json
import shutil

import pytest

from app.extensions import db
from app.models import Job, JobStatus
from core import image_client, script_images
from core.pipeline import DubbingConfig, DubbingPipeline
from core.script_images import NO_TEXT, plan
from core.transcriber import Segment
from core.tts_engine import TTSEngine
from tests.test_script_reader import _post_script, _tone


def _timings(count: int, sentence_sec: float = 4.0, pause: float = 0.35):
    out, t = [], 0.0
    for _ in range(count):
        out.append((t, t + sentence_sec))
        t += sentence_sec + pause
    return out


# ── Lập cảnh ─────────────────────────────────────────────────
def test_scenes_mode_one_image_per_scene_covering_whole_audio():
    timings = _timings(10)  # ~43 giây
    total = timings[-1][1]
    starts, slides = plan(timings, total, "scenes")
    assert starts[0] == 0 and len(starts) == len(slides) >= 3
    assert [s[0] for s in slides] == list(range(len(starts)))
    # Liền mạch từ 0 tới hết file, đổi ảnh đúng lúc một câu bắt đầu.
    assert slides[0][1] == 0.0 and slides[-1][2] == total
    assert all(a[2] == b[1] for a, b in zip(slides, slides[1:]))
    assert all(start in [t[0] for t in timings] for _, start, _ in slides[1:])


def test_scenes_mode_caps_image_count_for_long_scripts():
    timings = _timings(400)  # ~29 phút
    starts, _ = plan(timings, timings[-1][1], "scenes")
    assert len(starts) <= script_images.MAX_SCENES + 1


def test_few_mode_cycles_three_to_five_images():
    timings = _timings(60)  # ~4,3 phút
    starts, slides = plan(timings, timings[-1][1], "few")
    assert script_images.FEW_MIN <= len(starts) <= script_images.FEW_MAX
    assert [s[0] for s in slides][:len(starts) + 1] == list(range(len(starts))) + [0]
    assert len(slides) > len(starts)  # có lặp lại


def test_few_mode_short_script_never_more_images_than_sentences():
    starts, slides = plan(_timings(2), _timings(2)[-1][1], "few")
    assert len(starts) == 2 and {s[0] for s in slides} <= {0, 1}


# ── Prompt ───────────────────────────────────────────────────
def test_parse_prompts_adds_style_and_fills_missing():
    raw = json.dumps({"style": "watercolor", "prompts": ["a robot in a maze"]})
    prompts = script_images.parse_prompts(raw, ["Robot đi tìm đường.", "Robot nhận thưởng."])
    assert prompts[0] == f"a robot in a maze, watercolor, {NO_TEXT}"
    assert prompts[1].startswith("Robot nhận thưởng.") and NO_TEXT in prompts[1]


def test_write_prompts_falls_back_to_narration(monkeypatch):
    import config.settings as settings

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    prompts = script_images.write_prompts(["Câu một.", "Câu hai."])
    assert prompts == script_images.fallback_prompts(["Câu một.", "Câu hai."])


# ── Pipeline ─────────────────────────────────────────────────
def _fake_tts(monkeypatch, tmp_path):
    def fake_synthesize_all(self, segments):
        return [(seg, _tone(tmp_path / f"{i}.mp3", 1.0)) for i, seg in enumerate(segments)]

    monkeypatch.setattr(TTSEngine, "synthesize_all", fake_synthesize_all)


def _png(color: str) -> bytes:
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (image_client.WIDTH, image_client.HEIGHT), color).save(buf, format="PNG")
    return buf.getvalue()


def test_pipeline_makes_mp4_with_images(monkeypatch, tmp_path):
    pytest.importorskip("pydub")
    pytest.importorskip("PIL")
    if not shutil.which("ffmpeg"):
        pytest.skip("cần ffmpeg")
    _fake_tts(monkeypatch, tmp_path)
    seen: dict = {}
    monkeypatch.setattr(script_images, "write_prompts", lambda chunks: [f"p{i}" for i in range(len(chunks))])
    monkeypatch.setattr(image_client, "generate",
                        lambda prompts: seen.setdefault("prompts", prompts) and [_png("red") for _ in prompts])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    segments = [Segment(0, 0, f"Câu {i}.", f"Câu {i}.") for i in range(6)]
    result = DubbingPipeline(DubbingConfig(tts_engine="edge-tts", output_dir=out_dir, script_images="few")) \
        .synthesize_script(segments)

    assert result.success, result.error
    assert result.output_video.suffix == ".mp4" and result.output_video.exists()
    assert not result.output_video.with_suffix(".mp3").exists()
    assert result.srt_path.exists() and not result.notice
    assert seen["prompts"] == ["p0", "p1", "p2"]
    assert "images" in result.timings


def test_pipeline_keeps_mp3_when_images_fail(monkeypatch, tmp_path):
    pytest.importorskip("pydub")
    _fake_tts(monkeypatch, tmp_path)
    monkeypatch.setattr(script_images, "write_prompts", lambda chunks: ["p"] * len(chunks))

    def boom(prompts):
        raise RuntimeError("GPU hết chỗ")

    monkeypatch.setattr(image_client, "generate", boom)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = DubbingPipeline(DubbingConfig(tts_engine="edge-tts", output_dir=out_dir, script_images="scenes")) \
        .synthesize_script([Segment(0, 0, "Một.", "Một."), Segment(0, 0, "Hai.", "Hai.")])

    assert result.success and result.output_video.suffix == ".mp3" and result.output_video.exists()
    assert "GPU hết chỗ" in result.notice


# ── API + trang job ──────────────────────────────────────────
def test_script_job_passes_image_mode(as_user, monkeypatch):
    response, captured = _post_script(as_user, monkeypatch, script_images="scenes")
    assert response.status_code == 202 and captured["config"].script_images == "scenes"
    _, captured = _post_script(as_user, monkeypatch, script_images="lots")
    assert captured["config"].script_images == "none"


def test_illustrated_script_job_shows_video(app, as_user, monkeypatch, tmp_path):
    from app.jobs import run_script_job
    from core.pipeline import DubbingResult

    response, captured = _post_script(as_user, monkeypatch, script_images="few")
    job_id = response.get_json()["job_id"]
    video, srt = tmp_path / "x_script.mp4", tmp_path / "x_script.srt"
    video.write_bytes(b"mp4")
    srt.write_bytes(b"1\n")

    def fake(self, segments, progress_cb=None):
        return DubbingResult(output_video=video, srt_path=srt, success=True, segments=segments,
                             play_timings=[(0.0, 1.0), (1.35, 2.0), (2.9, 4.0)],
                             timings={"tts": 3.0, "images": 100.0})

    monkeypatch.setattr(DubbingPipeline, "synthesize_script", fake)
    run_script_job(app, job_id, captured["config"])
    with app.app_context():
        job = db.session.get(Job, job_id)
        assert job.status == JobStatus.DONE and not job.is_audio_only
        assert job.download_video_name.endswith("(minh hoạ).mp4")
        assert job.estimated_cost_usd > 0.05  # có tính tiền GPU vẽ ảnh
    page = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert '<video id="resultVideo"' in page and "Tải video" in page


def test_script_page_offers_image_modes(as_user):
    html = as_user.get("/doc-kich-ban").get_data(as_text=True)
    for mode in script_images.MODES:
        assert f'name="script_images" value="{mode}"' in html
