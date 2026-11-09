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


# ── Lịch chiếu ───────────────────────────────────────────────
def _sentences(count: int):
    return [f"Câu {i}." for i in range(count)]


def _no_gemini(monkeypatch):
    def fail(*args):
        raise RuntimeError("không có key")

    monkeypatch.setattr(script_images, "write_storyboard", fail)


def test_fallback_scenes_cover_whole_audio(monkeypatch):
    _no_gemini(monkeypatch)
    timings = _timings(10)  # ~43 giây
    prompts, slides = plan(_sentences(10), timings, "scenes")
    assert len(prompts) == len(slides) >= 3
    assert [s[0] for s in slides] == list(range(len(prompts)))
    # Liền mạch từ 0 tới hết file, đổi ảnh đúng lúc một câu bắt đầu.
    assert slides[0][1] == 0.0 and slides[-1][2] == timings[-1][1]
    assert all(a[2] == b[1] for a, b in zip(slides, slides[1:]))
    assert all(start in [t[0] for t in timings] for _, start, _ in slides[1:])
    assert all(script_images.PLAIN_BACKGROUND in p for p in prompts)


def test_fallback_few_cycles_three_to_five_images(monkeypatch):
    _no_gemini(monkeypatch)
    timings = _timings(60)  # ~4,3 phút
    prompts, slides = plan(_sentences(60), timings, "few")
    assert script_images.FEW_MIN <= len(prompts) <= script_images.FEW_MAX
    assert [s[0] for s in slides][:len(prompts) + 1] == list(range(len(prompts))) + [0]


def test_few_mode_short_script_never_more_images_than_sentences():
    assert script_images.few_count(_timings(2)) == 2


def test_tidy_starts_merges_short_scenes_and_caps_count():
    timings = _timings(400)  # ~29 phút
    kept = script_images.tidy_starts(list(range(400)), timings)
    assert kept[0] == 0 and len(kept) <= script_images.MAX_SCENES
    # Cảnh chỉ một câu (~4,35 giây) quá ngắn -> gộp vào cảnh trước.
    assert script_images.tidy_starts([0, 1, 2, 5], _timings(10)) == [0, 2, 3]


# ── Storyboard ───────────────────────────────────────────────
BOARD = {
    "main_character": {"look": "young maid with long hair and a ribbon."},
    "scenes": [
        {"start": 4, "main": False, "shot": "close-up", "action": "extreme close-up of an old man's furious eye"},
        {"start": 2, "main": True, "shot": "Full body shot", "action": "Daisy waters the flowers."},
        {"start": 9, "main": True, "action": ""},
    ],
}


def test_parse_storyboard_main_character_or_symbolic():
    board = script_images.parse_storyboard(json.dumps(BOARD), sentence_count=8)
    assert [start for start, _ in board] == [0, 3]  # sắp theo câu, cảnh đầu phủ từ câu 0
    daisy, symbolic = (prompt for _, prompt in board)
    assert daisy.startswith("full body shot, Daisy waters the flowers, The only character in the image: young maid")
    assert "ribbon, Solo" in daisy and script_images.PLAIN_BACKGROUND in daisy
    assert daisy.endswith(f"{script_images.MANGA_STYLE}, {NO_TEXT}")
    # Vắng nhân vật chính: cận cảnh tượng trưng, không chèn ngoại hình hay cỡ cảnh.
    assert script_images.SYMBOLIC_LOOK in symbolic and "maid" not in symbolic
    assert symbolic.startswith("extreme close-up of") and script_images.MANGA_STYLE in symbolic


def test_parse_storyboard_ignores_unknown_shot():
    raw = json.dumps({"main_character": "a boy",
                      "scenes": [{"start": 1, "main": True, "shot": "drone flyover", "action": "running"}]})
    [(_, prompt)] = script_images.parse_storyboard(raw, sentence_count=1)
    assert prompt.startswith("running, The only character")


def test_parse_storyboard_without_main_character_is_all_symbolic():
    raw = json.dumps({"main_character": None, "scenes": [{"start": 1, "main": True, "action": "a glowing star"}]})
    [(_, prompt)] = script_images.parse_storyboard(raw, sentence_count=3)
    assert script_images.SYMBOLIC_LOOK in prompt and script_images.MANGA_STYLE in prompt


@pytest.mark.parametrize("main_character, main", [
    ("young maid with blonde hair", True),
    ({"name": "Daisy", "appearance": "young maid with blonde hair"}, "true"),
])
def test_parse_storyboard_tolerates_loose_json(main_character, main):
    raw = json.dumps({"main_character": main_character,
                      "scenes": [{"start": 1, "main": main, "action": "sweeping"}]})
    [(_, prompt)] = script_images.parse_storyboard(raw, sentence_count=3)
    assert "The only character in the image: young maid with blonde hair" in prompt
    assert script_images.SYMBOLIC_LOOK not in prompt


def test_storyboard_request_carries_rules_and_numbered_sentences():
    request = script_images.storyboard_request(["Một.", "Hai."], [(0.0, 1.0), (61.0, 62.0)], "scenes", 2)
    assert "never show any other character and never show scenery" in request
    assert "black and white manga panel" in request and "dynamic action pose" in request
    assert "[1] (0:00) Một." in request and "[2] (1:01) Hai." in request


def test_plan_uses_storyboard_for_scenes(monkeypatch):
    timings = _timings(10)
    monkeypatch.setattr(script_images, "write_storyboard",
                        lambda sentences, timings, mode, count: [(0, "a"), (1, "b"), (5, "c")])
    prompts, slides = plan(_sentences(10), timings, "scenes")
    assert prompts == ["a", "c"]  # cảnh "b" chỉ một câu -> gộp vào "a"
    assert slides[1][1] == timings[5][0]


def test_write_storyboard_needs_key(monkeypatch):
    import config.settings as settings

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError):
        script_images.write_storyboard(["Câu."], [(0.0, 1.0)], "few", 1)


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
    monkeypatch.setattr(script_images, "write_storyboard",
                        lambda sentences, timings, mode, count: [(0, f"p{i}") for i in range(count)])
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
    _no_gemini(monkeypatch)

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
    # Giá trị lạ (hoặc không gửi) thì về mặc định của hệ thống: Manhwa recap.
    _, captured = _post_script(as_user, monkeypatch, script_images="lots")
    assert captured["config"].script_images == "recap"


def test_script_job_passes_recap_character_style(as_user, monkeypatch):
    _, captured = _post_script(as_user, monkeypatch, script_images="recap", recap_style="sunflower")
    assert captured["config"].recap_style == "sunflower"
    _, captured = _post_script(as_user, monkeypatch, script_images="recap", recap_style="robot")
    assert captured["config"].recap_style == "sunflower"


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


def test_script_page_always_uses_recap_with_character_choice(as_user):
    """Trang đọc kịch bản không còn cho chọn chế độ ảnh — luôn Manhwa recap,
    chỉ chọn kiểu nhân vật."""
    html = as_user.get("/doc-kich-ban").get_data(as_text=True)
    assert '<input type="hidden" name="script_images" value="recap">' in html
    assert 'type="radio" name="script_images"' not in html
    for style in script_images.RECAP_CHARACTER_STYLES:
        assert f'name="recap_style" value="{style}"' in html
