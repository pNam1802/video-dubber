"""Chế độ "Video giải thích" (Manim) của Đọc kịch bản: chia cảnh, lọc đặc tả, quyền admin."""
from __future__ import annotations

import json
import shutil

import pytest

from core import explainer as ex
from core import script_images

TIMINGS = [(0.0, 4.0), (4.0, 9.0), (9.0, 15.0), (15.0, 20.0)]
SENTENCES = ["RAG là gì?", "Đầu tiên tạo embedding.", "Sau đó tìm đoạn gần nhất.", "Cuối cùng LLM trả lời."]


def test_mode_is_registered():
    assert "explainer" in script_images.MODES


def test_parse_scenes_keeps_only_known_layouts_and_fields():
    raw = json.dumps({"scenes": [
        {"start": 2, "layout": "flow", "heading": "Luồng", "steps": ["A", "B", "", "C"], "code": "import os"},
        {"start": 1, "layout": "title", "title": "RAG"},
        {"start": 3, "layout": "python", "text": "x"},          # bố cục lạ -> bỏ
        {"start": 4, "layout": "bullets", "heading": "Rỗng"},   # không có gạch đầu dòng -> bỏ
        {"start": 99, "layout": "statement", "text": "ngoài"},  # quá số câu -> bỏ
        "rác",
    ]})
    board = ex.parse_scenes(raw, len(SENTENCES))
    assert board == [(0, {"layout": "title", "title": "RAG"}),
                     (1, {"layout": "flow", "heading": "Luồng", "steps": ["A", "B", "C"]})]


def test_long_text_and_lists_are_clamped():
    scene = ex.clean_scene({"layout": "vectors", "heading": "x" * 500, "query": "q",
                            "items": [f"doc {i}" for i in range(20)], "nearest": [0, "2", "a", -1]})
    assert len(scene["heading"]) == ex.FIELD_CHARS
    assert len(scene["items"]) == ex.LIST_LIMITS["items"]
    assert scene["nearest"] == [0, 2]


def test_plan_falls_back_to_statements_when_llms_fail(monkeypatch):
    import core.stock_footage as sf

    def boom(request):
        raise RuntimeError("hết hạn mức")

    monkeypatch.setattr(ex, "ask_gemini_json", boom)
    monkeypatch.setattr(sf, "ask_deepseek_json", boom)
    scenes, slides = ex.plan(SENTENCES, TIMINGS)
    assert scenes and all(s["layout"] == "statement" for s in scenes)
    assert slides[0][1] == 0.0 and slides[-1][2] == TIMINGS[-1][1]


def test_plan_uses_gemini_board_and_covers_whole_audio(monkeypatch):
    board = {"scenes": [{"start": 1, "layout": "title", "title": "RAG"},
                        {"start": 3, "layout": "flow", "steps": ["Hỏi", "Tìm", "Đáp"]}]}
    monkeypatch.setattr(ex, "ask_gemini_json", lambda request: json.dumps(board))
    scenes, slides = ex.plan(SENTENCES, TIMINGS)
    assert [s["layout"] for s in scenes] == ["title", "flow"]
    assert slides == [(0, 0.0, 9.0), (1, 9.0, 20.0)]
    assert sum(ex.frame_counts(slides)) == round(20.0 * ex.FPS)


@pytest.mark.parametrize("who, status", [("as_user", 403), ("as_admin", 202)])
def test_api_explainer_is_admin_only(request, monkeypatch, who, status):
    import app.api as api

    client = request.getfixturevalue(who)
    captured: dict = {}
    monkeypatch.setattr(api, "start_script_job", lambda flask_app, job_id, config: captured.update(config=config))
    res = client.post("/api/script", data={"script": "Xin chào. Đây là RAG.", "target_language": "vi",
                                           "tts_engine": "edge-tts", "tts_voice": "female",
                                           "script_images": "explainer"})
    assert res.status_code == status, res.get_json()
    if status == 202:
        assert captured["config"].script_images == "explainer"
    else:
        assert not captured


def test_script_form_shows_explainer_only_to_admin(as_user, as_admin):
    assert 'value="explainer"' not in as_user.get("/doc-kich-ban").get_data(as_text=True)
    assert 'value="explainer"' in as_admin.get("/doc-kich-ban").get_data(as_text=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")
def test_render_scene_has_exact_frame_count(tmp_path):
    pytest.importorskip("manim")
    import subprocess

    out = tmp_path / "scene_000.mp4"
    ex.render_scene({"layout": "statement", "text": "Chống bịa"}, 30, out, tmp_path)
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert int(frames) == 30


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")
def test_manim_render_lasts_the_whole_scene(tmp_path):
    """Manim tự ghi đè Scene.duration trong mỗi play(): từng làm cảnh chỉ dài ~1 giây rồi bị
    kéo khung cuối cho đủ — hình đứng im suốt phần lời đọc còn lại."""
    pytest.importorskip("manim")
    import subprocess

    ex.render_scene({"layout": "bullets", "heading": "Ý chính", "items": ["Một", "Hai", "Ba"],
                     "beats": [1.0, 2.0]}, 75, tmp_path / "scene_000.mp4", tmp_path)
    raw = next((tmp_path / "media_scene_000").rglob("scene_000.mp4"))
    seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(raw)], capture_output=True, text=True, check=True).stdout)
    assert seconds >= 2.9


def test_new_layouts_are_cleaned():
    probs = ex.clean_scene({"layout": "probabilities", "context": "Trời", "options": [
        {"label": "đẹp", "p": "60%"}, {"label": "mưa", "p": 40}, {"label": "", "p": 5}]})
    assert probs["options"] == [{"label": "đẹp", "p": 0.6}, {"label": "mưa", "p": 0.4}]
    tokens = ex.clean_scene({"layout": "tokens", "text": "Con mèo", "tokens": ["Con", " mèo", "  "]})
    assert tokens["tokens"] == ["Con", " mèo"]
    assert ex.clean_scene({"layout": "number", "value": "abc"}) is None
    assert ex.clean_scene({"layout": "code", "lines": "x = 1\n\n  y = 2"})["lines"] == ["x = 1", "  y = 2"]
    assert ex.clean_scene({"layout": "attention", "tokens": ["một"]}) is None
    net = ex.clean_scene({"layout": "network", "layers": [3, 99, "x", 2]})
    assert net["layers"] == [3, 6, 2]
    assert ex.scene_beats([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)], 1.0, 5.0) == [1.0, 3.0]


def test_estimate_timings_grow_with_text():
    timings = ex.estimate_timings(["Ngắn.", "Một câu dài hơn hẳn câu trước nhiều lần."])
    assert timings[0][0] == 0.0 and timings[1][0] > timings[0][1]
    assert timings[1][1] - timings[1][0] > timings[0][1] - timings[0][0]


def test_approved_board_is_used_instead_of_llm(monkeypatch):
    def boom(request):
        raise AssertionError("không được gọi LLM khi đã có phân cảnh duyệt")

    monkeypatch.setattr(ex, "ask_gemini_json", boom)
    approved = [{"start": 2, "layout": "number", "value": 175, "unit": "tỷ"},
                {"start": 0, "layout": "title", "title": "RAG"}, {"start": 9, "layout": "title", "title": "x"},
                {"start": 1, "layout": "python"}]
    scenes, slides = ex.plan(SENTENCES, TIMINGS, approved)
    assert [s["layout"] for s in scenes] == ["title", "number"]
    assert slides == [(0, 0.0, 9.0), (1, 9.0, 20.0)]


def test_api_passes_approved_storyboard_to_job(as_admin, monkeypatch):
    import app.api as api

    captured: dict = {}
    monkeypatch.setattr(api, "start_script_job", lambda flask_app, job_id, config: captured.update(config=config))
    data = {"script": "Một. Hai. Ba.", "target_language": "vi", "tts_engine": "edge-tts", "tts_voice": "female",
            "script_images": "explainer",
            "storyboard": json.dumps([{"start": 0, "layout": "title", "title": "A"},
                                      {"start": 2, "layout": "statement", "text": "B"}])}
    assert as_admin.post("/api/script", data=data).status_code == 202
    assert [s["start"] for s in captured["config"].explainer_board] == [0, 2]
    data["storyboard"] = "không phải json"
    assert as_admin.post("/api/script", data=data).status_code == 400
