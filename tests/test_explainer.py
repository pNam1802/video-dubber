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
