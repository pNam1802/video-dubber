"""Xưởng kịch bản: các bước LLM (core/studio.py) và API chỉ-admin (app/studio.py)."""
from __future__ import annotations

import pytest

from core import studio as cs

PLAN = {
    "title": "LLM & AI Agent", "description": "Từ token tới agent.", "style_guide": ["Mở bằng câu hỏi", "Một ví dụ xuyên suốt"],
    "glossary": [{"vi": "nhúng", "en": "embedding", "definition": "vector biểu diễn nghĩa"}, {"vi": "", "en": "x"}],
    "episodes": [{"title": "Token là gì", "goal": "Hiểu token", "key_points": ["BPE", "giá theo token"]},
                 {"title": "Embedding", "goal": "Hiểu embedding", "key_points": ["vector"]}, {"title": ""}],
}


def test_plan_series_cleans_llm_output():
    plan = cs.plan_series("LLM", "dev", 5, ask=lambda prompt: PLAN)
    assert [e["title"] for e in plan["episodes"]] == ["Token là gì", "Embedding"]
    assert plan["glossary"] == [{"vi": "nhúng", "en": "embedding", "definition": "vector biểu diễn nghĩa"}]
    assert plan["style_guide"] == "Mở bằng câu hỏi\nMột ví dụ xuyên suốt"


def test_clean_script_strips_markdown_and_blank_lines():
    assert cs.clean_script("# Mở đầu\n\n* Token là gì?  \n- Câu   hai.\n") == "Mở đầu\nToken là gì?\nCâu hai."


def test_merge_glossary_keeps_first_translation():
    old = [{"vi": "nhúng", "en": "Embedding", "definition": ""}]
    new = [{"vi": "véc-tơ nhúng", "en": "embedding", "definition": ""}, {"vi": "chú ý", "en": "attention", "definition": ""}]
    assert [g["vi"] for g in cs.merge_glossary(old, new)] == ["nhúng", "chú ý"]


def test_memory_context_has_glossary_and_earlier_summaries_but_not_current():
    series = {"title": "S", "topic": "t", "audience": "a", "style_guide": "ngắn gọn",
              "glossary": [{"vi": "nhúng", "en": "embedding", "definition": "d"}]}
    episodes = [{"number": 1, "title": "Token", "goal": "g1", "summary": "Đã giải thích token bằng ví dụ bánh mì."},
                {"number": 2, "title": "Embedding", "goal": "g2", "summary": "tóm tắt tập hiện tại"}]
    text = cs.memory_context(series, episodes, current_number=2)
    assert "nhúng = embedding" in text and "bánh mì" in text
    assert "tóm tắt tập hiện tại" not in text and "<- CURRENT" in text


def test_review_falls_back_to_draft_when_reviewer_returns_nothing():
    out = cs.review_script("ctx", {"number": 1, "title": "T"}, "notes", "Bản nháp.", ask=lambda p: {"issues": ["x"]})
    assert out == {"issues": ["x"], "script": "Bản nháp."}


def test_dedupe_sources():
    assert cs.dedupe_sources([{"title": "A", "url": "u"}, {"title": "A2", "url": "u"}, {"title": "", "url": ""}]) \
        == [{"title": "A", "url": "u"}]


# ── API ──────────────────────────────────────────────────────
@pytest.fixture()
def fake_agent(app, monkeypatch):
    """Chạy các bước đồng bộ, LLM giả."""
    import app.studio as st

    monkeypatch.setattr(st, "start_task", lambda flask_app, kind, target_id: st.run_task(flask_app, kind, target_id))
    monkeypatch.setattr(cs, "_ask_json", lambda prompt: {})
    real_plan = cs.plan_series
    monkeypatch.setattr(cs, "plan_series", lambda topic, audience, count: real_plan(topic, audience, count, ask=lambda p: PLAN))
    monkeypatch.setattr(cs, "research", lambda ctx, ep: ("Ghi chú về token.", [{"title": "Paper", "url": "https://arxiv.org/x"}]))
    monkeypatch.setattr(cs, "write_script", lambda ctx, ep, notes: {
        "outline": ["Mở"], "script": "Token là gì?\nHết.", "quiz": [{"q": "Token?", "a": "Mảnh chữ"}],
        "new_terms": [{"vi": "mã thông báo", "en": "token", "definition": "mảnh chữ"}]})
    monkeypatch.setattr(cs, "review_script", lambda ctx, ep, notes, script: {"issues": ["Sửa câu dài"], "script": script + "\nĐã soát."})
    monkeypatch.setattr(cs, "translate", lambda ctx, script: "What is a token?")
    monkeypatch.setattr(cs, "wrap_up", lambda script: {"summary": "Tập 1 giải thích token.", "new_terms": []})
    return st


def test_studio_is_admin_only(as_user):
    assert as_user.get("/api/studio/series").status_code == 403
    assert as_user.post("/api/studio/series", json={"topic": "x"}).status_code == 403
    assert as_user.get("/quan-tri/xuong-kich-ban").status_code == 403


def test_full_episode_flow_builds_series_memory(as_admin, fake_agent):
    res = as_admin.post("/api/studio/series", json={"topic": "LLM & AI Agent", "episodes": 2})
    assert res.status_code == 202
    series_id = res.get_json()["id"]
    series = as_admin.get(f"/api/studio/series/{series_id}").get_json()
    assert series["title"] == "LLM & AI Agent" and series["task"] is None
    assert [e["title"] for e in series["episodes"]] == ["Token là gì", "Embedding"]
    ep = series["episodes"][0]

    assert as_admin.post(f"/api/studio/episodes/{ep['id']}/write").status_code == 202
    ep = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]
    assert ep["status"] == "drafted" and ep["script_vi"].endswith("Đã soát.")
    assert ep["sources"][0]["url"].startswith("https://") and ep["review"] == ["Sửa câu dài"]

    as_admin.post(f"/api/studio/episodes/{ep['id']}/translate")
    as_admin.post(f"/api/studio/episodes/{ep['id']}/wrap")
    series = as_admin.get(f"/api/studio/series/{series_id}").get_json()
    ep = series["episodes"][0]
    assert ep["status"] == "done" and ep["script_en"] == "What is a token?" and ep["summary"]
    assert {g["en"] for g in series["glossary"]} == {"embedding", "token"}
    assert "/quan-tri/xuong-kich-ban" in as_admin.get(f"/quan-tri/xuong-kich-ban/{series_id}").get_data(as_text=True)


def test_step_error_is_recorded_not_raised(as_admin, fake_agent, monkeypatch):
    series_id = as_admin.post("/api/studio/series", json={"topic": "x", "episodes": 1}).get_json()["id"]
    ep_id = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]["id"]

    def boom(ctx, ep):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(cs, "research", boom)
    as_admin.post(f"/api/studio/episodes/{ep_id}/write")
    ep = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]
    assert ep["task"] is None and "429" in ep["error"] and ep["status"] == "planned"


def test_cannot_touch_other_admins_series(app, as_admin, fake_agent):
    from app.models import Series, User
    from app.extensions import db
    from tests.conftest import make_user

    with app.app_context():
        other = make_user("admin2", role="admin")
        series = Series(user_id=other.id, topic="riêng", title="riêng")
        db.session.add(series)
        db.session.commit()
        series_id = series.id
        assert User.query.count() >= 2
    assert as_admin.get(f"/api/studio/series/{series_id}").status_code == 404
    assert as_admin.delete(f"/api/studio/series/{series_id}").status_code == 404


def test_patch_episode_saves_script_and_only_own_jobs(as_admin, fake_agent, app):
    from tests.conftest import make_job, make_user

    series_id = as_admin.post("/api/studio/series", json={"topic": "x", "episodes": 1}).get_json()["id"]
    ep_id = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]["id"]
    with app.app_context():
        stranger_job_id = make_job(make_user("stranger").id).id
    res = as_admin.patch(f"/api/studio/episodes/{ep_id}", json={"script_vi": "## Xin chào\n\nCác bạn.",
                                                                 "job_vi_id": stranger_job_id})
    data = res.get_json()
    assert data["script_vi"] == "Xin chào\nCác bạn." and data["job_vi_id"] is None


def test_series_page_ships_voice_previews(as_admin, fake_agent, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "_voice_preview_urls", lambda: {"vieneu": {"female": "/static/voices/vieneu/female.mp3"}})
    series_id = as_admin.post("/api/studio/series", json={"topic": "x", "episodes": 1}).get_json()["id"]
    html = as_admin.get(f"/quan-tri/xuong-kich-ban/{series_id}").get_data(as_text=True)
    assert "/static/voices/vieneu/female.mp3" in html and "VOICE_SAMPLES" in html


def _drafted_episode(as_admin):
    series_id = as_admin.post("/api/studio/series", json={"topic": "x", "episodes": 1}).get_json()["id"]
    ep_id = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]["id"]
    as_admin.patch(f"/api/studio/episodes/{ep_id}", json={"script_vi": "Câu một. Câu hai.\nCâu ba."})
    return series_id, ep_id


def test_storyboard_step_preview_and_reset_on_script_change(as_admin, fake_agent, monkeypatch, tmp_path):
    import json as _json

    import app.studio as st
    from core import explainer

    series_id, ep_id = _drafted_episode(as_admin)
    board = {"scenes": [{"start": 1, "layout": "title", "title": "Mở"}, {"start": 3, "layout": "number", "value": 3}]}
    monkeypatch.setattr(explainer, "ask_gemini_json", lambda request: _json.dumps(board))
    assert as_admin.post(f"/api/studio/episodes/{ep_id}/board_vi").status_code == 202
    ep = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]
    assert [s["start"] for s in ep["storyboard_vi"]] == [0, 2] and len(ep["sentences_vi"]) == 3

    # Vẽ thử: render giả, file nằm trong OUTPUT_DIR và chỉ chủ tập xem được.
    monkeypatch.setattr("config.settings.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(explainer, "render_scene", lambda scene, frames, out, work: out.write_bytes(b"mp4"))
    monkeypatch.setattr("app.storage.commit_volume", lambda: None)
    assert as_admin.post(f"/api/studio/episodes/{ep_id}/preview", json={"lang": "vi", "index": 1}).status_code == 202
    ep = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]
    name = ep["previews"]["vi:1"]
    assert (tmp_path / name).read_bytes() == b"mp4"
    monkeypatch.setattr("app.storage.locate_file", lambda path: path)
    assert as_admin.get(f"/quan-tri/xuong-kich-ban/xem-thu/{name}").status_code == 200
    assert as_admin.get(f"/quan-tri/xuong-kich-ban/xem-thu/studio_{ep_id}_vi_9_deadbeef.mp4").status_code == 404
    assert as_admin.post(f"/api/studio/episodes/{ep_id}/preview", json={"lang": "vi", "index": 7}).status_code == 400

    # Sửa phân cảnh tay rồi đổi kịch bản: bảng cũ không còn khớp câu nên bị xoá.
    res = as_admin.patch(f"/api/studio/episodes/{ep_id}", json={"storyboard_vi": [{"start": 0, "layout": "python"}]})
    assert res.status_code == 400
    res = as_admin.patch(f"/api/studio/episodes/{ep_id}", json={"script_vi": "Khác hẳn."})
    assert res.get_json()["storyboard_vi"] == [] and res.get_json()["previews"] == {}
    assert st.TASK_LABELS["board"]
