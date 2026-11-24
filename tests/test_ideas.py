"""Ý tưởng mỗi ngày: core/ideas.py (không gọi mạng) và API trong app/studio.py."""
from __future__ import annotations

from core import ideas as ci

SOURCES = [{"title": "jiqizhixin.com", "url": "https://a"}, {"title": "zhihu.com", "url": "https://b"}]


def test_clean_ideas_keeps_only_real_sources():
    data = {"ideas": [
        {"title": "Agent tự sửa lỗi", "summary": "s", "angle": "a", "key_points": ["k1"], "sources": [1, 9, "x", 1]},
        {"title": "Không có nguồn", "sources": [7]},
        {"title": "", "sources": [0]},
        "rác",
    ]}
    out = ci.clean_ideas(data, SOURCES)
    assert [i["title"] for i in out] == ["Agent tự sửa lỗi"]
    assert out[0]["sources"] == [SOURCES[1]] and out[0]["key_points"] == ["k1"]
    assert ci.clean_ideas({"ideas": "x"}, SOURCES) == [] and ci.clean_ideas([], SOURCES) == []


def test_find_ideas_searches_chinese_sources_and_avoids_repeats():
    seen = {}

    def search(prompt):
        seen["search"] = prompt
        return "ghi chú", SOURCES

    def ask(prompt):
        seen["pick"] = prompt
        return {"ideas": [{"title": "T", "sources": [0]}]}

    out = ci.find_ideas("ctx", "LLM Agent", ["Ý tưởng cũ"], search=search, ask=ask)
    assert out[0]["sources"] == [SOURCES[0]]
    assert "机器之心" in seen["search"] and "智能体" in seen["search"]
    assert "Ý tưởng cũ" in seen["pick"] and "[1] zhihu.com — https://b" in seen["pick"]


def test_find_ideas_with_focus():
    seen = {}

    def search(prompt):
        seen["search"] = prompt
        return "n", SOURCES

    def ask(prompt):
        seen["pick"] = prompt
        return {"ideas": []}

    ci.find_ideas("ctx", "LLM Agent", [], search=search, ask=ask, focus="kiểm chứng output của AI")
    assert "kiểm chứng output của AI" in seen["search"] and "last 7 days" not in seen["search"]
    assert "focus: kiểm chứng output của AI" in seen["pick"]


def test_find_ideas_without_sources_fails():
    import pytest

    with pytest.raises(RuntimeError):
        ci.find_ideas("ctx", "t", [], search=lambda p: ("x", []), ask=lambda p: {})


# ── API ──────────────────────────────────────────────────────
def _series(app, username="boss"):
    from app.extensions import db
    from app.models import Episode, Series, User

    with app.app_context():
        owner = User.query.filter_by(username=username).one()
        series = Series(user_id=owner.id, topic="LLM Agent", title="S", audience="dev")
        series.episodes = [Episode(number=1, title="Tập 1", goal="g")]
        db.session.add(series)
        db.session.commit()
        return series.id


def test_ideas_flow(app, as_admin, monkeypatch):
    import app.studio as st

    monkeypatch.setattr(st, "start_task", lambda flask_app, kind, target_id: st.run_task(flask_app, kind, target_id))
    calls, focuses = [], []

    def fake_find(context, topic, avoid, focus=""):
        calls.append(list(avoid))
        focuses.append(focus)
        return [{"title": f"Ý tưởng {len(calls)}", "summary": "tóm tắt", "angle": "cách giảng", "key_points": ["a", "b"],
                 "sources": [{"title": "qbitai.com", "url": "https://q"}]}]

    monkeypatch.setattr("core.ideas.find_ideas", fake_find)
    sid = _series(app)
    assert as_admin.post(f"/api/studio/series/{sid}/ideas").status_code == 202
    assert as_admin.post(f"/api/studio/series/{sid}/ideas").status_code == 202
    assert calls == [[], ["Ý tưởng 1"]]  # lần sau tránh lặp ý tưởng đã gợi ý
    res = as_admin.post(f"/api/studio/series/{sid}/ideas", json={"focus": "  kiểm chứng:   output AI "})
    assert res.status_code == 202
    assert focuses == ["", "", "kiểm chứng: output AI"]  # gọn khoảng trắng, giữ dấu ":"
    rows = as_admin.get(f"/api/studio/series/{sid}/ideas").get_json()
    assert sorted(r["title"] for r in rows) == ["Ý tưởng 1", "Ý tưởng 2", "Ý tưởng 3"]
    first = next(r for r in rows if r["title"] == "Ý tưởng 1")

    assert as_admin.patch(f"/api/studio/ideas/{first['id']}", json={"status": "lung tung"}).status_code == 400
    assert as_admin.patch(f"/api/studio/ideas/{first['id']}", json={"status": "dismissed"}).get_json()["status"] == "dismissed"

    res = as_admin.post(f"/api/studio/ideas/{first['id']}/episode")
    assert res.status_code == 201
    ep = res.get_json()["episode"]
    assert ep["number"] == 2 and ep["title"] == "Ý tưởng 1" and "Tham khảo: https://q" in ep["key_points"]
    assert res.get_json()["idea"]["status"] == "used"
    assert as_admin.post(f"/api/studio/ideas/{first['id']}/episode").status_code == 409
    series = as_admin.get(f"/api/studio/series/{sid}").get_json()
    assert [e["number"] for e in series["episodes"]] == [1, 2] and series["error"] is None


def test_ideas_are_admin_and_owner_only(app, as_user, as_admin):
    sid = _series(app)
    assert as_user.get(f"/api/studio/series/{sid}/ideas").status_code == 403
    assert as_admin.get("/api/studio/series/9999/ideas").status_code == 404
    assert as_admin.patch("/api/studio/ideas/9999", json={"status": "new"}).status_code == 404


def test_daily_run_covers_active_admin_series_only(app, monkeypatch):
    import app.studio as st
    from app.extensions import db
    from app.models import Episode, Idea, Job, JobStatus, Role, Series, User

    monkeypatch.setattr("core.ideas.find_ideas", lambda context, topic, avoid, focus="": [
        {"title": "Ý tưởng", "summary": "", "angle": "", "key_points": [], "sources": [{"title": "a", "url": "u"}]}])
    with app.app_context():
        admin = User(username="boss2", role=Role.ADMIN)
        user = User(username="u2")
        db.session.add_all([admin, user])
        db.session.flush()
        job = Job(user_id=admin.id, status=JobStatus.DONE, message="xong")
        db.session.add(job)
        db.session.flush()
        active = Series(user_id=admin.id, topic="t", title="Có video")
        active.episodes = [Episode(number=1, title="e", job_vi_id=job.id)]
        idle = Series(user_id=admin.id, topic="t", title="Chưa có video")
        idle.episodes = [Episode(number=1, title="e")]
        other = Series(user_id=user.id, topic="t", title="Của người dùng thường")
        other.episodes = [Episode(number=1, title="e", job_vi_id=job.id)]
        db.session.add_all([active, idle, other])
        db.session.commit()
        active_id = active.id
    assert st.run_daily_ideas(app) == [active_id]
    with app.app_context():
        assert [i.series_id for i in Idea.query.all()] == [active_id]
        assert db.session.get(Series, active_id).task is None
