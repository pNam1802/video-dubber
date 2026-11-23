"""
app/studio.py
"Xưởng kịch bản" (chỉ admin): agent lên lộ trình series và viết kịch bản từng tập,
trí nhớ series nằm trong bảng Series/Episode. Logic LLM ở core/studio.py.

Mỗi bước LLM chạy nền (Composer trên Modal, thread khi chạy local) vì có thể lâu hơn
150 giây — giới hạn request web của Modal — và container web không cài google-genai.
Trang chỉ việc hỏi lại trạng thái (Series.task / Episode.task).
"""
from __future__ import annotations

import threading

from flask import Blueprint, Flask, current_app, jsonify, render_template, request
from flask_login import current_user

from app.extensions import db
from app.models import Episode, EpisodeStatus, Job, Series
from app.permissions import require_admin, require_admin_page
from config.settings import JOB_RUNNER, MODAL_APP_NAME

bp = Blueprint("studio", __name__)

EPISODE_STEPS = ("write", "translate", "wrap")
TASK_LABELS = {"plan": "Đang lên lộ trình series...", "write": "Đang nghiên cứu và viết kịch bản...",
               "translate": "Đang dịch sang tiếng Anh...", "wrap": "Đang chốt tập và cập nhật trí nhớ..."}


# ── Chạy nền ─────────────────────────────────────────────────
def start_task(flask_app: Flask, kind: str, target_id: int) -> None:
    if JOB_RUNNER == "modal":
        import modal

        composer = modal.Cls.from_name(MODAL_APP_NAME, "Composer")()
        composer.run_studio.spawn(kind, target_id)
        return
    threading.Thread(target=run_task, args=(flask_app, kind, target_id), daemon=True,
                     name=f"studio-{kind}-{target_id}").start()


def run_task(flask_app: Flask, kind: str, target_id: int) -> None:
    """Chạy một bước của agent và ghi kết quả. Lỗi thì ghi vào .error, không ném ra."""
    with flask_app.app_context():
        target = db.session.get(Series if kind == "plan" else Episode, target_id)
        if target is None:
            return
        try:
            _STEPS[kind](target)
            target.error = None
        except Exception as exc:
            db.session.rollback()
            target = db.session.get(Series if kind == "plan" else Episode, target_id)
            target.error = str(exc)[:2000]
            flask_app.logger.exception("Xưởng kịch bản: bước %s của #%s lỗi", kind, target_id)
        finally:
            target.task = None
            db.session.commit()


def _context(episode: Episode) -> str:
    from core.studio import memory_context

    series = episode.series
    return memory_context(series.as_dict(), [e.as_dict() for e in series.episodes], episode.number)


def _step_plan(series: Series) -> None:
    from core.studio import plan_series

    plan = plan_series(series.topic, series.audience, len(series.episodes) or 10)
    series.title, series.description = plan["title"], plan["description"]
    series.style_guide, series.glossary = plan["style_guide"], plan["glossary"]
    for ep in list(series.episodes):
        db.session.delete(ep)
    db.session.flush()
    for number, ep in enumerate(plan["episodes"], start=1):
        series.episodes.append(Episode(number=number, title=ep["title"], goal=ep["goal"],
                                       key_points=ep["key_points"]))


def _step_write(episode: Episode) -> None:
    from core.studio import merge_glossary, research, review_script, write_script

    context = _context(episode)
    data = episode.as_dict()
    notes, sources = research(context, data)
    draft = write_script(context, data, notes)
    reviewed = review_script(context, data, notes, draft["script"])
    episode.notes, episode.sources, episode.outline = notes, sources, draft["outline"]
    episode.script_vi, episode.quiz, episode.review = reviewed["script"], draft["quiz"], reviewed["issues"]
    episode.script_en = ""  # bản Anh cũ không còn khớp
    episode.status = EpisodeStatus.DRAFTED
    episode.series.glossary = merge_glossary(episode.series.glossary or [], draft["new_terms"])


def _step_translate(episode: Episode) -> None:
    from core.studio import translate

    if not episode.script_vi.strip():
        raise RuntimeError("Chưa có kịch bản tiếng Việt để dịch.")
    episode.script_en = translate(_context(episode), episode.script_vi)
    if episode.status != EpisodeStatus.DONE:
        episode.status = EpisodeStatus.TRANSLATED


def _step_wrap(episode: Episode) -> None:
    from core.studio import merge_glossary, wrap_up

    if not episode.script_vi.strip():
        raise RuntimeError("Chưa có kịch bản để chốt.")
    result = wrap_up(episode.script_vi)
    episode.summary = result["summary"]
    episode.status = EpisodeStatus.DONE
    episode.series.glossary = merge_glossary(episode.series.glossary or [], result["new_terms"])


_STEPS = {"plan": _step_plan, "write": _step_write, "translate": _step_translate, "wrap": _step_wrap}


# ── Trang ────────────────────────────────────────────────────
@bp.get("/quan-tri/xuong-kich-ban")
@require_admin_page
def page():
    return render_template("studio.html", series_id=None)


@bp.get("/quan-tri/xuong-kich-ban/<int:series_id>")
@require_admin_page
def series_page(series_id: int):
    _own_series(series_id)
    return render_template("studio.html", series_id=series_id)


# ── API ──────────────────────────────────────────────────────
def _own_series(series_id: int) -> Series:
    series = Series.query.filter_by(id=series_id, user_id=current_user.id).first()
    if series is None:
        from flask import abort

        abort(404)
    return series


def _own_episode(episode_id: int) -> Episode:
    episode = db.session.get(Episode, episode_id)
    if episode is None or episode.series.user_id != current_user.id:
        from flask import abort

        abort(404)
    return episode


def _busy(target) -> tuple | None:
    if target.task:
        return jsonify({"error": "Agent đang chạy một bước khác, hãy đợi xong.", "code": 409}), 409
    return None


@bp.get("/api/studio/series")
@require_admin
def list_series():
    rows = Series.query.filter_by(user_id=current_user.id).order_by(Series.created_at.desc()).all()
    return jsonify([{"id": s.id, "title": s.title or s.topic, "topic": s.topic, "task": s.task,
                     "episodes": len(s.episodes),
                     "done": sum(e.status == EpisodeStatus.DONE for e in s.episodes)} for s in rows])


@bp.post("/api/studio/series")
@require_admin
def create_series():
    body = request.get_json(silent=True) or {}
    topic = " ".join(str(body.get("topic", "")).split())[:1000]
    if not topic:
        return jsonify({"error": "Hãy nhập chủ đề series."}), 400
    try:
        count = max(1, min(30, int(body.get("episodes", 12))))
    except (TypeError, ValueError):
        count = 12
    series = Series(user_id=current_user.id, topic=topic, title=topic[:200],
                    audience=" ".join(str(body.get("audience", "")).split())[:500], task="plan")
    # Tập giữ chỗ: _step_plan đọc số tập muốn có từ đây rồi thay bằng lộ trình thật.
    series.episodes = [Episode(number=i + 1, title="") for i in range(count)]
    db.session.add(series)
    db.session.commit()
    start_task(current_app._get_current_object(), "plan", series.id)
    return jsonify(series.as_dict()), 202


@bp.get("/api/studio/series/<int:series_id>")
@require_admin
def get_series(series_id: int):
    data = _own_series(series_id).as_dict()
    data["task_label"] = TASK_LABELS.get(data["task"] or "")
    for ep in data["episodes"]:
        ep["task_label"] = TASK_LABELS.get(ep["task"] or "")
    return jsonify(data)


@bp.patch("/api/studio/series/<int:series_id>")
@require_admin
def update_series(series_id: int):
    series = _own_series(series_id)
    body = request.get_json(silent=True) or {}
    if "style_guide" in body:
        series.style_guide = str(body["style_guide"])[:4000]
    if "glossary" in body:
        from core.studio import clean_glossary

        series.glossary = clean_glossary(body["glossary"])
    db.session.commit()
    return jsonify(series.as_dict())


@bp.delete("/api/studio/series/<int:series_id>")
@require_admin
def delete_series(series_id: int):
    series = _own_series(series_id)
    db.session.delete(series)
    db.session.commit()
    return jsonify({"deleted": series_id})


@bp.patch("/api/studio/episodes/<int:episode_id>")
@require_admin
def update_episode(episode_id: int):
    """Sửa tay: tên, mục tiêu, kịch bản; gắn job video đã tạo từ kịch bản."""
    from core.studio import clean_script

    episode = _own_episode(episode_id)
    body = request.get_json(silent=True) or {}
    for key, limit in (("title", 200), ("goal", 1000)):
        if key in body:
            setattr(episode, key, " ".join(str(body[key]).split())[:limit])
    for key in ("script_vi", "script_en"):
        if key in body:
            setattr(episode, key, clean_script(body[key]))
    for key in ("job_vi_id", "job_en_id"):
        if key in body:
            job = Job.query.filter_by(id=body[key], user_id=current_user.id).first() if body[key] else None
            setattr(episode, key, job.id if job else None)
    db.session.commit()
    return jsonify(episode.as_dict())


@bp.post("/api/studio/episodes/<int:episode_id>/<step>")
@require_admin
def run_episode_step(episode_id: int, step: str):
    if step not in EPISODE_STEPS:
        return jsonify({"error": "Bước không hợp lệ."}), 400
    episode = _own_episode(episode_id)
    busy = _busy(episode) or _busy(episode.series)
    if busy:
        return busy
    episode.task, episode.error = step, None
    db.session.commit()
    start_task(current_app._get_current_object(), step, episode.id)
    return jsonify(episode.as_dict()), 202
