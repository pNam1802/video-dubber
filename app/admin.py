"""
app/admin.py
API quản trị: xem người dùng, đổi quyền, khoá tài khoản, thống kê chi phí.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import func

from app.extensions import db
from app.models import Job, JobStatus, Role, User
from app.permissions import require_admin
from app.quota import get_usage
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    JOB_RUNNER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    QUOTA_GPU_SECONDS_PER_DAY,
    QUOTA_JOBS_PER_DAY,
    QUOTA_STORAGE_MB,
    WHISPER_BACKEND,
)

bp = Blueprint("admin", __name__, url_prefix="/api")


@bp.get("/usage")
@login_required
def my_usage():
    """Mức tiêu thụ của chính người đang đăng nhập."""
    return jsonify(get_usage(current_user).as_dict())


@bp.get("/admin/users")
@login_required
@require_admin
def list_users():
    users = User.query.order_by(User.created_at.desc()).all()
    rows = []
    for user in users:
        usage = get_usage(user)
        rows.append(
            {
                "id": user.id,
                "username": user.username,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "quota_jobs_per_day": user.quota_jobs_per_day,
                "usage": usage.as_dict(),
            }
        )
    return jsonify({"users": rows, "total": len(rows)})


@bp.patch("/admin/users/<int:user_id>")
@login_required
@require_admin
def update_user(user_id: int):
    user = db.session.get(User, user_id)
    if user is None:
        return jsonify({"error": "Không tìm thấy tài khoản.", "code": 404}), 404

    payload = request.get_json(silent=True) or {}

    if "role" in payload:
        role = payload["role"]
        if role not in (Role.USER, Role.ADMIN):
            return jsonify({"error": "Role không hợp lệ.", "code": 400}), 400
        if user.id == current_user.id and role != Role.ADMIN:
            return jsonify({"error": "Không thể tự hạ quyền của chính mình.", "code": 400}), 400
        user.role = role

    if "is_active" in payload:
        is_active = bool(payload["is_active"])
        if user.id == current_user.id and not is_active:
            return jsonify({"error": "Không thể tự khoá tài khoản của chính mình.", "code": 400}), 400
        user.is_active = is_active

    if "quota_jobs_per_day" in payload:
        value = payload["quota_jobs_per_day"]
        if value is not None and (not isinstance(value, int) or value < 0):
            return jsonify({"error": "Hạn mức phải là số nguyên không âm.", "code": 400}), 400
        user.quota_jobs_per_day = value

    db.session.commit()
    return jsonify({"id": user.id, "username": user.username, "role": user.role, "is_active": user.is_active})


def _key_status(key: str) -> dict:
    """Bao trang thai key, KHONG bao gio tra ve gia tri key."""
    return {"configured": bool(key), "length": len(key) if key else 0}


@bp.get("/admin/config")
@login_required
@require_admin
def config_status():
    """Cau hinh dang chay: engine nao san sang, model nao, han muc bao nhieu."""
    return jsonify(
        {
            # MarianMT (fine-tune rieng EN->VI) da bi khoa khoi lua chon cua
            # nguoi dung — chi con Gemini/OpenAI cho job moi. Job cu van
            # hien du lieu binh thuong o /admin/stats (data-driven, khong
            # phu thuoc danh sach nay).
            "engines": {
                "gemini": {**_key_status(GEMINI_API_KEY), "model": GEMINI_MODEL},
                "openai": {**_key_status(OPENAI_API_KEY), "model": OPENAI_MODEL},
            },
            "runtime": {"job_runner": JOB_RUNNER, "whisper_backend": WHISPER_BACKEND},
            "quotas": {
                "jobs_per_day": QUOTA_JOBS_PER_DAY,
                "gpu_seconds_per_day": QUOTA_GPU_SECONDS_PER_DAY,
                "storage_mb": QUOTA_STORAGE_MB,
            },
        }
    )


#: Bat lay gia tri cua khoa "message" trong payload loi cua nha cung cap.
_MESSAGE_RE = re.compile(r"""["']message["']\s*:\s*(["'])(?P<text>.*?)\1(?=\s*[,}])""")


def humanise_error(raw: str | None, *, limit: int = 300) -> str:
    """Rút câu giải thích ra khỏi payload thô của nhà cung cấp.

    Lỗi từ Gemini/OpenAI được lưu nguyên cả dict, ví dụ:
        404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model ...', 'status': ...}}
    Chỉ câu trong "message" là đáng đọc; phần còn lại lặp lại mã lỗi đã có ở đầu chuỗi.
    Không khớp được thì trả lại nguyên văn — thà xấu còn hơn giấu mất thông tin.
    """
    text = " ".join((raw or "").split())
    match = _MESSAGE_RE.search(text)
    if not match:
        return text[:limit]
    code = text.split("{", 1)[0].strip(" .:")
    message = match.group("text").strip()
    return (f"{code}: {message}" if code else message)[:limit]


@bp.get("/admin/stats")
@login_required
@require_admin
def stats():
    """Số liệu cho dashboard: tổng quan, chi phí, và 14 ngày gần nhất."""
    since = datetime.now(timezone.utc) - timedelta(days=14)

    totals = db.session.query(
        func.count(Job.id),
        func.coalesce(func.sum(Job.elapsed_sec), 0.0),
        func.coalesce(func.sum(Job.estimated_cost_usd), 0.0),
        func.coalesce(func.avg(Job.elapsed_sec), 0.0),
    ).one()

    by_status = dict(
        db.session.query(Job.status, func.count(Job.id)).group_by(Job.status).all()
    )
    by_engine = dict(
        db.session.query(Job.translator_engine, func.count(Job.id))
        .filter(Job.translator_engine.isnot(None))
        .group_by(Job.translator_engine)
        .all()
    )

    daily = (
        db.session.query(
            func.date(Job.created_at).label("day"),
            func.count(Job.id),
            func.coalesce(func.sum(Job.estimated_cost_usd), 0.0),
        )
        .filter(Job.created_at >= since)
        .group_by("day")
        .order_by("day")
        .all()
    )

    top_users = (
        db.session.query(
            User.username,
            func.count(Job.id),
            func.coalesce(func.sum(Job.estimated_cost_usd), 0.0),
        )
        .join(Job, Job.user_id == User.id)
        .group_by(User.username)
        .order_by(func.coalesce(func.sum(Job.estimated_cost_usd), 0.0).desc())
        .limit(10)
        .all()
    )

    # Suc khoe tung engine: engine nao dang fail nhieu thi biet ngay,
    # vi du key het han hoac model bi ngung cap.
    engine_health: dict[str, dict[str, int]] = {}
    for engine, status, count in (
        db.session.query(Job.translator_engine, Job.status, func.count(Job.id))
        .filter(Job.translator_engine.isnot(None))
        .group_by(Job.translator_engine, Job.status)
        .all()
    ):
        row = engine_health.setdefault(engine, {"total": 0, "done": 0, "failed": 0, "unfinished": 0})
        row["total"] += count
        if status == JobStatus.DONE:
            row["done"] += count
        elif status == JobStatus.FAILED:
            row["failed"] += count
        else:
            # Dang chay, dang xep hang, bi huy hay bi ngat — chua ket luan duoc gi
            # ve engine, nen phai tach ra khoi mau tinh ty le loi.
            row["unfinished"] += count

    # Thoi gian dich trung binh theo engine — so sanh MarianMT voi Gemini.
    engine_speed = {
        engine: round(float(avg or 0), 1)
        for engine, avg in db.session.query(
            func.coalesce(Job.translator_actual, Job.translator_engine),
            func.avg(Job.translate_sec),
        )
        .filter(Job.translate_sec.isnot(None))
        .group_by(func.coalesce(Job.translator_actual, Job.translator_engine))
        .all()
        if engine
    }

    # Thoi gian trung binh tung buoc — cho biet nut that nam o dau.
    step_avgs = db.session.query(
        func.avg(Job.extract_sec),
        func.avg(Job.transcribe_sec),
        func.avg(Job.translate_sec),
        func.avg(Job.tts_sec),
        func.avg(Job.compose_sec),
    ).one()
    avg_steps = {
        key: round(float(value or 0), 1)
        for key, value in zip(
            ("extract", "transcribe", "translate", "tts", "compose"), step_avgs
        )
    }

    fallback_count = Job.query.filter(
        Job.translator_actual.isnot(None),
        Job.translator_actual != Job.translator_engine,
    ).count()

    last_error = (
        Job.query.filter(Job.status == JobStatus.FAILED, Job.error.isnot(None))
        .order_by(Job.id.desc())
        .first()
    )

    done = by_status.get(JobStatus.DONE, 0)
    return jsonify(
        {
            "totals": {
                "jobs": totals[0],
                "gpu_seconds": round(float(totals[1]), 1),
                "estimated_cost_usd": round(float(totals[2]), 4),
                "avg_seconds_per_job": round(float(totals[3]), 1),
                "success_rate": round(done / totals[0] * 100, 1) if totals[0] else 0.0,
                "users": User.query.count(),
            },
            "by_status": by_status,
            "by_engine": by_engine,
            "engine_health": engine_health,
            "engine_translate_seconds": engine_speed,
            "avg_seconds_per_step": avg_steps,
            "fallback_count": fallback_count,
            "last_error": (
                {
                    "job_id": last_error.id,
                    "engine": last_error.translator_engine,
                    "error": humanise_error(last_error.error),
                    "at": (last_error.finished_at or last_error.created_at).isoformat(),
                }
                if last_error
                else None
            ),
            "daily": [
                {"day": str(day), "jobs": count, "estimated_cost_usd": round(float(cost), 4)}
                for day, count, cost in daily
            ],
            "top_users": [
                {"username": name, "jobs": count, "estimated_cost_usd": round(float(cost), 4)}
                for name, count, cost in top_users
            ],
            "note": (
                "estimated_cost_usd chỉ tính thời gian pipeline chạy. Hoá đơn Modal thực tế "
                "cao hơn vì còn tính container nằm không, cold start, CPU, RAM và container web."
            ),
        }
    )
