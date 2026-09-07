"""
app/main.py
Trang chính, phục vụ file kết quả và healthcheck.
"""
from __future__ import annotations

import shutil

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_login import current_user, login_required
from sqlalchemy import text

from app.extensions import db
from app.legal import PRIVACY, TERMS, UPDATED
from app.models import Job, JobStatus
from app.permissions import require_admin_page
from app.quota import get_usage
from app.storage import refresh_outputs
from config.settings import (
    FFMPEG_BIN,
    MAX_UPLOAD_MB,
    FFPROBE_BIN,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUTPUT_DIR,
)

bp = Blueprint("main", __name__)


def _recent_outputs(limit: int = 8) -> list[dict[str, str | None]]:
    jobs = (
        Job.query.filter_by(user_id=current_user.id, status=JobStatus.DONE)
        .order_by(Job.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": job.id,
            "video_name": job.video_name,
            "video_url": job.video_url,
            "srt_name": job.srt_name,
            "srt_url": job.srt_url,
            "source_filename": job.source_filename,
        }
        for job in jobs
    ]


@bp.get("/")
@login_required
def index():
    return render_template(
        "index.html",
        outputs=_recent_outputs(),
        openai_key_exists=bool(OPENAI_API_KEY),
        default_openai_model=OPENAI_MODEL,
        gemini_key_exists=bool(GEMINI_API_KEY),
        default_gemini_model=GEMINI_MODEL,
        usage=get_usage(current_user).as_dict(),
        max_upload_mb=MAX_UPLOAD_MB,
    )


@bp.get("/lich-su")
@login_required
def history():
    return render_template("history.html")


@bp.route("/cai-dat", methods=["GET", "POST"])
@login_required
def account_settings():
    """Email nhận thông báo job — tài khoản đăng ký thường không có email
    cho tới khi tự thêm vào đây (khác Google, đã có sẵn email đã xác thực
    lúc đăng nhập, xem app/oauth.py)."""
    if request.method == "POST":
        if current_user.uses_google:
            # Email lay tu Google, khong cho sua tay o day — chi doi duoc
            # co nhan thong bao hay khong.
            current_user.notify_email = "notify_email" in request.form
        else:
            email = request.form.get("email", "").strip()
            if email and "@" not in email:
                flash("Email không hợp lệ.", "danger")
                return render_template("account_settings.html")
            current_user.email = email or None
            current_user.notify_email = "notify_email" in request.form
        db.session.commit()
        flash("Đã lưu cài đặt.", "success")
        return redirect(url_for("main.account_settings"))
    return render_template("account_settings.html")


@bp.get("/quan-tri/thong-ke")
@require_admin_page
def admin_stats():
    return render_template("admin_stats.html")


@bp.get("/quan-tri/nguoi-dung")
@require_admin_page
def admin_users():
    return render_template("admin_users.html")


@bp.get("/quan-tri/he-thong")
@require_admin_page
def admin_system():
    return render_template("admin_system.html")


@bp.get("/job/<int:job_id>")
@login_required
def job_detail(job_id: int):
    """Trang riêng cho một job: chia sẻ được link, tải lại trang không mất tiến trình."""
    job = Job.query.filter_by(id=job_id, user_id=current_user.id).first()
    if job is None:
        abort(404)
    return render_template("job.html", job=job)


@bp.get("/media/output/<path:filename>")
@login_required
def serve_output(filename: str):
    owned = Job.query.filter(
        Job.user_id == current_user.id,
        (Job.video_name == filename) | (Job.srt_name == filename),
    ).first()
    if owned is None:
        return jsonify({"error": "Bạn không có quyền truy cập hoặc file không tồn tại."}), 403

    # Tren Modal, file do container GPU ghi chi hien ra sau khi reload volume.
    # Chi reload khi chua thay file, de khong ton mot lan goi mang moi request.
    if not (OUTPUT_DIR / filename).exists():
        refresh_outputs()

    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@bp.get("/privacy")
def privacy():
    """Công khai: Google yêu cầu URL này mới cho publish ứng dụng OAuth."""
    return render_template("legal.html", page_title="Chính sách bảo mật",
                           sections=PRIVACY, updated=UPDATED)


@bp.get("/terms")
def terms():
    return render_template("legal.html", page_title="Điều khoản sử dụng",
                           sections=TERMS, updated=UPDATED)


@bp.get("/healthz")
def healthz():
    """Kiểm tra các phụ thuộc bắt buộc trước khi nhận traffic."""
    checks: dict[str, bool] = {}
    try:
        db.session.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        current_app.logger.exception("Healthcheck: database unreachable")
        checks["database"] = False
    checks["ffmpeg"] = shutil.which(FFMPEG_BIN) is not None
    checks["ffprobe"] = shutil.which(FFPROBE_BIN) is not None

    healthy = all(checks.values())
    return jsonify({"status": "ok" if healthy else "degraded", "checks": checks}), (200 if healthy else 503)
