"""
app/api.py
Endpoint JSON cho upload và theo dõi tiến trình.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app.extensions import db, limiter
from app.jobs import cancel_job, start_job
from app.models import Job, JobStatus
from app.quota import check_quota
from app.storage import commit_uploads, commit_volume
from config.settings import (
    OUTPUT_DIR,
    RATELIMIT_UPLOAD,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    UPLOAD_DIR,
)
from core.pipeline import DubbingConfig

bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
# MarianMT (fine-tune rieng EN->VI) bi khoa khoi lua chon cua nguoi dung: no
# dich sai trong im lang khi nguon khong phai tieng Anh (van "thanh cong"),
# va khong dich duoc sang Nhat/Trung/Han. Du an gio dung day du Gemini/OpenAI
# nen khong can phuong an mien phi/offline nay nua.
ALLOWED_TRANSLATORS = {"openai", "gemini"}
ALLOWED_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}
# "auto" = Whisper tu nhan dang; con lai la khi nguoi dung tu chon vi tu nhan
# dang doan sai (video co nhac nen dai, giong khong ro o dau video...).
ALLOWED_SOURCE_LANGUAGES = {"auto", "en", "vi", "ja", "zh", "ko"}
ALLOWED_TARGET_LANGUAGES = {"vi", "ja", "zh", "ko"}
ALLOWED_DEVICES = {"auto", "cuda", "cpu"}
ALLOWED_TTS_ENGINES = {"edge-tts", "gtts"}
ALLOWED_VOICES = {"female", "male"}
ALLOWED_SUBTITLE_MODES = {"bilingual", "target", "source", "none"}


def _pick(field: str, allowed: set[str], default: str) -> str:
    value = request.form.get(field, default)
    return value if value in allowed else default


@bp.post("/upload")
@login_required
@limiter.limit(RATELIMIT_UPLOAD)
def upload_video():
    video = request.files.get("video")
    if not video or not video.filename:
        return jsonify({"error": "Vui lòng chọn file video."}), 400

    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Định dạng video không được hỗ trợ."}), 400

    translator_engine = _pick("translator_engine", ALLOWED_TRANSLATORS, "gemini")
    target_language = _pick("target_language", ALLOWED_TARGET_LANGUAGES, "vi")

    openai_api_key = request.form.get("openai_api_key", "").strip() or OPENAI_API_KEY
    if translator_engine == "openai" and not openai_api_key:
        return jsonify({"error": "Cần OPENAI API key để dùng OpenAI translator."}), 400

    gemini_api_key = request.form.get("gemini_api_key", "").strip() or GEMINI_API_KEY
    if translator_engine == "gemini" and not gemini_api_key:
        return jsonify({"error": "Cần GEMINI API key để dùng Gemini translator."}), 400

    whisper_model = _pick("whisper_model", ALLOWED_WHISPER_MODELS, "base")
    source_language = _pick("source_language", ALLOWED_SOURCE_LANGUAGES, "auto")
    compute_device = _pick("compute_device", ALLOWED_DEVICES, "auto")
    tts_engine = _pick("tts_engine", ALLOWED_TTS_ENGINES, "edge-tts")
    tts_voice = _pick("tts_voice", ALLOWED_VOICES, "female")
    subtitle_mode = _pick("subtitle_mode", ALLOWED_SUBTITLE_MODES, "bilingual")

    try:
        original_volume = float(request.form.get("original_volume", "10"))
    except ValueError:
        original_volume = 10.0
    original_volume = max(0.0, min(50.0, original_volume)) / 100.0

    # Kiem tra han muc TRUOC khi ghi file: khong de mot nguoi vuot quota
    # van kip do day dia bang cac file rac.
    allowed, reason = check_quota(current_user, request.content_length or 0)
    if not allowed:
        return jsonify({"error": reason, "code": 429}), 429

    safe_name = secure_filename(video.filename)
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    video.save(upload_path)

    # Container GPU doc file nay qua Volume nen phai commit truoc khi spawn.
    commit_uploads()

    job = Job(
        user_id=current_user.id,
        status=JobStatus.QUEUED,
        progress=1,
        message="Đã nhận video, đang khởi tạo tác vụ...",
        source_filename=video.filename,
        file_size=upload_path.stat().st_size,
        # Luu lai duong dan that su tren dia/Volume: can dung o buoc duyet
        # transcript (giai doan 2 doc lai tu day, xem app/jobs.py) va o
        # /api/jobs (danh sach) neu can don file cua job bi bo do.
        upload_path=str(upload_path),
        translator_engine=translator_engine,
        tts_engine=tts_engine,
        whisper_model=whisper_model,
        target_language=target_language,
    )
    db.session.add(job)
    db.session.commit()

    config = DubbingConfig(
        translator_engine=translator_engine,
        openai_api_key=openai_api_key,
        openai_model=request.form.get("openai_model", OPENAI_MODEL) or OPENAI_MODEL,
        gemini_api_key=gemini_api_key,
        gemini_model=request.form.get("gemini_model", GEMINI_MODEL) or GEMINI_MODEL,
        whisper_model=whisper_model,
        source_language=source_language,
        target_language=target_language,
        compute_device=compute_device,
        tts_engine=tts_engine,
        tts_voice=tts_voice,
        original_volume=original_volume,
        subtitle_mode=subtitle_mode,
    )

    start_job(current_app._get_current_object(), job.id, upload_path, config)
    return jsonify({"job_id": job.id}), 202


@bp.get("/jobs")
@login_required
def list_jobs():
    """Lịch sử job của chính người dùng, có lọc và phân trang."""
    query = Job.query.filter_by(user_id=current_user.id)

    status = request.args.get("status", "").strip()
    if status and status != "all":
        query = query.filter(Job.status == status)

    engine = request.args.get("engine", "").strip()
    if engine and engine != "all":
        query = query.filter(
            db.or_(Job.translator_actual == engine, Job.translator_engine == engine)
        )

    keyword = request.args.get("q", "").strip()
    if keyword:
        query = query.filter(Job.source_filename.ilike(f"%{keyword}%"))

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    per_page = 20

    total = query.count()
    jobs = (
        query.order_by(Job.created_at.desc(), Job.id.desc())
        .limit(per_page)
        .offset((page - 1) * per_page)
        .all()
    )

    return jsonify(
        {
            "items": [job.to_list_dict() for job in jobs],
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": max(1, -(-total // per_page)),
        }
    )


@bp.delete("/jobs/<int:job_id>")
@login_required
def delete_job(job_id: int):
    job = Job.query.filter_by(id=job_id, user_id=current_user.id).first()
    if job is None:
        return jsonify({"error": "Không tìm thấy job.", "code": 404}), 404
    if job.status in JobStatus.OPEN:
        return jsonify({"error": "Job đang chạy, hãy huỷ trước khi xoá.", "code": 409}), 409

    # Xoá luôn file trên đĩa, không chỉ bản ghi — nếu không thì dung lượng
    # vẫn bị tính vào hạn mức mà người dùng không còn thấy video đâu.
    removed = 0
    for name in (job.video_name, job.srt_name):
        if not name:
            continue
        try:
            path = OUTPUT_DIR / name
            if path.exists():
                path.unlink()
                removed += 1
        except OSError:
            current_app.logger.exception("Không xoá được file %s", name)

    # upload_path bình thường đã được job.py dọn khi job kết thúc — còn sót
    # lại chỉ khi job bị INTERRUPTED (server tắt giữa chừng, không ai dọn).
    if job.upload_path:
        try:
            path = Path(job.upload_path)
            if path.exists():
                path.unlink()
                removed += 1
        except OSError:
            current_app.logger.exception("Không xoá được file upload %s", job.upload_path)

    if removed:
        commit_volume()

    db.session.delete(job)
    db.session.commit()
    return jsonify({"deleted": job_id, "files_removed": removed})


@bp.post("/jobs/<int:job_id>/cancel")
@login_required
def cancel(job_id: int):
    job = Job.query.filter_by(id=job_id, user_id=current_user.id).first()
    if job is None:
        return jsonify({"error": "Không tìm thấy job.", "code": 404}), 404
    if job.status not in JobStatus.OPEN:
        return jsonify({"error": "Job đã kết thúc, không huỷ được.", "code": 409}), 409

    stopped = cancel_job(current_app._get_current_object(), job)
    return jsonify({
        "status": job.status,
        # Runner thread khong dung giua chung duoc; Modal thi dung han.
        "stopped": stopped,
        "message": "Đã huỷ job." if stopped else "Đã đánh dấu huỷ, phần đang chạy sẽ dừng ở bước kế tiếp.",
    })


@bp.get("/progress/<int:job_id>")
@login_required
def job_progress(job_id: int):
    job = Job.query.filter_by(id=job_id, user_id=current_user.id).first()
    if job is None:
        return jsonify({"error": "Không tìm thấy job."}), 404

    payload = job.to_progress_dict()
    if job.status == JobStatus.QUEUED:
        # max_containers gioi han so container GPU, nen job den sau xep hang
        # that su. Khong bao thi nguoi dung tuong he thong bi treo.
        payload["queue_position"] = (
            Job.query.filter(Job.status == JobStatus.QUEUED, Job.id < job.id).count()
        )
    return jsonify(payload)


@bp.get("/jobs/<int:job_id>/segments")
@login_required
def job_segments(job_id: int):
    """Transcript song ngữ đã có mốc thời gian của một job.

    Job chạy trước khi có bảng này thì trả danh sách rỗng chứ không phải 404:
    job vẫn tồn tại, chỉ là không còn transcript. Cờ `available` để giao diện
    phân biệt "chưa từng lưu" với "video không có lời nào".
    """
    job = Job.query.filter_by(id=job_id, user_id=current_user.id).first()
    if job is None:
        return jsonify({"error": "Không tìm thấy job."}), 404

    segments = job.segments.all()
    return jsonify({
        "job_id": job.id,
        "count": len(segments),
        "available": bool(segments),
        "segments": [segment.to_dict() for segment in segments],
    })
