"""
app/voices.py
Giọng mẫu người dùng tải lên để VieNeu-TTS clone (giọng "Giọng của bạn").

Mỗi người một file voices/user_<id>.wav trên volume dữ liệu, đã chuẩn hoá về
WAV mono 24 kHz và cắt còn tối đa VOICE_SAMPLE_MAX_SEC — VieNeu chỉ cần 3–8
giây, file dài hơn chỉ tốn chỗ và chậm bước mã hoá giọng.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import Request
from werkzeug.datastructures import FileStorage

from app.extensions import db
from app.models import User, utcnow
from app.storage import commit_volume
from config.settings import (
    FFMPEG_BIN,
    FFPROBE_BIN,
    TEMP_DIR,
    VOICE_SAMPLE_MAX_MB,
    VOICE_SAMPLE_MAX_SEC,
    VOICE_SAMPLE_MIN_SEC,
)
from utils.proc import run_command

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".webm", ".flac"}


class VoiceSampleError(ValueError):
    """Lỗi do file người dùng gửi lên — thông báo hiển thị thẳng cho họ."""


def _duration(path: Path) -> float:
    result = run_command(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        timeout=30,
        error_message="Không đọc được file âm thanh.",
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise VoiceSampleError("Không đọc được độ dài file âm thanh.") from exc


def accept_voice_upload(user: User, req: Request) -> float:
    """Kiểm tra request tải giọng mẫu rồi lưu — dùng chung cho trang Cài đặt
    và nút tải ngay trong form tạo job. Bắt buộc xác nhận quyền dùng giọng:
    VieNeu (Apache 2.0) để trách nhiệm clone cho người dùng, và không được
    clone giọng người khác khi chưa có sự đồng ý."""
    if (req.content_length or 0) > VOICE_SAMPLE_MAX_MB * 1024 * 1024:
        raise VoiceSampleError(f"File quá lớn — tối đa {VOICE_SAMPLE_MAX_MB} MB.")
    if "voice_consent" not in req.form:
        raise VoiceSampleError("Bạn cần xác nhận đây là giọng của bạn hoặc đã được chủ giọng đồng ý.")
    upload = req.files.get("voice_sample")
    if not upload or not upload.filename:
        raise VoiceSampleError("Vui lòng chọn file giọng mẫu.")
    return save_voice_sample(user, upload)


def save_voice_sample(user: User, upload: FileStorage) -> float:
    """Chuẩn hoá và lưu giọng mẫu; trả về độ dài (giây) đã lưu."""
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise VoiceSampleError("Định dạng không hỗ trợ — dùng WAV, MP3, M4A, OGG, WEBM hoặc FLAC.")

    raw = TEMP_DIR / f"voice_{uuid.uuid4().hex}{ext}"
    clean = raw.with_name(raw.stem + "_clean.wav")
    upload.save(raw)
    try:
        try:
            run_command(
                [FFMPEG_BIN, "-v", "error", "-y", "-i", str(raw), "-t", str(VOICE_SAMPLE_MAX_SEC),
                 "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(clean)],
                timeout=60,
                error_message="Không đọc được file âm thanh.",
            )
        except RuntimeError as exc:
            raise VoiceSampleError("Không đọc được file âm thanh — file hỏng hoặc không phải âm thanh.") from exc
        seconds = _duration(clean)
        if seconds < VOICE_SAMPLE_MIN_SEC:
            raise VoiceSampleError(
                f"Giọng mẫu quá ngắn ({seconds:.1f} giây) — cần ít nhất {VOICE_SAMPLE_MIN_SEC:.0f} giây."
            )
        target = user.voice_sample_path
        target.parent.mkdir(parents=True, exist_ok=True)
        clean.replace(target)
    finally:
        raw.unlink(missing_ok=True)
        clean.unlink(missing_ok=True)

    user.voice_sample_at = utcnow()
    db.session.commit()
    # Container ghép video đọc file này qua volume.
    commit_volume()
    return seconds


def delete_voice_sample(user: User) -> None:
    user.voice_sample_path.unlink(missing_ok=True)
    user.voice_sample_at = None
    db.session.commit()
    commit_volume()
