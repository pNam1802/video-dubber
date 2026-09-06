"""
app/models.py
Schema: User và Job. Job là nguồn sự thật duy nhất về trạng thái xử lý —
không còn dict trong RAM, nên restart server không làm mất job.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask_login import UserMixin

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    #: Trạng thái chưa kết thúc — dùng để dò job mồ côi sau khi restart.
    ACTIVE = (QUEUED, PROCESSING)


class Role:
    USER = "user"
    ADMIN = "admin"


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    # Nguoi dang nhap bang Google khong co mat khau.
    password_hash = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(255), nullable=True, index=True)
    # Dinh danh bat bien cua Google. Khong dung email lam khoa vi email doi duoc.
    google_sub = db.Column(db.String(64), unique=True, nullable=True, index=True)
    avatar_url = db.Column(db.String(500), nullable=True)
    role = db.Column(db.String(20), nullable=False, default=Role.USER, server_default=Role.USER)
    # Flask-Login đọc thuộc tính is_active để quyết định có cho đăng nhập không.
    is_active = db.Column(db.Boolean, nullable=False, default=True, server_default=db.true())
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    # None = dung han muc mac dinh trong settings; admin co the dat rieng.
    quota_jobs_per_day = db.Column(db.Integer, nullable=True)

    jobs = db.relationship("Job", backref="user", lazy=True, cascade="all, delete-orphan")

    @property
    def uses_google(self) -> bool:
        return bool(self.google_sub)

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


class Job(db.Model):
    __tablename__ = "job"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    # ── Trạng thái ────────────────────────────────────────────
    status = db.Column(
        db.String(20), nullable=False, default=JobStatus.QUEUED, server_default=JobStatus.QUEUED, index=True
    )
    progress = db.Column(db.Integer, nullable=False, default=0, server_default="0")
    message = db.Column(db.String(500), nullable=False, default="", server_default="")
    error = db.Column(db.Text, nullable=True)

    # ── Đầu vào ───────────────────────────────────────────────
    source_filename = db.Column(db.String(255), nullable=True)
    file_size = db.Column(db.BigInteger, nullable=True)
    translator_engine = db.Column(db.String(30), nullable=True)
    tts_engine = db.Column(db.String(30), nullable=True)
    whisper_model = db.Column(db.String(30), nullable=True)
    # Ma ngon ngu NGUON da nhan dang/nguoi dung chon that su (vd "en", "ja").
    # Rong = job chay truoc khi co da ngon ngu, mac dinh coi la tieng Anh.
    source_language = db.Column(db.String(8), nullable=True)
    # Ngon ngu DICH — luon co gia tri, mac dinh tieng Viet nhu tu truoc gio.
    target_language = db.Column(db.String(8), nullable=False, default="vi", server_default="vi")

    # ── Kết quả ───────────────────────────────────────────────
    video_name = db.Column(db.String(255), nullable=True)
    video_url = db.Column(db.String(500), nullable=True)
    srt_name = db.Column(db.String(255), nullable=True)
    srt_url = db.Column(db.String(500), nullable=True)
    duration_sec = db.Column(db.Float, nullable=True)
    elapsed_sec = db.Column(db.Float, nullable=True)
    segment_count = db.Column(db.Integer, nullable=True)
    # Engine da dung THAT SU. Khac translator_engine khi API chinh hong va
    # phai lui sang nha cung cap con lai (Gemini <-> OpenAI) — giao dien
    # can noi ro cho nguoi dung biet.
    translator_actual = db.Column(db.String(30), nullable=True)
    # Thoi gian tung buoc, giay. Khong co may cot nay thi khong biet
    # nut that nam o Whisper, o buoc dich hay o TTS.
    extract_sec = db.Column(db.Float, nullable=True)
    transcribe_sec = db.Column(db.Float, nullable=True)
    translate_sec = db.Column(db.Float, nullable=True)
    tts_sec = db.Column(db.Float, nullable=True)
    compose_sec = db.Column(db.Float, nullable=True)
    # Uoc tinh tu elapsed_sec, KHONG phai hoa don Modal (xem app/quota.py).
    estimated_cost_usd = db.Column(db.Float, nullable=True)

    # ── Modal (dùng từ Phase 2) ───────────────────────────────
    modal_call_id = db.Column(db.String(100), nullable=True, index=True)

    # ── Mốc thời gian ─────────────────────────────────────────
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    started_at = db.Column(db.DateTime(timezone=True), nullable=True)
    finished_at = db.Column(db.DateTime(timezone=True), nullable=True)

    @property
    def is_finished(self) -> bool:
        return self.status not in JobStatus.ACTIVE

    def to_list_dict(self) -> dict[str, Any]:
        """Payload gọn cho bảng lịch sử — không kèm thời gian từng bước."""
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "source_filename": self.source_filename,
            "engine": self.translator_actual or self.translator_engine,
            "fallback_used": bool(
                self.translator_actual and self.translator_actual != self.translator_engine
            ),
            "elapsed_sec": round(self.elapsed_sec, 1) if self.elapsed_sec else None,
            "estimated_cost_usd": self.estimated_cost_usd,
            "file_size": self.file_size,
            "video_url": self.video_url,
            "srt_url": self.srt_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_progress_dict(self) -> dict[str, Any]:
        """Payload cho endpoint theo dõi tiến trình."""
        result = None
        if self.status == JobStatus.DONE and self.video_url:
            result = {
                "video_name": self.video_name,
                "video_url": self.video_url,
                "srt_name": self.srt_name,
                "srt_url": self.srt_url,
                "elapsed_seconds": round(self.elapsed_sec or 0, 2),
            }
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": result,
            "translator_engine": self.translator_engine,
            "translator_actual": self.translator_actual,
            "fallback_used": bool(
                self.translator_actual and self.translator_actual != self.translator_engine
            ),
            "timings": {
                "extract": self.extract_sec,
                "transcribe": self.transcribe_sec,
                "translate": self.translate_sec,
                "tts": self.tts_sec,
                "compose": self.compose_sec,
            },
        }

    def __repr__(self) -> str:
        return f"<Job {self.id} {self.status} {self.progress}%>"


class TranscriptSegment(db.Model):
    """Một dòng phụ đề đã có mốc thời gian, kèm cả bản gốc lẫn bản dịch.

    Trước đây nội dung này chỉ đi vào file .srt trên Volume: không truy vấn
    được, và mất hẳn khi người dùng chọn subtitle_mode "source" hoặc "none".
    Database mới là nơi giữ nó — đây là nền cho sửa phụ đề, tóm tắt và hỏi đáp.

    Tên cột trung lập (text_source/text_target) chứ không cứng text_en/text_vi
    — hỗ trợ đa ngôn ngữ đích (Nhật/Trung/Hàn...) từ đây, xem
    Job.source_language/target_language để biết đúng ngôn ngữ của từng cột.

    Tên khác với Segment trong core/transcriber.py cho khỏi nhầm: bên kia là
    dataclass chạy trong bộ nhớ, bên này là bảng.
    """

    __tablename__ = "transcript_segment"

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(
        db.Integer, db.ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Thu tu trong video, bat dau tu 0. Dung de trich dan "[#12]" khi hoi dap.
    idx = db.Column(db.Integer, nullable=False)
    start_sec = db.Column(db.Float, nullable=False)
    end_sec = db.Column(db.Float, nullable=False)

    text_source = db.Column(db.Text, nullable=False, default="", server_default="")
    text_target = db.Column(db.Text, nullable=False, default="", server_default="")

    #: Nguoi dung da sua tay dong nay chua — phan biet ban may dich voi ban da duyet.
    edited = db.Column(db.Boolean, nullable=False, default=False, server_default=db.text("false"))

    job = db.relationship(
        "Job",
        backref=db.backref(
            "segments",
            order_by="TranscriptSegment.idx",
            cascade="all, delete-orphan",
            passive_deletes=True,
            lazy="dynamic",
        ),
    )

    __table_args__ = (
        # Moi job chi co mot dong o moi vi tri; cung la index cho truy van theo thu tu.
        db.UniqueConstraint("job_id", "idx", name="uq_transcript_segment_job_idx"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "start": round(self.start_sec, 3),
            "end": round(self.end_sec, 3),
            "source": self.text_source,
            "target": self.text_target,
            "edited": self.edited,
        }

    def __repr__(self) -> str:
        return f"<TranscriptSegment job={self.job_id} #{self.idx}>"
