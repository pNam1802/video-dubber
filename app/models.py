"""
app/models.py
Schema: User và Job. Job là nguồn sự thật duy nhất về trạng thái xử lý —
không còn dict trong RAM, nên restart server không làm mất job.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from flask_login import UserMixin

from app.extensions import db
from config.settings import VOICE_SAMPLE_DIR


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    #: Đã dịch xong, dừng lại chờ người dùng duyệt/sửa transcript trước khi
    #: tốn thời gian TTS + ghép video. Không có gì đang CHẠY ở trạng thái
    #: này — khác PROCESSING, restart server không ảnh hưởng gì tới nó.
    AWAITING_REVIEW = "awaiting_review"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    #: Đang thực sự chạy trên container — dùng để dò job mồ côi sau khi
    #: restart. KHÔNG gồm AWAITING_REVIEW vì không có gì đang chạy cho nó cả.
    ACTIVE = (QUEUED, PROCESSING)

    #: Chưa tới trạng thái cuối cùng — còn huỷ được, còn chặn xoá, và trang
    #: job vẫn cần tiếp tục poll để biết khi nào chuyển trạng thái.
    OPEN = (QUEUED, PROCESSING, AWAITING_REVIEW)


class JobKind:
    #: Lồng tiếng video: nhận dạng → dịch → duyệt → đọc → ghép video.
    VIDEO = "video"
    #: Đọc kịch bản dán vào: chỉ đọc, kết quả là mp3 + SRT (core/script_reader.py).
    SCRIPT = "script"


class Role:
    USER = "user"
    # VIP: dùng không giới hạn tài nguyên (lượt, GPU, dung lượng) nhưng KHÔNG có
    # quyền quản trị. Chỉ cấp cho vài người tin cậy — không có trần chi phí.
    VIP = "vip"
    ADMIN = "admin"
    ALL = (USER, VIP, ADMIN)


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    # Nguoi dang nhap bang Google khong co mat khau.
    password_hash = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(255), nullable=True, index=True)
    # Tai khoan dang nhap thuong khong co email cho toi khi tu them o trang
    # cai dat (khac Google, da co san email da xac thuc). Bat mac dinh —
    # ai co email thi mac dinh nhan thong bao, tu tat duoc neu khong muon.
    notify_email = db.Column(db.Boolean, nullable=False, default=True, server_default=db.true())
    # Dinh danh bat bien cua Google. Khong dung email lam khoa vi email doi duoc.
    google_sub = db.Column(db.String(64), unique=True, nullable=True, index=True)
    avatar_url = db.Column(db.String(500), nullable=True)
    role = db.Column(db.String(20), nullable=False, default=Role.USER, server_default=Role.USER)
    # Flask-Login đọc thuộc tính is_active để quyết định có cho đăng nhập không.
    is_active = db.Column(db.Boolean, nullable=False, default=True, server_default=db.true())
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    # None = dung han muc mac dinh trong settings; admin co the dat rieng.
    quota_jobs_per_day = db.Column(db.Integer, nullable=True)
    # None = dùng QUOTA_GPU_SECONDS_PER_DAY chung.
    quota_gpu_seconds_per_day = db.Column(db.Integer, nullable=True)
    # Luc nguoi dung tai giong mau len VA xac nhan do la giong cua ho (hoac
    # da duoc chu giong dong y). None = chua co giong clone. File nam o
    # voice_sample_path.
    voice_sample_at = db.Column(db.DateTime(timezone=True), nullable=True)
    # Nhu tren, cho giong mau TIENG ANH (Chatterbox clone) — xem
    # voice_sample_file("en").
    voice_sample_en_at = db.Column(db.DateTime(timezone=True), nullable=True)

    jobs = db.relationship("Job", backref="user", lazy=True, cascade="all, delete-orphan")

    @property
    def uses_google(self) -> bool:
        return bool(self.google_sub)

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    @property
    def is_unlimited(self) -> bool:
        """Không bị hạn mức tài nguyên: admin và VIP."""
        return self.role in (Role.ADMIN, Role.VIP)

    @property
    def role_label(self) -> str:
        return {Role.ADMIN: "Quản trị viên", Role.VIP: "Thành viên VIP"}.get(self.role, "Thành viên")

    @property
    def voice_sample_path(self) -> Path:
        return self.voice_sample_file("vi")

    @property
    def has_voice_sample(self) -> bool:
        return self.has_voice_sample_in("vi")

    # Giọng mẫu theo ngôn ngữ (config.settings.VOICE_SAMPLE_LANGUAGES). Mẫu
    # tiếng Việt giữ tên file cũ để giọng mẫu đã tải lên trước đây còn dùng được.
    def voice_sample_file(self, language: str = "vi") -> Path:
        suffix = "" if language == "vi" else f"_{language}"
        return VOICE_SAMPLE_DIR / f"user_{self.id}{suffix}.wav"

    def voice_sample_time(self, language: str = "vi"):
        return self.voice_sample_en_at if language == "en" else self.voice_sample_at

    def set_voice_sample_time(self, language: str, value) -> None:
        if language == "en":
            self.voice_sample_en_at = value
        else:
            self.voice_sample_at = value

    def has_voice_sample_in(self, language: str = "vi") -> bool:
        return self.voice_sample_time(language) is not None

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

    kind = db.Column(db.String(10), nullable=False, default=JobKind.VIDEO, server_default=JobKind.VIDEO)

    # ── Đầu vào ───────────────────────────────────────────────
    source_filename = db.Column(db.String(255), nullable=True)
    file_size = db.Column(db.BigInteger, nullable=True)
    # Duong dan THAT SU tren dia/Volume cua file da upload — khac
    # source_filename (ten goc phia client). Can luu lai vi giai doan 2
    # (TTS + ghep video sau khi duyet) co the chay o mot request/container
    # hoan toan khac, rat lau sau luc upload — khong con bien video_path
    # nao trong scope de dung nua, phai tra lai tu DB.
    upload_path = db.Column(db.String(500), nullable=True)
    translator_engine = db.Column(db.String(30), nullable=True)
    # Kieu dich "natural" | "localized" (llm_common.TRANSLATION_STYLES).
    # None = job cu, coi nhu "natural".
    translation_style = db.Column(db.String(20), nullable=True)
    # Tom tat + bang tu cua buoc doc hieu truoc khi dich — "dich lai cau nay"
    # o buoc duyet doc lai de van hieu dung tieng long/ngu canh ca video.
    translation_notes = db.Column(db.Text, nullable=True)
    tts_engine = db.Column(db.String(30), nullable=True)
    whisper_model = db.Column(db.String(30), nullable=True)
    # Ma ngon ngu NGUON da nhan dang/nguoi dung chon that su (vd "en", "ja").
    # Rong = job chay truoc khi co da ngon ngu, mac dinh coi la tieng Anh.
    source_language = db.Column(db.String(8), nullable=True)
    # Ngon ngu DICH — luon co gia tri, mac dinh tieng Viet nhu tu truoc gio.
    target_language = db.Column(db.String(8), nullable=False, default="vi", server_default="vi")
    # Ba truong nay chi dung o GIAI DOAN 2 (TTS + ghep video), nhung phai luu
    # tu luc tao job: /jobs/<id>/continue co the chay o mot request hoan
    # toan khac, rat lau sau — khong con DubbingConfig cua giai doan 1 nao
    # trong scope de doc lai, phai dung dung lua chon nguoi dung da chon.
    tts_voice = db.Column(db.String(20), nullable=True)
    # Tua nhanh hinh o khoang lang toi da bay nhieu lan (core/video_retime.py).
    # None/<=1 = tat — job cu truoc khi co tinh nang nay giu cach ghep cu.
    max_video_speed = db.Column(db.Float, nullable=True)
    original_volume = db.Column(db.Float, nullable=True)
    subtitle_mode = db.Column(db.String(20), nullable=True)

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
    # Chi phi DICH THAT SU — tinh tu so token that API tra ve x gia cong bo
    # (xem core/translator/llm_common.estimate_llm_cost()), khac han
    # estimated_cost_usd o tren (chi la GPU-giay, khong lien quan API key).
    # NULL o ca 3 cot nghia la CHUA BIET (model la, hoac job cu truoc khi co
    # tinh nang nay) — khac 0.0 (that su mien phi), khong duoc danh dong.
    translate_prompt_tokens = db.Column(db.Integer, nullable=True)
    translate_completion_tokens = db.Column(db.Integer, nullable=True)
    translate_cost_usd = db.Column(db.Float, nullable=True)

    # ── Modal (dùng từ Phase 2) ───────────────────────────────
    modal_call_id = db.Column(db.String(100), nullable=True, index=True)

    # ── Mốc thời gian ─────────────────────────────────────────
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    started_at = db.Column(db.DateTime(timezone=True), nullable=True)
    finished_at = db.Column(db.DateTime(timezone=True), nullable=True)

    @property
    def is_finished(self) -> bool:
        return self.status not in JobStatus.OPEN

    # File trên đĩa mang tiền tố uuid (vd "77022eca…_What_is_a_GIS_dubbed.mp4")
    # để hai người upload trùng tên không ghi đè nhau — tên đó KHÔNG dành cho
    # người xem. Hiển thị và tải về luôn dùng tên video gốc của người dùng.
    @property
    def display_name(self) -> str:
        return self.source_filename or f"Job #{self.id}"

    @property
    def _download_stem(self) -> str:
        if self.is_script:
            # Tên job kịch bản là đầu câu đầu tiên, không phải tên file: có thể
            # chứa dấu chấm, gạch chéo — PurePath sẽ cắt mất một phần.
            stem = re.sub(r'[\\/:*?"<>|]+', " ", self.source_filename or "").strip(" .")
        else:
            stem = PurePath(self.source_filename or "").stem.strip()
        return stem or f"job-{self.id}"

    @property
    def is_script(self) -> bool:
        return self.kind == JobKind.SCRIPT

    @property
    def is_audio_only(self) -> bool:
        """Job kịch bản không có ảnh minh hoạ: video_name/video_url giữ file mp3."""
        return self.is_script and not (self.video_name or "").endswith(".mp4")

    @property
    def download_video_name(self) -> str:
        if self.is_audio_only:
            return f"{self._download_stem} (giọng đọc).mp3"
        if self.is_script:
            return f"{self._download_stem} (minh hoạ).mp4"
        return f"{self._download_stem} (lồng tiếng).mp4"

    @property
    def download_srt_name(self) -> str:
        return f"{self._download_stem} (phụ đề).srt"

    def to_list_dict(self) -> dict[str, Any]:
        """Payload gọn cho bảng lịch sử — không kèm thời gian từng bước."""
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "kind": self.kind,
            "source_filename": self.source_filename,
            # Job kịch bản không dịch — hiện công cụ đọc thay vào cột này.
            "engine": self.tts_engine if self.is_script else self.translator_actual or self.translator_engine,
            "fallback_used": bool(
                self.translator_actual and self.translator_actual != self.translator_engine
            ),
            "elapsed_sec": round(self.elapsed_sec, 1) if self.elapsed_sec else None,
            "estimated_cost_usd": self.estimated_cost_usd,
            "file_size": self.file_size,
            "video_url": self.video_url,
            "srt_url": self.srt_url,
            "download_video_name": self.download_video_name,
            "download_srt_name": self.download_srt_name,
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
    #: Moc cau trong VIDEO DA LONG TIENG (giai doan 2 ghi). Khac start/end_sec
    #: khi hinh bi tua nhanh — start/end_sec van la moc goc, giai doan 2 can.
    #: NULL = chua ghep video, hoac job cu: dung moc goc.
    play_start_sec = db.Column(db.Float, nullable=True)
    play_end_sec = db.Column(db.Float, nullable=True)

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
            # id (khoa chinh that su) can de sua/dich lai TUNG dong qua
            # PATCH/retranslate — idx chi la thu tu hien thi, khong dung de
            # dinh danh vi khong on dinh qua cac lan luu lai (save_segments()
            # xoa het roi chen lai, id moi nhung idx giu nguyen thu tu).
            "id": self.id,
            "idx": self.idx,
            "start": round(self.start_sec, 3),
            "end": round(self.end_sec, 3),
            "play_start": round(self.play_start_sec if self.play_start_sec is not None else self.start_sec, 3),
            "play_end": round(self.play_end_sec if self.play_end_sec is not None else self.end_sec, 3),
            "source": self.text_source,
            "target": self.text_target,
            "edited": self.edited,
        }

    def __repr__(self) -> str:
        return f"<TranscriptSegment job={self.job_id} #{self.idx}>"


class Series(db.Model):
    """Series video YouTube do "Xưởng kịch bản" (app/studio.py) lên ý tưởng.

    Đây là TRÍ NHỚ của agent viết kịch bản: văn phong và bảng thuật ngữ Việt–Anh
    nằm ở đây, mỗi tập chốt xong lại bổ sung thêm. Viết tập mới thì agent đọc lại
    toàn bộ — không phụ thuộc vào việc ai/ phiên chat nào còn nhớ tập trước nói gì.
    """

    __tablename__ = "series"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False, default="", server_default="")
    topic = db.Column(db.Text, nullable=False, default="", server_default="")
    audience = db.Column(db.Text, nullable=False, default="", server_default="")
    description = db.Column(db.Text, nullable=False, default="", server_default="")
    #: Văn phong chung của cả series (giọng kể, độ dài, cách mở/kết tập).
    style_guide = db.Column(db.Text, nullable=False, default="", server_default="")
    #: [{"vi": ..., "en": ..., "definition": ...}] — dịch thuật ngữ nhất quán mọi tập.
    glossary = db.Column(db.JSON, nullable=False, default=list)

    #: Bước agent đang chạy nền ("plan"), None = rảnh. error: lỗi của lần chạy gần nhất.
    task = db.Column(db.String(20), nullable=True)
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)

    episodes = db.relationship(
        "Episode", backref="series", order_by="Episode.number",
        cascade="all, delete-orphan", passive_deletes=True,
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "topic": self.topic, "audience": self.audience,
            "description": self.description, "style_guide": self.style_guide, "glossary": self.glossary or [],
            "task": self.task, "error": self.error,
            "episodes": [e.as_dict() for e in self.episodes],
        }


class EpisodeStatus:
    PLANNED = "planned"        # mới có tên + mục tiêu trong lộ trình
    DRAFTED = "drafted"        # đã nghiên cứu + viết + tự soát kịch bản tiếng Việt
    TRANSLATED = "translated"  # đã có bản tiếng Anh
    DONE = "done"              # đã chốt: tóm tắt + thuật ngữ vào trí nhớ series


class Episode(db.Model):
    __tablename__ = "episode"

    id = db.Column(db.Integer, primary_key=True)
    series_id = db.Column(db.Integer, db.ForeignKey("series.id", ondelete="CASCADE"), nullable=False, index=True)
    number = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(200), nullable=False, default="", server_default="")
    goal = db.Column(db.Text, nullable=False, default="", server_default="")
    key_points = db.Column(db.JSON, nullable=False, default=list)
    status = db.Column(db.String(20), nullable=False, default=EpisodeStatus.PLANNED,
                       server_default=EpisodeStatus.PLANNED)

    #: Ghi chú nghiên cứu (Gemini + Google Search) và nguồn [{"title", "url"}].
    notes = db.Column(db.Text, nullable=False, default="", server_default="")
    sources = db.Column(db.JSON, nullable=False, default=list)
    outline = db.Column(db.JSON, nullable=False, default=list)
    script_vi = db.Column(db.Text, nullable=False, default="", server_default="")
    script_en = db.Column(db.Text, nullable=False, default="", server_default="")
    #: Câu hỏi tự kiểm tra cuối tập [{"q", "a"}] — mục tiêu đầu tiên là học.
    quiz = db.Column(db.JSON, nullable=False, default=list)
    #: Những lỗi bước tự soát đã tìm thấy và sửa.
    review = db.Column(db.JSON, nullable=False, default=list)
    #: Bảng phân cảnh Video giải thích đã duyệt, mỗi bản một bảng:
    #: [{"start": chỉ số câu từ 0, "layout", ...}] (core/explainer.board_to_list). Rỗng = chưa phân cảnh.
    storyboard_vi = db.Column(db.JSON, nullable=False, default=list)
    storyboard_en = db.Column(db.JSON, nullable=False, default=list)
    #: Cảnh đã render thử: {"vi:3": tên file trong OUTPUT_DIR}.
    previews = db.Column(db.JSON, nullable=False, default=dict)
    #: Tóm tắt khi chốt tập — agent đọc lại khi viết các tập sau.
    summary = db.Column(db.Text, nullable=False, default="", server_default="")
    job_vi_id = db.Column(db.Integer, db.ForeignKey("job.id", ondelete="SET NULL"), nullable=True)
    job_en_id = db.Column(db.Integer, db.ForeignKey("job.id", ondelete="SET NULL"), nullable=True)

    task = db.Column(db.String(20), nullable=True)
    error = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (db.UniqueConstraint("series_id", "number", name="uq_episode_series_number"),)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "series_id": self.series_id, "number": self.number, "title": self.title,
            "goal": self.goal, "key_points": self.key_points or [], "status": self.status,
            "notes": self.notes, "sources": self.sources or [], "outline": self.outline or [],
            "script_vi": self.script_vi, "script_en": self.script_en, "quiz": self.quiz or [],
            "review": self.review or [], "summary": self.summary,
            "storyboard_vi": self.storyboard_vi or [], "storyboard_en": self.storyboard_en or [],
            "previews": self.previews or {},
            "job_vi_id": self.job_vi_id, "job_en_id": self.job_en_id,
            "task": self.task, "error": self.error,
        }
