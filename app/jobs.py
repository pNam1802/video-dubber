"""
app/jobs.py
Điều phối việc chạy pipeline lồng tiếng và ghi tiến trình vào DB.

Hai runner, cùng một hàm xử lý:
  - "thread": chạy trong thread nền của tiến trình web (dev, máy local).
  - "modal" : spawn một Modal Function chạy trên GPU (production).
Chọn bằng biến môi trường JOB_RUNNER.
"""
from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import quote

from flask import Flask

from app.extensions import db
from app.mailer import notify_job_status
from app.models import Job, JobStatus, TranscriptSegment, utcnow
from app.quota import estimate_cost
from config.settings import JOB_RUNNER, MODAL_APP_NAME
from core.pipeline import DubbingConfig, DubbingPipeline
from core.transcriber import Segment


# ── Điều phối ────────────────────────────────────────────────────
def start_job(flask_app: Flask, job_id: int, video_path: Path, config: DubbingConfig) -> None:
    """Khởi chạy job. Trả về ngay, tiến trình theo dõi qua DB."""
    if JOB_RUNNER == "modal":
        _spawn_on_modal(flask_app, job_id, video_path, config)
        return

    worker = threading.Thread(
        target=run_job,
        args=(flask_app, job_id, video_path, config),
        daemon=True,
        name=f"dubbing-job-{job_id}",
    )
    worker.start()


def _spawn_on_modal(flask_app: Flask, job_id: int, video_path: Path, config: DubbingConfig) -> None:
    """Đẩy job sang Modal GPU và lưu lại call id để tra cứu/huỷ."""
    import modal

    dubber = modal.Cls.from_name(MODAL_APP_NAME, "Dubber")()
    call = dubber.run.spawn(job_id, str(video_path), config.__dict__)

    job = db.session.get(Job, job_id)
    if job is not None:
        job.modal_call_id = call.object_id
        job.message = "Đã gửi sang GPU, đang chờ container..."
        db.session.commit()
    flask_app.logger.info("Job %s -> Modal call %s", job_id, call.object_id)


def continue_job(flask_app: Flask, job_id: int, config: DubbingConfig) -> None:
    """Chạy tiếp giai đoạn 2 (TTS + ghép video) sau khi người dùng xác nhận
    đã duyệt xong transcript. Không nhận video_path — job.upload_path
    trong DB là nguồn sự thật duy nhất tới lúc này, vì lệnh gọi phía trên
    (endpoint /jobs/<id>/continue) không có gì khác để dựa vào."""
    job = db.session.get(Job, job_id)
    if job is None or not job.upload_path:
        return
    video_path = Path(job.upload_path)

    if JOB_RUNNER == "modal":
        _spawn_phase2_on_modal(flask_app, job_id, video_path, config)
        return

    worker = threading.Thread(
        target=run_job_phase2,
        args=(flask_app, job_id, video_path, config),
        daemon=True,
        name=f"dubbing-job-{job_id}-phase2",
    )
    worker.start()


def _spawn_phase2_on_modal(flask_app: Flask, job_id: int, video_path: Path, config: DubbingConfig) -> None:
    """Như _spawn_on_modal(), nhưng gọi Dubber.run_phase2 — video đã có sẵn
    trên Volume từ lúc upload, không cần commit lại trước khi spawn."""
    import modal

    dubber = modal.Cls.from_name(MODAL_APP_NAME, "Dubber")()
    call = dubber.run_phase2.spawn(job_id, str(video_path), config.__dict__)

    job = db.session.get(Job, job_id)
    if job is not None:
        job.modal_call_id = call.object_id
        job.message = "Đã gửi sang GPU, đang chờ container..."
        db.session.commit()
    flask_app.logger.info("Job %s -> Modal call %s (giai đoạn 2)", job_id, call.object_id)


def _cleanup_upload(flask_app: Flask, video_path: Path) -> None:
    try:
        video_path.unlink(missing_ok=True)
    except OSError:
        flask_app.logger.warning("Không xoá được file upload tạm: %s", video_path)


def cancel_job(flask_app: Flask, job: Job) -> bool:
    """
    Huỷ job. Trả về True nếu thực sự dừng được tiến trình xử lý.

    Runner "thread" không dừng được giữa chừng: chỉ đánh dấu trạng thái để
    callback tiến trình ngừng ghi, phần việc đang chạy vẫn chạy nốt.
    """
    stopped = False
    was_awaiting_review = job.status == JobStatus.AWAITING_REVIEW
    if job.modal_call_id:
        try:
            import modal

            modal.FunctionCall.from_id(job.modal_call_id).cancel()
            stopped = True
        except Exception:
            flask_app.logger.exception("Không huỷ được Modal call %s", job.modal_call_id)

    job.status = JobStatus.CANCELLED
    job.message = "Đã huỷ theo yêu cầu."
    job.finished_at = utcnow()
    db.session.commit()

    # AWAITING_REVIEW không có gì đang chạy (call Modal của giai đoạn 1 đã
    # xong từ lâu) — dọn ngay file upload đang giữ lại chờ duyệt, không thì
    # nó nằm lại trên đĩa vô thời hạn mà không job nào còn nhắc tới.
    if was_awaiting_review and job.upload_path:
        _cleanup_upload(flask_app, Path(job.upload_path))
    return stopped


# ── Transcript ───────────────────────────────────────────────────
def save_segments(flask_app: Flask, job_id: int, segments) -> int:
    """Ghi transcript song ngữ vào DB, thay bản cũ nếu job được chạy lại.

    Không ném lỗi ra ngoài: tới bước này video đã lồng tiếng xong, mất
    transcript chỉ là mất một tính năng chứ không phải hỏng cả job.
    """
    if not segments:
        return 0

    try:
        TranscriptSegment.query.filter_by(job_id=job_id).delete(synchronize_session=False)
        db.session.add_all(
            [
                TranscriptSegment(
                    job_id=job_id,
                    idx=idx,
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text_source=(segment.text or "").strip(),
                    text_target=(segment.translated or "").strip(),
                )
                for idx, segment in enumerate(segments)
            ]
        )
        db.session.commit()
        return len(segments)
    except Exception:
        db.session.rollback()
        flask_app.logger.exception("Không lưu được transcript của job %s", job_id)
        return 0


# ── Thực thi ─────────────────────────────────────────────────────
def run_job(flask_app: Flask, job_id: int, video_path: Path, config: DubbingConfig) -> None:
    """Giai đoạn 1: tách audio, nhận dạng giọng nói, dịch. Dùng chung cho cả
    thread lẫn Modal. Dừng lại ở AWAITING_REVIEW — KHÔNG tự chạy tiếp
    TTS/ghép video, chờ người dùng duyệt rồi gọi continue_job()."""
    with flask_app.app_context():
        job = db.session.get(Job, job_id)
        if job is None:
            return

        job.status = JobStatus.PROCESSING
        job.started_at = utcnow()
        job.message = "Bắt đầu xử lý..."
        db.session.commit()

        def on_progress(percent: int, message: str) -> None:
            current = db.session.get(Job, job_id)
            if current is None or current.status == JobStatus.CANCELLED:
                return
            current.progress = percent
            current.message = message[:500]
            db.session.commit()

        result = DubbingPipeline(config).transcribe_and_translate(video_path, progress_cb=on_progress)

        job = db.session.get(Job, job_id)
        if job is None or job.status == JobStatus.CANCELLED:
            # Bị huỷ giữa chừng — đừng ghi đè trạng thái CANCELLED đã có.
            return

        job.extract_sec = result.timings.get("extract")
        job.transcribe_sec = result.timings.get("transcribe")
        job.translate_sec = result.timings.get("translate")
        job.segment_count = len(result.segments) if result.segments else None
        job.source_language = result.source_language_detected or None
        job.translator_actual = result.translator_used or None
        job.elapsed_sec = result.elapsed_seconds
        job.estimated_cost_usd = estimate_cost(result.elapsed_seconds)

        if not result.success:
            job.status = JobStatus.FAILED
            job.progress = 0
            job.error = result.error or "Không rõ nguyên nhân."
            # Kem theo ly do ngan gon: bao "Xu ly that bai" khong thi nguoi dung
            # khong biet la loi API dich, thieu key hay video hong.
            reason = " ".join(job.error.split())[:200]
            job.message = f"Xử lý thất bại: {reason}"[:500]
            job.finished_at = utcnow()
            db.session.commit()
            notify_job_status(job, "failed")
            _cleanup_upload(flask_app, Path(video_path))
            return

        job.status = JobStatus.AWAITING_REVIEW
        job.progress = 65
        if result.fallback_from:
            job.message = (
                f"Đã dịch xong (nhưng {result.fallback_from} gặp lỗi nên đã "
                f"chuyển sang {result.translator_used}) — đang chờ bạn duyệt transcript."
            )
        else:
            job.message = "Đã dịch xong, đang chờ bạn duyệt transcript trước khi tạo giọng đọc."
        db.session.commit()

        # Ghi transcript SAU khi trạng thái job đã commit: đây là điều bắt
        # buộc phải có để người dùng duyệt — hỏng bước này thì không có gì
        # để duyệt cả, khác với trước đây (chỉ để xem lại, mất cũng không sao).
        if save_segments(flask_app, job_id, result.segments) == 0:
            job = db.session.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.FAILED
                job.error = "Không lưu được transcript để duyệt."
                job.message = "Xử lý thất bại: không lưu được transcript để duyệt."
                job.finished_at = utcnow()
                db.session.commit()
                notify_job_status(job, "failed")
            _cleanup_upload(flask_app, Path(video_path))
        else:
            # Mốc quan trọng nhất để báo: không có gì tự chạy tiếp từ đây,
            # job đứng yên vô thời hạn cho tới khi người dùng tự quay lại
            # xác nhận — không báo thì họ không biết mà quay lại.
            notify_job_status(job, "awaiting_review")

        # KHÔNG xoá video_path ở đây — giai đoạn 2 (run_job_phase2) còn cần
        # nó để ghép video, có thể diễn ra rất lâu sau khi người dùng duyệt.


def run_job_phase2(flask_app: Flask, job_id: int, video_path: Path, config: DubbingConfig) -> None:
    """Giai đoạn 2: TTS + ghép video + phụ đề, dùng transcript đang có
    trong DB (đã qua duyệt/sửa ở AWAITING_REVIEW). Luôn dọn file upload khi
    xong — đây là lần cuối pipeline còn cần tới nó."""
    with flask_app.app_context():
        job = db.session.get(Job, job_id)
        if job is None:
            return

        job.status = JobStatus.PROCESSING
        job.message = "Đang tạo giọng đọc..."
        db.session.commit()

        def on_progress(percent: int, message: str) -> None:
            current = db.session.get(Job, job_id)
            if current is None or current.status == JobStatus.CANCELLED:
                return
            current.progress = percent
            current.message = message[:500]
            db.session.commit()

        rows = (
            TranscriptSegment.query.filter_by(job_id=job_id)
            .order_by(TranscriptSegment.idx)
            .all()
        )
        segments = [
            Segment(start=row.start_sec, end=row.end_sec, text=row.text_source, translated=row.text_target)
            for row in rows
        ]

        if not segments:
            job = db.session.get(Job, job_id)
            if job is not None and job.status != JobStatus.CANCELLED:
                job.status = JobStatus.FAILED
                job.progress = 0
                job.error = "Không còn transcript để tạo giọng đọc."
                job.message = "Xử lý thất bại: không còn transcript để tạo giọng đọc."
                job.finished_at = utcnow()
                db.session.commit()
                notify_job_status(job, "failed")
            _cleanup_upload(flask_app, Path(video_path))
            return

        result = DubbingPipeline(config).synthesize_and_compose(video_path, segments, progress_cb=on_progress)

        job = db.session.get(Job, job_id)
        if job is None or job.status == JobStatus.CANCELLED:
            _cleanup_upload(flask_app, Path(video_path))
            return

        # Cộng dồn với thời gian đã tốn ở giai đoạn 1 — KHÔNG tính thời gian
        # chờ người dùng duyệt ở giữa, vì đó không phải chi phí GPU/API.
        job.elapsed_sec = (job.elapsed_sec or 0) + result.elapsed_seconds
        job.estimated_cost_usd = estimate_cost(job.elapsed_sec)
        job.tts_sec = result.timings.get("tts")
        job.compose_sec = result.timings.get("compose")
        job.finished_at = utcnow()

        if result.success and result.output_video:
            output_name = result.output_video.name
            srt_name = result.srt_path.name if result.srt_path else None
            job.status = JobStatus.DONE
            job.progress = 100
            job.message = "Hoàn tất xử lý video."
            job.video_name = output_name
            job.video_url = f"/media/output/{quote(output_name)}"
            job.srt_name = srt_name
            job.srt_url = f"/media/output/{quote(srt_name)}" if srt_name else None
        else:
            job.status = JobStatus.FAILED
            job.progress = 0
            job.error = result.error or "Không rõ nguyên nhân."
            reason = " ".join(job.error.split())[:200]
            job.message = f"Xử lý thất bại: {reason}"[:500]

        db.session.commit()
        notify_job_status(job, "done" if job.status == JobStatus.DONE else "failed")

        # KHÔNG ghi lại transcript ở đây: segments không đổi qua TTS, và dữ
        # liệu trong DB đã là bản mới nhất người dùng duyệt — ghi đè bằng
        # bản đã đọc TRƯỚC KHI chạy TTS có thể xoá mất một sửa đổi khác vừa
        # xảy ra song song (dù hiếm).
        _cleanup_upload(flask_app, Path(video_path))


def mark_interrupted_jobs(flask_app: Flask) -> int:
    """
    Job đang chạy dở khi tiến trình chết thì không ai hoàn thành nó nữa —
    đánh dấu là 'interrupted' lúc khởi động thay vì để treo mãi ở 'processing'.

    Chỉ áp dụng cho runner "thread". Với Modal, job vẫn chạy tiếp trong container
    riêng nên trạng thái thật nằm ở FunctionCall, không được đụng vào.
    """
    if JOB_RUNNER == "modal":
        return 0

    with flask_app.app_context():
        stale = Job.query.filter(Job.status.in_(JobStatus.ACTIVE)).all()
        for job in stale:
            job.status = JobStatus.INTERRUPTED
            job.message = "Bị gián đoạn do server khởi động lại."
            job.finished_at = utcnow()
        if stale:
            db.session.commit()
            flask_app.logger.warning("Đã đánh dấu %d job bị gián đoạn.", len(stale))
        return len(stale)
