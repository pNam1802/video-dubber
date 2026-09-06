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
from app.models import Job, JobStatus, TranscriptSegment, utcnow
from app.quota import estimate_cost
from config.settings import JOB_RUNNER, MODAL_APP_NAME
from core.pipeline import DubbingConfig, DubbingPipeline


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


def cancel_job(flask_app: Flask, job: Job) -> bool:
    """
    Huỷ job. Trả về True nếu thực sự dừng được tiến trình xử lý.

    Runner "thread" không dừng được giữa chừng: chỉ đánh dấu trạng thái để
    callback tiến trình ngừng ghi, phần việc đang chạy vẫn chạy nốt.
    """
    stopped = False
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
    """Chạy trọn một job và ghi kết quả vào DB. Dùng chung cho cả thread lẫn Modal."""
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

        result = DubbingPipeline(config).run(video_path, progress_cb=on_progress)

        job = db.session.get(Job, job_id)
        if job is None:
            return

        job.elapsed_sec = result.elapsed_seconds
        job.estimated_cost_usd = estimate_cost(result.elapsed_seconds)
        job.translator_actual = result.translator_used or None
        job.extract_sec = result.timings.get("extract")
        job.transcribe_sec = result.timings.get("transcribe")
        job.translate_sec = result.timings.get("translate")
        job.tts_sec = result.timings.get("tts")
        job.compose_sec = result.timings.get("compose")
        job.segment_count = len(result.segments) if result.segments else None
        job.finished_at = utcnow()

        if result.success and result.output_video:
            output_name = result.output_video.name
            srt_name = result.srt_path.name if result.srt_path else None
            job.status = JobStatus.DONE
            job.progress = 100
            if result.fallback_from:
                job.message = (
                    f"Hoàn tất, nhưng {result.fallback_from} gặp lỗi nên đã "
                    "chuyển sang MarianMT."
                )
            else:
                job.message = "Hoàn tất xử lý video."
            job.video_name = output_name
            job.video_url = f"/media/output/{quote(output_name)}"
            job.srt_name = srt_name
            job.srt_url = f"/media/output/{quote(srt_name)}" if srt_name else None
        else:
            job.status = JobStatus.FAILED
            job.progress = 0
            job.error = result.error or "Không rõ nguyên nhân."
            # Kem theo ly do ngan gon: bao "Xu ly that bai" khong thi nguoi dung
            # khong biet la loi API dich, thieu key hay video hong.
            reason = " ".join(job.error.split())[:200]
            job.message = f"Xử lý thất bại: {reason}"[:500]

        db.session.commit()

        # Ghi transcript SAU khi trạng thái job đã commit: video đã xong rồi,
        # hỏng bước lưu transcript thì không được kéo theo mất kết quả.
        save_segments(flask_app, job_id, result.segments)

        try:
            Path(video_path).unlink(missing_ok=True)
        except OSError:
            flask_app.logger.warning("Không xoá được file upload tạm: %s", video_path)


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
