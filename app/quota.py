"""
app/quota.py
Hạn mức theo người dùng và ước tính chi phí GPU.

Mỗi job là tiền GPU thật, và trang đăng ký đang mở cho bất kỳ ai — nên phải có
trần trước khi đưa link ra ngoài.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from app.extensions import db
from app.models import Job, JobStatus, User
from config.settings import (
    CPU_COST_PER_SECOND,
    GPU_COST_PER_SECOND,
    IMAGE_GPU_COST_PER_SECOND,
    QUOTA_GPU_SECONDS_PER_DAY,
    QUOTA_JOBS_PER_DAY,
    QUOTA_STORAGE_MB,
)


@dataclass
class Usage:
    """Mức tiêu thụ trong 24 giờ gần nhất, kèm trần tương ứng."""

    jobs_today: int
    jobs_limit: int
    gpu_seconds_today: float
    gpu_seconds_limit: float
    storage_bytes: int
    storage_limit_bytes: int
    estimated_cost_usd: float

    def as_dict(self) -> dict:
        return {
            "jobs": {"used": self.jobs_today, "limit": self.jobs_limit},
            "gpu_seconds": {"used": round(self.gpu_seconds_today, 1), "limit": self.gpu_seconds_limit},
            "storage_mb": {
                "used": round(self.storage_bytes / 1024 / 1024, 1),
                "limit": round(self.storage_limit_bytes / 1024 / 1024, 1),
            },
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }


def estimate_cost(gpu_seconds: float | None) -> float:
    """Ước tính chi phí GPU của một job.

    Đây là CẬN DƯỚI, không phải hoá đơn: Modal còn tính thời gian container nằm
    không sau job, cold start, CPU, RAM và cả container web. Dùng để so sánh
    giữa các user và các video, không dùng để đối chiếu hoá đơn.
    """
    return round((gpu_seconds or 0) * GPU_COST_PER_SECOND, 6)


def estimate_cpu_cost(cpu_seconds: float | None) -> float:
    """Như estimate_cost(), cho phần chạy trên container CPU (giai đoạn 2)."""
    return round((cpu_seconds or 0) * CPU_COST_PER_SECOND, 6)


def estimate_image_cost(gpu_seconds: float | None) -> float:
    """Như estimate_cost(), cho ảnh minh hoạ vẽ trên GPU L40S."""
    return round((gpu_seconds or 0) * IMAGE_GPU_COST_PER_SECOND, 6)


def _limits(user: User) -> tuple[int, float, int]:
    jobs = user.quota_jobs_per_day if user.quota_jobs_per_day is not None else QUOTA_JOBS_PER_DAY
    return jobs, QUOTA_GPU_SECONDS_PER_DAY, QUOTA_STORAGE_MB * 1024 * 1024


def get_usage(user: User) -> Usage:
    since = datetime.now(timezone.utc) - timedelta(days=1)
    jobs_limit, gpu_limit, storage_limit = _limits(user)

    recent = Job.query.filter(Job.user_id == user.id, Job.created_at >= since)
    jobs_today = recent.count()
    gpu_seconds_today = (
        db.session.query(func.coalesce(func.sum(Job.elapsed_sec), 0.0))
        .filter(Job.user_id == user.id, Job.created_at >= since)
        .scalar()
        or 0.0
    )
    storage_bytes = (
        db.session.query(func.coalesce(func.sum(Job.file_size), 0))
        # DONE giu video ket qua; AWAITING_REVIEW van giu nguyen file upload
        # goc tren dia cho toi khi nguoi dung duyet xong (xem app/jobs.py) —
        # cả hai đều là dung lượng thật, không tính thì thành lỗ hổng: cứ
        # upload rồi bỏ đó không duyệt là né được hạn mức vô thời hạn.
        .filter(Job.user_id == user.id, Job.status.in_((JobStatus.DONE, JobStatus.AWAITING_REVIEW)))
        .scalar()
        or 0
    )
    cost = (
        db.session.query(func.coalesce(func.sum(Job.estimated_cost_usd), 0.0))
        .filter(Job.user_id == user.id)
        .scalar()
        or 0.0
    )

    return Usage(
        jobs_today=jobs_today,
        jobs_limit=jobs_limit,
        gpu_seconds_today=float(gpu_seconds_today),
        gpu_seconds_limit=gpu_limit,
        storage_bytes=int(storage_bytes),
        storage_limit_bytes=storage_limit,
        estimated_cost_usd=float(cost),
    )


class QuotaCheck(NamedTuple):
    allowed: bool
    reason: str = ""
    # "jobs" | "gpu" | "storage": giao diện chỉ gợi ý xoá video khi hết dung
    # lượng — xoá job không giúp gì với hạn mức lượt/GPU theo ngày.
    kind: str = ""


def check_quota(user: User, incoming_bytes: int = 0) -> QuotaCheck:
    """Trả về (cho_phép, lý_do, loại_hạn_mức). Admin không bị giới hạn."""
    if user.is_admin:
        return QuotaCheck(True)

    usage = get_usage(user)

    if usage.jobs_today >= usage.jobs_limit:
        return QuotaCheck(False, (
            f"Bạn đã dùng hết {usage.jobs_limit} lượt xử lý trong 24 giờ. "
            "Vui lòng thử lại sau."
        ), "jobs")

    if usage.gpu_seconds_today >= usage.gpu_seconds_limit:
        return QuotaCheck(False, (
            f"Bạn đã dùng hết {usage.gpu_seconds_limit:.0f} giây GPU trong 24 giờ. "
            "Vui lòng thử lại sau."
        ), "gpu")

    if incoming_bytes and usage.storage_bytes + incoming_bytes > usage.storage_limit_bytes:
        # Nói rõ cả kích thước file mới: chỉ ghi "1880/2048 MB" thì người dùng
        # tưởng còn chỗ mà vẫn bị chặn, không hiểu phải xoá bao nhiêu.
        mb = 1024 * 1024
        free_mb = max(usage.storage_limit_bytes - usage.storage_bytes, 0) / mb
        limit_mb = usage.storage_limit_bytes / mb
        return QuotaCheck(False, (
            f"Video này cần {incoming_bytes / mb:.0f} MB nhưng bạn chỉ còn trống "
            f"{free_mb:.0f}/{limit_mb:.0f} MB. Hãy xoá bớt video cũ, hoặc duyệt "
            "hay huỷ các video đang chờ duyệt, rồi thử lại."
        ), "storage")

    return QuotaCheck(True)
