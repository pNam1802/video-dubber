"""
app/mailer.py
Gửi email thông báo tiến trình job — SMTP thuần qua smtplib (mặc định
Gmail + app password), không cần thư viện hay dịch vụ ngoài nào.

Nguyên tắc: lỗi gửi mail KHÔNG BAO GIỜ được làm hỏng job. Job là việc
chính, email chỉ là tiện ích đi kèm — mọi hàm ở đây tự bắt lỗi, chỉ log,
không bao giờ ném ra ngoài cho nơi gọi.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText

from config.settings import (
    APP_BASE_URL,
    MAIL_ENABLED,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
)

logger = logging.getLogger(__name__)


def job_url(job_id: int) -> str:
    """Link đầy đủ tới trang job, dùng trong email. Thiếu APP_BASE_URL thì
    trả về đường dẫn tương đối — vẫn đọc được nếu người dùng đã đăng nhập
    sẵn trên cùng domain, chỉ là không bấm thẳng từ email ở máy khác được."""
    path = f"/job/{job_id}"
    return f"{APP_BASE_URL}{path}" if APP_BASE_URL else path


def send_email(to: str, subject: str, body: str) -> bool:
    """Gửi một email văn bản thuần. Trả về True nếu gửi thành công.

    Chưa cấu hình SMTP (MAIL_ENABLED=False) hoặc người nhận rỗng thì bỏ
    qua trong im lặng ngay từ đầu — tính năng phụ trợ, không phải một
    phần bắt buộc của pipeline, và im lặng đúng lúc mới không làm log
    ngập tràn cảnh báo vô ích trên máy dev chưa cấu hình mail.
    """
    if not MAIL_ENABLED or not to:
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to], msg.as_string())
        return True
    except Exception:
        logger.exception("Không gửi được email tới %s", to)
        return False


def notify_job_status(job, event: str) -> None:
    """Gửi email theo đúng mốc trạng thái của job, nếu người dùng còn bật
    nhận thông báo và có email. Không tự truy vấn job.user — gọi nơi đã
    có sẵn job trong session để tránh N+1 không cần thiết.

    event: "awaiting_review" | "done" | "failed"
    """
    user = job.user
    if user is None or not user.email or not user.notify_email:
        return

    name = job.source_filename or f"Job #{job.id}"
    link = job_url(job.id)

    if event == "awaiting_review":
        subject = f"[Nora] \"{name}\" đã dịch xong, cần bạn duyệt"
        body = (
            f"Video \"{name}\" đã nhận dạng và dịch xong.\n\n"
            "Vào xem/sửa transcript rồi xác nhận để tạo giọng đọc:\n"
            f"{link}\n\n"
            "Job sẽ đứng yên ở đây cho tới khi bạn xác nhận — không tự chạy tiếp."
        )
    elif event == "done":
        subject = f"[Nora] \"{name}\" đã lồng tiếng xong"
        body = f"Video \"{name}\" đã xử lý xong, tải về tại:\n{link}"
    elif event == "failed":
        reason = " ".join((job.error or job.message or "Không rõ nguyên nhân.").split())[:300]
        subject = f"[Nora] \"{name}\" xử lý thất bại"
        body = f"Video \"{name}\" xử lý thất bại.\n\nLý do: {reason}\n\nChi tiết: {link}"
    else:
        return

    send_email(user.email, subject, body)
