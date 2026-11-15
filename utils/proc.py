"""
utils/proc.py
Chạy lệnh ngoài (ffmpeg/ffprobe) với timeout bắt buộc.

Không có timeout thì một tiến trình ffmpeg treo sẽ giữ job ở trạng thái
"đang xử lý" vĩnh viễn — trên Modal nghĩa là trả tiền GPU cho một container
không làm gì cho tới khi hết timeout của cả function.
"""
from __future__ import annotations

import subprocess

#: Số ký tự cuối stderr được giữ lại trong thông báo lỗi.
STDERR_TAIL_CHARS = 1500


def run_command(
	cmd: list[str],
	*,
	timeout: float,
	error_message: str,
) -> subprocess.CompletedProcess:
	"""Chạy lệnh và chuyển mọi lỗi thành RuntimeError có thông tin rõ ràng."""
	try:
		return subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			check=True,
			timeout=timeout,
		)
	except FileNotFoundError as exc:
		raise RuntimeError(error_message) from exc
	except subprocess.TimeoutExpired as exc:
		raise RuntimeError(
			f"{error_message}\nLệnh chạy quá {timeout:.0f} giây nên đã bị dừng: {' '.join(cmd[:3])}..."
		) from exc
	except subprocess.CalledProcessError as exc:
		detail = (exc.stderr or "").strip()
		# Dòng lỗi thật của ffmpeg luôn nằm CUỐI stderr; phần đầu là banner
		# phiên bản/cấu hình dài hàng nghìn ký tự. Giữ đoạn cuối để nơi hiển
		# thị (cắt 200-300 ký tự) còn thấy được nguyên nhân — job #80 từng
		# mất hẳn lý do lỗi vì chỉ còn lại banner.
		if len(detail) > STDERR_TAIL_CHARS:
			detail = "…" + detail[-STDERR_TAIL_CHARS:]
		raise RuntimeError(f"{error_message}\n{detail}") from exc
