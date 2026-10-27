"""
core/video_retime.py
Tua nhanh HÌNH ở những khoảng lặng để khớp giọng đọc, thay vì kéo giãn giọng.

Giọng Việt (nhất là VieNeu) thường ngắn hơn lời nói gốc: đặt đúng mốc câu gốc
thì sau mỗi câu là một khoảng lặng (job #44: 21/26 chỗ nghỉ ≥ 0,5 giây). Ở đây
mỗi câu ứng với một "cửa sổ" hình — từ lúc câu đó bắt đầu tới lúc câu sau bắt
đầu — và cửa sổ được tua nhanh cho vừa câu tiếng Việt + một nhịp nghỉ, tối đa
MAX_VIDEO_SPEED lần. Không bao giờ làm CHẬM hình: câu Việt dài hơn cửa sổ thì
hình chạy tốc độ thường, giọng tự xử lý như cũ (plan_timeline tăng tốc).

Đoạn trước câu đầu tiên (intro) giữ nguyên tốc độ.

Hàm thuần, không gọi ffmpeg — VideoComposer dựng bộ lọc từ kết quả.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from core.voice_timeline import PAUSE

MAX_VIDEO_SPEED = 2.0
# Tua dưới mức này thì để nguyên: khác biệt không đáng một lần cắt ghép.
_MIN_SPEEDUP = 1.02


@dataclass
class Window:
    """Một đoạn hình gốc [src_start, src_end) phát với tốc độ speed, bắt đầu
    ở thời điểm out_start trong video mới."""
    src_start: float
    src_end: float
    speed: float
    out_start: float

    @property
    def out_end(self) -> float:
        return self.out_start + (self.src_end - self.src_start) / self.speed


def plan_retime(
    slots: Sequence[Tuple[float, float]],
    durations: Sequence[float],
    video_duration: float,
    max_speed: float = MAX_VIDEO_SPEED,
    pause: float = PAUSE,
) -> List[Window]:
    """slots: (start, end) câu gốc theo thứ tự; durations: độ dài giọng Việt
    tương ứng. Trả về các cửa sổ phủ kín [0, video_duration]."""
    windows: List[Window] = []
    out = 0.0

    def add(src_start: float, src_end: float, speed: float) -> None:
        nonlocal out
        if src_end - src_start <= 1e-6:
            return
        windows.append(Window(src_start, src_end, speed, out))
        out = windows[-1].out_end

    starts = [max(0.0, min(s, video_duration)) for s, _ in slots]
    if not starts:
        add(0.0, video_duration, 1.0)
        return windows

    add(0.0, starts[0], 1.0)  # intro
    for i, begin in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else video_duration
        end = max(begin, end)
        need = durations[i] + pause
        speed = (end - begin) / need if need > 0 else 1.0
        speed = min(max_speed, speed)
        add(begin, end, speed if speed >= _MIN_SPEEDUP else 1.0)
    return windows


def map_time(windows: Sequence[Window], t: float) -> float:
    """Thời điểm t trong video gốc -> thời điểm tương ứng trong video mới."""
    for w in windows:
        if t < w.src_end or w is windows[-1]:
            return w.out_start + (max(t, w.src_start) - w.src_start) / w.speed
    return t


def is_identity(windows: Sequence[Window]) -> bool:
    return all(abs(w.speed - 1.0) < 1e-6 for w in windows)
