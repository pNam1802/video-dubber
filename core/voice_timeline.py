"""
core/voice_timeline.py
Xếp giọng đọc lên dòng thời gian cho nghe LIỀN MẠCH như giọng kể (kiểu video
review trên TikTok), thay vì ép mỗi câu vào đúng khe thời gian của câu gốc.

Cách cũ đặt từng câu đúng thời điểm câu tiếng Anh bắt đầu. Tiếng Việt thường
đọc nhanh hơn, nên sau mỗi câu là một khoảng lặng — đo thật trên job #32: 34
chỗ nghỉ ≥ 0.5s, dài nhất 3.1s, phần lớn rơi giữa mạch kể. Ở đây:

- Câu sau bắt đầu ngay khi câu trước đọc xong + một nhịp nghỉ ngắn (PAUSE),
  được phép vào SỚM hơn câu gốc tối đa MAX_LEAD giây để lấp khoảng lặng.
- Không bao giờ hai câu chồng lên nhau.
- Câu nào nếu đọc tốc độ thường sẽ đẩy câu sau trễ quá MAX_LAG giây so với
  hình thì được yêu cầu đọc nhanh hơn (speed > 1), tối đa MAX_SPEED.

Chỉ dời thời điểm thì CHƯA đủ: giọng HoaiMy đọc nhanh hơn người kể tiếng Anh
(job #33: tổng giọng Việt = 0.92 lần thời gian nói gốc), phần thiếu vẫn thành
khoảng lặng — 51 chỗ nghỉ ≥ 1s. Nên thêm hai bước giãn tốc độ:

- base_speed(): một tốc độ CHUNG cho cả video để tổng giọng Việt ≈ thời gian
  nói gốc (giọng đều, không câu nhanh câu chậm).
- fill_gaps(): câu nào vẫn để hở ≥ FILL_GAP thì đọc chậm thêm một chút để
  lấp, không chậm hơn MIN_FILL_FACTOR.

Mô phỏng trên dữ liệu thật job #33: chỗ nghỉ ≥ 1s từ 51 còn 0, dài nhất 1.0s.

Hàm thuần, không gọi mạng hay ffmpeg — TTSEngine dùng kết quả để tổng hợp
lại các câu bằng tham số rate của chính edge-tts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

PAUSE = 0.25
MAX_LEAD = 2.0
MAX_LAG = 1.5
MAX_SPEED = 1.35
MIN_BASE_SPEED = 0.88
FILL_GAP = 0.6
MIN_FILL_FACTOR = 0.9


def base_speed(slots: Sequence[Tuple[float, float]], durations: Sequence[float]) -> float:
    """Tốc độ chung (≤ 1) để tổng độ dài giọng đọc khớp tổng thời gian nói
    của bản gốc. Giọng đã dài hơn bản gốc thì giữ 1.0 — phần dư do
    plan_timeline() tự tăng tốc từng câu khi cần."""
    spoken = sum(max(0.0, end - start) for start, end in slots)
    total = sum(durations)
    if spoken <= 0 or total <= 0:
        return 1.0
    return max(MIN_BASE_SPEED, min(1.0, total / spoken))


def fill_gaps(
    durations: Sequence[float],
    placements: Sequence["Placement"],
    pause: float = PAUSE,
    min_gap: float = FILL_GAP,
    min_factor: float = MIN_FILL_FACTOR,
) -> List[float]:
    """Hệ số tốc độ (≤ 1) cho từng câu để lấp khoảng hở phía sau nó. Câu đã
    phải đọc nhanh (speed > 1) thì không đụng tới. Câu sau không bị dời: nó
    đang đứng ở mốc MAX_LEAD (chính vì vậy mới có khoảng hở), câu trước dài
    ra thêm tối đa đúng phần hở trừ một nhịp nghỉ."""
    factors = [1.0] * len(placements)
    for i in range(len(placements) - 1):
        current = placements[i]
        if current.speed > 1.0 or durations[i] <= 0:
            continue
        end = current.start + durations[i] / current.speed
        gap = placements[i + 1].start - end
        if gap >= min_gap:
            target = durations[i] + gap - pause
            factors[i] = max(min_factor, durations[i] / target)
    return factors


@dataclass
class Placement:
    start: float
    speed: float


def plan_timeline(
    slots: Sequence[Tuple[float, float]],
    durations: Sequence[float],
    pause: float = PAUSE,
    max_lead: float = MAX_LEAD,
    max_lag: float = MAX_LAG,
    max_speed: float = MAX_SPEED,
) -> List[Placement]:
    """slots: (start, end) của câu gốc; durations: độ dài giọng đọc ở tốc độ
    thường (đã cắt khoảng lặng đầu/cuối). Trả về thời điểm bắt đầu và tốc độ
    cần đọc cho từng câu, cùng thứ tự đầu vào."""
    placements: List[Placement] = []
    cursor = float("-inf")  # thời điểm câu trước đọc xong
    for i, ((slot_start, _), duration) in enumerate(zip(slots, durations)):
        start = max(0.0, slot_start - max_lead, cursor + pause)

        # Hạn chót: câu sau (nếu có) không được bắt đầu trễ quá max_lag.
        if i + 1 < len(slots):
            deadline = slots[i + 1][0] + max_lag - pause
        else:
            deadline = slots[i][1] + max_lag
        available = deadline - start
        speed = 1.0
        if duration > 0 and available > 0 and duration > available:
            speed = min(duration / available, max_speed)
        elif duration > 0 and available <= 0:
            speed = max_speed

        placements.append(Placement(start=start, speed=speed))
        cursor = start + duration / speed
    return placements
