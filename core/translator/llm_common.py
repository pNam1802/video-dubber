"""
core/translator/llm_common.py
Prompt và tiện ích dùng chung cho các translator dựa trên LLM (OpenAI, Gemini...).
"""
from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, TypeVar

from config.settings import AI_PRESERVE_TERMS, LLM_MAX_RETRIES
from core.transcriber import Segment

T = TypeVar("T")


#: Tên hiển thị tiếng Việt cho từng mã ngôn ngữ — dùng để dựng câu lệnh cho
#: LLM ("dịch từ X sang Y"), không phải để hiện lên giao diện.
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "tiếng Anh",
    "vi": "tiếng Việt",
    "ja": "tiếng Nhật",
    "zh": "tiếng Trung",
    "ko": "tiếng Hàn",
}


def language_name(code: str) -> str:
    """Không nhận diện được thì coi như tiếng Anh — đúng với phần lớn video
    nguồn của sản phẩm, và không làm crash cả job vì một mã lạ."""
    return LANGUAGE_NAMES.get((code or "").strip().lower(), "tiếng Anh")


def build_system_prompt(source_language: str = "en", target_language: str = "vi") -> str:
    """Trước đây là hằng số cứng "tiếng Anh sang tiếng Việt" — giờ dựng theo
    đúng cặp ngôn ngữ thật của job (Whisper tự nhận dạng nguồn, người dùng
    tự chọn đích), để hỗ trợ thêm Nhật/Trung/Hàn."""
    source_name = language_name(source_language)
    target_name = language_name(target_language)
    preserve_terms = ", ".join(AI_PRESERVE_TERMS[:20])
    return f"""Bạn là chuyên gia dịch thuật {source_name} sang {target_name},
chuyên lĩnh vực Trí tuệ nhân tạo (AI) và Học máy (ML).

Quy tắc dịch:
1. Dịch tự nhiên, chuẩn ngữ pháp {target_name}.
2. Giữ nguyên các thuật ngữ kỹ thuật AI/ML bằng tiếng Anh (không dịch):
   {preserve_terms}
3. Không thêm giải thích hay chú thích, chỉ trả về văn bản đã dịch.
4. Giữ nguyên dấu câu và cấu trúc câu gốc.
5. Dịch ngắn gọn, phù hợp để đọc thành lời nói.
"""


def build_batch_prompt(texts: List[str], source_language: str = "en", target_language: str = "vi") -> str:
    """Ghép nhiều câu thành 1 prompt đánh số để dịch trong một request."""
    source_name = language_name(source_language)
    target_name = language_name(target_language)
    numbered = "\n".join(f"{idx+1}. {t}" for idx, t in enumerate(texts))
    return (
        f"Dịch {len(texts)} câu sau từ {source_name} sang {target_name}.\n"
        f"Trả về đúng {len(texts)} dòng, mỗi dòng bắt đầu bằng số thứ tự tương ứng "
        f"(ví dụ: '1. ...', '2. ...'). Không thêm hoặc bỏ dòng nào.\n\n"
        f"{numbered}"
    )


def parse_numbered_lines(raw: str) -> Dict[int, str]:
    """Tách kết quả dạng '1. ...' / '1) ...' thành dict {số thứ tự: bản dịch}."""
    parts: Dict[int, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in (". ", ") "):
            if line[0].isdigit() and sep in line:
                idx_str, _, translation = line.partition(sep)
                try:
                    parts[int(idx_str.strip())] = translation.strip()
                except ValueError:
                    pass
                break
    return parts


def fill_batch_gaps(batch: List[Segment], parts: Dict[int, str], translate_one: Callable[[str], str]) -> None:
    """Điền chỗ trống khi parse_numbered_lines() thiếu dòng cho một câu trong
    batch — mô hình đôi khi gộp hai câu vào một dòng hay bỏ sót một số thứ
    tự dù prompt đã yêu cầu đúng số dòng. Bug thật gặp trên production: câu
    thiếu dòng bị nhét thẳng nguyên văn tiếng Anh vào seg.translated, kéo
    theo TTS đọc tiếng Anh bằng giọng tiếng Việt — nghe sai hẳn, không phải
    "AI ảo giác" mà là dịch thất bại trong im lặng.

    Trước khi chấp nhận giữ nguyên bản gốc, thử dịch lại RIÊNG đúng câu đó
    một lần — translate_one (translate_text của chính engine đang dùng) đã
    có sẵn cơ chế thử lại khi lỗi tạm thời, nên đây gần như luôn sửa được
    trừ khi API thật sự không dùng được."""
    for j, seg in enumerate(batch):
        translated = parts.get(j + 1)
        if translated:
            seg.translated = translated
            continue
        try:
            seg.translated = translate_one(seg.text)
        except Exception:
            # Dịch lại cũng hỏng — thà còn tiếng Anh (ít nhất TTS còn phát ra
            # âm thanh có nghĩa) còn hơn để trống hẳn khoảng thời gian đó.
            seg.translated = seg.text


# ── Thử lại khi API trả lỗi tạm thời ─────────────────────────
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_RETRYABLE_NAME_HINTS = (
    "ratelimit",
    "timeout",
    "connection",
    "internalserver",
    "serviceunavailable",
    "unavailable",
    "overloaded",
    "resourceexhausted",
)

_RETRYABLE_MESSAGE_HINTS = (
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "try again",
    "resource exhausted",
)


def is_retryable(exc: BaseException) -> bool:
    """Lỗi tạm thời (429, 5xx, timeout, đứt kết nối) thì đáng thử lại; lỗi
    cấu hình hay sai key thì không — thử lại chỉ tốn thời gian."""
    for attr in ("status_code", "code", "http_status"):
        status = getattr(exc, attr, None)
        if isinstance(status, int) and status in RETRYABLE_STATUS:
            return True

    name = type(exc).__name__.lower()
    if any(hint in name for hint in _RETRYABLE_NAME_HINTS):
        return True

    message = str(exc).lower()
    return any(hint in message for hint in _RETRYABLE_MESSAGE_HINTS)


def call_with_retry(
    func: Callable[[], T],
    *,
    what: str = "API",
    attempts: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """
    Gọi func(), thử lại với backoff luỹ thừa khi gặp lỗi tạm thời.

    Không có bước này thì một lần dính 429 ở batch thứ 30/50 làm hỏng cả job,
    mất luôn phần đã dịch trước đó.
    """
    total = attempts if attempts is not None else LLM_MAX_RETRIES
    total = max(1, total)
    delay = base_delay

    for attempt in range(1, total + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= total or not is_retryable(exc):
                raise
            wait = min(delay + random.uniform(0, delay * 0.5), max_delay)
            print(f"[{what}] Lỗi tạm thời (lần {attempt}/{total}): {exc}. Thử lại sau {wait:.1f}s...")
            time.sleep(wait)
            delay = min(delay * 2, max_delay)

    raise RuntimeError(f"[{what}] Hết số lần thử lại.")  # không bao giờ tới đây
