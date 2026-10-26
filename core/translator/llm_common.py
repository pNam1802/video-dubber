"""
core/translator/llm_common.py
Prompt và tiện ích dùng chung cho các translator dựa trên LLM (OpenAI, Gemini...).
"""
from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

from config.settings import LLM_MAX_RETRIES, LLM_PRICING_PER_MILLION_TOKENS
from core.transcriber import Segment

T = TypeVar("T")


def estimate_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Chi phí THẬT của một lượt dịch, theo đúng giá công bố của nhà cung
    cấp (config.settings.LLM_PRICING_PER_MILLION_TOKENS) — khác hẳn
    app/quota.estimate_cost() vốn chỉ ước lượng theo thời gian GPU, không
    liên quan gì tới tiền API Gemini/OpenAI thật sự tốn.

    Trả về None (KHÔNG phải 0.0) khi model không có trong bảng giá: 0.0 dễ
    bị hiểu nhầm là "miễn phí", trong khi sự thật là "chưa biết giá" — hai
    ý nghĩa khác nhau, không được đánh đồng.
    """
    pricing = LLM_PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        return None
    cost = (prompt_tokens or 0) / 1_000_000 * pricing["input"] + (completion_tokens or 0) / 1_000_000 * pricing["output"]
    return round(cost, 6)


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
    tự chọn đích), để hỗ trợ thêm Nhật/Trung/Hàn.

    Không gắn với lĩnh vực nào: trước đây prompt chuyên AI/ML kèm danh sách
    thuật ngữ cố định, giờ video có thể là mọi chủ đề (nấu ăn, vlog, tin
    tức, bài giảng...) nên để mô hình tự nhận ra ngữ cảnh."""
    source_name = language_name(source_language)
    target_name = language_name(target_language)
    return f"""Bạn là chuyên gia dịch thuật {source_name} sang {target_name} cho lồng tiếng video
thuộc mọi chủ đề (giải trí, đời sống, giáo dục, tin tức, kỹ thuật...).

Quy tắc dịch:
1. Dịch tự nhiên như người bản xứ nói, chuẩn ngữ pháp {target_name}.
2. Tự nhận ra chủ đề và giọng điệu của video (trang trọng, thân mật, hài hước...)
   rồi dịch cho phù hợp; dùng xưng hô tự nhiên theo ngữ cảnh.
3. Thành ngữ, tiếng lóng, câu đùa: dịch theo NGHĨA, không dịch từng chữ.
4. Giữ nguyên tên riêng, tên thương hiệu/sản phẩm, và thuật ngữ chuyên ngành
   mà người nói {target_name} vẫn quen dùng nguyên gốc.
5. Không thêm giải thích hay chú thích, chỉ trả về văn bản đã dịch.
6. Dịch ngắn gọn, độ dài gần với câu gốc, phù hợp để đọc thành lời nói.
7. Giữ NHẤT QUÁN xưng hô (anh/cậu/cô/em...) và cách viết tên riêng xuyên suốt
   video — người xem nghe liền một mạch, đổi xưng hô giữa chừng rất khó chịu.
"""


#: Giới hạn token output. Trước đây 1000/lô và 500/câu — câu dài (vài trăm
#: chữ) bị API cắt ngang giữa chừng, bản dịch kết thúc kiểu "…những người
#: nhiễm bệ" mà không báo lỗi gì. Chỉ tính tiền theo token thật dùng nên
#: đặt rộng tay không tốn thêm.
BATCH_MAX_TOKENS = 8192
SINGLE_MAX_TOKENS = 4096


def drop_truncated_tail(parts: Dict[int, str]) -> Dict[int, str]:
    """Phản hồi bị cắt vì hết token thì dòng CUỐI đã parse gần như chắc chắn
    dở dang — bỏ nó đi để fill_batch_gaps() dịch lại riêng câu đó, thay vì
    nhận nửa câu như bản dịch hoàn chỉnh."""
    if parts:
        parts = dict(parts)
        parts.pop(max(parts))
    return parts


#: Số câu mỗi lô, và số cặp (gốc, dịch) của lô trước gửi kèm làm ngữ cảnh.
BATCH_SIZE = 15
CONTEXT_PAIRS = 6


def build_batch_prompt(
    texts: List[str],
    source_language: str = "en",
    target_language: str = "vi",
    context: Sequence[Tuple[str, str]] = (),
) -> str:
    """Ghép nhiều câu thành 1 prompt đánh số để dịch trong một request.

    context: vài cặp (gốc, đã dịch) ngay trước lô này. Mỗi lô là một request
    độc lập — không có ngữ cảnh thì lô sau tự chọn xưng hô lại từ đầu (bug
    thật, job #30: Minjun là "anh" ở lô 1, thành "cậu" ở lô 2, rồi lại "anh").
    """
    source_name = language_name(source_language)
    target_name = language_name(target_language)
    numbered = "\n".join(f"{idx+1}. {t}" for idx, t in enumerate(texts))
    context_block = ""
    if context:
        lines = "\n".join(f"- {src}\n  → {dst}" for src, dst in context)
        context_block = (
            "Các câu NGAY TRƯỚC đã được dịch như sau — chỉ để tham khảo, giữ đúng "
            "xưng hô và tên riêng như vậy, KHÔNG dịch lại các câu này:\n"
            f"{lines}\n\n"
        )
    return (
        context_block
        + f"Dịch {len(texts)} câu sau từ {source_name} sang {target_name}.\n"
        f"Trả về đúng {len(texts)} dòng, mỗi dòng bắt đầu bằng số thứ tự tương ứng "
        f"(ví dụ: '1. ...', '2. ...'). Không thêm hoặc bỏ dòng nào.\n\n"
        f"{numbered}"
    )


def batch_context(segments: List[Segment], start: int) -> List[Tuple[str, str]]:
    """CONTEXT_PAIRS cặp (gốc, đã dịch) ngay trước vị trí start."""
    previous = segments[max(0, start - CONTEXT_PAIRS):start]
    return [(seg.text, seg.translated) for seg in previous if seg.translated]


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
