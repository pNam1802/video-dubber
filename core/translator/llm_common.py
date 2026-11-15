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


#: Kiểu dịch người dùng chọn ở form. "natural": dịch theo nghĩa, tự nhiên.
#: "localized": bản địa hoá câu đùa/tiếng lóng sang meme của nước đích.
TRANSLATION_STYLES = ("natural", "localized")

LOCALIZED_RULES = """
Kiểu dịch: BẢN ĐỊA HOÁ kiểu mạng xã hội.
- Tiếng lóng, meme, câu cà khịa của bản gốc thì đổi sang cách nói/câu đùa TƯƠNG ĐƯƠNG mà
  dân mạng nói {target} thật sự hay dùng trên TikTok/YouTube (kiểu kênh recap phim), để người
  xem bản xứ thấy buồn cười như người xem bản gốc. Không bịa thêm tình tiết, không đổi nội dung.
- Meme của nước nguồn mà ở nước đích mang NGHĨA KHÁC hoặc không ai hiểu (vd "Skibidi" ở mạng
  Việt là nói giảm chuyện quan hệ, còn ở Mỹ chỉ là meme "Skibidi Toilet" vô nghĩa) thì KHÔNG
  giữ nguyên chữ: dùng cách nói bản xứ theo NGHĨA THẬT ("hook up"). Ẩn dụ chỉ có ở nước nguồn
  cũng vậy (vd "cắm sừng"/"tặng cặp sừng" -> "cheated on him").
- Đừng nhồi tiếng lóng vào mọi câu: câu kể bình thường thì dịch bình thường.
"""

NOTES_BLOCK = """
NGỮ CẢNH CẢ VIDEO (đã đọc trước toàn bộ transcript) — BẮT BUỘC dùng:
{notes}

Cách dùng bảng từ trên:
- Gặp từ có trong bảng thì dịch theo NGHĨA THẬT trong bảng, bằng cách nói tự nhiên của người bản xứ.
- Meme, tiếng lóng, ẩn dụ của mạng xã hội nguồn KHÔNG phải tên riêng hay thương hiệu: không giữ
  nguyên chữ khi người nghe bản dịch sẽ không hiểu đúng nghĩa đó.
- Được dài/ngắn hơn câu gốc một chút để rõ nghĩa; giữ đúng giọng điệu của video.
"""

#: Bước ĐỌC HIỂU trước khi dịch: đọc cả transcript, tóm tắt + lập bảng từ.
#: Đo 16/11/2026 trên đoạn review manhwa: không có bước này thì "đòi
#: Skibidi", "tặng cặp sừng 36 mét", "nờ tờ giờ (NTR) lại" bị dịch sát chữ;
#: có bảng từ (kèm cách dùng ở NOTES_BLOCK) thì ra "asked for a hookup",
#: "cheated on him big time", "NTR them right back". Bật suy nghĩ (thinking)
#: cho riêng bước này — suy ra "nờ tờ giờ" là N-T-R đọc từng chữ cần nó.
UNDERSTAND_MAX_TOKENS = 16384
#: Transcript dài hơn thì chỉ gửi phần đầu — đủ để nắm nhân vật/giọng điệu,
#: và không vượt ngữ cảnh của model rẻ.
UNDERSTAND_MAX_CHARS = 60_000
#: Bảng từ gắn vào MỖI request dịch sau đó, nên giới hạn độ dài.
NOTES_MAX_CHARS = 6_000


def build_understanding_prompt(
    texts: Sequence[str], source_language: str = "en", target_language: str = "vi", style: str = "natural"
) -> str:
    source_name = language_name(source_language)
    target_name = language_name(target_language)
    transcript = "\n".join(t.strip() for t in texts if t and t.strip())
    if len(transcript) > UNDERSTAND_MAX_CHARS:
        transcript = transcript[:UNDERSTAND_MAX_CHARS] + "\n[...transcript còn tiếp...]"
    equivalent = (
        f"tiếng lóng/meme TƯƠNG ĐƯƠNG mà dân mạng nói {target_name} thật sự dùng"
        if style == "localized"
        else f"cách nói tương đương tự nhiên bằng {target_name}"
    )
    return f"""Dưới đây là TOÀN BỘ transcript {source_name} của một video, sắp được dịch sang {target_name}.
Transcript do máy nhận dạng giọng nói nên có thể nghe nhầm (sai dấu, nhầm từ gần âm).
Đọc hết, hiểu câu chuyện, rồi trả về đúng hai phần, viết ngắn gọn:

TÓM TẮT: 2-4 câu — chủ đề, nhân vật (tên, giới tính, quan hệ), giọng điệu của video.
BẢNG TỪ: mỗi dòng "từ trong transcript = nghĩa thật = {equivalent}", cho mọi tiếng lóng/từ mạng,
từ bị nghe nhầm, chữ cái bị đánh vần thành âm (vd "ét o ét" = SOS), ẩn dụ, nói giảm, câu đùa.
- Cột cuối chỉ ghi MỘT cách nói, dùng thẳng được trong bản dịch và người bản xứ nghe hiểu ngay.
  Ẩn dụ chỉ nước nguồn mới có thì ghi NGHĨA, không dịch hình ảnh (vd "cắm sừng" -> "cheated on",
  không phải "horns").
- Cách gọi người trong lời kể (vd "bà chị", "ông anh", "thanh niên", "bà thím") thường là gọi
  vui, KHÔNG nói tuổi hay họ hàng thật — ghi đúng như vậy khi ngữ cảnh cho thấy thế.
Suy luận từ NGỮ CẢNH câu chuyện. Từ bình thường thì bỏ qua. Không có gì thì ghi "(không có)".

TRANSCRIPT:
{{transcript}}""".replace("{transcript}", transcript)


def clean_notes(raw: str) -> str:
    """Bảng từ đi kèm mọi request dịch sau đó — cắt bớt nếu model viết dài."""
    text = (raw or "").strip()
    return text[:NOTES_MAX_CHARS]


def build_system_prompt(
    source_language: str = "en",
    target_language: str = "vi",
    style: str = "natural",
    notes: str = "",
) -> str:
    """Trước đây là hằng số cứng "tiếng Anh sang tiếng Việt" — giờ dựng theo
    đúng cặp ngôn ngữ thật của job (Whisper tự nhận dạng nguồn, người dùng
    tự chọn đích), để hỗ trợ thêm Nhật/Trung/Hàn.

    Không gắn với lĩnh vực nào: trước đây prompt chuyên AI/ML kèm danh sách
    thuật ngữ cố định, giờ video có thể là mọi chủ đề (nấu ăn, vlog, tin
    tức, bài giảng...) nên để mô hình tự nhận ra ngữ cảnh.

    style: kiểu dịch (TRANSLATION_STYLES). notes: tóm tắt + bảng từ của bước
    đọc hiểu (build_understanding_prompt) — có thì gắn kèm cùng cách dùng."""
    source_name = language_name(source_language)
    target_name = language_name(target_language)
    prompt = f"""Bạn là chuyên gia dịch thuật {source_name} sang {target_name} cho lồng tiếng video
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
8. Văn bản nguồn là bản NHẬN DẠNG GIỌNG NÓI tự động nên có thể nghe nhầm: sai
   dấu thanh, hoặc nhầm sang từ gần âm (tiếng Việt hay gặp: "ba chị" hay "3 chị"/"bà chị",
   "nhỏ bộ"/"nhỏ bồ", "sinh đẹp"/"xinh đẹp"). Từ nào vô lý theo ngữ cảnh thì
   hiểu theo từ gần âm hợp lý nhất (đọc cả các câu xung quanh), rồi mới dịch.
   Đừng dịch nguyên văn một từ nghe nhầm.
"""
    if style == "localized":
        prompt += LOCALIZED_RULES.format(target=target_name)
    if notes.strip():
        prompt += NOTES_BLOCK.format(notes=notes.strip())
    return prompt


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
