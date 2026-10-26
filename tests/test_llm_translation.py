"""Câu thiếu dòng khi dịch theo batch — xem core/translator/llm_common.py.

Bug thật gặp trên production: Gemini gộp/bỏ sót một dòng trong phản hồi đánh
số dù prompt đã yêu cầu đúng số dòng. Chỗ thiếu dòng bị nhét thẳng nguyên văn
tiếng Anh vào segment.translated — không phải AI "ảo giác" mà là dịch thất
bại trong im lặng, kéo theo TTS đọc tiếng Anh bằng giọng tiếng Việt và bảng
lời thoại hiện hai dòng giống hệt nhau (transcript viewer, xem job.html).

Sửa: trước khi chấp nhận giữ nguyên bản gốc, thử dịch lại riêng đúng câu đó
một lần qua translate_text() của chính engine — parse_numbered_lines() và
fill_batch_gaps() không gọi API thật, chỉ cần các hàm giả lập trả về đúng
hình dạng dữ liệu là kiểm tra được logic, không cần key thật.
"""
from __future__ import annotations

from core.transcriber import Segment
from core.translator.gemini_translator import GeminiTranslator
from core.translator.llm_common import (
    build_batch_prompt,
    build_system_prompt,
    drop_truncated_tail,
    estimate_llm_cost,
    fill_batch_gaps,
    language_name,
)
from core.translator.openai_translator import OpenAITranslator


# ── fill_batch_gaps() độc lập với engine ─────────────────────
def test_uses_parsed_translation_when_line_present():
    batch = [Segment(start=0, end=1, text="Hello.")]
    fill_batch_gaps(batch, {1: "Xin chào."}, translate_one=lambda t: "KHÔNG ĐƯỢC GỌI")
    assert batch[0].translated == "Xin chào."


def test_retries_missing_line_via_translate_one():
    batch = [Segment(start=0, end=1, text="Hello."), Segment(start=1, end=2, text="Missing one.")]
    fill_batch_gaps(batch, {1: "Xin chào."}, translate_one=lambda t: "Dịch lại: " + t)
    assert batch[1].translated == "Dịch lại: Missing one."


def test_falls_back_to_english_when_retry_also_fails():
    """Dịch lại cũng hỏng thì giữ nguyên tiếng Anh — thà còn âm thanh có
    nghĩa (TTS vẫn đọc được) còn hơn để trống hẳn khoảng thời gian đó, và
    không được để lỗi này làm sập cả job."""
    def boom(text: str) -> str:
        raise RuntimeError("API vẫn lỗi")

    batch = [Segment(start=0, end=1, text="Missing entirely.")]
    fill_batch_gaps(batch, {}, translate_one=boom)
    assert batch[0].translated == "Missing entirely."


# ── GeminiTranslator: gộp dòng trong phản hồi thật gặp phải ──
def test_gemini_retries_gap_with_single_sentence_call(monkeypatch):
    translator = GeminiTranslator(api_key="test-key")
    calls: list[str] = []

    def fake_generate(prompt, max_output_tokens, what="Gemini"):
        calls.append(what)
        if what == "Gemini batch 1":
            # Mô phỏng đúng bug thật: gộp câu 2 và 3 vào một dòng, thiếu dòng số 3.
            return "1. Xin chào.\n2. Đây là câu hai và ba."
        return "Đây là câu ba."  # translate_text() gọi lại riêng câu thiếu

    monkeypatch.setattr(translator, "_generate", fake_generate)

    segments = [
        Segment(start=0, end=1, text="Hello."),
        Segment(start=1, end=2, text="This is two."),
        Segment(start=2, end=3, text="This is three."),
    ]
    result = translator.translate_segments_batch(segments)

    assert result[0].translated == "Xin chào."
    assert result[2].translated == "Đây là câu ba."  # được điền bằng dịch lại, KHÔNG còn là tiếng Anh
    assert "Gemini batch 1" in calls


def test_gemini_gap_retry_failure_keeps_english_not_crash(monkeypatch):
    translator = GeminiTranslator(api_key="test-key")

    def fake_generate(prompt, max_output_tokens, what="Gemini"):
        if what == "Gemini batch 1":
            return "1. Xin chào."  # chỉ 1 dòng cho 2 câu
        raise RuntimeError("API vẫn lỗi")

    monkeypatch.setattr(translator, "_generate", fake_generate)

    segments = [Segment(start=0, end=1, text="Hello."), Segment(start=1, end=2, text="Second.")]
    result = translator.translate_segments_batch(segments)

    assert result[0].translated == "Xin chào."
    assert result[1].translated == "Second."  # dịch lại cũng hỏng, giữ nguyên chứ không crash


# ── OpenAITranslator: cùng cơ chế, client khác ───────────────
class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeCompletions:
    """Trả lần lượt từng phản hồi đã xếp sẵn — batch trước, rồi các lần
    dịch lại riêng lẻ theo đúng thứ tự fill_batch_gaps() gọi tới."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def create(self, **kwargs):
        return _FakeResponse(self._responses.pop(0))


class _FakeOpenAIClient:
    def __init__(self, responses: list[str]):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(responses)})()


def test_openai_retries_gap_with_single_sentence_call(monkeypatch):
    translator = OpenAITranslator(api_key="test-key")
    translator.client = _FakeOpenAIClient([
        "1. Xin chào.\n2. Đây là câu hai và ba.",  # batch: thiếu dòng số 3
        "Đây là câu ba.",                            # dịch lại riêng câu thiếu
    ])

    segments = [
        Segment(start=0, end=1, text="Hello."),
        Segment(start=1, end=2, text="This is two."),
        Segment(start=2, end=3, text="This is three."),
    ]
    result = translator.translate_segments_batch(segments)

    assert result[0].translated == "Xin chào."
    assert result[2].translated == "Đây là câu ba."


# ── Prompt theo đúng cặp ngôn ngữ, không còn cứng EN→VI ───────
def test_language_name_maps_known_codes():
    assert language_name("vi") == "tiếng Việt"
    assert language_name("ja") == "tiếng Nhật"
    assert language_name("zh") == "tiếng Trung"
    assert language_name("ko") == "tiếng Hàn"
    assert language_name("en") == "tiếng Anh"


def test_language_name_falls_back_to_english_for_unknown_code():
    """Không nhận diện được thì coi như tiếng Anh — đúng với phần lớn video
    nguồn, và không được làm crash cả job vì một mã lạ."""
    assert language_name("xx") == "tiếng Anh"
    assert language_name("") == "tiếng Anh"


def test_system_prompt_reflects_target_language():
    prompt_vi = build_system_prompt("en", "vi")
    prompt_ja = build_system_prompt("en", "ja")
    assert "tiếng Anh sang tiếng Việt" in prompt_vi
    assert "tiếng Anh sang tiếng Nhật" in prompt_ja
    assert "chuẩn ngữ pháp tiếng Nhật" in prompt_ja


def test_batch_prompt_reflects_both_languages():
    prompt = build_batch_prompt(["Hello."], source_language="ja", target_language="ko")
    assert "từ tiếng Nhật sang tiếng Hàn" in prompt


def test_gemini_builds_prompt_for_chosen_target_language(monkeypatch):
    """Trước đây SYSTEM_PROMPT là hằng số cứng — giờ phải đúng theo
    target_language của từng translator, không phải luôn luôn tiếng Việt."""
    translator = GeminiTranslator(api_key="test-key", source_language="en", target_language="ja")
    seen_prompts: list[str] = []

    def fake_generate(prompt, max_output_tokens, what="Gemini"):
        seen_prompts.append(prompt)
        config = translator._config(max_output_tokens, False)
        seen_prompts.append(config.system_instruction)
        return "1. こんにちは。"

    monkeypatch.setattr(translator, "_generate", fake_generate)
    segments = [Segment(start=0, end=1, text="Hello.")]
    translator.translate_segments_batch(segments)

    assert any("tiếng Anh sang tiếng Nhật" in p for p in seen_prompts)


def test_openai_builds_prompt_for_chosen_target_language():
    translator = OpenAITranslator(api_key="test-key", source_language="en", target_language="zh")
    assert "tiếng Anh sang tiếng Trung" in translator._system_prompt()


# ── Chi phí dịch thật: ghi lại đúng token đã dùng qua API ─────
# (khác estimated_cost_usd trong DB, vốn chỉ ước lượng theo giây GPU —
# xem core/translator/llm_common.estimate_llm_cost() và app/jobs.py)
class _FakeGeminiUsage:
    def __init__(self, prompt=10, candidates=5, thoughts=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts


class _FakeGeminiResponse:
    def __init__(self, text, usage=None):
        self.text = text
        self.usage_metadata = usage


class _FakeGeminiModels:
    def __init__(self, response):
        self._response = response

    def generate_content(self, **kwargs):
        return self._response


class _FakeGeminiClient:
    def __init__(self, response):
        self.models = _FakeGeminiModels(response)


def test_gemini_records_real_token_usage():
    translator = GeminiTranslator(api_key="test-key")
    translator.client = _FakeGeminiClient(
        _FakeGeminiResponse("Xin chào.", _FakeGeminiUsage(prompt=20, candidates=8, thoughts=3))
    )
    translator.translate_text("Hello.")
    assert translator.usage_prompt_tokens == 20
    # thoughts_token_count (bật "thinking") TÍNH TIỀN như output — bỏ sót
    # thì chi phí tính ra thấp hơn hoá đơn thật.
    assert translator.usage_completion_tokens == 8 + 3


def test_gemini_missing_usage_metadata_does_not_crash():
    """Response thiếu usage_metadata (edge case của SDK) không được làm
    crash cả lượt dịch — chỉ đơn giản là không ghi được usage lần đó."""
    translator = GeminiTranslator(api_key="test-key")
    translator.client = _FakeGeminiClient(_FakeGeminiResponse("Xin chào.", usage=None))
    result = translator.translate_text("Hello.")
    assert result == "Xin chào."
    assert translator.usage_prompt_tokens == 0


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeOpenAICompletionsSingle:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeOpenAIClientSingle:
    def __init__(self, response):
        self.chat = type("Chat", (), {"completions": _FakeOpenAICompletionsSingle(response)})()


def test_openai_records_real_token_usage():
    translator = OpenAITranslator(api_key="test-key")
    response = _FakeResponse("Xin chào.")
    response.usage = _FakeUsage(prompt_tokens=15, completion_tokens=6)
    translator.client = _FakeOpenAIClientSingle(response)
    translator.translate_text("Hello.")
    assert translator.usage_prompt_tokens == 15
    assert translator.usage_completion_tokens == 6


def test_openai_missing_usage_does_not_crash():
    translator = OpenAITranslator(api_key="test-key")
    translator.client = _FakeOpenAIClientSingle(_FakeResponse("Xin chào."))  # không set .usage
    result = translator.translate_text("Hello.")
    assert result == "Xin chào."
    assert translator.usage_prompt_tokens == 0


# ── estimate_llm_cost(): giá thật, không đoán mò model lạ ─────
def test_estimate_llm_cost_known_model():
    # gemini-3.6-flash: $0.75/1M input, $3.75/1M output (config/settings.py)
    cost = estimate_llm_cost("gemini-3.6-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == 4.5


def test_estimate_llm_cost_unknown_model_returns_none_not_zero():
    """None (chưa biết giá) và 0.0 (thật sự miễn phí) là hai ý nghĩa khác
    nhau — model lạ, không có trong bảng giá, phải là None. Trả 0.0 sẽ bị
    hiểu nhầm thành "miễn phí", sai sự thật."""
    assert estimate_llm_cost("some-model-not-in-pricing-table", 1000, 1000) is None


def test_estimate_llm_cost_zero_tokens_is_zero_not_none():
    assert estimate_llm_cost("gpt-4o-mini", 0, 0) == 0.0


# ── DeepSeekTranslator: dùng lại OpenAITranslator, tắt thinking ──
class _RecordingCompletions:
    def __init__(self, text):
        self.text = text
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return _FakeResponse(self.text)


def test_deepseek_points_at_deepseek_api_and_disables_thinking():
    from core.translator.deepseek_translator import DeepSeekTranslator

    translator = DeepSeekTranslator(api_key="test-key")
    assert str(translator.client.base_url).startswith("https://api.deepseek.com")
    assert translator.name == "DeepSeek deepseek-flash"

    completions = _RecordingCompletions("Xin chào.")
    translator.client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()
    assert translator.translate_text("Hello.") == "Xin chào."
    assert completions.kwargs[0]["model"] == "deepseek-flash"
    assert completions.kwargs[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_sends_no_extra_body():
    translator = OpenAITranslator(api_key="test-key")
    completions = _RecordingCompletions("Xin chào.")
    translator.client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()
    translator.translate_text("Hello.")
    assert completions.kwargs[0]["extra_body"] is None


def test_estimate_llm_cost_deepseek_uses_peak_price():
    cost = estimate_llm_cost("deepseek-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == 1.5


# ── Phản hồi bị cắt vì hết token: không nhận nửa câu ─────────
def test_drop_truncated_tail_removes_last_line_only():
    assert drop_truncated_tail({1: "Một.", 2: "Hai.", 3: "Ba dở da"}) == {1: "Một.", 2: "Hai."}
    assert drop_truncated_tail({}) == {}


def test_openai_truncated_batch_retranslates_cut_off_line():
    """Bug thật (job #29): lô 10 câu dài chạm max_tokens, câu cuối kết thúc
    "…đang chờ ngay tr" mà vẫn được nhận như bản dịch hoàn chỉnh."""
    translator = OpenAITranslator(api_key="test-key")
    responses = [
        _FakeResponse("1. Xin chào.\n2. Đây là câu bị cắt ngay tr", finish_reason="length"),
        _FakeResponse("Đây là câu hai đầy đủ."),
    ]
    completions = type("Comp", (), {"create": lambda self, **kw: responses.pop(0)})()
    translator.client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()

    segments = [Segment(start=0, end=1, text="Hello."), Segment(start=1, end=2, text="This is two.")]
    result = translator.translate_segments_batch(segments)

    assert result[0].translated == "Xin chào."
    assert result[1].translated == "Đây là câu hai đầy đủ."


# ── Ngữ cảnh giữa các lô: giữ xưng hô nhất quán ──────────────
def test_batch_prompt_includes_previous_translations_as_context():
    prompt = build_batch_prompt(["He nods."], context=[("Minjun looks back.", "Minjun quay lại nhìn, anh mỉm cười.")])
    assert "Minjun quay lại nhìn, anh mỉm cười." in prompt
    assert "KHÔNG dịch lại" in prompt
    assert prompt.rstrip().endswith("1. He nods.")


def test_batch_prompt_without_context_has_no_context_block():
    assert "NGAY TRƯỚC" not in build_batch_prompt(["Hello."])


def test_second_batch_receives_first_batch_translations():
    """Bug thật (job #30): lô 2 không biết lô 1 gọi Minjun là "anh" nên tự
    chọn "cậu". Lô sau phải nhận các câu đã dịch của lô trước."""
    translator = OpenAITranslator(api_key="test-key")
    prompts: list[str] = []

    def create(**kwargs):
        user = kwargs["messages"][1]["content"]
        prompts.append(user)
        n = user.rstrip().splitlines()[-1].split(".")[0]
        return _FakeResponse("\n".join(f"{k}. Bản dịch {k}" for k in range(1, int(n) + 1)))

    translator.client = type("C", (), {"chat": type("Chat", (), {
        "completions": type("Comp", (), {"create": lambda self, **kw: create(**kw)})()})()})()
    segments = [Segment(start=i, end=i + 1, text=f"Line {i}.") for i in range(4)]
    translator.translate_segments_batch(segments, batch_size=2)

    assert "NGAY TRƯỚC" not in prompts[0]
    assert "Line 1." in prompts[1] and "→ Bản dịch 2" in prompts[1]
