"""
core/translator/gemini_translator.py
Dịch bằng Google Gemini API.
"""
from __future__ import annotations

from typing import List

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from core.translator.base import BaseTranslator
from core.translator.llm_common import (
    BATCH_MAX_TOKENS,
    BATCH_SIZE,
    SINGLE_MAX_TOKENS,
    batch_context,
    build_batch_prompt,
    build_system_prompt,
    call_with_retry,
    drop_truncated_tail,
    fill_batch_gaps,
    parse_numbered_lines,
)
from core.transcriber import Segment


def _hit_token_limit(response) -> bool:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return "MAX_TOKENS" in str(reason or "")


class GeminiTranslator(BaseTranslator):
    """Dịch bằng Google Gemini (google-genai SDK)."""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
        source_language: str = "en",
        target_language: str = "vi",
    ):
        super().__init__()
        if not api_key:
            raise ValueError("GEMINI_API_KEY chưa được thiết lập!")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self._cache: dict[str, str] = {}
        #: Lượt _generate() gần nhất có bị cắt vì chạm max_output_tokens không.
        self._last_truncated = False
        # Model 2.5 hỗ trợ tắt "thinking" cho nhanh; model cũ thì không.
        # Lần đầu gặp lỗi vì cấu hình này thì nhớ lại để khỏi thử nữa.
        self._thinking_supported = True

    @property
    def name(self) -> str:
        return f"Gemini {self.model}"

    def _config(self, max_output_tokens: int, use_thinking: bool):
        kwargs = dict(
            system_instruction=build_system_prompt(self.source_language, self.target_language),
            temperature=0.3,
            max_output_tokens=max_output_tokens,
        )
        if use_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def _generate(self, prompt: str, max_output_tokens: int, what: str = "Gemini") -> str:
        """Gọi Gemini, có thử lại khi gặp lỗi tạm thời (429/5xx/timeout)."""
        for use_thinking in ([True, False] if self._thinking_supported else [False]):
            config = self._config(max_output_tokens, use_thinking)
            try:
                response = call_with_retry(
                    lambda: self.client.models.generate_content(
                        model=self.model, contents=prompt, config=config
                    ),
                    what=what,
                )
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    # thoughts_token_count (khi bật "thinking") vẫn TÍNH TIỀN
                    # như output — bỏ sót thì chi phí tính ra thấp hơn hoá đơn thật.
                    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                    self._record_usage(usage.prompt_token_count, output_tokens)
                self._last_truncated = _hit_token_limit(response)
                return (response.text or "").strip()
            except Exception as exc:
                if use_thinking:
                    # Model không nhận thinking_config -> bỏ hẳn, không thử lại nữa.
                    print(f"[Gemini] Model không hỗ trợ thinking_config ({exc}). Dùng cấu hình cơ bản.")
                    self._thinking_supported = False
                    continue
                raise
        raise RuntimeError("[Gemini] Không gọi được model.")  # không bao giờ tới đây

    def translate_text(self, text: str) -> str:
        """Dịch một đoạn văn ngắn."""
        if text in self._cache:
            return self._cache[text]

        translated = self._generate(text, max_output_tokens=SINGLE_MAX_TOKENS) or text
        if self._last_truncated:
            print(f"[Gemini] Bản dịch bị cắt vì hết token ({len(text)} ký tự nguồn).")
        self._cache[text] = translated
        return translated

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Dịch theo batch để tiết kiệm API call."""
        if not texts:
            return []

        segments = [Segment(start=0, end=0, text=t) for t in texts]
        translated_segments = self.translate_segments_batch(segments)
        return [seg.translated for seg in translated_segments]

    def translate_segments_batch(self, segments: List[Segment], batch_size: int = BATCH_SIZE) -> List[Segment]:
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            texts = [seg.text for seg in batch]

            raw = self._generate(
                build_batch_prompt(
                    texts, self.source_language, self.target_language, context=batch_context(segments, i)
                ),
                max_output_tokens=BATCH_MAX_TOKENS,
                what=f"Gemini batch {i // batch_size + 1}",
            )
            parts = parse_numbered_lines(raw)
            if self._last_truncated:
                parts = drop_truncated_tail(parts)
            fill_batch_gaps(batch, parts, self.translate_text)

        return segments
