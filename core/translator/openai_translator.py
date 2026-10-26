"""
core/translator/openai_translator.py
Dịch bằng OpenAI GPT.
"""
from __future__ import annotations

from typing import List

from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from core.translator.base import BaseTranslator
from core.translator.llm_common import (
    build_batch_prompt,
    BATCH_MAX_TOKENS,
    BATCH_SIZE,
    SINGLE_MAX_TOKENS,
    batch_context,
    build_system_prompt,
    call_with_retry,
    drop_truncated_tail,
    fill_batch_gaps,
    parse_numbered_lines,
)
from core.transcriber import Segment


class OpenAITranslator(BaseTranslator):
    """Dịch bằng OpenAI GPT-4o."""

    #: Tên hiển thị/log của nhà cung cấp — lớp con (vd DeepSeek) đổi lại.
    provider = "OpenAI"
    #: Tham số riêng của nhà cung cấp gửi kèm mỗi request (extra_body).
    extra_body: dict | None = None

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        source_language: str = "en",
        target_language: str = "vi",
        base_url: str | None = None,
    ):
        super().__init__()
        if not api_key:
            raise ValueError(f"{self.provider.upper()}_API_KEY chưa được thiết lập!")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self._cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return f"{self.provider} {self.model}"

    def _system_prompt(self) -> str:
        return build_system_prompt(self.source_language, self.target_language)

    def _record_response_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(usage.prompt_tokens, usage.completion_tokens)

    def translate_text(self, text: str) -> str:
        """Dịch một đoạn văn ngắn."""
        if text in self._cache:
            return self._cache[text]

        response = call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=SINGLE_MAX_TOKENS,
                extra_body=self.extra_body,
            ),
            what=self.provider,
        )
        self._record_response_usage(response)
        if response.choices[0].finish_reason == "length":
            print(f"[{self.provider}] Bản dịch bị cắt vì hết token ({len(text)} ký tự nguồn).")
        translated = response.choices[0].message.content.strip()
        self._cache[text] = translated
        return translated

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Dịch theo batch để tiết kiệm API call.
        Ghép nhiều câu thành 1 request, phân tách bằng dấu hiệu đặc biệt.
        """
        if not texts:
            return []

        segments = [Segment(start=0, end=0, text=t) for t in texts]
        translated_segments = self.translate_segments_batch(segments)
        return [seg.translated for seg in translated_segments]

    def translate_segments_batch(self, segments: List[Segment], batch_size: int = BATCH_SIZE) -> List[Segment]:
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            texts = [seg.text for seg in batch]

            prompt = build_batch_prompt(
                texts, self.source_language, self.target_language, context=batch_context(segments, i)
            )
            response = call_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=BATCH_MAX_TOKENS,
                    extra_body=self.extra_body,
                ),
                what=f"{self.provider} batch {i // batch_size + 1}",
            )
            self._record_response_usage(response)

            raw = response.choices[0].message.content.strip()
            parts = parse_numbered_lines(raw)
            if response.choices[0].finish_reason == "length":
                parts = drop_truncated_tail(parts)
            fill_batch_gaps(batch, parts, self.translate_text)

        return segments
