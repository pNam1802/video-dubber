"""
core/translator/deepseek_translator.py
Dịch bằng DeepSeek — API tương thích OpenAI nên dùng lại toàn bộ
OpenAITranslator, chỉ đổi base_url và tắt chế độ suy nghĩ.
"""
from __future__ import annotations

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from core.translator.openai_translator import OpenAITranslator


class DeepSeekTranslator(OpenAITranslator):
    """Dịch bằng DeepSeek (mặc định deepseek-flash)."""

    provider = "DeepSeek"
    # deepseek-flash bật thinking mặc định: tốn thêm token suy nghĩ (tính phí
    # như output), chậm hơn và bỏ qua temperature — dịch phụ đề không cần.
    extra_body = {"thinking": {"type": "disabled"}}

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        model: str = DEEPSEEK_MODEL,
        source_language: str = "en",
        target_language: str = "vi",
        base_url: str = DEEPSEEK_BASE_URL,
    ):
        super().__init__(
            api_key=api_key, model=model,
            source_language=source_language, target_language=target_language,
            base_url=base_url,
        )
