from __future__ import annotations

def get_translator(engine: str, **kwargs):
    engine_key = (engine or "").strip().lower()
    if engine_key == "openai":
        from core.translator.openai_translator import OpenAITranslator

        return OpenAITranslator(**kwargs)
    if engine_key == "deepseek":
        from core.translator.deepseek_translator import DeepSeekTranslator

        return DeepSeekTranslator(**kwargs)
    if engine_key == "gemini":
        from core.translator.gemini_translator import GeminiTranslator

        return GeminiTranslator(**kwargs)
    if engine_key == "marian":
        from core.translator.marian_translator import MarianTranslator

        return MarianTranslator(**kwargs)
    raise ValueError(f"Unsupported translator engine: {engine}")


__all__ = ["get_translator"]
