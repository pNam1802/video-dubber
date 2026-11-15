"""Đọc hiểu cả transcript trước khi dịch + kiểu dịch "bản địa hoá"."""
from __future__ import annotations

from core.transcriber import Segment
from core.translator.llm_common import build_system_prompt, build_understanding_prompt
from core.translator.openai_translator import OpenAITranslator
from tests.test_jobs import _awaiting_review_job
from app.extensions import db
from app.models import Job


class _Recorder:
    """Client OpenAI giả: ghi lại mọi request, trả lần lượt các phản hồi."""

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        message = type("M", (), {"content": content})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice], "usage": None})()


def _segments():
    return [Segment(0, 1, "đòi Skibidi"), Segment(1, 2, "nờ tờ giờ lại hai người kia")]


def test_translate_segments_reads_whole_transcript_first_then_sends_notes():
    notes = "TÓM TẮT: trả đũa người yêu cũ.\nBẢNG TỪ:\nđòi Skibidi = rủ quan hệ = ask for a hookup"
    translator = OpenAITranslator(api_key="k", source_language="vi", target_language="en")
    translator.client = _Recorder([notes, "1. asked for a hookup\n2. NTR them right back"])

    result = translator.translate_segments(_segments())

    assert [s.translated for s in result] == ["asked for a hookup", "NTR them right back"]
    understand_call, batch_call = translator.client.calls
    # Lượt 1 thấy TOÀN BỘ transcript, không có system prompt dịch.
    assert "đòi Skibidi" in understand_call["messages"][0]["content"]
    assert "nờ tờ giờ" in understand_call["messages"][0]["content"]
    assert understand_call["messages"][0]["role"] == "user"
    # Lượt dịch mang theo bảng từ trong system prompt.
    assert "ask for a hookup" in batch_call["messages"][0]["content"]
    assert translator.notes == notes


def test_understanding_failure_still_translates():
    translator = OpenAITranslator(api_key="k", source_language="vi", target_language="en")
    translator.client = _Recorder(["1. a\n2. b"])

    def boom(texts):
        raise RuntimeError("API sập")

    translator.understand = boom
    result = translator.translate_segments(_segments())
    assert [s.translated for s in result] == ["a", "b"]
    assert translator.notes == ""


def test_preset_notes_skip_understanding():
    """Dịch lại ở bước duyệt: notes đọc từ DB, không đọc hiểu lại."""
    translator = OpenAITranslator(api_key="k", source_language="vi", target_language="en")
    translator.notes = "BẢNG TỪ: x = y"
    translator.client = _Recorder(["1. a\n2. b"])
    translator.translate_segments(_segments())
    assert len(translator.client.calls) == 1


def test_deepseek_keeps_thinking_for_understanding_only():
    from core.translator.deepseek_translator import DeepSeekTranslator

    translator = DeepSeekTranslator(api_key="k", source_language="vi", target_language="en")
    translator.client = _Recorder(["BẢNG TỪ: x = y", "1. a\n2. b"])
    translator.translate_segments(_segments())
    understand_call, batch_call = translator.client.calls
    assert understand_call["extra_body"] is None
    assert batch_call["extra_body"] == {"thinking": {"type": "disabled"}}


def test_localized_style_changes_prompts():
    natural = build_system_prompt("vi", "en", "natural")
    localized = build_system_prompt("vi", "en", "localized")
    assert "BẢN ĐỊA HOÁ" not in natural
    assert "BẢN ĐỊA HOÁ" in localized and "tiếng Anh" in localized
    assert "meme TƯƠNG ĐƯƠNG" in build_understanding_prompt(["x"], "vi", "en", "localized")
    assert "meme TƯƠNG ĐƯƠNG" not in build_understanding_prompt(["x"], "vi", "en", "natural")


def test_upload_saves_translation_style(app, as_user, monkeypatch):
    from tests.test_jobs import _upload_and_capture_config

    response, config = _upload_and_capture_config(as_user, monkeypatch, translation_style="localized")
    assert response.status_code == 202
    assert config.translation_style == "localized"
    with app.app_context():
        assert db.session.get(Job, response.get_json()["job_id"]).translation_style == "localized"


def test_upload_rejects_unknown_style_to_natural(app, as_user, monkeypatch):
    from tests.test_jobs import _upload_and_capture_config

    _, config = _upload_and_capture_config(as_user, monkeypatch, translation_style="hack")
    assert config.translation_style == "natural"


def test_retranslate_reuses_saved_notes_and_style(app, as_user, user, monkeypatch, tmp_path):
    import app.api as api_mod

    captured = {}

    class Fake:
        usage_prompt_tokens = 0
        usage_completion_tokens = 0

        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def translate_text(self, text):
            captured["notes"] = self.notes
            return "ok"

    monkeypatch.setattr(api_mod, "get_translator", lambda engine, **kwargs: Fake(**kwargs))
    job_id, seg_id, _ = _awaiting_review_job(
        app, user, tmp_path, translation_style="localized", translation_notes="BẢNG TỪ: a = b",
    )
    response = as_user.post(f"/api/jobs/{job_id}/segments/{seg_id}/retranslate", json={"gemini_api_key": "k"})
    assert response.status_code == 200
    assert captured["kwargs"]["style"] == "localized"
    assert captured["notes"] == "BẢNG TỪ: a = b"


def test_index_has_translation_style_select(as_user):
    html = as_user.get("/").get_data(as_text=True)
    assert 'name="translation_style"' in html
    assert '<option value="natural" selected>' in html
