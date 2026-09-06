"""Phần lõi, không cần Flask: fallback dịch, timeout lệnh ngoài, retry API, thứ tự TTS."""
from __future__ import annotations

import sys
import time

import pytest

from core.pipeline import DubbingConfig, DubbingPipeline
from core.transcriber import Segment
from core.translator.llm_common import call_with_retry, is_retryable, parse_numbered_lines
from utils.proc import run_command


# ── Fallback khi API dịch hỏng ───────────────────────────────
class Boom(Exception):
    pass


class FakeTranslator:
    def __init__(self, engine, calls):
        self.engine = engine
        self.calls = calls

    def translate_segments(self, segments):
        self.calls.append(self.engine)
        if self.engine != "marian":
            raise Boom("404 model không còn khả dụng")
        for segment in segments:
            segment.translated = "Học tăng cường rất hữu ích."
        return segments


@pytest.fixture()
def fake_translators(monkeypatch):
    calls: list[str] = []
    import core.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_translator",
                        lambda engine, **kwargs: FakeTranslator(engine, calls))
    return calls


def segments():
    return [Segment(start=0, end=2, text="Reinforcement learning is useful.")]


def test_api_failure_falls_back_to_marian(fake_translators):
    config = DubbingConfig(translator_engine="gemini", gemini_api_key="x")
    out, used, fell_from = DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)

    assert (used, fell_from) == ("marian", "gemini")
    assert fake_translators == ["gemini", "marian"]
    assert out[0].translated.startswith("Học tăng cường")


def test_non_vietnamese_target_never_falls_back_to_marian(fake_translators):
    """MarianMT là model fine-tune riêng EN→VI — dùng cho Nhật/Trung/Hàn sẽ
    âm thầm ra tiếng Việt trong khi job báo ngôn ngữ khác. Không có gì để
    lùi về thì phải để lỗi thật hiện ra, không được lùi sai ngôn ngữ."""
    config = DubbingConfig(translator_engine="gemini", gemini_api_key="x", target_language="ja")
    with pytest.raises(Boom):
        DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert fake_translators == ["gemini"]  # chưa từng thử marian


def test_vietnamese_target_still_falls_back_to_marian(fake_translators):
    """target_language mặc định "vi" — hành vi cũ (giai đoạn trước khi có đa
    ngôn ngữ) phải giữ nguyên."""
    config = DubbingConfig(translator_engine="gemini", gemini_api_key="x", target_language="vi")
    out, used, fell_from = DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert (used, fell_from) == ("marian", "gemini")


def test_marian_runs_without_fallback(fake_translators):
    config = DubbingConfig(translator_engine="marian")
    _, used, fell_from = DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert (used, fell_from) == ("marian", "")
    assert fake_translators == ["marian"]


def test_fallback_can_be_disabled(fake_translators):
    config = DubbingConfig(translator_engine="gemini", translator_fallback=False)
    with pytest.raises(Boom):
        DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert fake_translators == ["gemini"]


# ── Timeout cho lệnh ngoài ───────────────────────────────────
def test_command_is_killed_when_it_hangs():
    """Không có timeout thì một tiến trình ffmpeg treo giữ job ở 'đang xử lý' vĩnh viễn."""
    started = time.time()
    with pytest.raises(RuntimeError, match="quá 2 giây"):
        run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                    timeout=2, error_message="Lệnh test")
    assert time.time() - started < 6


def test_missing_binary_reported_clearly():
    with pytest.raises(RuntimeError, match="Không tìm thấy"):
        run_command(["lenh_khong_ton_tai_abc"], timeout=5, error_message="Không tìm thấy lệnh")


def test_failed_command_includes_stderr():
    with pytest.raises(RuntimeError, match="loi that"):
        run_command([sys.executable, "-c", "import sys; sys.stderr.write('loi that'); sys.exit(3)"],
                    timeout=10, error_message="Lệnh thất bại")


def test_successful_command_returns_output():
    assert run_command([sys.executable, "-c", "print('xin chao')"],
                       timeout=10, error_message="x").stdout.strip() == "xin chao"


# ── Thử lại khi API trả lỗi tạm thời ─────────────────────────
class RateLimitError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


class ServerError(Exception):
    status_code = 503


@pytest.mark.parametrize("error,retryable", [
    (ServerError("boom"), True),
    (RateLimitError("slow down"), True),
    (Exception("Rate limit exceeded"), True),
    (BadRequestError("bad key"), False),
    (ValueError("API key không hợp lệ"), False),
])
def test_retryable_classification(error, retryable):
    assert is_retryable(error) is retryable


def test_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ServerError("service unavailable")
        return "xong"

    assert call_with_retry(flaky, what="Test", base_delay=0.05, attempts=4) == "xong"
    assert attempts["n"] == 3


def test_non_retryable_raises_immediately():
    attempts = {"n": 0}

    def always_bad():
        attempts["n"] += 1
        raise BadRequestError("sai key")

    with pytest.raises(BadRequestError):
        call_with_retry(always_bad, what="Test", base_delay=0.05, attempts=4)
    assert attempts["n"] == 1


def test_gives_up_and_reraises_original():
    attempts = {"n": 0}

    def always_flaky():
        attempts["n"] += 1
        raise ServerError("still down")

    with pytest.raises(ServerError):
        call_with_retry(always_flaky, what="Test", base_delay=0.02, attempts=3)
    assert attempts["n"] == 3


# ── Ghép lại kết quả dịch theo lô ────────────────────────────
def test_parse_numbered_lines_handles_both_separators():
    assert parse_numbered_lines("1. Xin chào\n2) Thế giới") == {1: "Xin chào", 2: "Thế giới"}


def test_parse_numbered_lines_ignores_noise():
    assert parse_numbered_lines("Đây là bản dịch:\n1. Một\n\n2. Hai") == {1: "Một", 2: "Hai"}


# ── Ngôn ngữ nguồn: "auto" ↔ Whisper language=None ────────────
def test_auto_source_language_becomes_none_for_whisper():
    """"auto" là giá trị chọn qua UI — Whisper không hiểu chuỗi này, nó cần
    language=None để tự nhận dạng."""
    assert DubbingPipeline._whisper_language_arg("auto") is None


def test_manual_source_language_passed_through_unchanged():
    assert DubbingPipeline._whisper_language_arg("ja") == "ja"
