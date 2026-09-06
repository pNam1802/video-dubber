"""Phần lõi, không cần Flask: fallback dịch, timeout lệnh ngoài, retry API, thứ tự TTS."""
from __future__ import annotations

import sys
import time
from pathlib import Path

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
        if self.engine != "openai":
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


def test_api_failure_falls_back_to_other_engine(fake_translators):
    config = DubbingConfig(translator_engine="gemini", gemini_api_key="x", openai_api_key="y")
    out, used, fell_from = DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)

    assert (used, fell_from) == ("openai", "gemini")
    assert fake_translators == ["gemini", "openai"]
    assert out[0].translated.startswith("Học tăng cường")


def test_fallback_works_for_non_vietnamese_target_too(fake_translators):
    """MarianMT (fine-tune riêng EN→VI) từng không dùng được cho Nhật/Trung/
    Hàn, nên trước đây phải chặn fallback cho các đích đó. Giờ dự phòng là
    nhà cung cấp còn lại (Gemini ⇄ OpenAI), vốn dịch được mọi ngôn ngữ đích,
    nên không còn lý do gì để đối xử khác với target_language != "vi"."""
    config = DubbingConfig(
        translator_engine="gemini", gemini_api_key="x", openai_api_key="y", target_language="ja"
    )
    out, used, fell_from = DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert (used, fell_from) == ("openai", "gemini")


def test_no_fallback_key_lets_error_surface(fake_translators):
    """Không có key cho nhà cung cấp còn lại thì không có gì để lùi về —
    lỗi thật phải hiện ra thay vì thử một engine chắc chắn cũng hỏng."""
    config = DubbingConfig(translator_engine="gemini", gemini_api_key="x")
    with pytest.raises(Boom):
        DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert fake_translators == ["gemini"]


def test_openai_runs_without_fallback(fake_translators):
    config = DubbingConfig(translator_engine="openai", openai_api_key="y")
    _, used, fell_from = DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert (used, fell_from) == ("openai", "")
    assert fake_translators == ["openai"]


def test_fallback_can_be_disabled(fake_translators):
    config = DubbingConfig(translator_engine="gemini", translator_fallback=False)
    with pytest.raises(Boom):
        DubbingPipeline(config)._translate(segments(), config, lambda p, m: None)
    assert fake_translators == ["gemini"]


# ── Tách pipeline: duyệt transcript giữa dịch và TTS ──────────
class FakeExtractor:
    def extract(self, video_path, output_path=None):
        Path(output_path).write_bytes(b"fake-audio")
        return Path(output_path)

    def get_duration(self, media_path):
        return 12.0


class FakeTranscriber:
    def __init__(self, model_size="base", device="auto"):
        pass

    def transcribe(self, audio_path, language=None, sentence_resegment=True, silence_threshold=0.45):
        return [Segment(start=0, end=1, text="Hello.")], "en"


class FakeTTS:
    def __init__(self, engine="edge-tts", voice="female", tts_dir=None):
        self.tts_dir = Path(tts_dir)

    def synthesize_all(self, segs):
        self.tts_dir.mkdir(parents=True, exist_ok=True)
        out = self.tts_dir / "seg_00000.mp3"
        out.write_bytes(b"fake-audio")
        return [(segs[0], out)]


class FakeComposer:
    def compose(self, video_path, segments, tts_paths, original_volume=0.1, tts_volume=1.6, video_duration=None):
        out = Path(video_path).parent / "out_dubbed.mp4"
        out.write_bytes(b"fake-video")
        return out


@pytest.fixture()
def fake_collaborators(monkeypatch):
    """Giả toàn bộ collaborator nặng (ffmpeg, Whisper, TTS) để test chỉ đo
    hành vi ĐIỀU PHỐI của pipeline: dừng đúng chỗ, không dịch lại, gộp đúng
    timings — không phải chạy pipeline thật."""
    import core.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "AudioExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_mod, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline_mod, "TTSEngine", FakeTTS)
    monkeypatch.setattr(pipeline_mod, "VideoComposer", FakeComposer)
    calls: list[str] = []
    monkeypatch.setattr(pipeline_mod, "get_translator",
                        lambda engine, **kwargs: FakeTranslator(engine, calls))
    return calls


def test_transcribe_and_translate_never_touches_tts_or_compose(fake_collaborators, tmp_path, monkeypatch):
    """Nửa đầu pipeline phải DỪNG LẠI sau khi dịch — không được tự chạy tiếp
    TTS/ghép video, đó là lý do người dùng có cơ hội duyệt/sửa ở giữa."""
    import core.pipeline as pipeline_mod

    def boom_tts(*a, **k):
        raise AssertionError("Không được gọi TTS ở giai đoạn 1")

    monkeypatch.setattr(pipeline_mod, "TTSEngine", boom_tts)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    config = DubbingConfig(translator_engine="openai", openai_api_key="x")

    result = DubbingPipeline(config).transcribe_and_translate(video_path)

    assert result.success
    assert result.output_video is None
    assert result.segments[0].translated.startswith("Học tăng cường")
    assert set(result.timings) == {"extract", "transcribe", "translate"}


def test_synthesize_and_compose_does_not_retranslate(fake_collaborators, tmp_path):
    """Nửa sau pipeline phải dùng ĐÚNG segments truyền vào (có thể đã được
    người dùng sửa tay) — không được gọi translator lần nữa."""
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    config = DubbingConfig(translator_engine="openai", openai_api_key="x")

    edited_segments = [Segment(start=0, end=1, text="Hello.", translated="Bản đã người dùng sửa tay.")]
    result = DubbingPipeline(config).synthesize_and_compose(video_path, edited_segments)

    assert result.success
    assert result.output_video is not None
    assert result.segments[0].translated == "Bản đã người dùng sửa tay."
    assert fake_collaborators == []  # get_translator() chưa từng được gọi


def test_run_combines_both_phases(fake_collaborators, tmp_path):
    """run() (dùng cho CLI, không có bước duyệt) phải cho kết quả tương
    đương chạy 2 nửa nối tiếp: gộp timings, cộng dồn elapsed_seconds."""
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    config = DubbingConfig(translator_engine="openai", openai_api_key="x")

    result = DubbingPipeline(config).run(video_path)

    assert result.success
    assert result.output_video is not None
    assert result.source_language_detected == "en"
    assert set(result.timings) == {"extract", "transcribe", "translate", "tts", "compose"}


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
