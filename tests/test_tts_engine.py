"""TTSEngine: đổi tốc độ bằng rate của edge-tts, và chặn âm thanh rỗng/cụt."""
from __future__ import annotations

import pytest

from core.tts_engine import TTSEngine


def test_edge_rate_rounds_toward_shorter_audio():
    assert TTSEngine.edge_rate(1.0) == "+0%"
    assert TTSEngine.edge_rate(1.121) == "+13%"   # nhanh: tròn LÊN
    assert TTSEngine.edge_rate(0.917) == "-8%"    # chậm: bớt chậm (8.3% -> 8%)
    assert TTSEngine.edge_rate(0.9) == "-10%"


def test_check_audio_rejects_near_empty_clip(monkeypatch, tmp_path):
    """Bug thật (job #33): edge-tts trả về file có dung lượng nhưng gần như
    không có tiếng — câu #13 biến mất, video im lặng 7.6s."""
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda path: 0.2))
    with pytest.raises(RuntimeError, match="bất thường"):
        TTSEngine._check_audio(tmp_path / "x.mp3", "Vẫn còn kinh hãi, cô tuyệt vọng đề nghị chờ thêm vài ngày")


def test_check_audio_accepts_normal_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda path: 3.0))
    TTSEngine._check_audio(tmp_path / "x.mp3", "Vẫn còn kinh hãi, cô tuyệt vọng đề nghị chờ thêm vài ngày")


def test_synthesize_checked_retries_bad_audio(monkeypatch, tmp_path):
    engine = TTSEngine(tts_dir=tmp_path)
    calls: list[float] = []
    durations = iter([0.1, 3.0])
    monkeypatch.setattr(engine, "synthesize_text", lambda text, out, speed=1.0: calls.append(speed))
    monkeypatch.setattr(TTSEngine, "_trim_silence", staticmethod(lambda path: None))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda path: next(durations)))
    engine._synthesize_checked("Một câu đủ dài để kiểm tra", tmp_path / "x.mp3")
    assert len(calls) == 2


# ── Danh mục giọng + Gemini ───────────────────────────────────
def test_voice_names_from_catalog(tmp_path):
    assert TTSEngine(engine="gemini", voice="male", tts_dir=tmp_path).voice_name == "Charon"
    assert TTSEngine(engine="gemini", voice="female", tts_dir=tmp_path).voice_name == "Leda"
    edge = TTSEngine(engine="edge-tts", voice="florian", tts_dir=tmp_path)
    assert edge.edge_voice == "de-DE-FlorianMultilingualNeural"


def test_unknown_voice_falls_back_to_engine_default(tmp_path):
    assert TTSEngine(engine="edge-tts", voice="khong-co", tts_dir=tmp_path).edge_voice == "vi-VN-HoaiMyNeural"


def test_gemini_engine_falls_back_to_hoaimy_on_edge(tmp_path):
    assert TTSEngine(engine="gemini", voice="male", tts_dir=tmp_path).edge_voice == "vi-VN-HoaiMyNeural"


def test_gemini_speed_change_uses_atempo_not_another_api_call(monkeypatch, tmp_path):
    """Đổi tốc độ giọng Gemini không được gọi API lần nữa — tốn tiền gấp đôi
    và Gemini đọc lại mỗi lần một kiểu."""
    from core.transcriber import Segment

    engine = TTSEngine(engine="gemini", voice="female", tts_dir=tmp_path)
    path = tmp_path / "seg_00000.mp3"
    path.write_bytes(b"audio")
    tempos: list[float] = []
    monkeypatch.setattr(engine, "synthesize_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("API gọi lại")))
    monkeypatch.setattr(TTSEngine, "_atempo", staticmethod(lambda p, speed: tempos.append(speed)))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 3.0))
    duration = engine._resynthesize((Segment(0, 3, "Hello.", "Xin chào các bạn."), path), 0.92)
    assert tempos == [0.92]
    assert duration == 3.0
