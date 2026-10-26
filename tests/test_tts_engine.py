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
