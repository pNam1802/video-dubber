"""Chatterbox-Turbo: giọng clone tiếng Anh (giọng có sẵn + giọng của người dùng)."""
from __future__ import annotations

import pytest

from core import chatterbox_client
from core.transcriber import Segment
from core.tts_engine import TTSEngine
from tests.test_jobs import _awaiting_review_job, _upload_and_capture_config
from tests.test_vieneu import _set_sample


# ── TTSEngine ────────────────────────────────────────────────
def test_chatterbox_builtin_voice_sends_no_sample(tmp_path):
    assert TTSEngine(engine="chatterbox", voice="female", tts_dir=tmp_path, language="en")._chatterbox_ref() is None


def test_chatterbox_mine_reads_user_sample(tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFFme")
    engine = TTSEngine(engine="chatterbox", voice="mine", tts_dir=tmp_path, ref_audio=sample, language="en")
    assert engine._chatterbox_ref() == b"RIFFme"


def test_chatterbox_reads_whole_video_in_one_call(monkeypatch, tmp_path):
    calls: list[tuple[list[str], bytes | None]] = []

    def fake_synthesize(texts, ref_wav=None):
        calls.append((list(texts), ref_wav))
        return [f"mp3:{t}".encode() for t in texts]

    monkeypatch.setattr(chatterbox_client, "synthesize", fake_synthesize)
    monkeypatch.setattr(TTSEngine, "_trim_silence", staticmethod(lambda p: None))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 3.0))

    engine = TTSEngine(engine="chatterbox", voice="female", tts_dir=tmp_path, concurrency=4, language="en")
    segments = [Segment(i, i + 1, "x", f"Sentence number {i} is long enough.") for i in range(5)]
    results = engine.synthesize_all(segments)
    assert len(calls) == 1 and len(calls[0][0]) == 5 and calls[0][1] is None
    assert [p.read_bytes() for _, p in results] == [
        f"mp3:Sentence number {i} is long enough.".encode() for i in range(5)
    ]


def test_chatterbox_failure_falls_back_to_english_edge_voice(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("GPU hết chỗ")

    edge_calls: list[str] = []
    monkeypatch.setattr(chatterbox_client, "synthesize", boom)
    monkeypatch.setattr(TTSEngine, "_synthesize_edge",
                        lambda self, text, out, speed=1.0: (edge_calls.append(self.edge_voice), out.write_bytes(b"e")))
    monkeypatch.setattr(TTSEngine, "_trim_silence", staticmethod(lambda p: None))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 3.0))

    engine = TTSEngine(engine="chatterbox", voice="female", tts_dir=tmp_path, language="en")
    results = engine.synthesize_all([Segment(0, 1, "x", "A sentence long enough to read.")])
    assert len(results) == 1
    assert edge_calls == ["en-US-AriaNeural"]


def test_chatterbox_never_slows_down(monkeypatch, tmp_path):
    """Như VieNeu: đổi tốc độ bằng atempo, giãn chậm nghe lê — chỉ tăng tốc."""
    engine = TTSEngine(engine="chatterbox", voice="female", tts_dir=tmp_path, language="en")
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 1.0))
    monkeypatch.setattr(engine, "_resynthesize", lambda pair, speed: (_ for _ in ()).throw(AssertionError(speed)))
    # Câu 1 giây trong khung 3 giây: engine khác sẽ đọc chậm lại cho lấp chỗ trống.
    pair = (Segment(0, 3, "Hi.", "Hello there."), tmp_path / "a.mp3")
    engine.place_on_timeline([pair])


# ── chatterbox_client (không cần model thật) ─────────────────
class _FakeTensor:
    """Đủ cho .squeeze(0).cpu().numpy() — không cần cài torch để chạy test."""

    def squeeze(self, dim):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.zeros(2400)


class _FakeModel:
    sr = 24000

    def __init__(self):
        self.builtin_conds = "builtin"
        self.conds = "builtin"
        self.used: list[str] = []

    def prepare_conditionals(self, path):
        self.conds = f"clone:{path}"

    def generate(self, text):
        self.used.append(self.conds)
        return _FakeTensor()


def test_builtin_voice_restored_after_a_clone(monkeypatch):
    """Model dùng chung giữa các job: job clone trước không được để lại giọng
    của người đó cho job dùng giọng có sẵn sau."""
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    monkeypatch.setattr(chatterbox_client, "_padded_ref", lambda ref, folder: folder / "ref.wav")
    monkeypatch.setattr(chatterbox_client.subprocess, "run",
                        lambda cmd, **k: __import__("pathlib").Path(cmd[-1]).write_bytes(b"mp3"))
    model = _FakeModel()
    chatterbox_client.synthesize_with(model, ["One."], ref_wav=b"RIFF")
    chatterbox_client.synthesize_with(model, ["Two."])
    assert model.used[0].startswith("clone:") and model.used[1] == "builtin"


def test_short_sample_is_padded_past_turbo_minimum(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    import io

    buf = io.BytesIO()
    sf.write(buf, np.zeros(24000 * 3), 24000, format="WAV")
    audio, sr = sf.read(str(chatterbox_client._padded_ref(buf.getvalue(), tmp_path)))
    assert len(audio) / sr > 5.0


# ── Upload / tiếp tục job ────────────────────────────────────
def test_upload_chatterbox_for_english(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(
        as_user, monkeypatch, target_language="en", tts_engine="chatterbox", tts_voice="female"
    )
    assert response.status_code == 202
    assert (config.tts_engine, config.tts_voice) == ("chatterbox", "female")


def test_upload_chatterbox_for_vietnamese_rejected(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch, target_language="vi", tts_engine="chatterbox")
    assert response.status_code == 400
    assert config is None


def test_continue_passes_user_sample_to_chatterbox(app, as_user, user, monkeypatch, tmp_path):
    import app.api as api_mod

    captured = {}
    monkeypatch.setattr(api_mod, "continue_job", lambda flask_app, job_id, config: captured.update(config=config))
    path = _set_sample(app, user)
    job_id, _, _ = _awaiting_review_job(app, user, tmp_path, tts_engine="chatterbox", tts_voice="mine",
                                        target_language="en")
    assert as_user.post(f"/api/jobs/{job_id}/continue").status_code == 200
    assert captured["config"].tts_ref_audio == str(path)
