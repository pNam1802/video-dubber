"""VieNeu-TTS: giọng mặc định Nora, giọng có sẵn, giọng clone của người dùng."""
from __future__ import annotations

import io
import subprocess

import pytest

from app.extensions import db
from app.models import User, utcnow
from core.transcriber import Segment
from core.tts_engine import TTSEngine
from tests.test_jobs import _awaiting_review_job, _upload_and_capture_config


def _tone(seconds: float) -> bytes:
    """WAV thật (ffmpeg tạo) — bước chuẩn hoá giọng mẫu gọi ffmpeg/ffprobe thật."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-f", "wav", "-"],
        check=True, capture_output=True,
    ).stdout


# ── TTSEngine ────────────────────────────────────────────────
def test_vieneu_default_voice_is_nora_clone(tmp_path):
    engine = TTSEngine(engine="vieneu", voice="female", tts_dir=tmp_path)
    voice, ref = engine._vieneu_args()
    assert voice is None and ref  # clone từ core/voices/nora.wav
    assert TTSEngine(engine="vieneu", voice="ngochuyen", tts_dir=tmp_path)._vieneu_args() == ("Ngọc Huyền", None)


def test_vieneu_mine_reads_user_sample(tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFFme")
    engine = TTSEngine(engine="vieneu", voice="mine", tts_dir=tmp_path, ref_audio=sample)
    assert engine._vieneu_args() == (None, b"RIFFme")


def test_vieneu_reads_whole_video_in_one_call(monkeypatch, tmp_path):
    """GPU chỉ nhanh khi đọc theo lô — phải gửi mọi câu trong MỘT lần gọi."""
    from core import vieneu_client

    calls: list[list[str]] = []

    def fake_synthesize(texts, voice=None, ref_wav=None):
        calls.append(list(texts))
        return [f"mp3:{t}".encode() for t in texts]

    monkeypatch.setattr(vieneu_client, "synthesize", fake_synthesize)
    monkeypatch.setattr(TTSEngine, "_trim_silence", staticmethod(lambda p: None))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 3.0))

    engine = TTSEngine(engine="vieneu", voice="ngochuyen", tts_dir=tmp_path, concurrency=4)
    segments = [Segment(i, i + 1, "x", f"Câu số {i} đủ dài.") for i in range(5)]
    results = engine.synthesize_all(segments)
    assert len(calls) == 1 and len(calls[0]) == 5
    assert [p.read_bytes() for _, p in results] == [f"mp3:Câu số {i} đủ dài.".encode() for i in range(5)]


def test_vieneu_failure_falls_back_to_edge(monkeypatch, tmp_path):
    from core import vieneu_client

    def boom(*a, **k):
        raise RuntimeError("GPU hết chỗ")

    edge_calls: list[str] = []
    monkeypatch.setattr(vieneu_client, "synthesize", boom)
    monkeypatch.setattr(TTSEngine, "_synthesize_edge",
                        lambda self, text, out, speed=1.0: (edge_calls.append(self.edge_voice), out.write_bytes(b"e")))
    monkeypatch.setattr(TTSEngine, "_trim_silence", staticmethod(lambda p: None))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 3.0))

    engine = TTSEngine(engine="vieneu", voice="female", tts_dir=tmp_path)
    results = engine.synthesize_all([Segment(0, 1, "x", "Một câu đủ dài để đọc.")])
    assert len(results) == 1
    assert edge_calls == ["vi-VN-HoaiMyNeural"]


def test_vieneu_speed_change_uses_atempo(monkeypatch, tmp_path):
    engine = TTSEngine(engine="vieneu", voice="female", tts_dir=tmp_path)
    path = tmp_path / "seg_00000.mp3"
    path.write_bytes(b"audio")
    tempos: list[float] = []
    monkeypatch.setattr(engine, "synthesize_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gọi lại")))
    monkeypatch.setattr(TTSEngine, "_atempo", staticmethod(lambda p, speed: tempos.append(speed)))
    monkeypatch.setattr(TTSEngine, "_duration", staticmethod(lambda p: 3.0))
    engine._resynthesize((Segment(0, 3, "Hi.", "Xin chào các bạn."), path), 1.1)
    assert tempos == [1.1]


# ── Giọng mẫu ở trang cài đặt ────────────────────────────────
def _post_sample(client, seconds: float = 5.0, consent: bool = True, name: str = "me.wav"):
    data = {"voice_sample": (io.BytesIO(_tone(seconds)), name)}
    if consent:
        data["voice_consent"] = "on"
    return client.post("/cai-dat/giong", data=data, content_type="multipart/form-data", follow_redirects=True)


def test_voice_sample_requires_consent(app, as_user, user):
    body = _post_sample(as_user, consent=False).get_data(as_text=True)
    assert "xác nhận" in body
    with app.app_context():
        assert db.session.get(User, user).voice_sample_at is None


def test_voice_sample_too_short_rejected(app, as_user, user):
    body = _post_sample(as_user, seconds=1.0).get_data(as_text=True)
    assert "quá ngắn" in body
    with app.app_context():
        assert db.session.get(User, user).voice_sample_at is None


def test_voice_sample_saved_trimmed_and_deletable(app, as_user, user):
    from app.voices import _duration

    assert "Đã lưu giọng mẫu" in _post_sample(as_user, seconds=20.0).get_data(as_text=True)
    with app.app_context():
        u = db.session.get(User, user)
        assert u.has_voice_sample
        path = u.voice_sample_path
    assert path.exists() and _duration(path) <= 15.05

    assert as_user.get("/cai-dat/giong.wav").status_code == 200
    # Giọng của bạn giờ hiện trong danh sách chọn giọng.
    assert '"mine"' in as_user.get("/").get_data(as_text=True)

    as_user.post("/cai-dat/giong/xoa")
    with app.app_context():
        assert db.session.get(User, user).voice_sample_at is None
    assert not path.exists()
    assert as_user.get("/cai-dat/giong.wav").status_code == 404


def test_voice_sample_rejects_non_audio_extension(app, as_user):
    assert "Định dạng không hỗ trợ" in _post_sample(as_user, name="me.exe").get_data(as_text=True)


def test_mine_voice_hidden_without_sample(as_user):
    assert '"mine"' not in as_user.get("/").get_data(as_text=True)


def test_other_user_cannot_hear_my_sample(app, as_user, as_other, user):
    _post_sample(as_user)
    assert as_other.get("/cai-dat/giong.wav").status_code == 404


# ── Upload / tiếp tục job ────────────────────────────────────
def test_upload_mine_without_sample_refused(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch, tts_engine="vieneu", tts_voice="mine")
    assert response.status_code == 400
    assert config is None


def test_upload_vieneu_default_voice(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch, tts_engine="vieneu", tts_voice="female")
    assert response.status_code == 202
    assert (config.tts_engine, config.tts_voice) == ("vieneu", "female")


def _set_sample(app, user_id):
    with app.app_context():
        u = db.session.get(User, user_id)
        u.voice_sample_at = utcnow()
        db.session.commit()
        return u.voice_sample_path


def test_continue_passes_user_sample_to_phase2(app, as_user, user, monkeypatch, tmp_path):
    import app.api as api_mod

    captured = {}
    monkeypatch.setattr(api_mod, "continue_job", lambda flask_app, job_id, config: captured.update(config=config))
    path = _set_sample(app, user)
    job_id, _, _ = _awaiting_review_job(app, user, tmp_path, tts_engine="vieneu", tts_voice="mine")
    assert as_user.post(f"/api/jobs/{job_id}/continue").status_code == 200
    assert captured["config"].tts_ref_audio == str(path)


def test_continue_refused_when_sample_deleted(app, as_user, user, tmp_path):
    job_id, _, _ = _awaiting_review_job(app, user, tmp_path, tts_engine="vieneu", tts_voice="mine")
    response = as_user.post(f"/api/jobs/{job_id}/continue")
    assert response.status_code == 409
    assert "Giọng mẫu" in response.get_json()["error"]


@pytest.mark.parametrize("voice", ["female", "ngochuyen", "male"])
def test_continue_preset_voices_need_no_sample(app, as_user, user, monkeypatch, tmp_path, voice):
    import app.api as api_mod

    captured = {}
    monkeypatch.setattr(api_mod, "continue_job", lambda flask_app, job_id, config: captured.update(config=config))
    job_id, _, _ = _awaiting_review_job(app, user, tmp_path, tts_engine="vieneu", tts_voice=voice)
    assert as_user.post(f"/api/jobs/{job_id}/continue").status_code == 200
    assert captured["config"].tts_ref_audio is None
