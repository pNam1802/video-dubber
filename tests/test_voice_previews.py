"""Nghe thử giọng ở form tạo job (core/voice_previews.py)."""
from __future__ import annotations

from core.voice_previews import PREVIEW_DIR, preview_name, preview_voices


def test_every_voice_has_a_preview_file():
    # Thêm giọng vào TTS_VOICES mà quên chạy scripts/make_voice_previews.py
    # thì giọng đó mất nút nghe thử — bắt ở đây.
    missing = [preview_name(e, k) for e, k in preview_voices() if not (PREVIEW_DIR / preview_name(e, k)).is_file()]
    assert missing == []


def test_clone_voice_has_no_static_preview():
    assert ("vieneu", "mine") not in preview_voices()


def test_index_links_previews(as_user):
    html = as_user.get("/").get_data(as_text=True)
    assert 'id="voicePreviewBtn"' in html
    assert "/static/voices/vieneu/female.mp3" in html
    assert "/static/voices/edge-tts/male.mp3" in html


def test_preview_file_is_served(client):
    response = client.get("/static/voices/vieneu/female.mp3")
    assert response.status_code == 200
    assert response.mimetype == "audio/mpeg"
