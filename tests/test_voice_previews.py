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
    assert 'id="voiceGrid"' in html  # mỗi thẻ giọng có nút nghe thử riêng
    assert "/static/voices/vieneu/female.mp3" in html
    assert "/static/voices/edge-tts/male.mp3" in html


def test_preview_urls_carry_content_version(as_user):
    """File tĩnh cache 1 năm: link phải đổi khi nội dung file nghe thử đổi."""
    import re
    import zlib

    html = as_user.get("/").get_data(as_text=True)
    match = re.search(r"/static/voices/chatterbox/narrator\.mp3\?v=([0-9a-f]{8})", html)
    assert match
    expected = format(zlib.crc32((PREVIEW_DIR / "chatterbox" / "narrator.mp3").read_bytes()), "08x")
    assert match.group(1) == expected


def test_preview_file_is_served(client):
    response = client.get("/static/voices/vieneu/female.mp3")
    assert response.status_code == 200
    assert response.mimetype == "audio/mpeg"
