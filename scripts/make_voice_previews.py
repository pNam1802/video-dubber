"""Tạo lại file nghe thử giọng ở static/voices/ (xem core/voice_previews.py).

    python scripts/make_voice_previews.py

Chạy trên Modal (Gemini cần key trong secret, VieNeu cần GPU) rồi ghi file về
máy này. Đổi PREVIEW_TEXT hoặc thêm giọng vào TTS_VOICES thì chạy lại.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.voice_previews import write_all  # noqa: E402
from modal_app import app, voice_previews  # noqa: E402

if __name__ == "__main__":
    with app.run():
        write_all(voice_previews.remote())
