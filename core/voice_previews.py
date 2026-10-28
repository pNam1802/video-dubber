"""
core/voice_previews.py
Câu nghe thử cho từng giọng ở form tạo job: mọi giọng đọc CÙNG một câu để
người dùng so sánh. File tạo sẵn một lần (scripts/make_voice_previews.py) và
phục vụ tĩnh — bấm là nghe ngay, không chờ GPU nạp model hay tốn API.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from config.settings import ROOT_DIR, TTS_VOICES

PREVIEW_TEXT = "Xin chào, mình là Nora. Hôm nay là ngày 27 tháng 10 — bạn đã sẵn sàng xem video chưa?"
# Engine chỉ đọc tiếng Anh nghe thử bằng câu tiếng Anh tương đương.
PREVIEW_TEXT_EN = "Hi, I'm Nora. Today is October 27th — are you ready to watch the video?"
ENGLISH_ONLY = {"chatterbox"}
PREVIEW_DIR = ROOT_DIR / "static" / "voices"
# Giọng clone của từng người dùng: nghe thử bằng chính file mẫu họ tải lên.
SKIP = {("vieneu", "mine"), ("chatterbox", "mine")}


def preview_voices() -> list[tuple[str, str]]:
    return [(engine, key) for engine, voices in TTS_VOICES.items() for key in voices if (engine, key) not in SKIP]


def preview_name(engine: str, key: str) -> str:
    return f"{engine}/{key}.mp3"


def render_all(engines: list[str] | None = None) -> dict[str, bytes]:
    """Đọc PREVIEW_TEXT bằng từng giọng (chỉ các engine trong engines nếu có —
    thêm giọng mới không cần đọc lại, và trả tiền Gemini lại, mọi giọng cũ).
    KHÔNG rơi về giọng dự phòng như khi lồng tiếng: mẫu của Gemini mà lại là
    giọng edge-tts thì sai hẳn mục đích."""
    from core.tts_engine import TTSEngine

    out: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for engine, key in preview_voices():
            if engines and engine not in engines:
                continue
            english = engine in ENGLISH_ONLY
            tts = TTSEngine(engine=engine, voice=key, tts_dir=tmp, concurrency=1, language="en" if english else "vi")
            path = Path(tmp) / f"{engine}_{key}.mp3"
            tts._synthesize_checked(PREVIEW_TEXT_EN if english else PREVIEW_TEXT, path)
            out[preview_name(engine, key)] = path.read_bytes()
            print(f"[Preview] {engine}/{key}: {tts._duration(path):.1f}s")
    return out


def write_all(files: dict[str, bytes]) -> None:
    for name, data in files.items():
        target = PREVIEW_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"Đã ghi {target} ({len(data) // 1024} KB)")
