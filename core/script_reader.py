"""
core/script_reader.py
"Đọc kịch bản": tách văn bản người dùng dán vào thành câu cho TTS, rồi nối
các câu đã đọc thành một file mp3 liền mạch kèm phụ đề SRT.

Không có video nên không có dòng thời gian để khớp: câu nối tiếp nhau với
khoảng nghỉ cố định, dài hơn ở chỗ xuống đoạn.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from core.transcriber import Segment

#: Câu dài hơn thì chia nhỏ ở dấu phẩy/chấm phẩy — model TTS đọc câu quá dài
#: hay bị hụt hơi, đọc sót hoặc lặp từ.
MAX_SENTENCE_CHARS = 250
SENTENCE_PAUSE_SEC = 0.35
PARAGRAPH_PAUSE_SEC = 0.9

# Khoảng trắng sau dấu kết câu, kể cả khi có ngoặc/nháy đóng ngay sau dấu.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?…。])|(?<=[.!?…。][\"'”’)\]]))\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _split_long(sentence: str) -> List[str]:
    if len(sentence) <= MAX_SENTENCE_CHARS:
        return [sentence]
    pieces: List[str] = []
    current = ""
    for clause in _CLAUSE_END.split(sentence):
        candidate = f"{current} {clause}".strip()
        if current and len(candidate) > MAX_SENTENCE_CHARS:
            pieces.append(current)
            current = clause
        else:
            current = candidate
    if current:
        pieces.append(current)
    # Một vế vẫn quá dài (không có dấu câu nào): cắt ở khoảng trắng.
    out: List[str] = []
    for piece in pieces:
        while len(piece) > MAX_SENTENCE_CHARS:
            cut = piece.rfind(" ", 0, MAX_SENTENCE_CHARS)
            cut = cut if cut > 0 else MAX_SENTENCE_CHARS
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


def split_script(text: str) -> List[tuple[int, str]]:
    """Trả về (số thứ tự đoạn văn, câu) theo đúng thứ tự trong kịch bản.
    Mỗi dòng không trống là một đoạn — người viết kịch bản xuống dòng ở đâu
    thì nghỉ dài ở đó."""
    out: List[tuple[int, str]] = []
    paragraphs = [" ".join(line.split()) for line in (text or "").splitlines()]
    for para_idx, paragraph in enumerate(p for p in paragraphs if p):
        for sentence in _SENTENCE_END.split(paragraph):
            for piece in _split_long(sentence.strip()):
                if piece:
                    out.append((para_idx, piece))
    return out


def assemble(pairs: List[tuple[Segment, Path]], output_path: Path) -> List[tuple[float, float]]:
    """Nối các câu đã đọc thành một mp3. `Segment.start` giữ số thứ tự đoạn
    văn (xem app/api.py::create_script_job) để biết chỗ nào nghỉ dài.
    Trả về (bắt đầu, kết thúc) của từng câu trong file kết quả."""
    from pydub import AudioSegment

    audio = AudioSegment.silent(duration=0)
    timings: List[tuple[float, float]] = []
    previous_paragraph = None
    for segment, path in pairs:
        if previous_paragraph is not None:
            pause = PARAGRAPH_PAUSE_SEC if segment.start != previous_paragraph else SENTENCE_PAUSE_SEC
            audio += AudioSegment.silent(duration=int(pause * 1000))
        previous_paragraph = segment.start
        clip = AudioSegment.from_file(str(path))
        start = len(audio) / 1000.0
        audio += clip
        timings.append((start, len(audio) / 1000.0))
    audio.export(str(output_path), format="mp3", bitrate="128k")
    return timings
