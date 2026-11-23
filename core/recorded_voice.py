"""
core/recorded_voice.py
"Giọng tự thu" cho Đọc kịch bản (Xưởng kịch bản): người dùng tự đọc kịch bản trước mic,
mỗi ĐOẠN (một dòng của kịch bản) một bản thu, thay cho giọng AI.

Phụ đề và hình minh hoạ cần mốc thời gian từng CÂU, nên mỗi bản thu đoạn được cắt thành
từng câu tại các chỗ ngắt hơi (ffmpeg silencedetect): ranh giới câu thứ k là chỗ lặng gần
nhất với vị trí ước lượng theo tỉ lệ số ký tự. Cắt ở giữa chỗ lặng thì không bao giờ cắt
ngang chữ; không có chỗ lặng nào đủ gần (đọc liền một hơi) thì đành cắt theo tỉ lệ.
Sau đó các câu đi qua đúng đường ghép của giọng AI (core/script_reader.assemble).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

from utils.proc import run_command

#: Mức âm coi là lặng và độ dài lặng tối thiểu để tính là chỗ ngắt giữa hai câu.
SILENCE_DB = -35
SILENCE_SEC = 0.25
#: Chỗ lặng chỉ được nhận làm ranh giới câu nếu cách vị trí ước lượng không quá bấy nhiêu giây.
MAX_SHIFT_SEC = 2.5
#: Một bản thu đoạn dài tối đa bấy nhiêu giây.
MAX_PARAGRAPH_SEC = 180
SAMPLE_RATE = 24000


def text_hash(text: str) -> str:
    """Mã của nội dung đoạn: sửa chữ trong đoạn thì bản thu cũ hết khớp."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:8]


def recording_name(episode_id: int, lang: str, index: int, text: str) -> str:
    return f"rec_{episode_id}_{lang}_{index}_{text_hash(text)}.wav"


def _ffmpeg() -> str:
    from config.settings import FFMPEG_BIN

    return FFMPEG_BIN


def normalize_take(src: Path, dst: Path) -> float:
    """webm/ogg/m4a từ trình duyệt -> wav mono 24 kHz, cắt lặng hai đầu. Trả về độ dài (giây)."""
    trim = (f"silenceremove=start_periods=1:start_threshold={SILENCE_DB}dB:start_silence=0.1,"
            f"areverse,silenceremove=start_periods=1:start_threshold={SILENCE_DB}dB:start_silence=0.15,areverse")
    run_command([_ffmpeg(), "-v", "error", "-y", "-i", str(src), "-t", str(MAX_PARAGRAPH_SEC), "-af", trim,
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)],
                timeout=120, error_message="Không đọc được bản thu âm.")
    return duration(dst)


def duration(path: Path) -> float:
    import soundfile

    return float(soundfile.info(str(path)).duration)


def silences(path: Path) -> List[Tuple[float, float]]:
    """Các khoảng lặng (bắt đầu, kết thúc) trong bản thu."""
    result = run_command([_ffmpeg(), "-v", "info", "-hide_banner", "-i", str(path), "-af",
                          f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_SEC}", "-f", "null", "-"],
                         timeout=120, error_message="Không dò được chỗ ngắt hơi trong bản thu.")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    return list(zip(starts, ends))


def sentence_cuts(sentences: List[str], total: float, gaps: List[Tuple[float, float]]) -> List[float]:
    """Điểm cắt giữa các câu (len(sentences) - 1 điểm, tăng dần)."""
    chars = [max(1, len(s)) for s in sentences]
    cuts, used, acc, last = [], set(), 0, 0.0
    for k in range(len(sentences) - 1):
        acc += chars[k]
        guess = total * acc / sum(chars)
        best = None
        for i, (a, b) in enumerate(gaps):
            mid = (a + b) / 2
            if i in used or mid <= last + 0.2 or abs(mid - guess) > MAX_SHIFT_SEC:
                continue
            # Ưu tiên chỗ gần vị trí ước lượng, chỗ lặng dài (ngắt câu thật) được cộng điểm.
            score = abs(mid - guess) - min(b - a, 1.0)
            if best is None or score < best[0]:
                best = (score, i, mid)
        if best is not None:
            used.add(best[1])
            cut = best[2]
        else:
            cut = max(guess, last + 0.2)
        cuts.append(min(cut, total - 0.05))
        last = cuts[-1]
    return cuts


def split_take(path: Path, sentences: List[str], work_dir: Path) -> List[Path]:
    """Cắt bản thu một đoạn thành từng câu."""
    if len(sentences) <= 1:
        return [path]
    total = duration(path)
    cuts = sentence_cuts(sentences, total, silences(path))
    bounds = [0.0] + cuts + [total]
    pieces = []
    for k in range(len(sentences)):
        piece = work_dir / f"{path.stem}_s{k}.wav"
        run_command([_ffmpeg(), "-v", "error", "-y", "-i", str(path), "-ss", f"{bounds[k]:.3f}",
                     "-to", f"{bounds[k + 1]:.3f}", "-c", "copy", str(piece)],
                    timeout=60, error_message="Không cắt được bản thu thành từng câu.")
        pieces.append(piece)
    return pieces


def synthesize_recorded(segments, recordings: Dict[str, str], work_dir: Path):
    """Thay TTSEngine.synthesize_all: [(Segment, file âm thanh)] lấy từ bản tự thu.

    segments: các câu của kịch bản, Segment.start = số thứ tự đoạn (xem split_script).
    recordings: {"<số thứ tự đoạn>": đường dẫn bản thu}. Thiếu đoạn nào thì báo lỗi."""
    work_dir.mkdir(parents=True, exist_ok=True)
    groups: Dict[int, list] = {}
    for seg in segments:
        groups.setdefault(int(seg.start), []).append(seg)
    pairs = []
    for para, segs in sorted(groups.items()):
        path = recordings.get(str(para))
        if not path or not Path(path).is_file():
            raise RuntimeError(f"Thiếu bản tự thu của đoạn {para + 1}.")
        pieces = split_take(Path(path), [s.translated or s.text for s in segs], work_dir)
        pairs += list(zip(segs, pieces))
    return pairs
