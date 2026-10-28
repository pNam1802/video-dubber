"""
core/chatterbox_client.py
Gọi Chatterbox-Turbo (mã nguồn mở, MIT — github.com/resemble-ai/chatterbox):
clone giọng TIẾNG ANH từ file mẫu.

Chatterbox ghim torch 2.6, khác cả Whisper (2.5) lẫn VieNeu (2.8), nên trên
Modal nó chạy ở container riêng (modal_app.py::EnglishCloner); container ghép
video chỉ gửi văn bản sang và nhận lại mp3. Chạy local (JOB_RUNNER=thread) thì
nạp thẳng trong tiến trình nếu đã `pip install chatterbox-tts`.

File này KHÔNG import config/Flask ở đầu: container EnglishCloner chỉ có
chatterbox-tts + numpy + soundfile.
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional

#: Turbo từ chối giọng mẫu không dài hơn 5 giây (assert trong
#: prepare_conditionals) — mẫu ngắn hơn thì lặp lại cho đủ.
MIN_REF_SEC = 5.5

_local_tts: Any = None
_local_lock = threading.Lock()


def load_model() -> Any:
    """Nạp Chatterbox-Turbo (350M). Lần đầu tải trọng số từ Hugging Face."""
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model = ChatterboxTurboTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    # generate() dùng model.conds khi không có giọng mẫu — giọng clone lần
    # trước sẽ ghi đè nó, nên giữ riêng giọng có sẵn để khôi phục.
    model.builtin_conds = model.conds
    return model


def _padded_ref(ref_wav: bytes, folder: Path) -> Path:
    import numpy as np
    import soundfile as sf

    raw = folder / "ref_raw.wav"
    raw.write_bytes(ref_wav)
    audio, sr = sf.read(str(raw))
    if len(audio) and len(audio) / sr < MIN_REF_SEC:
        repeats = int(MIN_REF_SEC * sr // len(audio)) + 1
        audio = np.concatenate([audio] * repeats)
    ref = folder / "ref.wav"
    sf.write(str(ref), audio, sr)
    return ref


def synthesize_with(tts: Any, texts: List[str], ref_wav: Optional[bytes] = None) -> List[bytes]:
    """Đọc từng câu bằng giọng có sẵn hoặc giọng clone từ ref_wav.
    Trả về mp3 cho từng câu, đúng thứ tự đầu vào."""
    import soundfile as sf

    if not texts:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        if ref_wav:
            tts.prepare_conditionals(str(_padded_ref(ref_wav, Path(tmp))))
        elif tts.builtin_conds is not None:
            tts.conds = tts.builtin_conds
        else:
            raise RuntimeError("Chatterbox không có giọng có sẵn — cần giọng mẫu để clone")

        out: List[bytes] = []
        for i, text in enumerate(texts):
            # Turbo không đọc theo lô: mỗi câu một lần generate(). Âm thanh
            # mang watermark Perth (dấu nhận biết giọng AI) — giữ nguyên.
            audio = tts.generate(text).squeeze(0).cpu().numpy()
            wav = Path(tmp) / f"{i}.wav"
            mp3 = Path(tmp) / f"{i}.mp3"
            sf.write(str(wav), audio, tts.sr)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "24000",
                 "-b:a", "64k", str(mp3)],
                check=True, capture_output=True, timeout=120,
            )
            out.append(mp3.read_bytes())
        return out


def synthesize(texts: List[str], ref_wav: Optional[bytes] = None) -> List[bytes]:
    """Điểm vào cho TTSEngine: tự chọn chạy trên Modal hay tại chỗ."""
    from config.settings import JOB_RUNNER, MODAL_APP_NAME

    if JOB_RUNNER == "modal":
        import modal

        cloner = modal.Cls.from_name(MODAL_APP_NAME, "EnglishCloner")()
        return cloner.synthesize.remote(texts, ref_wav)

    global _local_tts
    with _local_lock:
        if _local_tts is None:
            try:
                _local_tts = load_model()
            except ImportError as exc:
                raise RuntimeError("Chưa cài Chatterbox: pip install chatterbox-tts") from exc
        return synthesize_with(_local_tts, texts, ref_wav)
