"""
core/vieneu_client.py
Gọi VieNeu-TTS (mã nguồn mở, Apache 2.0 — github.com/pnnbao97/VieNeu-TTS).

VieNeu cần torch 2.8 + GPU, khác hẳn môi trường của phần còn lại (torch 2.5
cho Whisper), nên trên Modal nó chạy ở container riêng
(modal_app.py::VoiceCloner); container ghép video chỉ gửi văn bản sang và nhận
lại mp3. Chạy local (JOB_RUNNER=thread) thì nạp thẳng trong tiến trình nếu đã
`pip install vieneu`.

File này KHÔNG import config/Flask ở đầu: container VoiceCloner chỉ có
vieneu + numpy + soundfile.
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional

# Gom câu thành lô: GPU chỉ nhanh khi đọc nhiều câu một lượt (RTF ~0,04 trên
# T4 với cả 103 câu một lần — xem README).
BATCH_SIZE = 32

_local_tts: Any = None
_local_lock = threading.Lock()


def load_model() -> Any:
    """Nạp VieNeu v3 Turbo (mặc định). Lần đầu tải ~vài trăm MB từ Hugging Face."""
    from vieneu import Vieneu

    return Vieneu(max_batch_size=BATCH_SIZE)


def synthesize_with(tts: Any, texts: List[str], voice: Optional[str] = None,
                    ref_wav: Optional[bytes] = None) -> List[bytes]:
    """Đọc từng câu bằng giọng có sẵn (voice) hoặc giọng clone từ ref_wav.
    Trả về mp3 cho từng câu, đúng thứ tự đầu vào."""
    import soundfile as sf

    if not texts:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        kwargs: dict[str, Any] = {}
        if ref_wav:
            ref_path = Path(tmp) / "ref.wav"
            ref_path.write_bytes(ref_wav)
            kwargs["ref_audio"] = str(ref_path)
        else:
            kwargs["voice"] = voice
        # apply_watermark để mặc định (True): âm thanh mang dấu nhận biết là
        # giọng AI — giữ nguyên, nhất là với giọng clone.
        audios = tts.infer_batch(texts, batch_size=BATCH_SIZE, **kwargs)

        out: List[bytes] = []
        for i, audio in enumerate(audios):
            wav = Path(tmp) / f"{i}.wav"
            mp3 = Path(tmp) / f"{i}.mp3"
            sf.write(str(wav), audio, tts.sample_rate)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "24000",
                 "-b:a", "64k", str(mp3)],
                check=True, capture_output=True, timeout=120,
            )
            out.append(mp3.read_bytes())
        return out


def synthesize(texts: List[str], voice: Optional[str] = None, ref_wav: Optional[bytes] = None) -> List[bytes]:
    """Điểm vào cho TTSEngine: tự chọn chạy trên Modal hay tại chỗ."""
    from config.settings import JOB_RUNNER, MODAL_APP_NAME

    if JOB_RUNNER == "modal":
        import modal

        cloner = modal.Cls.from_name(MODAL_APP_NAME, "VoiceCloner")()
        return cloner.synthesize.remote(texts, voice, ref_wav)

    global _local_tts
    with _local_lock:
        if _local_tts is None:
            try:
                _local_tts = load_model()
            except ImportError as exc:
                raise RuntimeError("Chưa cài VieNeu-TTS: pip install vieneu") from exc
        return synthesize_with(_local_tts, texts, voice, ref_wav)
