"""
core/pipeline.py
Orchestrator: điều phối toàn bộ pipeline lồng tiếng từ đầu đến cuối.
Dùng được cả từ CLI lẫn Streamlit app.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from config.settings import TEMP_DIR, OUTPUT_DIR
from core.audio_extractor import AudioExtractor
from core.transcriber import Transcriber, Segment
from core.translator import get_translator
from core.tts_engine import TTSEngine
from core.video_composer import VideoComposer
from utils.subtitle_utils import segments_to_srt
import uuid
import shutil


@dataclass
class DubbingConfig:
    """Toàn bộ cấu hình cho một lần chạy pipeline.
    tts_engine: "edge-tts" | "gtts"
    """
    translator_engine: str = "openai"       # "openai" | "gemini" | "marian"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    whisper_model: str = "base"
    # "auto" = Whisper tu nhan dang tu am thanh; hoac ma ISO cu the ("en",
    # "vi", "ja", "zh", "ko"...) khi nguoi dung tu chon vi tu nhan dang doan.
    source_language: str = "auto"
    # Ngon ngu DICH. "vi" duoc MarianMT (fine-tune rieng EN->VI) ho tro; cac
    # ngon ngu khac bat buoc dung Gemini/OpenAI — xem _translate() ve viec
    # khoa MarianMT khi target != "vi".
    target_language: str = "vi"
    compute_device: str = "auto"           # "auto" | "cuda" | "cpu"
    sentence_resegment: bool = True
    silence_threshold: float = 0.45         # split by pauses >= threshold (seconds)
    tts_engine: str = "edge-tts"            # "edge-tts" | "gtts"
    tts_voice: str = "female"               # "female" | "male"
    original_volume: float = 0.1            # 0.0 – 1.0
    subtitle_mode: str = "bilingual"        # "bilingual" | "target" | "source" | "none"
    # API dich hong thi lui ve MarianMT (chay offline) thay vi hong ca job.
    translator_fallback: bool = True
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)


@dataclass
class DubbingResult:
    """Kết quả trả về sau khi pipeline hoàn thành."""
    output_video: Optional[Path] = None
    srt_path: Optional[Path] = None
    segments: List[Segment] = field(default_factory=list)
    #: Ma ngon ngu THAT SU da dung de nhan dang — luon la ma cu the (vd "en"),
    #: kha nang khac config.source_language khi nguoi dung de "auto".
    source_language_detected: str = ""
    elapsed_seconds: float = 0.0
    success: bool = False
    error: str = ""
    #: Thoi gian tung buoc, giay. Dung de biet nut that nam o dau.
    timings: dict = field(default_factory=dict)
    #: Engine da dung that su — khac translator_engine khi phai fallback.
    translator_used: str = ""
    #: Engine ban dau, chi set khi da fallback.
    fallback_from: str = ""


ProgressCallback = Callable[[int, str], None]  # (percent, message)


class DubbingPipeline:
    """
    Pipeline lồng tiếng end-to-end.

    Ví dụ sử dụng:
        config = DubbingConfig(translator_engine="openai", openai_api_key="sk-...")
        pipeline = DubbingPipeline(config)
        result = pipeline.run("my_video.mp4")
        print(result.output_video)
    """

    def __init__(self, config: DubbingConfig):
        self.config = config

    @staticmethod
    def _whisper_language_arg(source_language: str) -> str | None:
        """"auto" (mặc định, chọn qua UI) nghĩa là để Whisper tự nhận dạng —
        Whisper hiểu "tự nhận dạng" là language=None, không phải chuỗi "auto"."""
        return None if source_language == "auto" else source_language

    def _translate(
        self, segments, cfg: "DubbingConfig", progress, source_language: str = "en"
    ) -> tuple[list, str, str]:
        """Dịch, và lùi về MarianMT nếu API bên ngoài hỏng.

        Một lần Gemini đổi model hay dính rate limit không nên làm hỏng cả job:
        Whisper đã chạy xong, TTS vẫn chạy được, chỉ thiếu bản dịch. MarianMT
        chạy offline nên luôn sẵn sàng làm phương án dự phòng — NHƯNG chỉ khi
        đích là tiếng Việt: model đó tự fine-tune riêng cho EN→VI, dùng cho
        Nhật/Trung/Hàn sẽ ra tiếng Việt trong khi job báo là ngôn ngữ khác —
        sai còn khó nhận ra hơn cả việc dịch thất bại hẳn.
        """
        target = cfg.target_language

        def build(engine: str):
            if engine == "openai":
                return get_translator(
                    engine, api_key=cfg.openai_api_key, model=cfg.openai_model,
                    source_language=source_language, target_language=target,
                )
            if engine == "gemini":
                return get_translator(
                    engine, api_key=cfg.gemini_api_key, model=cfg.gemini_model,
                    source_language=source_language, target_language=target,
                )
            return get_translator("marian", device=cfg.compute_device)

        engine = cfg.translator_engine
        try:
            return build(engine).translate_segments(segments), engine, ""
        except Exception as exc:
            # target != "vi" thì marian không phải phương án dự phòng hợp lệ —
            # không có gì để lùi về, để lỗi thật hiện ra còn hơn âm thầm sai ngôn ngữ.
            if engine == "marian" or not cfg.translator_fallback or target != "vi":
                raise
            print(f"[Pipeline] {engine} lỗi ({exc}). Chuyển sang MarianMT.")
            progress(50, f"{engine} gặp lỗi, đang chuyển sang MarianMT...")
            return build("marian").translate_segments(segments), "marian", engine

    def run(
        self,
        video_path: str | Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> DubbingResult:
        """
        Chạy toàn bộ pipeline.

        Args:
            video_path:   Đường dẫn video đầu vào.
            progress_cb:  Hàm callback (percent, message) để cập nhật tiến trình.

        Returns:
            DubbingResult với thông tin kết quả.
        """
        result = DubbingResult()
        t0 = time.time()

        def _progress(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)
            else:
                print(f"[{pct:3d}%] {msg}")

        try:
            video_path = Path(video_path)
            cfg = self.config
            job_id = uuid.uuid4().hex
            job_temp_dir = TEMP_DIR / job_id
            job_temp_dir.mkdir(parents=True, exist_ok=True)

            # ── Bước 1: Tách audio ─────────────────────────────────────
            _progress(10, "Tách audio từ video...")
            step_started = time.time()
            extractor = AudioExtractor()
            audio_path = extractor.extract(video_path, output_path=job_temp_dir / f"{video_path.stem}.wav")
            video_duration = extractor.get_duration(video_path)
            result.timings["extract"] = round(time.time() - step_started, 2)

            # ── Bước 2: Transcribe ─────────────────────────────────────
            _progress(25, f"Nhận dạng giọng nói (Whisper {cfg.whisper_model})...")
            step_started = time.time()
            transcriber = Transcriber(model_size=cfg.whisper_model, device=cfg.compute_device)
            whisper_language = self._whisper_language_arg(cfg.source_language)
            segments, result.source_language_detected = transcriber.transcribe(
                audio_path,
                language=whisper_language,
                sentence_resegment=cfg.sentence_resegment,
                silence_threshold=cfg.silence_threshold,
            )
            result.timings["transcribe"] = round(time.time() - step_started, 2)
            _progress(40, f"Tìm thấy {len(segments)} segment ({result.source_language_detected}).")

            # ── Bước 3: Dịch ───────────────────────────────────────────
            _progress(45, "Đang dịch...")
            step_started = time.time()
            segments, result.translator_used, result.fallback_from = self._translate(
                segments, cfg, _progress, source_language=result.source_language_detected
            )
            result.timings["translate"] = round(time.time() - step_started, 2)
            _progress(65, "Dịch hoàn tất.")

            # ── Bước 4: TTS ────────────────────────────────────────────
            _progress(70, "Tổng hợp giọng nói tiếng Việt...")
            step_started = time.time()
            tts = TTSEngine(engine=cfg.tts_engine, voice=cfg.tts_voice, tts_dir=job_temp_dir / "tts")
            tts_results = tts.synthesize_all(segments)
            tts_segments = [seg for seg, _ in tts_results]
            tts_paths = [path for _, path in tts_results]
            result.timings["tts"] = round(time.time() - step_started, 2)
            _progress(85, "Tổng hợp giọng xong.")

            # ── Bước 5: Ghép video ─────────────────────────────────────
            _progress(88, "Ghép audio vào video...")
            step_started = time.time()
            composer = VideoComposer()
            output_video = composer.compose(
                video_path=video_path,
                segments=tts_segments,
                tts_paths=tts_paths,
                original_volume=cfg.original_volume,
                video_duration=video_duration,
            )

            result.timings["compose"] = round(time.time() - step_started, 2)

            # ── Bước 6: Tạo phụ đề ────────────────────────────────────
            srt_path = None
            if cfg.subtitle_mode != "none":
                _progress(95, "Tạo file phụ đề...")
                srt_path = output_video.with_suffix(".srt")
                segments_to_srt(segments, mode=cfg.subtitle_mode, output_path=srt_path)

            # ── Hoàn thành ─────────────────────────────────────────────
            _progress(100, "✅ Hoàn tất!")
            result.output_video = output_video
            result.srt_path = srt_path
            result.segments = segments
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False
            _progress(0, f"❌ Lỗi: {e}")

        finally:
            result.elapsed_seconds = time.time() - t0
            # Dọn file tạm riêng của job
            if 'job_temp_dir' in locals() and job_temp_dir.exists():
                shutil.rmtree(job_temp_dir, ignore_errors=True)

        return result