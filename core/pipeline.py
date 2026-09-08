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
    translator_engine: str = "gemini"       # "openai" | "gemini"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    whisper_model: str = "base"
    # "auto" = Whisper tu nhan dang tu am thanh; hoac ma ISO cu the ("en",
    # "vi", "ja", "zh", "ko"...) khi nguoi dung tu chon vi tu nhan dang doan.
    source_language: str = "auto"
    # Ngon ngu DICH — Gemini/OpenAI deu dich duoc ca 4 (vi/ja/zh/ko), khong
    # con rang buoc nhu MarianMT (fine-tune rieng EN->VI) truoc day.
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
    #: Ten model THAT SU da dich (vd "gemini-3.6-flash") — dung de tra gia
    #: dung dong trong LLM_PRICING_PER_MILLION_TOKENS. Rong = khong dich
    #: qua API nao (vd job that bai truoc buoc dich).
    translate_model: str = ""
    #: So token that da dung qua API dich — xem core/translator/base.py.
    translate_prompt_tokens: int = 0
    translate_completion_tokens: int = 0


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
    ) -> tuple[list, str, str, object]:
        """Dịch, và lùi sang nhà cung cấp còn lại nếu API chính hỏng.

        Trả thêm translator (phần tử thứ 4) để nơi gọi đọc được
        usage_prompt_tokens/usage_completion_tokens/model — tính chi phí
        dịch THẬT (xem core/translator/llm_common.estimate_llm_cost()),
        khác hẳn ước lượng theo giây GPU vốn có sẵn từ trước.

        Trước đây lùi về MarianMT (chạy offline) khi Gemini/OpenAI hỏng —
        model đó đã bị khoá khỏi lựa chọn của người dùng (fine-tune riêng
        EN→VI, dịch sai trong im lặng khi nguồn không phải tiếng Anh hoặc
        đích không phải tiếng Việt). Giờ dự án dùng đầy đủ API key nên
        phương án dự phòng là nhà cung cấp CÒN LẠI: Gemini lỗi thì thử
        OpenAI, và ngược lại — vẫn giữ đúng tinh thần ban đầu (một lần
        rate limit hay đổi model không nên làm hỏng cả job), chỉ đổi nơi
        lùi về.
        """
        target = cfg.target_language

        def build(engine: str):
            if engine == "openai":
                return get_translator(
                    "openai", api_key=cfg.openai_api_key, model=cfg.openai_model,
                    source_language=source_language, target_language=target,
                )
            return get_translator(
                "gemini", api_key=cfg.gemini_api_key, model=cfg.gemini_model,
                source_language=source_language, target_language=target,
            )

        engine = cfg.translator_engine
        other = "openai" if engine == "gemini" else "gemini"
        other_key = cfg.openai_api_key if other == "openai" else cfg.gemini_api_key

        try:
            translator = build(engine)
            return translator.translate_segments(segments), engine, "", translator
        except Exception as exc:
            # Không có key cho nhà cung cấp còn lại thì không có gì để lùi
            # về — để lỗi thật hiện ra còn hơn thử một engine chắc chắn hỏng.
            if not cfg.translator_fallback or not other_key:
                raise
            print(f"[Pipeline] {engine} lỗi ({exc}). Chuyển sang {other}.")
            progress(50, f"{engine} gặp lỗi, đang chuyển sang {other}...")
            translator = build(other)
            return translator.translate_segments(segments), other, engine, translator

    @staticmethod
    def _progress_fn(progress_cb: Optional[ProgressCallback]) -> Callable[[int, str], None]:
        def _progress(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)
            else:
                print(f"[{pct:3d}%] {msg}")
        return _progress

    def transcribe_and_translate(
        self,
        video_path: str | Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> DubbingResult:
        """Nửa đầu pipeline: tách audio, nhận dạng giọng nói, dịch.

        Dừng lại ở đây thay vì chạy tiếp TTS/ghép video — tốn ít công sức và
        thời gian hơn hẳn (không có TTS/ffmpeg), hợp để người dùng dừng lại
        xem/sửa transcript trước khi tốn công đoạn tốn thời gian nhất.
        Không cần giữ lại audio đã tách: synthesize_and_compose() ở dưới đọc
        thẳng từ video_path gốc (composer tự trích audio gốc bằng ffmpeg),
        nên temp dir của bước này dọn ngay khi xong, không phải chờ tới
        giai đoạn 2 — có thể diễn ra rất lâu sau, ở container khác.
        """
        result = DubbingResult()
        t0 = time.time()
        _progress = self._progress_fn(progress_cb)

        job_temp_dir = None
        try:
            video_path = Path(video_path)
            cfg = self.config
            job_temp_dir = TEMP_DIR / uuid.uuid4().hex
            job_temp_dir.mkdir(parents=True, exist_ok=True)

            # ── Bước 1: Tách audio ─────────────────────────────────────
            _progress(10, "Tách audio từ video...")
            step_started = time.time()
            extractor = AudioExtractor()
            audio_path = extractor.extract(video_path, output_path=job_temp_dir / f"{video_path.stem}.wav")
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
            segments, result.translator_used, result.fallback_from, translator = self._translate(
                segments, cfg, _progress, source_language=result.source_language_detected
            )
            result.timings["translate"] = round(time.time() - step_started, 2)
            result.translate_model = getattr(translator, "model", "") or ""
            # MarianTranslator.model KHÔNG phải chuỗi tên model (là object
            # PyTorch đã nạp) — chỉ Gemini/OpenAI mới có .model dạng chuỗi,
            # đường dẫn duy nhất _translate() thực sự đi qua từ khi MarianMT
            # bị khoá khỏi lựa chọn của người dùng.
            if not isinstance(result.translate_model, str):
                result.translate_model = ""
            result.translate_prompt_tokens = getattr(translator, "usage_prompt_tokens", 0)
            result.translate_completion_tokens = getattr(translator, "usage_completion_tokens", 0)
            _progress(65, "Dịch hoàn tất, đang chờ duyệt.")

            result.segments = segments
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.success = False
            _progress(0, f"❌ Lỗi: {e}")

        finally:
            result.elapsed_seconds = time.time() - t0
            if job_temp_dir is not None and job_temp_dir.exists():
                shutil.rmtree(job_temp_dir, ignore_errors=True)

        return result

    def synthesize_and_compose(
        self,
        video_path: str | Path,
        segments: List[Segment],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> DubbingResult:
        """Nửa sau pipeline: TTS rồi ghép video, dùng `segments` đã có sẵn
        (từ transcribe_and_translate(), có thể đã được người dùng sửa tay).

        Không đụng tới translator ở đây — segments coi như bản dịch cuối
        cùng, đã qua bước duyệt. video_duration được ĐO LẠI từ video_path
        (ffprobe, rẻ) thay vì nhận qua tham số, vì lệnh gọi này có thể tới
        rất lâu sau transcribe_and_translate(), thậm chí ở container khác —
        không có gì đảm bảo giữ được biến số đó qua ranh giới đó.
        """
        result = DubbingResult()
        t0 = time.time()
        _progress = self._progress_fn(progress_cb)

        job_temp_dir = None
        try:
            video_path = Path(video_path)
            cfg = self.config
            job_temp_dir = TEMP_DIR / uuid.uuid4().hex
            job_temp_dir.mkdir(parents=True, exist_ok=True)

            video_duration = AudioExtractor().get_duration(video_path)

            # ── Bước 4: TTS ────────────────────────────────────────────
            _progress(70, "Tổng hợp giọng đọc...")
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
            if job_temp_dir is not None and job_temp_dir.exists():
                shutil.rmtree(job_temp_dir, ignore_errors=True)

        return result

    def run(
        self,
        video_path: str | Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> DubbingResult:
        """
        Chạy toàn bộ pipeline end-to-end, không dừng lại để duyệt — dùng cho
        CLI hay bất kỳ chỗ nào không cần bước duyệt transcript giữa chừng.
        Ghép kết quả của transcribe_and_translate() + synthesize_and_compose().

        Args:
            video_path:   Đường dẫn video đầu vào.
            progress_cb:  Hàm callback (percent, message) để cập nhật tiến trình.

        Returns:
            DubbingResult với thông tin kết quả.
        """
        first = self.transcribe_and_translate(video_path, progress_cb=progress_cb)
        if not first.success:
            return first

        second = self.synthesize_and_compose(video_path, first.segments, progress_cb=progress_cb)
        second.source_language_detected = first.source_language_detected
        second.translator_used = first.translator_used
        second.fallback_from = first.fallback_from
        second.elapsed_seconds += first.elapsed_seconds
        second.timings = {**first.timings, **second.timings}
        return second