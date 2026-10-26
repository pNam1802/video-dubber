from __future__ import annotations

import asyncio
import base64
import io
import math
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

# Fix Windows CP1252 charmap error: force UTF-8 for stdout/stderr
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from config.settings import (
	FFMPEG_BIN,
	FFMPEG_TIMEOUT,
	GEMINI_API_KEY,
	GEMINI_TTS_MODEL,
	GEMINI_TTS_STYLE,
	TEMP_DIR,
	TTS_CONCURRENCY,
	TTS_VOICES,
)
from core.transcriber import Segment
from core.voice_timeline import base_speed, fill_gaps, plan_timeline
from utils.proc import run_command

# edge-tts chap chon ("No audio was received") kha thuong xuyen, nhat la khi
# goi song song. Het luot thu thi segment roi ve gTTS — GIONG KHAC HAN giua
# chung video — nen thu lai nhieu hon va gian cach dai hon truoc khi bo cuoc.
_EDGE_ATTEMPTS = 6
_EDGE_BACKOFF = 1.0


class TTSEngine:
	"""Vietnamese text-to-speech wrapper for edge-tts, Gemini TTS and gTTS."""

	def __init__(
		self,
		engine: str = "edge-tts",
		voice: str = "female",
		tts_dir: str | Path | None = None,
		concurrency: int | None = None,
	):
		self.engine = (engine or "edge-tts").strip().lower()
		self.voice = voice
		self.tts_dir = Path(tts_dir) if tts_dir else TEMP_DIR / "tts"
		self.tts_dir.mkdir(parents=True, exist_ok=True)
		self.concurrency = max(1, int(concurrency or TTS_CONCURRENCY))

		voices = TTS_VOICES.get(self.engine, {})
		#: Tên giọng thật của nhà cung cấp (vd "Leda", "vi-VN-HoaiMyNeural").
		self.voice_name = voices.get(voice) or voices.get("female", "")
		# edge-tts còn là phương án dự phòng khi Gemini lỗi — khi đó đọc bằng
		# HoaiMy, gần giọng Việt tự nhiên nhất trong các giọng miễn phí.
		self.edge_voice = self.voice_name if self.engine == "edge-tts" else TTS_VOICES["edge-tts"]["female"]
		self._gemini_client = None
		self._gemini_lock = threading.Lock()
		#: Token âm thanh Gemini đã dùng — để ghi log chi phí thật.
		self.gemini_audio_tokens = 0

	def synthesize_all(self, segments: List[Segment]) -> List[tuple]:
		"""Return list of (segment, path) pairs — only for successfully synthesized segments.

		Các segment được tổng hợp song song: cả edge-tts lẫn gTTS đều chủ yếu là
		thời gian chờ mạng, nên chạy tuần tự sẽ khiến bước này chiếm phần lớn
		thời gian của cả pipeline.
		"""
		tasks: list[tuple[int, Segment, str]] = []
		for idx, segment in enumerate(segments):
			text = (segment.translated or segment.text).strip()
			if self._is_tts_text_valid(text):
				tasks.append((idx, segment, text))

		valid_segments = len(tasks)
		results: List[tuple] = []
		error_samples: list[str] = []

		if tasks:
			workers = min(self.concurrency, len(tasks))
			if workers > 1:
				print(f"[TTS] Tổng hợp {len(tasks)} segment với {workers} luồng song song...")
				with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts") as pool:
					# pool.map giữ nguyên thứ tự đầu vào nên segment không bị đảo.
					outcomes = list(pool.map(lambda task: self._synthesize_segment(*task), tasks))
			else:
				outcomes = [self._synthesize_segment(*task) for task in tasks]

			for pair, err_msg in outcomes:
				if pair is not None:
					results.append(pair)
				elif err_msg and len(error_samples) < 5:
					error_samples.append(err_msg)

		if not results:
			detail = (
				f"TTS generated 0 files (engine={self.engine}, "
				f"segments={len(segments)}, valid_segments={valid_segments})."
			)
			if error_samples:
				detail += " Sample errors: " + " | ".join(error_samples)
			raise RuntimeError(detail)
		if self.gemini_audio_tokens:
			print(f"[TTS] Gemini: {self.gemini_audio_tokens} token âm thanh (xem chi phí thật trên AI Studio).")
		return results

	def _synthesize_segment(self, idx: int, segment: Segment, text: str) -> tuple[tuple | None, str | None]:
		"""Tổng hợp một segment. Trả về ((segment, path), None) hoặc (None, thông báo lỗi)."""
		out = self.tts_dir / f"seg_{idx:05d}.mp3"
		try:
			self._synthesize_checked(text, out)
			return (segment, out), None
		except Exception as exc:
			# Fallback to gTTS for non-gTTS engines
			if self.engine != "gtts":
				# Gemini lỗi thì thử edge-tts trước (giọng tốt hơn gTTS nhiều).
				fallbacks = (["edge-tts"] if self.engine == "gemini" else []) + ["gtts"]
				err_msg = f"segment {idx}: {exc}"
				for fallback in fallbacks:
					try:
						if fallback == "edge-tts":
							self._synthesize_edge(text, out)
						else:
							self._synthesize_gtts(text, out)
						self._trim_silence(out)
						self._check_audio(out, text)
						print(f"[TTS] Segment {idx}: fallback {fallback} succeeded (primary error: {exc})")
						return (segment, out), None
					except Exception as fb_exc:
						err_msg += f"; fallback {fallback} failed: {fb_exc}"
			else:
				err_msg = f"segment {idx}: {exc}"
			print(f"[TTS] Skipping {err_msg}")
			return None, err_msg

	def _synthesize_checked(self, text: str, out: Path, speed: float = 1.0, attempts: int = 3) -> None:
		"""Tổng hợp + cắt khoảng lặng + kiểm tra âm thanh không rỗng/cụt; thử lại
		vài lần trước khi báo lỗi."""
		last_exc: Exception | None = None
		for _ in range(attempts):
			try:
				self.synthesize_text(text, out, speed=speed)
				self._trim_silence(out)
				self._check_audio(out, text)
				return
			except Exception as exc:
				last_exc = exc
		raise RuntimeError(str(last_exc)) from last_exc

	@classmethod
	def _check_audio(cls, path: Path, text: str) -> None:
		"""edge-tts đôi khi trả về file có dung lượng nhưng gần như không có
		tiếng — vượt qua kiểm tra "file > 0 byte" và câu đó biến mất khỏi
		video (job #33: câu #13 mất, im lặng 7.6s). Không ai nói nổi quá
		~12 âm tiết/giây, nên ngắn hơn mức đó là âm thanh hỏng."""
		duration = cls._duration(path)
		syllables = len(text.split())
		if duration < max(0.3, syllables / 12.0):
			raise RuntimeError(f"Âm thanh TTS bất thường: {duration:.2f}s cho {syllables} âm tiết")

	def place_on_timeline(self, results: List[tuple]) -> tuple[List[tuple], List[float]]:
		"""Xếp các câu đã tổng hợp lên dòng thời gian cho nghe liền mạch (xem
		core/voice_timeline.py). Đổi tốc độ bằng cách TỔNG HỢP LẠI với tham số
		rate của edge-tts — giọng vẫn tự nhiên — thay vì cắt ghép mẫu âm thanh
		(pydub.speedup) vốn làm giọng rè.

		Hai lượt: lượt đầu (synthesize_all) ở tốc độ thường để ĐO độ dài thật;
		từ đó tính tốc độ chung + tốc độ riêng từng câu, rồi tổng hợp lại một
		lần các câu cần đổi. Độ dài ở tốc độ s xấp xỉ độ dài thường / s (đo
		thật: -10% cho 1.11 lần).

		Trả về (results, thời điểm bắt đầu của từng câu)."""
		slots = [(seg.start, seg.end) for seg, _ in results]
		durations = [self._duration(path) for _, path in results]

		base = base_speed(slots, durations)
		estimated = [d / base for d in durations]
		plan = plan_timeline(slots, estimated)
		factors = fill_gaps(estimated, plan)
		speeds = [base * (p.speed if p.speed > 1.0 else f) for p, f in zip(plan, factors)]

		redo = [(i, s) for i, s in enumerate(speeds) if abs(s - 1.0) > 0.02]
		if redo:
			print(f"[TTS] Tốc độ chung {base:.2f}; tổng hợp lại {len(redo)}/{len(results)} câu để giữ nhịp liền mạch...")
			workers = min(self.concurrency, len(redo))
			with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts-rate") as pool:
				new_durations = list(pool.map(lambda job: self._resynthesize(results[job[0]], job[1]), redo))
			for (i, _), duration in zip(redo, new_durations):
				if duration is not None:
					durations[i] = duration

		# Lần xếp cuối với tốc độ cố định 1.0: chỉ dời thời điểm bắt đầu, nên
		# không câu nào chồng lên câu trước dù rate của edge-tts không khớp
		# tuyệt đối với tốc độ đã yêu cầu.
		final = plan_timeline(slots, durations, max_speed=1.0)
		starts = [p.start for p in final]
		gaps = [starts[i + 1] - (starts[i] + durations[i]) for i in range(len(starts) - 1)]
		lags = [start - slot[0] for start, slot in zip(starts, slots)]
		if gaps:
			print(
				f"[TTS] Nhịp đọc: {sum(g >= 0.5 for g in gaps)}/{len(gaps)} chỗ nghỉ ≥0.5s, "
				f"nghỉ dài nhất {max(gaps):.2f}s, lệch hình từ {min(lags):+.2f}s tới {max(lags):+.2f}s"
			)
		return results, starts

	def _resynthesize(self, pair: tuple, speed: float) -> float | None:
		segment, path = pair
		text = (segment.translated or segment.text).strip()
		# Ghi ra file tạm: lỗi giữa chừng thì bản tốc độ thường vẫn còn nguyên.
		tmp = path.with_name(path.stem + "_rate" + path.suffix)
		try:
			if self.engine == "edge-tts":
				self._synthesize_checked(text, tmp, speed=speed)
			else:
				# Gemini/gTTS không có tham số tốc độ chính xác: chỉnh bằng atempo
				# trên bản đã có, KHÔNG gọi API lần nữa — đỡ tốn tiền, và giữ
				# nguyên ngữ điệu (gọi lại thì Gemini đọc mỗi lần một kiểu).
				shutil.copy(path, tmp)
				self._atempo(tmp, speed)
				self._check_audio(tmp, text)
			tmp.replace(path)
			return self._duration(path)
		except Exception as exc:
			# Giữ bản tốc độ thường đã có — lệch nhịp một chút còn hơn mất câu.
			tmp.unlink(missing_ok=True)
			print(f"[TTS] Đổi tốc độ {path.name} thất bại, giữ tốc độ thường: {exc}")
			return None

	@staticmethod
	def _duration(path: Path) -> float:
		from pydub import AudioSegment

		return len(AudioSegment.from_file(str(path))) / 1000.0

	@staticmethod
	def _trim_silence(audio_path: Path) -> None:
		"""Cắt khoảng lặng đầu/cuối mà TTS tự chèn — để nhịp nghỉ giữa các câu
		do place_on_timeline() quyết định, không cộng dồn khoảng lặng thừa."""
		try:
			from pydub import AudioSegment
			from pydub.silence import detect_nonsilent

			audio = AudioSegment.from_file(str(audio_path))
			ranges = detect_nonsilent(audio, min_silence_len=150, silence_thresh=-45)
			if not ranges:
				return
			audio[ranges[0][0]:ranges[-1][1]].export(str(audio_path), format=audio_path.suffix.lstrip(".") or "mp3")
		except Exception as exc:
			print(f"[TTS] Cắt khoảng lặng thất bại cho {audio_path}: {exc}")

	@staticmethod
	def _is_tts_text_valid(text: str) -> bool:
		"""Filter out noisy/non-speech-like text that frequently breaks neural TTS."""
		normalized = re.sub(r"\s+", " ", text).strip()
		if not normalized:
			return False
		# Keep only text with meaningful alphanumeric content.
		alnum_count = sum(ch.isalnum() for ch in normalized)
		if alnum_count < 2:
			return False
		# Skip pure punctuation/symbol segments.
		if not re.search(r"[\w\d]", normalized, flags=re.UNICODE):
			return False
		return True

	def synthesize_text(self, text: str, output_path: str | Path, speed: float = 1.0) -> Path:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		if self.engine == "edge-tts":
			self._synthesize_edge(text, output_path, speed=speed)
			return output_path
		if self.engine in ("gtts", "gemini"):
			if self.engine == "gtts":
				self._synthesize_gtts(text, output_path)
			else:
				self._synthesize_gemini(text, output_path)
			if abs(speed - 1.0) > 1e-3:
				self._atempo(output_path, speed)
			return output_path
		raise ValueError(f"Unsupported TTS engine: {self.engine}")

	def _synthesize_gemini(self, text: str, output_path: Path) -> None:
		"""Gemini TTS: giọng AI có cảm xúc, làm theo GEMINI_TTS_STYLE. Trả về WAV
		24 kHz mono — chuyển sang định dạng của output_path (mp3) để các bước
		sau (cắt khoảng lặng, ghép video) xử lý như mọi giọng khác."""
		from pydub import AudioSegment

		from core.translator.llm_common import call_with_retry

		if not GEMINI_API_KEY:
			raise RuntimeError("GEMINI_API_KEY chưa được thiết lập!")
		with self._gemini_lock:
			if self._gemini_client is None:
				from google import genai

				self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
		client = self._gemini_client

		interaction = call_with_retry(
			lambda: client.interactions.create(
				model=GEMINI_TTS_MODEL,
				input=[{
					"type": "user_input",
					"content": [{
						"type": "text",
						"text": text,
						"annotations": [{"type": "speech_metadata", "style": GEMINI_TTS_STYLE}],
					}],
				}],
				response_format={"type": "audio"},
				generation_config={"speech_config": [{"voice": self.voice_name}]},
			),
			what="Gemini TTS",
		)
		audio = AudioSegment.from_file(io.BytesIO(base64.b64decode(interaction.output_audio.data)), format="wav")
		audio.export(str(output_path), format=output_path.suffix.lstrip(".") or "mp3")

		usage = getattr(interaction, "usage", None)
		for item in getattr(usage, "output_tokens_by_modality", None) or []:
			if str(getattr(item, "modality", "")).lower().endswith("audio"):
				with self._gemini_lock:
					self.gemini_audio_tokens += int(getattr(item, "tokens", 0) or 0)

	def _synthesize_gtts(self, text: str, output_path: Path) -> None:
		"""Synthesize speech using Google TTS (gTTS)."""
		try:
			from gtts import gTTS
		except ImportError as exc:
			raise RuntimeError("Missing package gTTS. Install with: pip install gTTS") from exc
		tts_obj = gTTS(text=text, lang="vi", slow=False)
		tts_obj.save(str(output_path))
		if not output_path.exists() or output_path.stat().st_size == 0:
			raise RuntimeError("gTTS produced no valid audio file.")

	@staticmethod
	def _atempo(path: Path, speed: float) -> None:
		"""gTTS không có tham số tốc độ: dùng atempo của ffmpeg (nhanh hoặc chậm, giữ cao độ,
		ít rè hơn nhiều so với pydub.speedup)."""
		tmp = path.with_name(path.stem + "_tempo" + path.suffix)
		run_command(
			[FFMPEG_BIN, "-y", "-i", str(path), "-filter:a", f"atempo={speed:.3f}", str(tmp)],
			timeout=FFMPEG_TIMEOUT,
			error_message="Đổi tốc độ giọng gTTS thất bại",
		)
		tmp.replace(path)

	@staticmethod
	def edge_rate(speed: float) -> str:
		"""rate của edge-tts là % so với tốc độ thường. Luôn làm tròn về phía
		câu NGẮN hơn (nhanh: tròn lên; chậm: bớt chậm) để không lấn sang câu sau."""
		if speed >= 1.0:
			return f"+{math.ceil(round((speed - 1.0) * 100, 6))}%"
		return f"-{math.floor(round((1.0 - speed) * 100, 6))}%"

	# Fix asyncio in _synthesize_edge:
	def _synthesize_edge(self, text: str, output_path: Path, speed: float = 1.0) -> None:
		try:
			import edge_tts
		except Exception as exc:
			raise RuntimeError("Missing dependency 'edge-tts'. Install with: pip install edge-tts") from exc

		rate = self.edge_rate(speed)

		async def _run() -> None:
			communicate = edge_tts.Communicate(text=text, voice=self.edge_voice, rate=rate)
			await communicate.save(str(output_path))

		last_exc: Exception | None = None
		for attempt in range(_EDGE_ATTEMPTS):
			try:
				# Mỗi lần gọi tạo event loop mới riêng cho thread hiện tại.
				# Tránh dùng asyncio.run() vì có thể conflict với loop của Flask thread.
				loop = asyncio.new_event_loop()
				try:
					loop.run_until_complete(_run())
				finally:
					loop.close()

				if output_path.exists() and output_path.stat().st_size > 0:
					return
				raise RuntimeError("edge-tts returned empty audio output")
			except Exception as exc:
				last_exc = exc
				msg = str(exc)
				if output_path.exists() and output_path.stat().st_size == 0:
					output_path.unlink(missing_ok=True)
				is_retryable = (
					"No audio was received" in msg
					or "empty audio output" in msg
					or "WinError" in msg
					or "ConnectionResetError" in msg
				)
				if not is_retryable or attempt == _EDGE_ATTEMPTS - 1:
					raise RuntimeError(f"edge-tts failed after {attempt + 1} attempt(s): {msg}") from exc
				time.sleep(_EDGE_BACKOFF * (attempt + 1))

		raise RuntimeError(f"edge-tts failed: {last_exc}")

