from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

from config.settings import WHISPER_BACKEND


@dataclass
class Segment:
	start: float
	end: float
	text: str
	translated: str = ""


class Transcriber:
	"""Speech-to-text using Whisper.

	Hai backend, cùng một API:
	  - "faster"  : faster-whisper (CTranslate2) — nhanh hơn nhiều ở cùng model size.
	  - "openai"  : openai-whisper gốc.
	Mặc định WHISPER_BACKEND="auto": ưu tiên faster-whisper, tự lùi về openai-whisper
	nếu chưa cài hoặc không khởi tạo được.
	"""

	_SENTENCE_END_RE = re.compile(r"[.!?][\"'\)\]]*$")
	_LEADING_PUNCT_RE = re.compile(r"^[,.;:!?%\)\]\}]+")

	def __init__(self, model_size: str = "base", device: str = "auto", backend: str | None = None):
		try:
			import torch
		except Exception as exc:
			raise RuntimeError(
				"Missing dependency 'torch'. Install with: pip install torch"
			) from exc

		requested_device = (device or "auto").strip().lower()
		if requested_device not in {"auto", "cuda", "cpu"}:
			raise ValueError(f"Unsupported device: {device}. Use one of: auto, cuda, cpu")

		if requested_device == "auto":
			resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
		elif requested_device == "cuda":
			if not torch.cuda.is_available():
				print("[Whisper] CUDA requested but unavailable. Falling back to CPU.")
				resolved_device = "cpu"
			else:
				resolved_device = "cuda"
		else:
			resolved_device = "cpu"

		self._torch = torch
		self.model_size = model_size
		self.device = resolved_device
		self.use_fp16 = self.device == "cuda"

		requested_backend = (backend or WHISPER_BACKEND or "auto").strip().lower()
		if requested_backend not in {"auto", "faster", "openai"}:
			raise ValueError(f"Unsupported whisper backend: {backend}. Use one of: auto, faster, openai")

		self.backend = ""
		if requested_backend in {"auto", "faster"}:
			if self._load_faster_whisper():
				self.backend = "faster"
			elif requested_backend == "faster":
				raise RuntimeError(
					"Backend 'faster' được yêu cầu nhưng không khởi tạo được. "
					"Cài bằng: pip install faster-whisper"
				)

		if not self.backend:
			self._load_openai_whisper()
			self.backend = "openai"

	# ── Nạp model ────────────────────────────────────────────────
	def _load_faster_whisper(self) -> bool:
		"""Thử nạp faster-whisper. Trả về False để caller lùi về openai-whisper."""
		try:
			from faster_whisper import WhisperModel
		except Exception as exc:
			print(f"[Whisper] Không dùng được faster-whisper ({exc}). Dùng openai-whisper.")
			return False

		for device, compute_type in self._faster_device_plan():
			try:
				self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
				self.device = device
				self.use_fp16 = device == "cuda"
				self.compute_type = compute_type
				print(f"[Whisper] faster-whisper '{self.model_size}' trên {device} ({compute_type}).")
				return True
			except Exception as exc:
				print(f"[Whisper] faster-whisper không chạy được trên {device}/{compute_type}: {exc}")
				continue
		return False

	def _faster_device_plan(self) -> list[tuple[str, str]]:
		"""Thứ tự thử: GPU trước (nếu có), rồi CPU int8."""
		plan: list[tuple[str, str]] = []
		if self.device == "cuda":
			plan.append(("cuda", "float16"))
		plan.append(("cpu", "int8"))
		return plan

	def _load_openai_whisper(self) -> None:
		try:
			import whisper
		except Exception as exc:
			raise RuntimeError(
				"Missing dependency 'openai-whisper'. Install with: pip install openai-whisper"
			) from exc

		self._whisper = whisper
		try:
			self.model = whisper.load_model(self.model_size, device=self.device)
		except RuntimeError as exc:
			msg = str(exc).lower()
			if self.device == "cuda" and (
				"out of memory" in msg or "unable to find an engine" in msg
			):
				print("[Whisper] CUDA failed while loading model. Falling back to CPU.")
				self.device = "cpu"
				self.use_fp16 = False
				if self._torch.cuda.is_available():
					self._torch.cuda.empty_cache()
				self.model = whisper.load_model(self.model_size, device=self.device)
			else:
				raise

	@classmethod
	def _join_text_fragments(cls, texts: List[str]) -> str:
		joined = ""
		for part in texts:
			part = (part or "").strip()
			if not part:
				continue
			if not joined:
				joined = part
				continue
			# Avoid adding a space before punctuation-starting fragments.
			if cls._LEADING_PUNCT_RE.match(part):
				joined += part
			else:
				joined += " " + part
		return joined.strip()

	@classmethod
	def _merge_segments_into_sentences(
		cls,
		segments: List[Segment],
		silence_threshold: float = 0.45,
	) -> List[Segment]:
		if not segments:
			return []

		ordered = sorted(segments, key=lambda s: (s.start, s.end))
		merged: List[Segment] = []
		bucket: List[Segment] = []
		bucket_texts: List[str] = []

		for idx, seg in enumerate(ordered):
			if seg.end <= seg.start:
				continue

			text = (seg.text or "").strip()
			if not text:
				continue

			bucket.append(seg)
			bucket_texts.append(text)

			current_text = cls._join_text_fragments(bucket_texts)
			duration = bucket[-1].end - bucket[0].start
			is_last = idx == len(ordered) - 1
			next_gap = 0.0
			next_starts_lower = False
			if not is_last:
				nxt = ordered[idx + 1]
				next_gap = max(0.0, nxt.start - seg.end)
				next_text = (nxt.text or "").lstrip()
				next_starts_lower = bool(next_text) and next_text[0].islower()

			should_flush = False
			if is_last:
				should_flush = True
			elif cls._SENTENCE_END_RE.search(current_text):
				should_flush = True
			elif next_gap >= silence_threshold and not next_starts_lower:
				should_flush = True

			if should_flush and bucket:
				merged.append(
					Segment(
						start=bucket[0].start,
						end=bucket[-1].end,
						text=current_text,
					)
				)
				bucket = []
				bucket_texts = []

		return merged

	# ── Nhận dạng ────────────────────────────────────────────────
	def _transcribe_faster(self, audio_path: Path, language: str | None) -> tuple[List[Segment], str]:
		segments, info = self.model.transcribe(str(audio_path), language=language, beam_size=5)
		# transcribe() trả về generator — phải duyệt hết thì mới thực sự chạy,
		# và info.language chỉ có giá trị thật (khác None) SAU khi duyệt xong
		# nếu language=None (tự nhận dạng dựa trên đoạn âm thanh đầu tiên).
		result = [
			Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
			for s in segments
			if (s.text or "").strip()
		]
		return result, (info.language or language or "en")

	def _transcribe_openai(self, audio_path: Path, language: str | None) -> tuple[List[Segment], str]:
		try:
			result = self.model.transcribe(str(audio_path), language=language, fp16=self.use_fp16)
		except RuntimeError as exc:
			msg = str(exc).lower()
			if self.device == "cuda" and (
				"out of memory" in msg or "unable to find an engine" in msg
			):
				print("[Whisper] CUDA failed while transcribing. Retrying on CPU.")
				try:
					if self._torch.cuda.is_available():
						self._torch.cuda.empty_cache()
				except Exception:
					pass
				self.device = "cpu"
				self.use_fp16 = False
				self.model = self._whisper.load_model(self.model_size, device=self.device)
				result = self.model.transcribe(str(audio_path), language=language, fp16=self.use_fp16)
			else:
				raise

		segments: List[Segment] = []
		for seg in result.get("segments", []):
			text = (seg.get("text") or "").strip()
			if not text:
				continue
			segments.append(
				Segment(
					start=float(seg.get("start", 0.0)),
					end=float(seg.get("end", 0.0)),
					text=text,
				)
			)
		return segments, (result.get("language") or language or "en")

	def transcribe(
		self,
		audio_path: str | Path,
		language: str | None = None,
		sentence_resegment: bool = True,
		silence_threshold: float = 0.45,
	) -> tuple[List[Segment], str]:
		"""Nhận dạng giọng nói. language=None (mặc định) là TỰ NHẬN DẠNG —
		Whisper đoán ngôn ngữ từ ~30 giây âm thanh đầu. Truyền mã ISO cụ thể
		("en", "ja"...) để ép ngôn ngữ khi tự nhận dạng đoán sai (video có
		nhạc nền dài, giọng không rõ ở đầu video...).

		Trả về (segments, ngôn_ngữ_đã_dùng) — ngôn ngữ này luôn là mã cụ thể,
		kể cả khi gọi bằng language=None, để lưu lại đúng ngôn ngữ thật của
		video thay vì giữ nguyên "auto"."""
		audio_path = Path(audio_path)
		if not audio_path.exists():
			raise FileNotFoundError(f"Audio not found: {audio_path}")

		if self.backend == "faster":
			segments, detected_language = self._transcribe_faster(audio_path, language)
		else:
			segments, detected_language = self._transcribe_openai(audio_path, language)

		if not sentence_resegment:
			return segments, detected_language

		merged = self._merge_segments_into_sentences(
			segments,
			silence_threshold=silence_threshold,
		)
		return merged, detected_language
