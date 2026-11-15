from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.transcriber import Segment


class BaseTranslator(ABC):
	def __init__(self) -> None:
		# So token THAT SU da dung qua API — engine khong goi API (MarianMT
		# chay local) thi giu nguyen 0. Cac subclass tu ghi lai bang
		# _record_usage() sau moi lan goi. Dung de tinh chi phi dich that,
		# xem core/translator/llm_common.estimate_llm_cost().
		self.usage_prompt_tokens: int = 0
		self.usage_completion_tokens: int = 0
		#: Kiểu dịch (llm_common.TRANSLATION_STYLES) và tóm tắt + bảng từ của
		#: bước đọc hiểu. notes gán sẵn từ ngoài (vd dịch lại một câu ở bước
		#: duyệt, đọc lại từ DB) thì không đọc hiểu lại.
		self.style: str = "natural"
		self.notes: str = ""

	def _record_usage(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
		self.usage_prompt_tokens += prompt_tokens or 0
		self.usage_completion_tokens += completion_tokens or 0

	@property
	@abstractmethod
	def name(self) -> str:
		raise NotImplementedError

	@abstractmethod
	def translate_text(self, text: str) -> str:
		raise NotImplementedError

	def translate_batch(self, texts: List[str]) -> List[str]:
		return [self.translate_text(t) for t in texts]

	def understand(self, texts: List[str]) -> str:
		"""Đọc cả transcript trước khi dịch, trả về tóm tắt + bảng từ (xem
		llm_common.build_understanding_prompt). Engine không phải LLM thì
		không làm gì."""
		return ""

	def translate_segments(self, segments: List[Segment]) -> List[Segment]:
		texts = [seg.text for seg in segments]
		if not self.notes:
			# Bước phụ: hỏng thì vẫn dịch như cũ, không làm hỏng cả job.
			try:
				self.notes = self.understand(texts) or ""
			except Exception as exc:
				print(f"[{self.name}] Bước đọc hiểu lỗi, dịch không kèm bảng từ: {exc}")
				self.notes = ""
			if self.notes:
				print(f"[{self.name}] Đọc hiểu xong ({len(self.notes)} ký tự):\n{self.notes}")
		translated = self.translate_batch(texts)
		for seg, vi_text in zip(segments, translated):
			seg.translated = vi_text
		return segments

