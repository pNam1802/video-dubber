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

	def translate_segments(self, segments: List[Segment]) -> List[Segment]:
		texts = [seg.text for seg in segments]
		translated = self.translate_batch(texts)
		for seg, vi_text in zip(segments, translated):
			seg.translated = vi_text
		return segments

