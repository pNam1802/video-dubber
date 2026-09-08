"""
core/translator/marian_translator.py
Dịch EN → VI bằng MarianMT (Helsinki-NLP) — chạy offline, không cần API key.
"""
from __future__ import annotations

from typing import List

import torch
from transformers import MarianMTModel, MarianTokenizer

from config.settings import MARIAN_MODEL_EN_VI, HF_TOKEN
from core.translator.base import BaseTranslator


def _is_cuda_runtime_failure(exc: BaseException) -> bool:
    """Detect common CUDA runtime/init failures that should trigger CPU fallback."""
    msg = str(exc).lower()
    cuda_markers = (
        "cuda error",
        "cublas_status_not_initialized",
        "cublascreate",
        "cublas",
        "cuda initialization",
        "unable to find an engine",
        "out of memory",
    )
    return any(marker in msg for marker in cuda_markers)


class MarianTranslator(BaseTranslator):
    """Dịch offline bằng Helsinki-NLP MarianMT (en → vi)."""

    def __init__(
        self,
        model_name: str = MARIAN_MODEL_EN_VI,
        device: str = "auto",
        batch_size: int = 8,
    ):
        super().__init__()
        requested_device = (device or "auto").strip().lower()
        if requested_device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}. Use one of: auto, cuda, cpu")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if requested_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                print("[MarianMT] CUDA requested but unavailable. Falling back to CPU.")
                resolved_device = "cpu"
            else:
                resolved_device = "cuda"
        else:
            resolved_device = "cpu"

        self.device = torch.device(resolved_device)
        self.hf_token = HF_TOKEN or None
        print(f"[MarianMT] Đang tải model '{model_name}'...")
        tokenizer_kwargs = {"token": self.hf_token} if self.hf_token else {}
        self.tokenizer = MarianTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model_kwargs = {"token": self.hf_token} if self.hf_token else {}
        if self.device.type == "cuda":
            # fp16 giúp giảm VRAM đáng kể trên GPU nhỏ.
            model_kwargs["torch_dtype"] = torch.float16
        # low_cpu_mem_usage=False tránh lỗi "meta tensor" trên một số phiên bản transformers
        model_kwargs["low_cpu_mem_usage"] = False
        try:
            self.model = MarianMTModel.from_pretrained(model_name, **model_kwargs)
            self.model.to(self.device)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if self.device.type != "cuda" or not _is_cuda_runtime_failure(exc):
                raise
            print("[MarianMT] CUDA lỗi khi nạp model. Falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.device = torch.device("cpu")
            cpu_model_kwargs = {"token": self.hf_token} if self.hf_token else {}
            cpu_model_kwargs["low_cpu_mem_usage"] = False
            self.model = MarianMTModel.from_pretrained(model_name, **cpu_model_kwargs)
            self.model.to(self.device)
        self.model.eval()
        self._model_name = model_name
        self.batch_size = batch_size
        print(f"[MarianMT] Tải xong trên thiết bị: {self.device}!")

    def _move_to_cpu_and_reload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.device = torch.device("cpu")
        cpu_model_kwargs = {"token": self.hf_token} if self.hf_token else {}
        cpu_model_kwargs["low_cpu_mem_usage"] = False
        self.model = MarianMTModel.from_pretrained(self._model_name, **cpu_model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "MarianMT (Helsinki-NLP, offline)"

    def translate_text(self, text: str) -> str:
        """Dịch một câu trực tiếp bằng MarianMT."""

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                translated_tokens = self.model.generate(**inputs)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if self.device.type == "cuda" and _is_cuda_runtime_failure(exc):
                print("[MarianMT] CUDA failed while translating. Retrying on CPU.")
                self._move_to_cpu_and_reload()
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    translated_tokens = self.model.generate(**inputs)
            else:
                raise
        translated = self.tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True,
        )
        return translated

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Dịch nhiều câu theo lô nhỏ để giảm peak RAM/VRAM."""
        if not texts:
            return []

        results: List[str] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            try:
                with torch.no_grad():
                    translated_tokens = self.model.generate(**inputs)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if self.device.type == "cuda" and _is_cuda_runtime_failure(exc):
                    print("[MarianMT] CUDA failed while batch translating. Retrying on CPU.")
                    self._move_to_cpu_and_reload()
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        translated_tokens = self.model.generate(**inputs)
                else:
                    raise

            for j, tokens in enumerate(translated_tokens):
                translated = self.tokenizer.decode(tokens, skip_special_tokens=True)
                results.append(translated)

        return results