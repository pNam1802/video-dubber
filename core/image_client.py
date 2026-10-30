"""
core/image_client.py
Sinh ảnh minh hoạ bằng Z-Image-Turbo (Tongyi-MAI, Apache 2.0).

Model ~20 GB, cần GPU 24 GB+ nên trên Modal nó chạy ở container riêng
(modal_app.py::ImageGenerator); container đọc kịch bản chỉ gửi prompt sang và
nhận lại PNG. Chạy local (JOB_RUNNER=thread) thì nạp thẳng trong tiến trình
nếu đã `pip install diffusers` và có GPU đủ lớn.

File này KHÔNG import config/Flask ở đầu: container ImageGenerator chỉ có
torch + diffusers.
"""
from __future__ import annotations

import io
import threading
from typing import Any, List

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
# Khung 16:9 cho video; Z-Image cần cạnh chia hết cho 16.
WIDTH, HEIGHT = 1280, 720
# Turbo được chưng cất cho ~8 bước, không dùng CFG. Đo trên L40S (30/10/2026):
# ~3,2 giây/ảnh 1280x720.
STEPS = 9

_local_pipe: Any = None
_local_lock = threading.Lock()


def load_model() -> Any:
    import torch
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    return pipe.to("cuda")


def generate_with(pipe: Any, prompts: List[str], seed: int = 42) -> List[bytes]:
    """Vẽ từng prompt, trả PNG đúng thứ tự đầu vào. Cùng một seed cho mọi ảnh
    để bố cục và nét vẽ đồng đều hơn giữa các cảnh."""
    import torch

    out: List[bytes] = []
    for prompt in prompts:
        image = pipe(
            prompt=prompt, width=WIDTH, height=HEIGHT, num_inference_steps=STEPS, guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def generate(prompts: List[str]) -> List[bytes]:
    """Điểm vào cho pipeline: tự chọn chạy trên Modal hay tại chỗ."""
    from config.settings import JOB_RUNNER, MODAL_APP_NAME

    if JOB_RUNNER == "modal":
        import modal

        generator = modal.Cls.from_name(MODAL_APP_NAME, "ImageGenerator")()
        return generator.generate.remote(prompts)

    global _local_pipe
    with _local_lock:
        if _local_pipe is None:
            try:
                _local_pipe = load_model()
            except ImportError as exc:
                raise RuntimeError("Chưa cài diffusers để sinh ảnh: pip install diffusers") from exc
        return generate_with(_local_pipe, prompts)
