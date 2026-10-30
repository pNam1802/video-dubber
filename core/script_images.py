"""
core/script_images.py
Ảnh minh hoạ cho "Đọc kịch bản": chia giọng đọc thành từng cảnh, nhờ Gemini
viết prompt vẽ cho mỗi cảnh, rồi ghép ảnh + mp3 thành một video slideshow.

Hai chế độ:
  "few"    — 3-5 ảnh cho cả kịch bản, chiếu lần lượt rồi lặp lại từ đầu.
  "scenes" — mỗi cảnh (~12 giây giọng đọc) một ảnh riêng.

Ảnh chỉ đổi ở ranh giới câu, không bao giờ đổi giữa chừng một câu.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

from utils.proc import run_command

MODES = ("none", "few", "scenes")
SCENE_SEC = 12.0
SLIDE_SEC = 8.0
# Trần số ảnh của chế độ "scenes": kịch bản dài thì giãn cảnh ra thay vì vẽ
# thêm — mỗi ảnh ~3 giây GPU L40S, cộng thêm thời gian nạp model.
MAX_SCENES = 30
FEW_MIN, FEW_MAX = 3, 5
DEFAULT_STYLE = ("flat vector illustration, clean shapes, soft pastel colors, "
                 "educational explainer video style")
# Model ảnh hay vẽ chữ méo mó — luôn dặn không vẽ chữ, kể cả khi Gemini quên.
NO_TEXT = "no text, no letters, no watermark"
# Gửi Gemini tối đa bấy nhiêu ký tự cho mỗi phần, đủ để nắm ý chính.
MAX_CHUNK_CHARS = 1500

#: (chỉ số ảnh, bắt đầu, kết thúc) — giây, trong file âm thanh kết quả.
Slide = Tuple[int, float, float]


def group_by_duration(timings: List[Tuple[float, float]], target_sec: float) -> List[int]:
    """Chia các câu liền nhau thành nhóm dài khoảng target_sec, chỉ cắt ở
    đầu câu. Trả về chỉ số câu mở đầu mỗi nhóm."""
    starts = [0]
    group_start = timings[0][0]
    for i in range(1, len(timings)):
        if timings[i][0] - group_start >= target_sec:
            starts.append(i)
            group_start = timings[i][0]
    return starts


def _slides(starts: List[int], timings: List[Tuple[float, float]], total: float,
            image_of: List[int]) -> List[Slide]:
    """Nhóm câu -> khoảng chiếu liền mạch từ 0 tới hết file, kể cả quãng nghỉ."""
    bounds = [0.0] + [timings[i][0] for i in starts[1:]] + [total]
    return [(image_of[k], bounds[k], bounds[k + 1]) for k in range(len(starts))]


def plan(timings: List[Tuple[float, float]], total: float, mode: str) -> Tuple[List[int], List[Slide]]:
    """Trả về (chỉ số câu mở đầu mỗi phần cần một ảnh, lịch chiếu ảnh)."""
    if mode == "scenes":
        starts = group_by_duration(timings, max(SCENE_SEC, total / MAX_SCENES))
        return starts, _slides(starts, timings, total, list(range(len(starts))))

    count = min(FEW_MAX, max(FEW_MIN, math.ceil(total / 60)), len(timings))
    chunk_starts = group_by_duration(timings, total / count)[:count]
    slide_starts = group_by_duration(timings, SLIDE_SEC)
    image_of = [k % len(chunk_starts) for k in range(len(slide_starts))]
    return chunk_starts, _slides(slide_starts, timings, total, image_of)


def chunk_texts(sentences: List[str], starts: List[int]) -> List[str]:
    bounds = starts + [len(sentences)]
    return [" ".join(sentences[bounds[k]:bounds[k + 1]])[:MAX_CHUNK_CHARS] for k in range(len(starts))]


def _prompt_request(chunks: List[str]) -> str:
    parts = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunks))
    return (
        "You are an illustrator planning still images for a narrated explainer video.\n"
        f"Below are {len(chunks)} consecutive parts of the narration (it may be in Vietnamese or English).\n"
        "Write one English text-to-image prompt per part that depicts its main idea visually: "
        "concrete subjects, setting and composition, 1-3 sentences. "
        "Abstract ideas should become simple visual metaphors. "
        "Never ask for text, letters, numbers, logos or captions in the image. "
        "If a recurring character fits the story, describe it with the exact same words in every prompt.\n"
        "Also write one short shared art-style line used for every image so they look consistent.\n"
        f'Return JSON only: {{"style": "...", "prompts": ["...", ...]}} with exactly {len(chunks)} prompts.\n\n'
        f"{parts}"
    )


def parse_prompts(raw: str, chunks: List[str]) -> List[str]:
    """JSON của Gemini -> prompt hoàn chỉnh cho từng phần. Thiếu prompt nào
    thì dùng chính lời đọc của phần đó (model ảnh hiểu được cả tiếng Việt)."""
    data = json.loads(raw)
    style = str(data.get("style") or "").strip() or DEFAULT_STYLE
    prompts = [str(p).strip() for p in (data.get("prompts") or []) if str(p).strip()]
    prompts += [text[:300] for text in chunks[len(prompts):]]
    return [f"{prompt}, {style}, {NO_TEXT}" for prompt in prompts[:len(chunks)]]


def fallback_prompts(chunks: List[str]) -> List[str]:
    return [f"{text[:300]}, {DEFAULT_STYLE}, {NO_TEXT}" for text in chunks]


def write_prompts(chunks: List[str]) -> List[str]:
    """Nhờ Gemini viết prompt vẽ; không có key hoặc Gemini lỗi thì vẽ thẳng
    từ lời đọc — ảnh kém sát ý hơn nhưng job không hỏng."""
    from config.settings import GEMINI_API_KEY, GEMINI_MODEL

    if not GEMINI_API_KEY:
        return fallback_prompts(chunks)
    try:
        from google import genai
        from google.genai import types

        from core.translator.llm_common import call_with_retry

        client = genai.Client(api_key=GEMINI_API_KEY)
        config = types.GenerateContentConfig(temperature=0.7, response_mime_type="application/json")
        response = call_with_retry(
            lambda: client.models.generate_content(model=GEMINI_MODEL, contents=_prompt_request(chunks),
                                                   config=config),
            what="Gemini (prompt ảnh)",
        )
        return parse_prompts(response.text or "", chunks)
    except Exception as exc:
        print(f"[Ảnh minh hoạ] Gemini không viết được prompt ({exc}), vẽ thẳng từ lời đọc.")
        return fallback_prompts(chunks)


def make_slideshow(images: List[Path], slides: List[Slide], audio: Path, output: Path) -> None:
    """Ghép ảnh theo lịch chiếu với giọng đọc thành mp4."""
    listing = output.with_suffix(".txt")
    lines = []
    for image_idx, start, end in slides:
        lines += [f"file '{images[image_idx].resolve().as_posix()}'", f"duration {end - start:.3f}"]
    # concat demuxer bỏ qua duration của dòng cuối nếu không lặp lại file đó.
    lines.append(f"file '{images[slides[-1][0]].resolve().as_posix()}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run_command(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
             "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
             "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(output)],
            timeout=1800, error_message="Không ghép được ảnh minh hoạ thành video.",
        )
    finally:
        listing.unlink(missing_ok=True)


def illustrate(sentences: List[str], timings: List[Tuple[float, float]], audio: Path, output: Path,
               work_dir: Path, mode: str, progress=lambda percent, message: None) -> dict:
    """Toàn bộ bước ảnh minh hoạ. Trả về thời gian từng bước (giây)."""
    import time

    from core import image_client

    t0 = time.time()
    starts, slides = plan(timings, timings[-1][1], mode)
    chunks = chunk_texts(sentences, starts)

    progress(75, f"Đang lên ý tưởng cho {len(chunks)} ảnh minh hoạ...")
    prompts = write_prompts(chunks)
    t1 = time.time()

    progress(80, f"Đang vẽ {len(prompts)} ảnh minh hoạ...")
    work_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, png in enumerate(image_client.generate(prompts)):
        path = work_dir / f"image_{i:02d}.png"
        path.write_bytes(png)
        images.append(path)
    t2 = time.time()

    progress(95, "Đang ghép ảnh và giọng đọc thành video...")
    make_slideshow(images, slides, audio, output)
    return {"image_prompts": round(t1 - t0, 2), "images": round(t2 - t1, 2), "slideshow": round(time.time() - t2, 2)}
