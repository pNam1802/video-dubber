"""
core/recap.py
Chế độ "Manhwa recap" của ảnh minh hoạ, theo kiểu các kênh kể truyện manhwa:
ảnh webtoon MÀU, mỗi ảnh một khung dọc đặt giữa màn hình, hai bên là chính
ảnh đó phóng to + làm mờ, đổi ảnh nhanh (~RECAP_SEC giây) và zoom chậm cho
khỏi đứng hình.

Giống core/manga.py, Gemini lập danh sách nhân vật và bối cảnh với mô tả cố
định (code chèn nguyên văn vào mọi ảnh) rồi chia lời đọc thành từng cảnh.
Khác manga: có màu, nên mô tả nhân vật GỒM màu tóc/quần áo — đó là thứ giữ
nhân vật nhận ra được giữa các ảnh.
"""
from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.manga import _bible, _sentence
from core.script_images import NO_TEXT, Slide, ask_gemini_json, group_by_duration, numbered_lines, tidy_starts
from utils.proc import run_command

RECAP_SEC = 6.0
MIN_SHOT_SEC = 3.0
# 20.000 ký tự kịch bản ~ 20 phút đọc. Mỗi ảnh ~2,6 giây GPU L40S ở khổ
# DRAW_SIZE, 150 ảnh ~7 phút — dưới timeout 30 phút của ImageGenerator.
MAX_SHOTS = 150
MAX_CHARACTERS, MAX_LOCATIONS = 4, 6
# Khung dọc 4:5, cạnh chia hết cho 16 (Z-Image).
DRAW_SIZE = (768, 960)
FRAME_W, FRAME_H = 1280, 720
FPS = 25
# Zoom từ 1.0 lên chừng này trong suốt thời gian một ảnh.
ZOOM_TO = 1.06
# Nền hai bên: ảnh phủ kín khung, làm mờ và tối bớt để khung giữa nổi lên.
BLUR_RADIUS = 28
BACKGROUND_DIM = 0.55
ENCODE_WORKERS = 4

# Kiểu "anime key visual" (ánh ngược, hạt sáng, mắt long lanh) — chọn sau khi
# vẽ thử A/B trên Z-Image ngày 05/11/2026 theo ảnh mẫu khách đưa. "anime face,
# not photorealistic": thiếu nó thì cận mặt hay ngả sang ảnh chụp.
STYLE = ("stunning anime key visual, full color, highly detailed painterly digital illustration, "
         "delicate thin line art, anime face, not photorealistic, "
         "dramatic backlighting with strong rim light, glowing translucent hair strands, "
         "intense bloom and soft lens flare, floating golden light particles and sparkles, "
         "crystal-clear glossy eyes with detailed iris reflections, glistening highlights, "
         "rich saturated warm color grading, soft dreamy depth of field, emotional cinematic composition, "
         "masterpiece, best quality")
PANEL_NO_TEXT = f"{NO_TEXT}, no speech bubbles"
SHOTS = ("wide establishing shot", "full body shot", "medium shot", "close-up", "extreme close-up",
         "low angle shot", "high angle shot", "over the shoulder shot", "dynamic action shot")


# ── Storyboard ───────────────────────────────────────────────
def recap_request(sentences: List[str], timings: List[Tuple[float, float]], count: int) -> str:
    return (
        "You are the editor of a manhwa recap video: a narrator retells a Korean webtoon while full color webtoon "
        "panels change every few seconds.\n"
        "The narration is below, one numbered sentence per line with its start time. It may be in Vietnamese "
        "or English. Read the WHOLE script first and understand the story, then answer in JSON only:\n"
        '{"characters": [{"id": "...", "look": "..."}], "locations": [{"id": "...", "look": "..."}], '
        '"shots": [{"start": 1, "shot": "...", "characters": ["..."], "location": "...", "action": "..."}]}\n\n'
        "Rules:\n"
        f"- characters: at most {MAX_CHARACTERS} recurring characters who are drawn, each with a short id and a "
        "fixed English appearance INCLUDING colors (gender, age, build, hair color and style, eye color, "
        "clothing with colors, distinctive features). Make them easy to tell apart. The look is reused word for "
        "word in every image.\n"
        f"- locations: at most {MAX_LOCATIONS} places, each with an id and a fixed English description with "
        "lighting and colors, reused in every image set there.\n"
        f"- shots: about {count} shots, one every 4 to 8 seconds of narration, following the story beat by "
        "beat. \"start\" is the sentence number where the shot begins; the first shot starts at sentence 1 and "
        "starts increase.\n"
        f"- shot: one of {', '.join(SHOTS)}. Vary it like a webtoon: close-ups for emotions and reactions, wide "
        "shots when the place changes, dynamic action shots for fights.\n"
        "- characters: ids of the 0 to 2 characters visible. Never more than two people, never unnamed crowds. "
        "Two characters must never touch or fight in the same image: split an attack into the attacker striking "
        "and the target reacting.\n"
        "- location: a location id, or null for a close-up on a blurred or abstract background.\n"
        "- action: what is visible: pose, expression, key objects. Refer to characters by id and do not repeat "
        "their appearance. No art style.\n"
        "- Draw only what happens at that moment of the story, never what is merely mentioned in dialogue, "
        "thoughts or memories. Never replace a person or event with an unrelated object or metaphor.\n"
        "- Nothing written: no text, letters, numbers, logos, captions or speech bubbles.\n\n"
        f"{numbered_lines(sentences, timings)}"
    )


def compose_shot(shot: dict, characters: Dict[str, str], locations: Dict[str, str]) -> str:
    kind = str(shot.get("shot") or "").strip().lower()
    parts = [kind] if kind in SHOTS else []
    parts.append(str(shot["action"]).strip().rstrip(". "))

    ids = [str(c).strip().lower() for c in shot.get("characters") or [] if isinstance(c, str)]
    cast = [(i, characters[i]) for i in dict.fromkeys(ids) if i in characters][:2]
    if len(cast) == 1:
        parts += [f"{cast[0][0]}: {cast[0][1]}", "solo, single character"]
    elif len(cast) == 2:
        parts += [f"two characters only, {cast[0][0]} on the left: {cast[0][1]}",
                  f"{cast[1][0]} on the right: {cast[1][1]}"]

    location = locations.get(str(shot.get("location") or "").strip().lower())
    parts.append(f"background: {location}" if location else "softly blurred background")
    return ", ".join(parts + [STYLE, PANEL_NO_TEXT])


def parse_recap(raw: str, sentence_count: int) -> List[Tuple[int, str]]:
    """JSON của Gemini -> [(chỉ số câu mở đầu, prompt)] theo thứ tự câu.
    Ảnh hỏng (thiếu action) bị bỏ; ảnh đầu luôn phủ từ câu 0."""
    data = json.loads(raw)
    characters = _bible(data.get("characters"))
    locations = _bible(data.get("locations"))
    shots: Dict[int, str] = {}
    for shot in data.get("shots") or []:
        if not isinstance(shot, dict) or not str(shot.get("action") or "").strip():
            continue
        start = _sentence(shot.get("start"), -1, sentence_count)
        if start >= 0:
            shots.setdefault(start, compose_shot(shot, characters, locations))
    board = sorted(shots.items())
    if board:
        board[0] = (0, board[0][1])
    return board


def shot_count(timings: List[Tuple[float, float]]) -> int:
    return min(MAX_SHOTS, max(1, round(timings[-1][1] / RECAP_SEC)))


def fallback_board(sentences: List[str], timings: List[Tuple[float, float]]) -> List[Tuple[int, str]]:
    """Không có storyboard: chia đều theo thời lượng, vẽ thẳng từ lời đọc."""
    starts = group_by_duration(timings, max(RECAP_SEC, timings[-1][1] / MAX_SHOTS))
    bounds = starts + [len(sentences)]
    return [(starts[k], f"{' '.join(sentences[bounds[k]:bounds[k + 1]])[:300]}, {STYLE}, {PANEL_NO_TEXT}")
            for k in range(len(starts))]


def plan(sentences: List[str], timings: List[Tuple[float, float]]) -> Tuple[List[str], List[Slide]]:
    """(prompt cho từng ảnh, lịch chiếu). Gemini lỗi thì vẽ thẳng từ lời đọc."""
    try:
        board = parse_recap(ask_gemini_json(recap_request(sentences, timings, shot_count(timings))),
                            len(sentences))
        if not board:
            raise RuntimeError("storyboard rỗng")
    except Exception as exc:
        print(f"[Ảnh minh hoạ] Không dựng được storyboard recap ({exc}), vẽ thẳng từ lời đọc.")
        board = fallback_board(sentences, timings)
    starts = [start for start, _ in board]
    kept = tidy_starts(starts, timings, min_sec=MIN_SHOT_SEC, max_count=MAX_SHOTS)
    total = timings[-1][1]
    bounds = [0.0] + [timings[starts[k]][0] for k in kept[1:]] + [total]
    return [board[k][1] for k in kept], [(i, bounds[i], bounds[i + 1]) for i in range(len(kept))]


# ── Dựng khung + video ───────────────────────────────────────
def compose_frame(panel: bytes, path: Path) -> None:
    """Khung dọc ở giữa, hai bên là chính ảnh đó phủ kín, mờ và tối."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    img = Image.open(io.BytesIO(panel)).convert("RGB")
    frame = ImageOps.fit(img, (FRAME_W, FRAME_H)).filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    frame = ImageEnhance.Brightness(frame).enhance(BACKGROUND_DIM)
    front = ImageOps.contain(img, (FRAME_W, FRAME_H))
    frame.paste(front, ((FRAME_W - front.width) // 2, (FRAME_H - front.height) // 2))
    frame.save(path)


def clip_frames(slides: List[Slide]) -> List[int]:
    """Số khung hình của từng ảnh, làm tròn theo mốc tích luỹ để tổng không trôi."""
    return [max(1, round(end * FPS) - round(start * FPS)) for _, start, end in slides]


def zoom_filter(frames: int) -> str:
    # Phóng to gấp đôi trước khi zoompan để zoom chậm không bị giật từng pixel.
    step = (ZOOM_TO - 1) / max(1, frames - 1)
    return (f"scale={FRAME_W * 2}:{FRAME_H * 2},"
            f"zoompan=z='1+{step:.6f}*on':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={FRAME_W}x{FRAME_H}:fps={FPS},format=yuv420p")


def encode_clip(frame: Path, frames: int, output: Path) -> None:
    run_command(
        ["ffmpeg", "-v", "error", "-y", "-i", str(frame), "-vf", zoom_filter(frames), "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2", "-r", str(FPS), str(output)],
        timeout=300, error_message="Không dựng được đoạn video từ ảnh minh hoạ.",
    )


def make_video(frames_png: List[Path], slides: List[Slide], audio: Path, output: Path, work_dir: Path) -> None:
    """Mỗi ảnh một đoạn zoom chậm, nối lại (không mã hoá lại) rồi ghép giọng đọc."""
    counts = clip_frames(slides)
    clips = [work_dir / f"clip_{k:03d}.mp4" for k in range(len(slides))]
    with ThreadPoolExecutor(ENCODE_WORKERS) as pool:
        list(pool.map(lambda k: encode_clip(frames_png[slides[k][0]], counts[k], clips[k]), range(len(slides))))
    listing = work_dir / "clips.txt"
    listing.write_text("".join(f"file '{clip.resolve().as_posix()}'\n" for clip in clips), encoding="utf-8")
    run_command(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
         "-movflags", "+faststart", str(output)],
        timeout=1800, error_message="Không ghép được ảnh minh hoạ thành video.",
    )


def render(sentences: List[str], timings: List[Tuple[float, float]], audio: Path, output: Path, work_dir: Path,
           progress=lambda percent, message: None) -> dict:
    """Toàn bộ chế độ recap. Trả về thời gian từng bước (giây)."""
    import time

    from core import image_client

    t0 = time.time()
    prompts, slides = plan(sentences, timings)
    t1 = time.time()

    progress(80, f"Đang vẽ {len(prompts)} ảnh manhwa...")
    drawn = image_client.generate(prompts, [DRAW_SIZE] * len(prompts))
    t2 = time.time()

    progress(92, "Đang dựng video kiểu recap...")
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_png = []
    for i, png in enumerate(drawn):
        path = work_dir / f"frame_{i:03d}.png"
        compose_frame(png, path)
        frames_png.append(path)
    make_video(frames_png, slides, audio, output, work_dir)
    return {"image_prompts": round(t1 - t0, 2), "images": round(t2 - t1, 2),
            "slideshow": round(time.time() - t2, 2)}
