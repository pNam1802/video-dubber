"""
core/explainer.py
Chế độ ảnh minh hoạ "explainer" của "Đọc kịch bản": video giải thích kiểu 3Blue1Brown,
vẽ bằng Manim (thư viện mã nguồn mở của 3Blue1Brown). Hiện chỉ admin dùng được.

  1. Gemini (lỗi thì DeepSeek) đọc cả kịch bản, chia thành cảnh ~8 giây và chọn cho mỗi
     cảnh MỘT bố cục có sẵn (tiêu đề, gạch đầu dòng, sơ đồ luồng, không gian vector, so
     sánh, hội thoại, câu chốt) kèm chữ hiện trên màn hình. LLM chỉ điền dữ liệu, không
     sinh code: hình vẽ cố định trong core/explainer_scene.py.
  2. Mỗi cảnh render bằng `manim` trong tiến trình riêng, dài đúng bằng đoạn lời đọc.
  3. Cắt/kéo từng cảnh cho khớp số khung, nối lại rồi ghép giọng đọc.

Chỉ dùng CPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from core.script_images import Slide, ask_gemini_json, group_by_duration, numbered_lines, tidy_starts
from utils.proc import run_command

SCENE_FILE = Path(__file__).with_name("explainer_scene.py")
SCENE_SEC = 8.0
MIN_SCENE_SEC = 3.0
MAX_SCENES = 80
FPS = 25
RESOLUTION = "1280,720"
RENDER_WORKERS = 2
FIELD_CHARS = 80

#: Bố cục -> các trường chữ (str) và danh sách (list[str]) của nó.
LAYOUTS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "title": (("title", "subtitle"), ()),
    "bullets": (("heading",), ("items",)),
    "flow": (("heading",), ("steps",)),
    "vectors": (("heading", "query"), ("items",)),
    "compare": (("heading", "left_title", "right_title"), ("left", "right")),
    "chat": (("question", "answer"), ()),
    "statement": (("text", "caption"), ()),
}
LIST_LIMITS = {"items": 6, "steps": 5, "left": 4, "right": 4}


def scene_count(timings: List[Tuple[float, float]]) -> int:
    return min(MAX_SCENES, max(1, round(timings[-1][1] / SCENE_SEC)))


def explainer_request(sentences: List[str], timings: List[Tuple[float, float]], count: int) -> str:
    return (
        "You are designing an animated explainer video in the style of 3Blue1Brown for a narrated script.\n"
        f"Split the narration into about {count} scenes (roughly {SCENE_SEC:.0f} seconds each), cutting only "
        "at the start of a sentence. For every scene pick the ONE layout that best visualises what is said "
        "during that scene:\n"
        '- "title": {"title", "subtitle"} — opening or a new chapter.\n'
        '- "bullets": {"heading", "items": 2-5 short phrases} — a list of points.\n'
        '- "flow": {"heading", "steps": 2-5 short step names in order} — a process, pipeline or sequence.\n'
        '- "vectors": {"heading", "query", "items": 3-6 short document names, "nearest": indexes (0-based) '
        "of the 1-3 items most similar to the query} — embeddings, similarity, semantic search.\n"
        '- "compare": {"heading", "left_title", "left": 1-4 phrases, "right_title", "right": 1-4 phrases} '
        "— before/after, problem/solution, A versus B.\n"
        '- "chat": {"question", "answer"} — a user asking and a system answering.\n'
        '- "statement": {"text", "caption"} — one key idea or conclusion.\n'
        "Rules: write all on-screen text in the SAME language as the script, short and concrete "
        "(items at most 40 characters, headings at most 60). Vary layouts; prefer flow/vectors/compare "
        "when the narration describes them. On-screen text summarises, never copies whole sentences.\n"
        'Return JSON: {"scenes": [{"start": <number of the sentence where the scene begins>, '
        '"layout": "...", ...fields}]}. The first scene starts at 1.\n\n'
        f"Script (numbered sentences with start time):\n{numbered_lines(sentences, timings)}"
    )


def _clean(value, limit: int = FIELD_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def clean_scene(raw: dict) -> dict | None:
    """Chỉ giữ các trường của bố cục đã chọn, cắt chữ dài. Bố cục lạ thì bỏ."""
    layout = str(raw.get("layout", "")).strip().lower()
    if layout not in LAYOUTS:
        return None
    texts, lists = LAYOUTS[layout]
    scene = {"layout": layout}
    for key in texts:
        if raw.get(key):
            scene[key] = _clean(raw[key], 160 if key in ("answer", "question", "text") else FIELD_CHARS)
    for key in lists:
        values = raw.get(key) if isinstance(raw.get(key), list) else []
        scene[key] = [_clean(v) for v in values if _clean(v)][:LIST_LIMITS[key]]
    if layout == "vectors":
        nearest = raw.get("nearest") if isinstance(raw.get("nearest"), list) else []
        scene["nearest"] = [int(i) for i in nearest if str(i).isdigit()]
    if lists and not any(scene[key] for key in lists):
        return None
    if not lists and not any(scene.get(key) for key in texts):
        return None
    return scene


def parse_scenes(raw: str, sentence_count: int) -> List[Tuple[int, dict]]:
    data = json.loads(raw)
    items = data.get("scenes", []) if isinstance(data, dict) else data
    board: Dict[int, dict] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", 0)) - 1
        except (TypeError, ValueError):
            continue
        scene = clean_scene(item)
        if scene is not None and 0 <= start < sentence_count and start not in board:
            board[start] = scene
    result = sorted(board.items())
    if result:
        result[0] = (0, result[0][1])
    return result


def fallback_board(sentences: List[str], timings: List[Tuple[float, float]]) -> List[Tuple[int, dict]]:
    """LLM lỗi: mỗi ~SCENE_SEC hiện câu mở đầu đoạn đó làm câu chốt."""
    starts = group_by_duration(timings, max(SCENE_SEC, timings[-1][1] / MAX_SCENES))
    return [(i, {"layout": "statement", "text": _clean(sentences[i], 120)}) for i in starts]


def plan(sentences: List[str], timings: List[Tuple[float, float]]) -> Tuple[List[dict], List[Slide]]:
    """(đặc tả từng cảnh, lịch chiếu)."""
    from core.stock_footage import ask_deepseek_json

    request = explainer_request(sentences, timings, scene_count(timings))
    board: List[Tuple[int, dict]] = []
    for name, ask in (("Gemini", ask_gemini_json), ("DeepSeek", ask_deepseek_json)):
        try:
            board = parse_scenes(ask(request), len(sentences))
            if board:
                break
            print(f"[Video giải thích] {name}: không có cảnh nào")
        except Exception as exc:
            print(f"[Video giải thích] {name}: {exc}")
    if not board:
        board = fallback_board(sentences, timings)
    starts = [start for start, _ in board]
    kept = tidy_starts(starts, timings, min_sec=MIN_SCENE_SEC, max_count=MAX_SCENES)
    total = timings[-1][1]
    bounds = [0.0] + [timings[starts[k]][0] for k in kept[1:]] + [total]
    return [board[k][1] for k in kept], [(i, bounds[i], bounds[i + 1]) for i in range(len(kept))]


# ── Render ───────────────────────────────────────────────────
def frame_counts(slides: List[Slide]) -> List[int]:
    """Làm tròn theo mốc tích luỹ để tổng số khung không trôi so với giọng đọc."""
    return [max(1, round(end * FPS) - round(start * FPS)) for _, start, end in slides]


def render_scene(scene: dict, frames: int, output: Path, work_dir: Path) -> None:
    """Render một cảnh bằng Manim rồi cắt/kéo (giữ khung cuối) cho đúng số khung."""
    name = output.stem
    spec = work_dir / f"{name}.json"
    spec.write_text(json.dumps({**scene, "duration": frames / FPS}, ensure_ascii=False), encoding="utf-8")
    media = work_dir / f"media_{name}"
    run_command(
        [sys.executable, "-m", "manim", "render", str(SCENE_FILE), "ExplainerScene", "--format", "mp4",
         "-r", RESOLUTION, "--fps", str(FPS), "--media_dir", str(media), "-o", name,
         "--disable_caching", "--progress_bar", "none", "-v", "WARNING"],
        timeout=600, error_message="Không vẽ được cảnh video giải thích.",
        env={**os.environ, "EXPLAINER_SPEC": str(spec)},
    )
    rendered = next(media.rglob(f"{name}.mp4"), None)
    if rendered is None:
        raise RuntimeError("Manim không tạo ra file video cho cảnh.")
    run_command(
        ["ffmpeg", "-v", "error", "-y", "-i", str(rendered), "-an",
         "-vf", f"tpad=stop_mode=clone:stop_duration=30,fps={FPS},format=yuv420p",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-r", str(FPS), str(output)],
        timeout=300, error_message="Không chuẩn hoá được cảnh video giải thích.",
    )


def make_video(parts: List[Path], audio: Path, output: Path, work_dir: Path) -> None:
    listing = work_dir / "scenes.txt"
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    run_command(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
         "-movflags", "+faststart", str(output)],
        timeout=1800, error_message="Không ghép được video giải thích với giọng đọc.",
    )


def render(sentences: List[str], timings: List[Tuple[float, float]], audio: Path, output: Path, work_dir: Path,
           progress=lambda percent, message: None) -> dict:
    """Toàn bộ chế độ "explainer". Trả về thời gian từng bước (giây)."""
    t0 = time.time()
    scenes, slides = plan(sentences, timings)
    t1 = time.time()

    progress(80, f"Đang vẽ {len(scenes)} cảnh giải thích bằng Manim...")
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = frame_counts(slides)
    parts = [work_dir / f"scene_{k:03d}.mp4" for k in range(len(scenes))]
    with ThreadPoolExecutor(RENDER_WORKERS) as pool:
        list(pool.map(lambda k: render_scene(scenes[k], counts[k], parts[k], work_dir), range(len(scenes))))
    t2 = time.time()

    progress(95, "Đang ghép các cảnh với giọng đọc...")
    make_video(parts, audio, output, work_dir)
    # Không trả khoá "images": app/jobs.py tính khoá đó theo giá GPU, còn đây chỉ là CPU.
    return {"image_prompts": round(t1 - t0, 2), "scenes": round(t2 - t1, 2),
            "slideshow": round(time.time() - t2, 2)}
