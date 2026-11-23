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

#: Bố cục -> các trường chữ (str) và danh sách chữ (list[str]) của nó. Trường có cấu trúc
#: khác (số, danh sách dict) do clean_scene() xử lý riêng.
LAYOUTS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "title": (("title", "subtitle"), ()),
    "bullets": (("heading",), ("items",)),
    "flow": (("heading",), ("steps",)),
    "vectors": (("heading", "query"), ("items",)),
    "compare": (("heading", "left_title", "right_title"), ("left", "right")),
    "chat": (("question", "answer"), ()),
    "statement": (("text", "caption"), ()),
    "tokens": (("heading", "text"), ("tokens",)),
    "probabilities": (("heading", "context"), ()),
    "attention": (("heading", "caption"), ("tokens",)),
    "network": (("heading", "input_label", "output_label"), ()),
    "code": (("heading", "language"), ()),
    "timeline": (("heading",), ()),
    "number": (("heading", "unit", "caption"), ()),
}
LIST_LIMITS = {"items": 6, "steps": 5, "left": 4, "right": 4, "tokens": 12}
#: Bố cục chủ yếu là chữ: không để quá nhiều, không đứng liền nhau.
TEXT_LAYOUTS = frozenset({"title", "bullets", "statement"})


def scene_count(timings: List[Tuple[float, float]]) -> int:
    return min(MAX_SCENES, max(1, round(timings[-1][1] / SCENE_SEC)))


def explainer_request(sentences: List[str], timings: List[Tuple[float, float]], count: int) -> str:
    return (
        "You are designing an animated explainer video in the style of 3Blue1Brown for a narrated script.\n"
        f"Split the narration into about {count} scenes (roughly {SCENE_SEC:.0f} seconds each), cutting only "
        "at the start of a sentence. For every scene pick the ONE layout that best SHOWS (not tells) what is "
        "said during that scene:\n"
        '- "title": {"title", "subtitle"} — opening or a new chapter.\n'
        '- "bullets": {"heading", "items": 2-5 short phrases} — a list of points.\n'
        '- "flow": {"heading", "steps": 2-5 short step names in order} — a process, pipeline or sequence.\n'
        '- "vectors": {"heading", "query", "items": 3-6 short document names, "nearest": indexes (0-based) '
        "of the 1-3 items most similar to the query} — embeddings, similarity, semantic search.\n"
        '- "compare": {"heading", "left_title", "left": 1-4 phrases, "right_title", "right": 1-4 phrases} '
        "— before/after, problem/solution, A versus B.\n"
        '- "chat": {"question", "answer"} — a user asking and a system answering.\n'
        '- "statement": {"text", "caption"} — one key idea or conclusion.\n'
        '- "tokens": {"heading", "text": a short sentence, "tokens": how a tokenizer splits it (2-12 pieces, '
        "keep leading spaces)} — tokenization, how a model reads text.\n"
        '- "probabilities": {"heading", "context": an unfinished sentence, "options": [{"label": next word, '
        '"p": probability 0-1}] 3-6 options} — next-token prediction, sampling, choosing among outcomes.\n'
        '- "attention": {"heading", "tokens": 3-8 words of a sentence, "focus": index of the word that looks '
        'around, "attends": indexes it pays most attention to, "caption"} — attention, context, which words '
        "relate.\n"
        '- "network": {"heading", "layers": node counts per layer e.g. [3,5,5,2], "input_label", '
        '"output_label"} — neural networks, layers, training, signals flowing through a model.\n'
        '- "code": {"heading", "language", "lines": 2-8 short lines of real code, commands or API calls} — '
        "anything a developer would type or run.\n"
        '- "timeline": {"heading", "events": [{"when", "label"}] 3-6 events in order} — history, versions, '
        "steps over time.\n"
        '- "number": {"heading", "value": a number, "unit", "caption"} — one striking figure, size, cost, '
        "speed.\n"
        "Rules: this is a VISUAL video. Prefer the drawn layouts (tokens, probabilities, attention, network, "
        "vectors, flow, code, timeline, number, compare, chat); text-only layouts (title, bullets, statement) "
        "at most one scene in four and never two in a row. Write all on-screen text in the SAME language as "
        "the script, at most 6 words per item and 50 characters per heading — on-screen text labels the "
        "picture, the voice explains it; never copy whole sentences.\n"
        'Return JSON: {"scenes": [{"start": <number of the sentence where the scene begins>, '
        '"layout": "...", ...fields}]}. The first scene starts at 1.\n\n'
        f"Script (numbered sentences with start time):\n{numbered_lines(sentences, timings)}"
    )


def _clean(value, limit: int = FIELD_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _int_list(value) -> List[int]:
    return [int(i) for i in value if str(i).isdigit()] if isinstance(value, list) else []


def _number(value) -> float | None:
    try:
        number = float(str(value).replace(",", ".").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) < 1e15 else None  # loại NaN


def _token(value) -> str:
    """Token giữ nguyên khoảng trắng đầu (" mèo") — đó chính là điều cảnh tokens muốn cho thấy."""
    text = str(value or "").replace("\n", " ")
    return text[:20] if text.strip() else ""


def clean_scene(raw: dict) -> dict | None:
    """Chỉ giữ các trường của bố cục đã chọn, cắt chữ dài. Bố cục lạ hoặc thiếu dữ liệu thì bỏ."""
    layout = str(raw.get("layout", "")).strip().lower()
    if layout not in LAYOUTS:
        return None
    texts, lists = LAYOUTS[layout]
    scene: dict = {"layout": layout}
    for key in texts:
        if raw.get(key):
            scene[key] = _clean(raw[key], 160 if key in ("answer", "question", "text", "context") else FIELD_CHARS)
    for key in lists:
        values = raw.get(key) if isinstance(raw.get(key), list) else []
        clean = _token if key == "tokens" else _clean
        scene[key] = [clean(v) for v in values if clean(v)][:LIST_LIMITS[key]]

    if layout == "vectors":
        scene["nearest"] = _int_list(raw.get("nearest"))
    elif layout == "attention":
        focus = _number(raw.get("focus"))
        scene["focus"] = int(focus) if focus is not None and focus >= 0 else len(scene["tokens"]) - 1
        scene["attends"] = _int_list(raw.get("attends"))
        if len(scene["tokens"]) < 2:
            return None
    elif layout == "probabilities":
        options = []
        for opt in raw.get("options", []) if isinstance(raw.get("options"), list) else []:
            p = _number(opt.get("p")) if isinstance(opt, dict) else None
            if p is not None and p >= 0 and _clean(opt.get("label"), 30):
                options.append({"label": _clean(opt["label"], 30), "p": p})
        total = sum(o["p"] for o in options)
        if not options or total <= 0:
            return None
        # LLM lúc trả 0-1, lúc trả phần trăm: >1.5 thì hiểu là phần trăm.
        scale = 100.0 if total > 1.5 else 1.0
        scene["options"] = [{"label": o["label"], "p": min(1.0, o["p"] / scale)} for o in options[:6]]
    elif layout == "network":
        scene["layers"] = [max(1, min(6, n)) for n in _int_list(raw.get("layers"))][:5] or [3, 5, 5, 2]
    elif layout == "code":
        lines = raw.get("lines") if isinstance(raw.get("lines"), list) else str(raw.get("lines") or "").splitlines()
        scene["lines"] = [str(line).rstrip()[:70] for line in lines if str(line).strip()][:10]
        if not scene["lines"]:
            return None
    elif layout == "timeline":
        events = []
        for ev in raw.get("events", []) if isinstance(raw.get("events"), list) else []:
            if isinstance(ev, dict) and (_clean(ev.get("label")) or _clean(ev.get("when"))):
                events.append({"when": _clean(ev.get("when"), 20), "label": _clean(ev.get("label"), 40)})
        if not events:
            return None
        scene["events"] = events[:6]
    elif layout == "number":
        value = _number(raw.get("value"))
        if value is None:
            return None
        scene["value"] = value

    if lists and not any(scene[key] for key in lists):
        return None
    if not lists and layout in TEXT_LAYOUTS | {"chat", "compare"} and not any(scene.get(k) for k in texts):
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


#: Tốc độ đọc ước lượng (ký tự/giây) để phân cảnh TRƯỚC khi có giọng đọc thật.
CHARS_PER_SEC = {"vi": 14.0, "en": 15.0}


def estimate_timings(sentences: List[str], language: str = "vi") -> List[Tuple[float, float]]:
    """Mốc thời gian ước lượng theo độ dài câu (kèm khoảng nghỉ), dùng cho bước duyệt phân cảnh."""
    rate, t, out = CHARS_PER_SEC.get(language, 14.0), 0.0, []
    for sentence in sentences:
        length = max(1.0, len(sentence) / rate)
        out.append((round(t, 2), round(t + length, 2)))
        t += length + 0.35
    return out


def write_board(sentences: List[str], timings: List[Tuple[float, float]]) -> List[Tuple[int, dict]]:
    """Gemini (lỗi thì DeepSeek) phân cảnh; cả hai lỗi thì mỗi ~8 giây một câu chốt."""
    from core.stock_footage import ask_deepseek_json

    request = explainer_request(sentences, timings, scene_count(timings))
    for name, ask in (("Gemini", ask_gemini_json), ("DeepSeek", ask_deepseek_json)):
        try:
            board = parse_scenes(ask(request), len(sentences))
            if board:
                return board
            print(f"[Video giải thích] {name}: không có cảnh nào")
        except Exception as exc:
            print(f"[Video giải thích] {name}: {exc}")
    return fallback_board(sentences, timings)


def board_from_list(items, sentence_count: int) -> List[Tuple[int, dict]]:
    """Bảng phân cảnh đã duyệt (danh sách {"start": chỉ số câu từ 0, "layout", ...}) -> board.
    Cảnh lỗi bị bỏ; cảnh đầu luôn phủ từ câu đầu tiên."""
    board: Dict[int, dict] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", -1))
        except (TypeError, ValueError):
            continue
        scene = clean_scene(item)
        if scene is not None and 0 <= start < sentence_count:
            board[start] = scene
    result = sorted(board.items())
    if result:
        result[0] = (0, result[0][1])
    return result


def board_to_list(board: List[Tuple[int, dict]]) -> List[dict]:
    return [{"start": start, **scene} for start, scene in board]


def plan(sentences: List[str], timings: List[Tuple[float, float]],
         approved: List[dict] | None = None) -> Tuple[List[dict], List[Slide]]:
    """(đặc tả từng cảnh, lịch chiếu). Có bảng phân cảnh đã duyệt thì dùng đúng bảng đó."""
    board = board_from_list(approved, len(sentences)) if approved else []
    if not board:
        board = write_board(sentences, timings)
    starts = [start for start, _ in board]
    kept = tidy_starts(starts, timings, min_sec=MIN_SCENE_SEC, max_count=MAX_SCENES)
    total = timings[-1][1]
    bounds = [0.0] + [timings[starts[k]][0] for k in kept[1:]] + [total]
    return [board[k][1] for k in kept], [(i, bounds[i], bounds[i + 1]) for i in range(len(kept))]


def scene_beats(timings: List[Tuple[float, float]], start: float, end: float) -> List[float]:
    """Mốc bắt đầu các câu nằm trong cảnh, tính từ đầu cảnh — để hình hiện đúng lúc được nói tới."""
    return [round(t0 - start, 3) for t0, _ in timings if start < t0 < end]


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
           progress=lambda percent, message: None, approved: List[dict] | None = None) -> dict:
    """Toàn bộ chế độ "explainer". Trả về thời gian từng bước (giây)."""
    t0 = time.time()
    scenes, slides = plan(sentences, timings, approved)
    t1 = time.time()

    progress(80, f"Đang vẽ {len(scenes)} cảnh giải thích bằng Manim...")
    work_dir.mkdir(parents=True, exist_ok=True)
    scenes = [{**scene, "beats": scene_beats(timings, start, end)} for scene, (_, start, end) in zip(scenes, slides)]
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
