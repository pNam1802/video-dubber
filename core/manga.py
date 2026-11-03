"""
core/manga.py
Chế độ "Trang manga" của ảnh minh hoạ: Gemini làm đạo diễn manga — lập danh
sách nhân vật và bối cảnh (mô tả cố định, code chèn nguyên văn vào mọi khung),
rồi chia kịch bản thành các trang 2-4 khung. Code vẽ từng khung đúng tỉ lệ ô
của nó, ghép thành trang đen trắng có viền, và dựng video: cả trang trước,
sau đó phóng lần lượt từng khung theo lời đọc.

Luật rút ra khi vẽ thử (31/10/2026):
  - Mỗi khung tối đa 2 nhân vật. Model không vẽ được hai người chạm nhau (ra
    hai Daisy, mất cú tát), nên hành động va chạm tách thành khung ra đòn và
    khung trúng đòn — vốn cũng là kỹ thuật manga thật.
  - Bối cảnh chỉ lấy từ danh sách đã khai báo, nên không nhảy chỗ vô lý;
    khung cận cảnh dùng nền hiệu ứng (tram, nét tập trung) thay vì bối cảnh.
"""
from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.script_images import NO_TEXT, Slide, ask_gemini_json, make_slideshow, numbered_lines, tidy_starts

PAGE_W, PAGE_H = 1920, 1080
GUTTER, BORDER = 24, 6
# Mỗi trang phủ khoảng chừng ấy giây lời đọc; kịch bản dài thì giãn trang ra.
PAGE_SEC = 20.0
MIN_PAGE_SEC = 8.0
MAX_PAGES = 20
MAX_PANELS = 4
MAX_CHARACTERS, MAX_LOCATIONS = 4, 4
# Đầu mỗi trang chiếu cả trang chừng ấy giây rồi mới phóng vào từng khung.
OVERVIEW_SEC = 2.0
# Diện tích vẽ mỗi khung (~ khung 1280x720): ô nhỏ trên trang vẫn đủ nét khi phóng to.
PANEL_PIXELS = 1280 * 720

STYLE = ("seinen manga panel, black and white, G-pen ink line art, screentone shading, crosshatching, "
         "high contrast, professional manga artist")
PANEL_NO_TEXT = f"{NO_TEXT}, no speech bubbles"
EFFECT_BACKGROUND = "abstract background of screentone and focus lines, no scenery"
SHOTS = ("wide establishing shot", "full body shot", "medium shot", "close-up", "extreme close-up",
         "low angle shot", "high angle shot", "over the shoulder shot", "dynamic action shot")

Cell = Tuple[int, int, int, int]  # x, y, rộng, cao trên trang


class Panel:
    def __init__(self, start: int, prompt: str):
        self.start = start  # chỉ số câu lời đọc tới khung này
        self.prompt = prompt


class Page:
    def __init__(self, start: int, panels: List[Panel]):
        self.start = start
        self.panels = panels


# ── Bố cục ───────────────────────────────────────────────────
def layout(count: int) -> List[Cell]:
    """Ô khung theo thứ tự đọc (trái -> phải, trên -> dưới); khung cuối — cao
    trào của trang — luôn là ô lớn nhất."""
    g, w, h = GUTTER, PAGE_W, PAGE_H
    if count <= 1:
        return [(g, g, w - 2 * g, h - 2 * g)]
    if count == 2:
        left = int((w - 3 * g) * 0.42)
        return [(g, g, left, h - 2 * g), (2 * g + left, g, w - 3 * g - left, h - 2 * g)]
    right = 620
    left = w - 3 * g - right
    if count == 3:
        top = (h - 3 * g) // 2
        return [(g, g, left, top), (g, 2 * g + top, left, h - 3 * g - top), (2 * g + left, g, right, h - 2 * g)]
    top = 520
    small = (left - g) // 2
    return [(g, g, left, top), (g, 2 * g + top, small, h - 3 * g - top),
            (2 * g + small, 2 * g + top, left - g - small, h - 3 * g - top), (2 * g + left, g, right, h - 2 * g)]


def draw_size(cell: Cell) -> Tuple[int, int]:
    """Kích thước vẽ cho một ô: cùng tỉ lệ, diện tích ~PANEL_PIXELS, cạnh chia hết cho 16."""
    ratio = cell[2] / cell[3]
    width = math.sqrt(PANEL_PIXELS * ratio)
    return max(16, round(width / 16) * 16), max(16, round(width / ratio / 16) * 16)


# ── Storyboard ───────────────────────────────────────────────
def manga_request(sentences: List[str], timings: List[Tuple[float, float]], count: int) -> str:
    return (
        "You are a manga director adapting a narrated script into black and white manga pages shown in a video.\n"
        "The narration is below, one numbered sentence per line with its start time. It may be in Vietnamese "
        "or English. Read the WHOLE script first and understand the story, then answer in JSON only:\n"
        '{"characters": [{"id": "...", "look": "..."}], "locations": [{"id": "...", "look": "..."}], '
        '"pages": [{"start": 1, "location": "...", "panels": [{"start": 1, "shot": "...", "characters": ["..."], '
        '"background": "location", "action": "...", "effects": "..."}]}]}\n\n'
        "Rules:\n"
        f"- characters: at most {MAX_CHARACTERS} recurring characters who are drawn, each with a short id and a "
        "fixed English appearance (gender, age, build, hairstyle, clothing, distinctive features; no colors). "
        "Make them easy to tell apart. The look is reused word for word in every panel.\n"
        f"- locations: at most {MAX_LOCATIONS} places, each with an id and a fixed English description, reused "
        "in every panel set there.\n"
        f"- pages: about {count} pages, each covering about 15 to 25 seconds of narration. \"start\" is the "
        "sentence number where the page begins; the first page starts at sentence 1 and starts increase. "
        "\"location\" is a location id or null.\n"
        f"- Each page has 2 to {MAX_PANELS} panels in reading order, like a real manga page: often open with a "
        "wide establishing shot, then close-ups and reactions, and end with the biggest, most dramatic panel. "
        "A panel's \"start\" is the sentence number where the narration reaches it, not before the page start.\n"
        f"- shot: one of {', '.join(SHOTS)}.\n"
        "- characters: ids of the 0 to 2 characters visible in the panel. Never more than two people, never "
        "unnamed crowds. Two characters must never touch or fight in the same panel: split an attack into one "
        "panel of the attacker striking and one panel of the target reacting.\n"
        "- background: \"location\" draws the page location; \"effect\" draws abstract screentone or speed lines "
        "instead (best for close-ups, emotions and action).\n"
        "- action: what is visible: pose, expression, key objects. Refer to characters by id and do not repeat "
        "their appearance. No colors, no art style.\n"
        "- effects: manga effects such as speed lines, focus lines, impact burst, sweat drops, dark aura; or "
        "an empty string.\n"
        "- Draw only what happens at that moment of the story, never what is merely mentioned in dialogue, "
        "thoughts or memories. Never replace a person or event with an unrelated object or metaphor.\n"
        "- Nothing written: no text, letters, numbers, logos, captions or speech bubbles.\n\n"
        f"{numbered_lines(sentences, timings)}"
    )


def _bible(items) -> Dict[str, str]:
    """[{"id", "look"}] -> {id: look}; bỏ mục thiếu id hoặc mô tả."""
    out: Dict[str, str] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            key = str(item.get("id") or "").strip().lower()
            look = str(item.get("look") or item.get("appearance") or item.get("description") or "")
            if key and look.strip():
                out.setdefault(key, look.strip().rstrip(". "))
    return out


def compose_panel(panel: dict, characters: Dict[str, str], location: Optional[str]) -> str:
    shot = str(panel.get("shot") or "").strip().lower()
    parts = [shot] if shot in SHOTS else []
    parts.append(str(panel["action"]).strip().rstrip(". "))
    effects = str(panel.get("effects") or "").strip().rstrip(". ")
    if effects:
        parts.append(effects)

    ids = [str(c).strip().lower() for c in panel.get("characters") or [] if isinstance(c, str)]
    cast = [(i, characters[i]) for i in dict.fromkeys(ids) if i in characters][:2]
    if len(cast) == 1:
        parts += [f"{cast[0][0]}: {cast[0][1]}", "solo, single character"]
    elif len(cast) == 2:
        parts += [f"two characters only, {cast[0][0]} on the left: {cast[0][1]}",
                  f"{cast[1][0]} on the right: {cast[1][1]}"]

    use_location = location and str(panel.get("background") or "location").strip().lower() != "effect"
    parts.append(f"background: {location}" if use_location else EFFECT_BACKGROUND)
    return ", ".join(parts + [STYLE, PANEL_NO_TEXT])


def _sentence(value, default: int, sentence_count: int) -> int:
    """Số câu 1-based của Gemini -> chỉ số 0-based trong [0, sentence_count)."""
    try:
        return min(max(int(value), 1), sentence_count) - 1
    except (TypeError, ValueError):
        return default


def parse_manga(raw: str, sentence_count: int) -> List[Page]:
    """JSON của Gemini -> các trang theo thứ tự câu. Khung hỏng (thiếu action)
    bị bỏ, trang không còn khung nào bị bỏ; cảnh đầu luôn phủ từ câu 0."""
    data = json.loads(raw)
    characters = _bible(data.get("characters"))
    locations = _bible(data.get("locations"))
    pages: Dict[int, Page] = {}
    for page in data.get("pages") or []:
        if not isinstance(page, dict):
            continue
        start = _sentence(page.get("start"), -1, sentence_count)
        if start < 0 or start in pages:
            continue
        location = locations.get(str(page.get("location") or "").strip().lower())
        panels, last = [], start
        for panel in page.get("panels") or []:
            if not isinstance(panel, dict) or not str(panel.get("action") or "").strip():
                continue
            last = max(last, _sentence(panel.get("start"), last, sentence_count))
            panels.append(Panel(last, compose_panel(panel, characters, location)))
        if panels:
            pages[start] = Page(start, panels[:MAX_PANELS])
    board = [pages[k] for k in sorted(pages)]
    if board:
        board[0].start = 0
    return board


def plan(sentences: List[str], timings: List[Tuple[float, float]]) -> List[Page]:
    """Nhờ Gemini dựng các trang; lỗi thì ném ra để gọi bên ngoài lùi về "scenes"."""
    count = min(MAX_PAGES, max(1, round(timings[-1][1] / PAGE_SEC)))
    pages = parse_manga(ask_gemini_json(manga_request(sentences, timings, count)), len(sentences))
    if not pages:
        raise RuntimeError("storyboard manga rỗng")
    kept = tidy_starts([page.start for page in pages], timings, min_sec=MIN_PAGE_SEC, max_count=MAX_PAGES)
    return [pages[k] for k in kept]


# ── Lịch chiếu ───────────────────────────────────────────────
def page_timeline(page: Page, begin: float, end: float,
                  timings: List[Tuple[float, float]]) -> List[Tuple[Optional[int], float, float]]:
    """Một trang chiếu trong [begin, end) -> [(khung hay None = cả trang, từ, đến)].
    Cả trang chiếu trước; mỗi khung bắt đầu khi lời đọc tới câu của nó. Gemini
    cho câu không tăng dần (hay dồn hết vào một câu) thì chia đều thời gian."""
    overview_end = begin + min(OVERVIEW_SEC, (end - begin) / 3)
    count = len(page.panels)
    cuts = [overview_end] + [min(max(timings[p.start][0], overview_end), end) for p in page.panels[1:]] + [end]
    if any(b - a < 1.0 for a, b in zip(cuts, cuts[1:])):
        step = (end - overview_end) / count
        cuts = [overview_end + k * step for k in range(count)] + [end]
    return [(None, begin, overview_end)] + [(k, cuts[k], cuts[k + 1]) for k in range(count)]


# ── Dựng trang + video ───────────────────────────────────────
def compose_page(panels: List[bytes], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    pen = ImageDraw.Draw(page)
    for data, (x, y, w, h) in zip(panels, layout(len(panels))):
        page.paste(ImageOps.fit(Image.open(io.BytesIO(data)).convert("L"), (w, h)).convert("RGB"), (x, y))
        pen.rectangle([x, y, x + w - 1, y + h - 1], outline="black", width=BORDER)
    page.save(path)


def compose_closeup(panel: bytes, path: Path) -> None:
    """Một khung phóng to giữa nền trắng, giữ viền đen như trên trang."""
    from PIL import Image, ImageDraw, ImageOps

    frame = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    img = ImageOps.contain(Image.open(io.BytesIO(panel)).convert("L"), (PAGE_W - 4 * GUTTER, PAGE_H - 4 * GUTTER))
    x, y = (PAGE_W - img.width) // 2, (PAGE_H - img.height) // 2
    frame.paste(img.convert("RGB"), (x, y))
    ImageDraw.Draw(frame).rectangle([x, y, x + img.width - 1, y + img.height - 1], outline="black", width=BORDER)
    frame.save(path)


def render(pages: List[Page], timings: List[Tuple[float, float]], audio: Path, output: Path, work_dir: Path,
           progress=lambda percent, message: None, started: Optional[float] = None) -> dict:
    """Vẽ mọi khung trong một lượt GPU, ghép trang, dựng video. Trả về thời gian từng bước."""
    from core import image_client

    t1 = time.time()
    prompts, sizes = [], []
    for page in pages:
        for panel, cell in zip(page.panels, layout(len(page.panels))):
            prompts.append(panel.prompt)
            sizes.append(draw_size(cell))
    progress(80, f"Đang vẽ {len(prompts)} khung truyện cho {len(pages)} trang manga...")
    drawn = image_client.generate(prompts, sizes)
    t2 = time.time()

    progress(95, "Đang ghép trang truyện và giọng đọc thành video...")
    work_dir.mkdir(parents=True, exist_ok=True)
    total = timings[-1][1]
    bounds = [0.0] + [timings[page.start][0] for page in pages[1:]] + [total]
    views: List[Path] = []
    slides: List[Slide] = []
    offset = 0
    for p, page in enumerate(pages):
        panels = drawn[offset:offset + len(page.panels)]
        offset += len(page.panels)
        page_view = work_dir / f"page_{p:02d}.png"
        compose_page(panels, page_view)
        first_view = len(views)
        views.append(page_view)
        for k, data in enumerate(panels):
            closeup = work_dir / f"page_{p:02d}_panel_{k}.png"
            compose_closeup(data, closeup)
            views.append(closeup)
        for panel_idx, begin, end in page_timeline(page, bounds[p], bounds[p + 1], timings):
            slides.append((first_view if panel_idx is None else first_view + 1 + panel_idx, begin, end))
    make_slideshow(views, slides, audio, output)
    t0 = started if started is not None else t1
    return {"image_prompts": round(t1 - t0, 2), "images": round(t2 - t1, 2),
            "slideshow": round(time.time() - t2, 2)}
