"""
core/script_images.py
Ảnh minh hoạ cho "Đọc kịch bản": nhờ Gemini đọc cả kịch bản và dựng
storyboard, vẽ từng ảnh, rồi ghép ảnh + mp3 thành một video slideshow.

Ba chế độ:
  "few"    — 3-5 khoảnh khắc tiêu biểu của cả kịch bản, chiếu lần lượt rồi lặp lại.
  "scenes" — mỗi cảnh (đổi sự kiện) một ảnh riêng.
  "manga"  — trang truyện nhiều khung, có nhân vật phụ và bối cảnh: core/manga.py.
             Không dựng được trang manga thì lùi về "scenes".
  "recap"  — kiểu video kể truyện manhwa: ảnh webtoon màu ~6 giây một ảnh,
             khung dọc giữa nền mờ, zoom chậm: core/recap.py.

Trừ "recap", mọi ảnh cùng một nét vẽ manga đen trắng (MANGA_STYLE) cố định trong code; Gemini
chỉ chọn cỡ cảnh/góc máy cho từng ảnh để chúng đa dạng như các khung truyện.

Storyboard giữ ảnh nhất quán bằng cách vẽ ít thứ nhất có thể:
  - Chỉ NHÂN VẬT CHÍNH, không bao giờ vẽ nhân vật phụ — model ảnh không nhớ
    mặt giữa các ảnh, càng nhiều người càng lệch. Ngoại hình do Gemini viết MỘT
    lần, code chèn nguyên văn vào mọi ảnh có nhân vật chính.
  - Không vẽ bối cảnh: nền trơn, nên không bao giờ "nhảy" sai chỗ giữa hai cảnh.
  - Cảnh không có nhân vật chính: cận cảnh tượng trưng (đôi mắt, nắm tay, bóng
    người, đồ vật) trên nền tối — gợi không khí thay vì kể lại toàn cảnh.

Ảnh chỉ đổi ở ranh giới câu, không bao giờ đổi giữa chừng một câu.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

from utils.proc import run_command

MODES = ("none", "few", "scenes", "manga", "recap")
SCENE_SEC = 12.0
SLIDE_SEC = 8.0
# Cảnh ngắn hơn thì gộp vào cảnh trước — ảnh lướt qua quá nhanh không ai kịp xem.
MIN_SCENE_SEC = 5.0
# Trần số ảnh của chế độ "scenes": kịch bản dài thì giãn cảnh ra thay vì vẽ
# thêm — mỗi ảnh ~3 giây GPU L40S, cộng thêm thời gian nạp model.
MAX_SCENES = 30
FEW_MIN, FEW_MAX = 3, 5
# Người dùng chọn manga đen trắng (31/10/2026) — giữ cố định để mọi ảnh cùng nét.
MANGA_STYLE = ("black and white manga panel, monochrome ink drawing, clean bold line art, screentone shading, "
               "high contrast, japanese manga style")
PLAIN_BACKGROUND = "plain white background, no scenery"
# Cảnh vắng nhân vật chính, theo kiểu ảnh mẫu người dùng chọn (30/10/2026):
# cận cảnh nửa chìm trong bóng tối, nền đen.
SYMBOLIC_LOOK = ("symbolic extreme close-up, partial view, solid black background, heavy ink shadows, "
                 "half in shadow, dramatic moody atmosphere, no scenery")
# Cỡ cảnh/góc máy Gemini được chọn cho ảnh nhân vật chính; ngoài danh sách thì bỏ qua.
SHOTS = ("close-up", "medium shot", "full body shot", "low angle shot", "high angle shot",
         "dynamic action pose")
# Model ảnh hay vẽ chữ méo mó — luôn dặn không vẽ chữ, kể cả khi Gemini quên.
NO_TEXT = "no text, no letters, no watermark"
# Dự phòng khi không có storyboard: vẽ thẳng từ lời đọc, mỗi phần tối đa bấy nhiêu ký tự.
MAX_CHUNK_CHARS = 1500

#: (chỉ số ảnh, bắt đầu, kết thúc) — giây, trong file âm thanh kết quả.
Slide = Tuple[int, float, float]


# ── Lịch chiếu ───────────────────────────────────────────────
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


def few_count(timings: List[Tuple[float, float]]) -> int:
    total = timings[-1][1]
    return min(FEW_MAX, max(FEW_MIN, math.ceil(total / 60)), len(timings))


def cycle_slides(timings: List[Tuple[float, float]], image_count: int) -> List[Slide]:
    """Chế độ "few": đổi ảnh mỗi ~SLIDE_SEC, lần lượt rồi lặp lại từ đầu."""
    starts = group_by_duration(timings, SLIDE_SEC)
    return _slides(starts, timings, timings[-1][1], [k % image_count for k in range(len(starts))])


def scene_slides(starts: List[int], timings: List[Tuple[float, float]]) -> List[Slide]:
    return _slides(starts, timings, timings[-1][1], list(range(len(starts))))


def tidy_starts(starts: List[int], timings: List[Tuple[float, float]],
                min_sec: float = MIN_SCENE_SEC, max_count: int = MAX_SCENES) -> List[int]:
    """Chỉ số câu mở đầu cảnh (đã sắp, câu 0 luôn mở đầu) -> bỏ những cảnh quá
    ngắn hoặc vượt trần max_count; cảnh bị bỏ gộp vào cảnh đứng trước nó.
    Trả về VỊ TRÍ (trong starts) của các cảnh được giữ."""
    min_sec = max(min_sec, timings[-1][1] / (max_count + 1))
    kept = [0]
    for k in range(1, len(starts)):
        if timings[starts[k]][0] - timings[starts[kept[-1]]][0] >= min_sec:
            kept.append(k)
    return kept[:max_count]


# ── Storyboard ───────────────────────────────────────────────
def _clock(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def numbered_lines(sentences: List[str], timings: List[Tuple[float, float]]) -> str:
    return "\n".join(f"[{i + 1}] ({_clock(t[0])}) {text}" for i, (text, t) in enumerate(zip(sentences, timings)))


def storyboard_request(sentences: List[str], timings: List[Tuple[float, float]], mode: str, count: int) -> str:
    lines = numbered_lines(sentences, timings)
    if mode == "scenes":
        task = (f"Split the script into scenes wherever the event changes. A scene should last roughly 6 to 25 "
                f"seconds of narration; aim for about {count} scenes. Each scene's \"start\" is the number of the "
                "sentence where it begins; the first scene starts at sentence 1 and starts increase.")
    else:
        task = (f"Choose exactly {count} key moments that best represent the whole script, in story order. "
                "Set \"start\" to the sentence number each moment comes from.")
    return (
        "You are a storyboard artist turning a narrated script into still illustrations for a video.\n"
        "The narration is below, one numbered sentence per line with its start time. It may be in Vietnamese "
        "or English. Read the WHOLE script first and understand the story, then answer in JSON only:\n"
        '{"main_character": {"look": "..."} or null, '
        '"scenes": [{"start": 1, "main": true, "shot": "...", "action": "..."}]}\n\n'
        "Rules:\n"
        "- Every image is a black and white manga panel: describe content only, never colors or art style.\n"
        "- main_character: the protagonist of the script, with a fixed English description of appearance "
        "(gender, age, hairstyle, clothing, distinctive features; no colors). It is reused word for word in "
        "every image. "
        "null if the script has no protagonist.\n"
        "- Images never show any other character and never show scenery or places.\n"
        "- main = true when the main character is physically present at that moment: action is her/his pose, "
        "expression and the key object or gesture, on a plain background. Do not repeat the appearance there. "
        f"shot: one of {', '.join(SHOTS)}; vary it from scene to scene like manga panels: close-up for "
        "emotions, full body shot or dynamic action pose for movement.\n"
        "- main = false otherwise: action is a symbolic extreme close-up that conveys the mood of the moment: "
        "one partial detail such as eyes, a clenched fist, a silhouette or a key object, half hidden in shadow "
        "on a dark background. Never a full figure, never a group, never a place.\n"
        "- Draw only what happens at that moment of the story, never what is merely mentioned in dialogue, "
        "thoughts or memories.\n"
        "- Never replace a person or an event with an unrelated object or metaphor (e.g. no barbell for carrying "
        "people): show the main character's own part of it, such as her arms gripping a heavy load on her "
        "shoulder.\n"
        "- Nothing written: no text, letters, numbers, logos or captions.\n"
        f"- {task}\n\n"
        f"{lines}"
    )


def compose_prompt(action: str, main_look: Optional[str], shot: str = "") -> str:
    """main_look có giá trị: ảnh nhân vật chính; None: cận cảnh tượng trưng."""
    if main_look:
        parts = [action.rstrip(". "), f"The only character in the image: {main_look}",
                 "Solo, single character", PLAIN_BACKGROUND]
        if shot in SHOTS:
            parts.insert(0, shot)
    else:
        parts = [action.rstrip(". "), SYMBOLIC_LOOK]
    return ", ".join(parts + [MANGA_STYLE, NO_TEXT])


def _main_look(main) -> str:
    """Gemini không phải lúc nào cũng theo đúng khuôn: main_character có khi
    là chuỗi, có khi dùng khoá khác "look"."""
    if isinstance(main, dict):
        main = next((main[key] for key in ("look", "appearance", "description") if main.get(key)), "")
    return str(main or "").strip().rstrip(". ")


def parse_storyboard(raw: str, sentence_count: int) -> List[Tuple[int, str]]:
    """JSON của Gemini -> [(chỉ số câu mở đầu, prompt)] theo thứ tự câu.
    Cảnh hỏng (thiếu action, start sai) bị bỏ; không mô tả được nhân vật
    chính thì mọi cảnh đều thành cận cảnh tượng trưng."""
    data = json.loads(raw)
    main_look = _main_look(data.get("main_character"))
    scenes: dict[int, str] = {}
    for scene in data.get("scenes") or []:
        if not isinstance(scene, dict) or not str(scene.get("action") or "").strip():
            continue
        try:
            start = min(max(int(scene.get("start") or 1), 1), sentence_count) - 1
        except (TypeError, ValueError):
            continue
        look = main_look if str(scene.get("main")).strip().lower() in ("true", "1", "yes") else None
        shot = str(scene.get("shot") or "").strip().lower()
        scenes.setdefault(start, compose_prompt(str(scene["action"]).strip(), look or None, shot))
    board = sorted(scenes.items())
    if board:
        board[0] = (0, board[0][1])  # cảnh đầu luôn phủ từ câu đầu tiên
    return board


def ask_gemini_json(request: str) -> str:
    """Gửi yêu cầu storyboard, trả về chuỗi JSON. Lỗi (kể cả thiếu key) thì ném ra."""
    from config.settings import GEMINI_API_KEY, GEMINI_MODEL

    if not GEMINI_API_KEY:
        raise RuntimeError("chưa có GEMINI_API_KEY")
    from google import genai
    from google.genai import types

    from core.translator.llm_common import call_with_retry

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(temperature=0.5, response_mime_type="application/json")
    response = call_with_retry(
        lambda: client.models.generate_content(model=GEMINI_MODEL, config=config, contents=request),
        what="Gemini (storyboard)",
    )
    return response.text or ""


def write_storyboard(sentences: List[str], timings: List[Tuple[float, float]], mode: str,
                     count: int) -> List[Tuple[int, str]]:
    """Nhờ Gemini dựng storyboard. Lỗi thì ném ra để illustrate() lùi về cách vẽ
    thẳng từ lời đọc."""
    board = parse_storyboard(ask_gemini_json(storyboard_request(sentences, timings, mode, count)), len(sentences))
    if not board:
        raise RuntimeError("storyboard rỗng")
    return board


def chunk_texts(sentences: List[str], starts: List[int]) -> List[str]:
    bounds = starts + [len(sentences)]
    return [" ".join(sentences[bounds[k]:bounds[k + 1]])[:MAX_CHUNK_CHARS] for k in range(len(starts))]


def fallback_prompts(chunks: List[str]) -> List[str]:
    return [f"{text[:300]}, {PLAIN_BACKGROUND}, {MANGA_STYLE}, {NO_TEXT}" for text in chunks]


def fallback_plan(sentences: List[str], timings: List[Tuple[float, float]], mode: str) -> Tuple[List[str], List[Slide]]:
    """Không có storyboard: chia theo thời lượng, vẽ thẳng từ lời đọc."""
    total = timings[-1][1]
    if mode == "scenes":
        starts = group_by_duration(timings, max(SCENE_SEC, total / MAX_SCENES))
        return fallback_prompts(chunk_texts(sentences, starts)), scene_slides(starts, timings)
    count = few_count(timings)
    starts = group_by_duration(timings, total / count)[:count]
    return fallback_prompts(chunk_texts(sentences, starts)), cycle_slides(timings, len(starts))


def plan(sentences: List[str], timings: List[Tuple[float, float]], mode: str) -> Tuple[List[str], List[Slide]]:
    """(prompt cho từng ảnh, lịch chiếu ảnh)."""
    total = timings[-1][1]
    count = few_count(timings) if mode == "few" else min(MAX_SCENES, max(1, round(total / SCENE_SEC)))
    try:
        board = write_storyboard(sentences, timings, mode, count)
    except Exception as exc:
        print(f"[Ảnh minh hoạ] Không dựng được storyboard ({exc}), vẽ thẳng từ lời đọc.")
        return fallback_plan(sentences, timings, mode)

    if mode == "few":
        prompts = [prompt for _, prompt in board][:count]
        return prompts, cycle_slides(timings, len(prompts))
    starts = [start for start, _ in board]
    kept = tidy_starts(starts, timings)
    return [board[k][1] for k in kept], scene_slides([starts[k] for k in kept], timings)


# ── Ghép video ───────────────────────────────────────────────
def make_slideshow(images: List[Path], slides: List[Slide], audio: Path, output: Path) -> None:
    """Ghép ảnh theo lịch chiếu với giọng đọc thành mp4. hue=s=0 ép về đen
    trắng hẳn — model ảnh đôi khi vẫn ám chút màu dù prompt dặn monochrome."""
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
             "-vf", "fps=25,hue=s=0,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
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
    progress(75, "Đang đọc hiểu kịch bản để dựng cảnh minh hoạ...")
    if mode == "recap":
        from core import recap

        return recap.render(sentences, timings, audio, output, work_dir, progress)
    if mode == "manga":
        from core import manga

        try:
            pages = manga.plan(sentences, timings)
        except Exception as exc:
            print(f"[Ảnh minh hoạ] Không dựng được trang manga ({exc}), chuyển sang ảnh theo cảnh.")
            mode = "scenes"
        else:
            return manga.render(pages, timings, audio, output, work_dir, progress, started=t0)
    prompts, slides = plan(sentences, timings, mode)
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
