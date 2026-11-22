"""
core/stock_footage.py
Chế độ ảnh minh hoạ "stock" của "Đọc kịch bản": thay vì vẽ ảnh bằng GPU, ghép
các clip quay sẵn từ Pixabay (miễn phí, Pixabay Content License) theo nội dung
từng đoạn lời đọc.

  1. Gemini (lỗi thì DeepSeek) đọc cả kịch bản, chia thành cảnh ~5 giây và viết 2-3 cụm từ khoá
     tiếng Anh tả HÌNH ẢNH cụ thể cho từng cảnh ("person writing checklist",
     không phải "discipline") — từ khoá ý niệm hay ra clip lạc đề.
  2. Tìm clip trên Pixabay theo từng cụm từ khoá, lấy clip chưa dùng trong video,
     ưu tiên clip đủ dài. Kết quả tìm kiếm cache 24 giờ (điều khoản API).
  3. Tải clip bản 1280x720 về volume (dùng lại cho job sau), cắt/lặp cho vừa
     thời lượng cảnh, phủ kín khung 16:9, nối lại rồi ghép giọng đọc.

Chỉ dùng CPU: không tốn GPU như các chế độ vẽ ảnh.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.script_images import Slide, ask_gemini_json, group_by_duration, numbered_lines, tidy_starts
from utils.proc import run_command

SHOT_SEC = 5.0
MIN_SHOT_SEC = 3.0
MAX_SHOTS = 150
FRAME_W, FRAME_H = 1280, 720
FPS = 25
ENCODE_WORKERS = 4
SEARCH_URL = "https://pixabay.com/api/videos/"
SEARCH_TTL_SEC = 24 * 3600
# Bản "small" của Pixabay thường là 1280x720 — vừa khung ra, nhẹ hơn bản 1080p ~3 lần.
RENDITIONS = ("small", "medium", "tiny")
# Không tìm được clip nào cho cảnh thì thử các cảnh chung chung này.
GENERIC_QUERIES = ("sky clouds time lapse", "city aerial", "nature landscape", "ocean waves", "forest sunlight")
# Clip mang màu sắc tôn giáo/chính trị dễ lệch ý kịch bản (thử 22/11/2026: "woman reading
# book" ra cảnh đọc kinh Qur'an đầu bảng) — loại theo tag của Pixabay.
EXCLUDED_TAGS = frozenset({
    "quran", "koran", "bible", "prayer", "pray", "praying", "religion", "religious", "islam", "muslim",
    "mosque", "church", "christian", "christianity", "god", "jesus", "buddha", "buddhism", "temple",
    "hindu", "ramadan", "worship", "flag", "politics", "protest", "war", "weapon", "gun", "blood",
})


class StockFootageError(RuntimeError):
    pass


# ── Kế hoạch cảnh ────────────────────────────────────────────
def shot_count(timings: List[Tuple[float, float]]) -> int:
    return min(MAX_SHOTS, max(1, round(timings[-1][1] / SHOT_SEC)))


def stock_request(sentences: List[str], timings: List[Tuple[float, float]], count: int) -> str:
    return (
        "You are a video editor cutting stock footage to a narrated script.\n"
        "The narration is below, one numbered sentence per line with its start time. It may be in Vietnamese "
        "or English. Read the WHOLE script first, then answer in JSON only:\n"
        '{"shots": [{"start": 1, "queries": ["...", "..."]}]}\n\n'
        "Rules:\n"
        f"- About {count} shots, one every 4 to 7 seconds of narration. \"start\" is the sentence number where "
        "the shot begins; the first shot starts at sentence 1 and starts increase.\n"
        "- queries: 2 or 3 short English search phrases (2 to 4 words) for a stock video site, best first. "
        "Describe what the camera SEES: people, actions, objects, places — e.g. \"woman writing notebook\", "
        "\"running shoes street\", \"city night traffic\". Never abstract ideas such as \"discipline\", "
        "\"success\" or \"motivation\".\n"
        "- The second and third phrases are broader backups for the same moment.\n"
        "- No brand names, no text, no famous people.\n\n"
        f"{numbered_lines(sentences, timings)}"
    )


def parse_shots(raw: str, sentence_count: int) -> List[Tuple[int, List[str]]]:
    """JSON của Gemini -> [(chỉ số câu mở đầu, [từ khoá])], theo thứ tự câu."""
    data = json.loads(raw)
    shots: Dict[int, List[str]] = {}
    for shot in data.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        queries = [str(q).strip()[:100] for q in (shot.get("queries") or []) if str(q).strip()]
        if not queries:
            continue
        try:
            start = min(max(int(shot.get("start") or 1), 1), sentence_count) - 1
        except (TypeError, ValueError):
            continue
        shots.setdefault(start, queries[:3])
    board = sorted(shots.items())
    if board:
        board[0] = (0, board[0][1])
    return board


def fallback_board(timings: List[Tuple[float, float]]) -> List[Tuple[int, List[str]]]:
    """Gemini lỗi: chia đều theo thời lượng, xoay vòng các cảnh chung chung."""
    starts = group_by_duration(timings, max(SHOT_SEC, timings[-1][1] / MAX_SHOTS))
    return [(start, [GENERIC_QUERIES[k % len(GENERIC_QUERIES)]]) for k, start in enumerate(starts)]


def ask_deepseek_json(request: str) -> str:
    """Như ask_gemini_json nhưng hỏi DeepSeek (API tương thích OpenAI, chế độ JSON)."""
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    if not DEEPSEEK_API_KEY:
        raise RuntimeError("chưa có DEEPSEEK_API_KEY")
    from openai import OpenAI

    from core.translator.llm_common import call_with_retry

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = call_with_retry(
        lambda: client.chat.completions.create(
            model=DEEPSEEK_MODEL, temperature=0.5, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": request}],
            extra_body={"thinking": {"type": "disabled"}},
        ),
        what="DeepSeek (chia cảnh)",
    )
    return response.choices[0].message.content or ""


def plan_shots(request: str, sentence_count: int) -> List[Tuple[int, List[str]]]:
    """Gemini chia cảnh; Gemini lỗi (hết hạn mức, mất key...) thì DeepSeek.
    Cả hai lỗi thì ném ra để plan() lùi về cảnh chung chung."""
    errors = []
    for name, ask in (("Gemini", ask_gemini_json), ("DeepSeek", ask_deepseek_json)):
        try:
            board = parse_shots(ask(request), sentence_count)
            if board:
                return board
            errors.append(f"{name}: không có cảnh nào")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
        print(f"[Video stock] {errors[-1]}")
    raise RuntimeError("; ".join(errors))


def plan(sentences: List[str], timings: List[Tuple[float, float]]) -> Tuple[List[List[str]], List[Slide]]:
    """(từ khoá cho từng cảnh, lịch chiếu)."""
    try:
        board = plan_shots(stock_request(sentences, timings, shot_count(timings)), len(sentences))
    except Exception as exc:
        print(f"[Video stock] Không chia được cảnh ({exc}), dùng cảnh chung chung.")
        board = fallback_board(timings)
    starts = [start for start, _ in board]
    kept = tidy_starts(starts, timings, min_sec=MIN_SHOT_SEC, max_count=MAX_SHOTS)
    total = timings[-1][1]
    bounds = [0.0] + [timings[starts[k]][0] for k in kept[1:]] + [total]
    return [board[k][1] for k in kept], [(i, bounds[i], bounds[i + 1]) for i in range(len(kept))]


# ── Pixabay ──────────────────────────────────────────────────
class Pixabay:
    """Tìm + tải clip, cache trên đĩa (trên Modal: volume /data)."""

    def __init__(self, api_key: str, cache_dir: Path, session=None):
        if not api_key:
            raise StockFootageError("Chưa cấu hình PIXABAY_API_KEY nên không lấy được video stock.")
        import requests

        self.key = api_key
        self.http = session or requests.Session()
        self.search_dir = cache_dir / "search"
        self.clip_dir = cache_dir / "clips"
        self.search_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str) -> List[dict]:
        cache = self.search_dir / f"{hashlib.sha1(query.lower().encode()).hexdigest()}.json"
        if cache.exists() and time.time() - cache.stat().st_mtime < SEARCH_TTL_SEC:
            return json.loads(cache.read_text(encoding="utf-8"))
        response = self.http.get(SEARCH_URL, timeout=20, params={
            "key": self.key, "q": query, "video_type": "film", "safesearch": "true", "per_page": 30})
        if response.status_code == 429:
            raise StockFootageError("Pixabay báo đã hết lượt tìm kiếm, thử lại sau ít phút.")
        response.raise_for_status()
        hits = [hit for hit in response.json().get("hits") or [] if pick_rendition(hit) and not excluded(hit)]
        cache.write_text(json.dumps(hits), encoding="utf-8")
        return hits

    def download(self, hit: dict) -> Path:
        path = self.clip_dir / f"{hit['id']}.mp4"
        if path.exists():
            return path
        partial = path.with_name(f"{path.name}.{time.time_ns()}.part")
        try:
            with self.http.get(pick_rendition(hit)["url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("wb") as fileobj:
                    for chunk in response.iter_content(1 << 20):
                        fileobj.write(chunk)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        return path


def excluded(hit: dict) -> bool:
    tags = {tag.strip().lower() for tag in str(hit.get("tags") or "").split(",")}
    words = {word for tag in tags for word in tag.split()}
    return bool((tags | words) & EXCLUDED_TAGS)


def pick_rendition(hit: dict) -> Optional[dict]:
    videos = hit.get("videos") or {}
    return next((videos[name] for name in RENDITIONS if (videos.get(name) or {}).get("url")), None)


def choose_clip(hits: List[dict], need_sec: float, used: set) -> Optional[dict]:
    """Clip chưa dùng trong video này; ưu tiên đủ dài (không phải lặp), rồi giữ thứ hạng Pixabay."""
    fresh = [hit for hit in hits if hit.get("id") not in used]
    long_enough = [hit for hit in fresh if (hit.get("duration") or 0) >= need_sec]
    return (long_enough or fresh or [None])[0]


def find_clips(pixabay: Pixabay, queries: List[List[str]], slides: List[Slide]) -> List[dict]:
    """Một clip cho mỗi cảnh. Cảnh không tìm được gì thì dùng lại clip cảnh trước."""
    used: set = set()
    chosen: List[Optional[dict]] = []
    for shot_queries, (_, start, end) in zip(queries, slides):
        hit = None
        for query in list(shot_queries) + list(GENERIC_QUERIES):
            hit = choose_clip(pixabay.search(query), end - start, used)
            if hit:
                break
        if hit:
            used.add(hit["id"])
        chosen.append(hit)
    if not any(chosen):
        raise StockFootageError("Không tìm được clip nào trên Pixabay cho kịch bản này.")
    for k, hit in enumerate(chosen):  # điền chỗ trống bằng clip gần nhất phía trước/sau
        if hit is None:
            chosen[k] = next(h for h in chosen[k - 1::-1] + chosen[k:] if h)
    return chosen


# ── Dựng video ───────────────────────────────────────────────
def frame_counts(slides: List[Slide]) -> List[int]:
    """Làm tròn theo mốc tích luỹ để tổng số khung không trôi so với giọng đọc."""
    return [max(1, round(end * FPS) - round(start * FPS)) for _, start, end in slides]


def encode_shot(clip: Path, frames: int, output: Path) -> None:
    run_command(
        ["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", str(clip), "-an",
         "-vf", f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=increase,crop={FRAME_W}:{FRAME_H},"
                f"setsar=1,fps={FPS},format=yuv420p",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
         "-r", str(FPS), str(output)],
        timeout=300, error_message="Không cắt được clip video stock.",
    )


def make_video(clips: List[Path], slides: List[Slide], audio: Path, output: Path, work_dir: Path) -> None:
    counts = frame_counts(slides)
    parts = [work_dir / f"shot_{k:03d}.mp4" for k in range(len(slides))]
    with ThreadPoolExecutor(ENCODE_WORKERS) as pool:
        list(pool.map(lambda k: encode_shot(clips[k], counts[k], parts[k]), range(len(slides))))
    listing = work_dir / "shots.txt"
    listing.write_text("".join(f"file '{part.resolve().as_posix()}'\n" for part in parts), encoding="utf-8")
    run_command(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
         "-movflags", "+faststart", str(output)],
        timeout=1800, error_message="Không ghép được video stock với giọng đọc.",
    )


def render(sentences: List[str], timings: List[Tuple[float, float]], audio: Path, output: Path, work_dir: Path,
           progress=lambda percent, message: None, pixabay: Optional[Pixabay] = None) -> dict:
    """Toàn bộ chế độ "stock". Trả về thời gian từng bước (giây)."""
    if pixabay is None:
        from config.settings import DATA_DIR, PIXABAY_API_KEY

        pixabay = Pixabay(PIXABAY_API_KEY, DATA_DIR / "stock_cache")
    t0 = time.time()
    queries, slides = plan(sentences, timings)
    t1 = time.time()

    progress(80, f"Đang tìm {len(slides)} clip trên Pixabay...")
    hits = find_clips(pixabay, queries, slides)
    with ThreadPoolExecutor(ENCODE_WORKERS) as pool:
        clips = list(pool.map(pixabay.download, hits))
    t2 = time.time()

    progress(92, "Đang cắt ghép clip với giọng đọc...")
    work_dir.mkdir(parents=True, exist_ok=True)
    make_video(clips, slides, audio, output, work_dir)
    # Không trả khoá "images": app/jobs.py tính khoá đó theo giá GPU L40S, còn ở đây
    # thời gian tìm + tải clip chỉ là CPU (đã nằm trong elapsed_seconds).
    return {"image_prompts": round(t1 - t0, 2), "clips": round(t2 - t1, 2),
            "slideshow": round(time.time() - t2, 2)}
