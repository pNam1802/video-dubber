"""
scripts/stock_video_demo.py
Bản thử "văn bản -> video": mỗi cảnh một câu đọc + từ khoá, tìm clip trên
Pixabay, đọc bằng edge-tts, cắt/ghép bằng ffmpeg, cháy phụ đề. Chạy local:

    python scripts/stock_video_demo.py                       # kịch bản mẫu (giấc ngủ)
    python scripts/stock_video_demo.py kichban.json ten_video  # [[câu đọc, [từ khoá...]], ...]

Cần PIXABAY_API_KEY trong .env. Kết quả ở data/outputs/stock_demo/.
Bản thật sẽ để LLM tự chia cảnh + sinh từ khoá; ở đây cảnh viết tay để thử
chất lượng hình/tiếng trước.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import dotenv
import edge_tts
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "outputs" / "stock_demo"
CACHE = OUT / "clips"
W, H, FPS = 720, 1280, 30
VOICE = "vi-VN-HoaiMyNeural"

SCENES = [
    ("Bạn có biết, mỗi đêm thiếu ngủ đều để lại dấu vết trên cơ thể bạn?",
     ["tired man bed night", "insomnia"]),
    ("Khi bạn ngủ, não bộ dọn sạch chất thải tích tụ suốt cả ngày, như một đội vệ sinh làm ca đêm.",
     ["city night timelapse", "night street cleaning"]),
    ("Vì vậy, sau một đêm ngủ đủ giấc, bạn tập trung tốt hơn và nhớ lâu hơn.",
     ["woman working laptop", "student studying"]),
    ("Ngủ dưới sáu tiếng mỗi đêm làm tăng nguy cơ béo phì, bệnh tim và suy giảm miễn dịch.",
     ["heart health doctor", "hospital"]),
    ("Ánh sáng xanh từ điện thoại khiến não tưởng trời vẫn sáng, nên bạn càng khó ngủ.",
     ["phone screen dark", "smartphone night"]),
    ("Hãy cất điện thoại ba mươi phút trước khi ngủ, giữ phòng tối và mát.",
     ["bedroom night", "sleeping woman"]),
    ("Ngủ đủ bảy đến tám tiếng, và cơ thể sẽ cảm ơn bạn mỗi sáng thức dậy.",
     ["sunrise morning", "morning stretching"]),
]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        sys.exit(f"Lệnh lỗi: {' '.join(cmd[:6])}...\n{res.stderr[-1500:]}")
    return res.stdout


def duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(path)]).strip())


async def speak(text: str, out: Path, attempts: int = 5) -> None:
    # edge-tts hay trả "No audio was received" ngẫu nhiên; thử lại là qua.
    for n in range(attempts):
        try:
            await edge_tts.Communicate(text, VOICE, rate="+8%").save(str(out))
            return
        except edge_tts.exceptions.NoAudioReceived:
            if n == attempts - 1:
                raise
            await asyncio.sleep(1.5 * (n + 1))


def pick_clip(key: str, queries: list[str], need: float, used: set[int]) -> dict:
    """Clip đầu tiên chưa dùng, đủ dài, ưu tiên clip Full HD trở lên."""
    for q in queries:
        r = requests.get("https://pixabay.com/api/videos/", timeout=20, params={
            "key": key, "q": q, "video_type": "film", "safesearch": "true", "per_page": 30})
        r.raise_for_status()
        hits = [h for h in r.json()["hits"] if h["id"] not in used and h["videos"]["medium"]["url"]]
        hits.sort(key=lambda h: (h["duration"] < need, h["videos"]["medium"]["height"] < 1080))
        if hits:
            return {"query": q, **hits[0]}
    sys.exit(f"Không tìm được clip cho {queries}")


def download(hit: dict) -> Path:
    path = CACHE / f"{hit['id']}.mp4"
    if not path.exists():
        with requests.get(hit["videos"]["medium"]["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
    return path


def srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def chunks(text: str, size: int = 7) -> list[str]:
    words = text.split()
    n = max(1, round(len(words) / size))
    step = -(-len(words) // n)
    return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]


def main() -> None:
    scenes = SCENES
    name = "giac_ngu_demo"
    if len(sys.argv) > 1:
        scenes = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        name = sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).stem
    dotenv.load_dotenv(ROOT / ".env")
    key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not key:
        sys.exit("Thiếu PIXABAY_API_KEY trong .env")
    CACHE.mkdir(parents=True, exist_ok=True)

    used: set[int] = set()
    parts, srt, credits, t = [], [], [], 0.0
    for i, (text, queries) in enumerate(scenes, 1):
        voice = OUT / f"voice_{i}.mp3"
        asyncio.run(speak(text, voice))
        dur = duration(voice) + 0.35  # nghỉ ngắn giữa hai câu
        hit = pick_clip(key, queries, dur, used)
        used.add(hit["id"])
        clip = download(hit)
        print(f"Cảnh {i}: {dur:4.1f}s  '{hit['query']}' -> #{hit['id']} ({hit['duration']}s)")

        part = OUT / f"part_{i}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", str(clip), "-i", str(voice),
             "-t", f"{dur:.3f}", "-map", "0:v", "-map", "1:a",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS}",
             "-af", "apad", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
             "-c:a", "aac", "-ar", "44100", "-ac", "2", str(part)])
        parts.append(part)

        pieces = chunks(text)
        total = sum(len(p) for p in pieces)
        start = t
        for p in pieces:
            end = start + (dur - 0.35) * len(p) / total
            srt.append(f"{len(srt) + 1}\n{srt_time(start)} --> {srt_time(end)}\n{p}\n")
            start = end
        t += dur
        credits.append(f"Cảnh {i}: {hit['user']} trên Pixabay — {hit['pageURL']}")

    (OUT / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    (OUT / "subs.srt").write_text("\n".join(srt), encoding="utf-8")
    style = ("FontName=Arial,FontSize=13,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=70")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
         "-vf", f"subtitles=subs.srt:force_style='{style}'",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "copy",
         "-movflags", "+faststart", f"{name}.mp4"], cwd=OUT)
    (OUT / f"{name}_credits.txt").write_text("Video từ Pixabay\n" + "\n".join(credits) + "\n", encoding="utf-8")
    (OUT / f"{name}_scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nXong: {OUT / (name + '.mp4')}  ({t:.1f}s)")


if __name__ == "__main__":
    main()
