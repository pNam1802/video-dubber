"""
Xem trước cảnh viết tay của một tập (core/episodes/<module>.py) ngay trên máy, không tốn Modal.

    python scripts/preview_episode.py s1e01                 # mọi cảnh, 720p
    python scripts/preview_episode.py s1e01 --only takeaway  # một cảnh
    python scripts/preview_episode.py s1e01 --board          # in bảng phân cảnh JSON để dán vào Xưởng

Mốc thời gian ước lượng theo độ dài câu (giống bước duyệt phân cảnh), nên hình khớp lời gần đúng.
Ra: <out>/<cảnh>.mp4 và <out>/<cảnh>.png (4 khung ở 20/45/70/95% ghép ngang).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.explainer import FPS, estimate_timings, render_scene, scene_beats  # noqa: E402
from core.script_reader import split_script  # noqa: E402


def load_board(module: str):
    spec = importlib.util.spec_from_file_location(module, ROOT / "core" / "episodes" / f"{module}.py")
    # Chỉ cần BOARD: đọc bằng ast để không phải nạp Manim.
    import ast

    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BOARD":
            return ast.literal_eval(node.value)
    raise SystemExit(f"{module}.py không có BOARD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--only", default="")
    ap.add_argument("--quality", default="720p")
    ap.add_argument("--out", default="")
    ap.add_argument("--board", action="store_true")
    args = ap.parse_args()

    board = load_board(args.module)
    if args.board:
        print(json.dumps([{"start": start, "layout": "custom", "module": args.module, "scene": name}
                          for start, name in board], ensure_ascii=False))
        return
    text = (ROOT / "core" / "episodes" / f"{args.module}_{args.lang}.txt").read_text(encoding="utf-8")
    sentences = [s for _, s in split_script(text)]
    timings = estimate_timings(sentences, args.lang)
    out = Path(args.out or tempfile.mkdtemp(prefix=f"preview-{args.module}-"))
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, (start, name) in enumerate(board):
        if args.only and name != args.only:
            continue
        t0 = timings[start][0]
        t1 = timings[board[i + 1][0]][0] if i + 1 < len(board) else timings[-1][1]
        scene = {"layout": "custom", "module": args.module, "scene": name, "beats": scene_beats(timings, t0, t1)}
        jobs.append((name, scene, max(25, round((t1 - t0) * FPS))))

    def run(job):
        name, scene, frames = job
        video = out / f"{name}.mp4"
        work = out / f"work_{name}"
        work.mkdir(exist_ok=True)
        try:
            render_scene(scene, frames, video, work, args.quality)
        except Exception as exc:  # in lỗi từng cảnh, không dừng cả loạt
            return f"LỖI {name}: {exc}"
        sec = frames / FPS
        picks = "+".join(f"eq(n\\,{round(sec * p * FPS)})" for p in (0.2, 0.45, 0.7, 0.95))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                        f"select='{picks}',scale=640:-1,tile=4x1", "-frames:v", "1", "-vsync", "0",
                        str(out / f"{name}.png")], check=False)
        return f"OK {name} ({sec:.1f}s)"

    with ThreadPoolExecutor(4) as pool:
        for line in pool.map(run, jobs):
            print(line, flush=True)
    print(out)


if __name__ == "__main__":
    main()
