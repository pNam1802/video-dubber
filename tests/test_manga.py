"""Chế độ "Trang manga": storyboard, bố cục trang, lịch chiếu, ghép video."""
from __future__ import annotations

import json
import shutil

import pytest

from core import image_client, manga, script_images
from core.pipeline import DubbingConfig, DubbingPipeline
from core.transcriber import Segment
from tests.test_script_images import _fake_tts, _no_gemini, _png, _timings

BOARD = {
    "characters": [{"id": "Daisy", "look": "young maid with a ponytail."},
                   {"id": "thug", "look": "burly bald man with a scar"}],
    "locations": [{"id": "garden", "look": "mansion garden with hedges"}],
    "pages": [
        {"start": 3, "location": "garden", "panels": [
            {"start": 3, "shot": "close-up", "characters": ["thug"], "background": "effect",
             "action": "thug sneers", "effects": "focus lines"},
        ]},
        {"start": 1, "location": "garden", "panels": [
            {"start": 1, "shot": "wide establishing shot", "characters": ["daisy", "thug", "ghost"],
             "action": "thug looms over daisy"},
            {"start": 2, "shot": "drone shot", "characters": [], "action": ""},
            {"start": 2, "shot": "extreme close-up", "characters": ["daisy"], "action": "daisy's cold eyes."},
        ]},
        {"start": 5, "location": None, "panels": [{"start": 5, "action": ""}]},
    ],
}


def test_parse_manga_injects_character_and_location_bible():
    pages = manga.parse_manga(json.dumps(BOARD), sentence_count=6)
    assert [page.start for page in pages] == [0, 2]  # trang không còn khung hợp lệ bị bỏ
    wide, eyes = (panel.prompt for panel in pages[0].panels)
    assert wide.startswith("wide establishing shot, thug looms over daisy, two characters only, "
                           "daisy on the left: young maid with a ponytail, thug on the right: burly bald man")
    assert "ghost" not in wide and "background: mansion garden with hedges" in wide
    assert wide.endswith(f"{manga.STYLE}, {manga.PANEL_NO_TEXT}")
    assert eyes.startswith("extreme close-up, daisy's cold eyes, daisy: young maid") and "solo" in eyes
    [sneer] = pages[1].panels
    assert "focus lines" in sneer.prompt and manga.EFFECT_BACKGROUND in sneer.prompt
    assert "mansion garden" not in sneer.prompt
    assert [panel.start for panel in pages[0].panels] == [0, 1]


def test_manga_request_carries_rules():
    request = manga.manga_request(["Một.", "Hai."], [(0.0, 1.0), (2.0, 3.0)], 2)
    assert "Two characters must never touch or fight in the same panel" in request
    assert "[2] (0:02) Hai." in request and "about 2 pages" in request


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_layout_fits_page_and_last_panel_is_biggest(count):
    cells = manga.layout(count)
    assert len(cells) == count
    for x, y, w, h in cells:
        assert x >= manga.GUTTER and y >= manga.GUTTER
        assert x + w <= manga.PAGE_W - manga.GUTTER and y + h <= manga.PAGE_H - manga.GUTTER
    areas = [w * h for _, _, w, h in cells]
    assert areas[-1] == max(areas)
    for w, h in map(manga.draw_size, cells):
        assert w % 16 == 0 and h % 16 == 0


def test_page_timeline_overview_then_panels():
    timings = _timings(10)
    page = manga.Page(0, [manga.Panel(0, "a"), manga.Panel(3, "b"), manga.Panel(6, "c")])
    timeline = manga.page_timeline(page, 0.0, 40.0, timings)
    assert [k for k, _, _ in timeline] == [None, 0, 1, 2]
    assert timeline[0] == (None, 0.0, manga.OVERVIEW_SEC)
    assert timeline[2][1] == timings[3][0] and timeline[-1][2] == 40.0
    assert all(a[2] == b[1] for a, b in zip(timeline, timeline[1:]))


def test_page_timeline_splits_evenly_when_panels_share_a_sentence():
    page = manga.Page(0, [manga.Panel(0, "a"), manga.Panel(0, "b")])
    timeline = manga.page_timeline(page, 0.0, 12.0, _timings(3))
    assert [round(end - begin, 2) for _, begin, end in timeline[1:]] == [5.0, 5.0]


def test_manga_falls_back_to_scenes_without_gemini(monkeypatch, tmp_path):
    _no_gemini(monkeypatch)
    monkeypatch.setattr(script_images, "ask_gemini_json", lambda request: (_ for _ in ()).throw(RuntimeError("x")))
    seen = {}
    monkeypatch.setattr(script_images, "make_slideshow", lambda images, slides, audio, output: None)

    def fake_generate(prompts, sizes=None):
        seen["sizes"] = sizes
        return [b"png"] * len(prompts)

    monkeypatch.setattr(image_client, "generate", fake_generate)
    timings = _timings(10)
    script_images.illustrate([f"Câu {i}." for i in range(10)], timings, tmp_path / "a.mp3",
                             tmp_path / "a.mp4", tmp_path / "w", "manga")
    assert seen["sizes"] is None  # đã lùi về ảnh theo cảnh 16:9


def test_pipeline_makes_manga_video(monkeypatch, tmp_path):
    pytest.importorskip("pydub")
    pytest.importorskip("PIL")
    if not shutil.which("ffmpeg"):
        pytest.skip("cần ffmpeg")
    _fake_tts(monkeypatch, tmp_path)
    board = {"characters": [{"id": "a", "look": "a boy"}], "locations": [],
             "pages": [{"start": 1, "panels": [{"start": 1, "characters": ["a"], "action": "runs"},
                                               {"start": 2, "action": "a door"}]},
                       {"start": 4, "panels": [{"start": 4, "action": "rain"}]}]}
    monkeypatch.setattr(script_images, "ask_gemini_json", lambda request: json.dumps(board))
    monkeypatch.setattr(manga, "ask_gemini_json", lambda request: json.dumps(board))
    seen = {}

    def fake_generate(prompts, sizes=None):
        seen["sizes"] = sizes
        return [_png("gray") for _ in prompts]

    monkeypatch.setattr(image_client, "generate", fake_generate)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    segments = [Segment(0, 0, f"Câu {i}.", f"Câu {i}.") for i in range(6)]
    result = DubbingPipeline(DubbingConfig(tts_engine="edge-tts", output_dir=out_dir, script_images="manga")) \
        .synthesize_script(segments)

    assert result.success and not result.notice, result.notice
    assert result.output_video.suffix == ".mp4" and result.output_video.exists()
    # Audio giả chỉ vài giây: trang 2 quá ngắn nên gộp vào trang 1, còn một trang 2 khung.
    assert seen["sizes"] == [manga.draw_size(c) for c in manga.layout(2)]
