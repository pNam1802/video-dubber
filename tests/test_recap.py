"""Chế độ "Manhwa recap": storyboard, lịch chiếu, khung hình."""
from __future__ import annotations

import io
import json

from core import recap, script_images


def _timings(count: int, sentence_sec: float = 2.6, pause: float = 0.4):
    out, t = [], 0.0
    for _ in range(count):
        out.append((t, t + sentence_sec))
        t += sentence_sec + pause
    return out


def _board(shots, characters=None, locations=None) -> str:
    return json.dumps({"characters": characters or [{"id": "jun", "look": "young man, black hair, grey hoodie"}],
                       "locations": locations or [{"id": "room", "look": "dim bedroom at night"}],
                       "shots": shots})


def test_recap_is_an_image_mode():
    assert "recap" in script_images.MODES


def test_compose_shot_inserts_fixed_look_and_color_style():
    prompt = recap.compose_shot({"shot": "close-up", "characters": ["Jun"], "location": "room",
                                 "action": "jun looks shocked."},
                                {"jun": "young man, black hair"}, {"room": "dim bedroom"})
    assert prompt.startswith("close-up, jun looks shocked, jun: young man, black hair, solo")
    assert "background: dim bedroom" in prompt
    assert "full color" in prompt and "no speech bubbles" in prompt


def test_character_style_follows_gender_in_look():
    cast = recap.style_cast({"a": "young woman, long brown hair", "b": "teenage girl",
                             "c": "old man, grey beard", "d": "a black cat"})
    assert cast["a"] == f"young woman, long brown hair, {recap.FEMALE_STYLE}"
    assert cast["b"].endswith(recap.FEMALE_STYLE)
    assert cast["c"].endswith(recap.MALE_STYLE)
    assert cast["d"] == "a black cat"


def test_sunflower_style_dresses_only_the_first_woman_like_the_sample():
    cast = recap.style_cast({"jun": "young man", "yena": "young woman, brown hair, lavender cardigan",
                             "mom": "middle-aged woman, short grey hair"}, "sunflower")
    assert cast["yena"] == f"{recap.SUNFLOWER_HEROINE}, {recap.FEMALE_STYLE}"
    assert cast["mom"].startswith("middle-aged woman, short grey hair")
    assert cast["jun"].endswith(recap.MALE_STYLE)


def test_parse_recap_applies_character_style():
    raw = _board([{"start": 1, "characters": ["yena"], "action": "yena smiles"}],
                 characters=[{"id": "yena", "look": "young woman, brown hair"}])
    assert recap.SUNFLOWER_HEROINE in recap.parse_recap(raw, 3, "sunflower")[0][1]
    assert recap.SUNFLOWER_HEROINE not in recap.parse_recap(raw, 3)[0][1]


def test_every_character_style_has_a_preview_image():
    from config.settings import ROOT_DIR

    for style in script_images.RECAP_CHARACTER_STYLES:
        assert (ROOT_DIR / "static" / "recap_styles" / f"{style}.jpg").is_file()


def test_compose_shot_drops_unknown_characters_and_caps_two():
    looks = {"a": "A", "b": "B", "c": "C"}
    prompt = recap.compose_shot({"characters": ["a", "ghost", "b", "c"], "action": "x"}, looks, {})
    assert "a on the left: A" in prompt and "b on the right: B" in prompt and "C" not in prompt.split(", ")
    assert "softly blurred background" in prompt


def test_parse_recap_sorts_skips_broken_and_starts_at_zero():
    raw = _board([{"start": 4, "action": "later"}, {"start": 2, "action": "first"},
                  {"start": 3, "action": ""}, {"start": "abc", "action": "bad"}, "junk"])
    board = recap.parse_recap(raw, 10)
    assert [start for start, _ in board] == [0, 3]
    assert board[0][1].startswith("first")


def test_plan_covers_whole_audio_without_gaps(monkeypatch):
    timings = _timings(20)
    monkeypatch.setattr(recap, "ask_gemini_json",
                        lambda req: _board([{"start": i, "action": f"beat {i}"} for i in range(1, 21, 2)]))
    prompts, slides = recap.plan([f"S{i}." for i in range(20)], timings)
    assert len(prompts) == len(slides)
    assert slides[0][1] == 0.0 and slides[-1][2] == timings[-1][1]
    assert all(a[2] == b[1] for a, b in zip(slides, slides[1:]))
    # bước 2 câu = 6 giây, đúng nhịp recap
    assert all(end - start >= recap.MIN_SHOT_SEC for _, start, end in slides[:-1])


def test_plan_falls_back_when_gemini_fails(monkeypatch):
    def boom(req):
        raise RuntimeError("không có key")

    monkeypatch.setattr(recap, "ask_gemini_json", boom)
    timings = _timings(30)
    prompts, slides = recap.plan([f"S{i}." for i in range(30)], timings)
    assert len(prompts) == len(slides) >= 10
    assert slides[-1][2] == timings[-1][1]
    assert all("full color" in p for p in prompts)


def test_long_script_is_capped_at_max_shots(monkeypatch):
    timings = _timings(800)  # ~40 phút
    monkeypatch.setattr(recap, "ask_gemini_json",
                        lambda req: _board([{"start": i, "action": "x"} for i in range(1, 801)]))
    prompts, slides = recap.plan([f"S{i}." for i in range(800)], timings)
    assert len(prompts) <= recap.MAX_SHOTS
    assert slides[-1][2] == timings[-1][1]


def test_clip_frames_do_not_drift():
    slides = [(i, i * 6.013, (i + 1) * 6.013) for i in range(100)]
    assert sum(recap.clip_frames(slides)) == round(slides[-1][2] * recap.FPS)


def test_compose_frame_keeps_portrait_centered_on_blurred_background(tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", recap.DRAW_SIZE, (200, 40, 40)).save(buf, "PNG")
    recap.compose_frame(buf.getvalue(), tmp_path / "f.png")
    frame = Image.open(tmp_path / "f.png").convert("RGB")
    assert frame.size == (recap.FRAME_W, recap.FRAME_H)
    center, edge = frame.getpixel((recap.FRAME_W // 2, recap.FRAME_H // 2)), frame.getpixel((5, recap.FRAME_H // 2))
    assert center == recap.grade(Image.new("RGB", (8, 8), (200, 40, 40))).getpixel((4, 4))
    assert edge[0] < center[0]  # hai bên tối hơn


def test_grade_makes_washed_out_colors_bolder():
    from PIL import Image

    pastel = Image.new("RGB", (8, 8), (220, 190, 180))
    graded = recap.grade(pastel).getpixel((4, 4))
    assert graded[0] - graded[2] > 220 - 180  # màu đậm hơn
