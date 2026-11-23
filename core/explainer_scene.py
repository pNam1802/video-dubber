"""
core/explainer_scene.py
Một cảnh của chế độ "explainer" (video giải thích kiểu 3Blue1Brown, vẽ bằng Manim).

File này do `manim render` nạp trong một tiến trình riêng (xem core/explainer.render_scene):
đặc tả cảnh nằm trong file JSON có đường dẫn ở biến môi trường EXPLAINER_SPEC. LLM chỉ
chọn bố cục và điền chữ — không bao giờ sinh code — nên mọi hình vẽ đều nằm cố định ở đây.

Đặc tả: {"layout": ..., <các trường của bố cục>, "duration": giây}
Bố cục: title, bullets, flow, vectors, compare, chat, statement (xem core/explainer.LAYOUTS).
"""
from __future__ import annotations

import json
import math
import os
import random
import textwrap
import unicodedata
from pathlib import Path

import manimpango
from manim import (
    BOLD, DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Circle, Create, Dot, FadeIn, GrowArrow,
    GrowFromCenter, Indicate, Line, NumberPlane, RoundedRectangle, Scene, Text, VGroup, Write, config,
)

FONT = "Be Vietnam Pro"
FONT_DIR = Path(__file__).parent / "fonts"
BG = "#0f1117"
INK = "#e8eaf0"
MUTED = "#9aa3b5"
ACCENT = "#58c4dd"   # xanh kiểu 3Blue1Brown
GOLD = "#ffd166"
GREEN = "#83c167"
PINK = "#fc6255"
FRAME_W = 12.6       # bề ngang dùng được (đơn vị Manim, khung 14.2 x 8)

for _font in FONT_DIR.glob("*.ttf"):
    manimpango.register_font(str(_font))
config.background_color = BG


#: Ký tự LLM hay dùng mà Be Vietnam Pro không có (hiện thành ô vuông) -> ký tự font có.
GLYPH_FALLBACK = str.maketrans({"→": "»", "⇒": "»", "➜": "»", "←": "«", "⇐": "«", "≈": "~", "✓": "•",
                                "✔": "•", "✗": "×", "❌": "×", "✅": "•"})


def _nfc(text: str) -> str:
    # Pango vẽ dấu tách rời (ô vuông) nếu chữ tiếng Việt ở dạng tổ hợp NFD.
    return unicodedata.normalize("NFC", str(text or "")).translate(GLYPH_FALLBACK).strip()


def label(text: str, size: float = 30, color: str = INK, bold: bool = False, width_chars: int = 0,
          max_width: float = FRAME_W) -> Text:
    text = _nfc(text)
    if width_chars:
        text = "\n".join(textwrap.wrap(text, width_chars)) or " "
    mob = Text(text, font=FONT, font_size=size, color=color, weight=BOLD if bold else "NORMAL",
               line_spacing=0.9)
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


class ExplainerScene(Scene):
    def setup(self):
        self.spec = json.loads(Path(os.environ["EXPLAINER_SPEC"]).read_text(encoding="utf-8"))
        self.duration = max(1.0, float(self.spec.get("duration", 5)))
        self.clock = 0.0

    # ── Nhịp thời gian: mọi cảnh phải dài đúng bằng đoạn lời đọc của nó ──
    def run(self, *animations, run_time: float = 0.8):
        run_time = max(0.2, min(run_time, self.remaining() - 0.05))
        if run_time < 0.2:
            self.add(*[a.mobject for a in animations])
            return
        self.play(*animations, run_time=run_time)
        self.clock += run_time

    def remaining(self) -> float:
        return self.duration - self.clock

    def wait_until(self, t: float):
        gap = min(t, self.duration) - self.clock
        if gap > 0.04:
            self.wait(gap)
            self.clock += gap

    def reveal_times(self, n: int, start: float = 0.9) -> list[float]:
        """Hiện n phần tử rải đều trong ~70% đầu cảnh, phần còn lại để người xem đọc."""
        end = max(start + 0.1, self.duration * 0.7)
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    def heading(self, text: str):
        if not text:
            return None
        head = label(text, 40, INK, bold=True, max_width=FRAME_W).to_edge(UP, buff=0.5)
        rule = Line(LEFT, RIGHT, color=ACCENT, stroke_width=3).set_width(min(head.width, FRAME_W))
        rule.next_to(head, DOWN, buff=0.15)
        self.run(Write(head), Create(rule), run_time=0.8)
        return VGroup(head, rule)

    # ── Bố cục ──
    def construct(self):
        draw = getattr(self, "layout_" + str(self.spec.get("layout")), self.layout_statement)
        draw(self.spec)
        self.wait_until(self.duration)

    def layout_title(self, s):
        title = label(s.get("title"), 60, INK, bold=True, width_chars=28)
        bar = Line(LEFT, RIGHT, color=ACCENT, stroke_width=4).set_width(min(title.width, 8) * 0.6)
        sub = label(s.get("subtitle"), 30, MUTED, width_chars=50) if s.get("subtitle") else None
        VGroup(title, bar, *([sub] if sub else [])).arrange(DOWN, buff=0.4).move_to(ORIGIN)
        self.run(Write(title), run_time=1.2)
        self.run(GrowFromCenter(bar), run_time=0.5)
        if sub:
            self.run(FadeIn(sub, shift=UP * 0.2), run_time=0.7)

    def layout_bullets(self, s):
        self.heading(s.get("heading"))
        items = [i for i in s.get("items", []) if i][:5]
        rows = VGroup(*[
            VGroup(Dot(color=ACCENT, radius=0.08), label(text, 32, INK, width_chars=46)).arrange(RIGHT, buff=0.3)
            for text in items
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        if rows.height > 5.6:
            rows.scale_to_fit_height(5.6)
        rows.move_to(DOWN * 0.4 if s.get("heading") else ORIGIN)
        for row, t in zip(rows, self.reveal_times(len(rows))):
            self.wait_until(t)
            self.run(FadeIn(row, shift=RIGHT * 0.3), run_time=0.6)

    def layout_flow(self, s):
        self.heading(s.get("heading"))
        steps = [i for i in s.get("steps", []) if i][:5]
        boxes = VGroup()
        for i, text in enumerate(steps):
            txt = label(text, 26, INK, bold=True, width_chars=12)
            box = RoundedRectangle(corner_radius=0.18, width=max(2.0, txt.width + 0.5),
                                   height=max(1.2, txt.height + 0.5), color=ACCENT, stroke_width=3)
            box.set_fill(ACCENT, opacity=0.08)
            boxes.add(VGroup(box, txt))
        boxes.arrange(RIGHT, buff=0.8)
        if boxes.width > FRAME_W:
            boxes.scale_to_fit_width(FRAME_W)
        boxes.move_to(DOWN * 0.3)
        arrows = [Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1, color=MUTED,
                        stroke_width=4, max_tip_length_to_length_ratio=0.3) for i in range(len(boxes) - 1)]
        times = self.reveal_times(len(boxes), start=0.9)
        for i, (box, t) in enumerate(zip(boxes, times)):
            self.wait_until(t)
            anims = [FadeIn(box, shift=UP * 0.2)]
            if i:
                anims.insert(0, GrowArrow(arrows[i - 1]))
            self.run(*anims, run_time=0.6)
        # Dữ liệu "chảy" qua các bước: chấm vàng nhảy từ ô này sang ô kia.
        if len(boxes) > 1 and self.remaining() > 2.5:
            dot = Dot(boxes[0].get_bottom() + DOWN * 0.35, color=GOLD, radius=0.12)
            self.run(FadeIn(dot), run_time=0.3)
            hop = min(0.8, (self.remaining() - 0.5) / len(boxes))
            for box in boxes[1:]:
                self.run(dot.animate.move_to(box.get_bottom() + DOWN * 0.35),
                         Indicate(box[0], color=GOLD, scale_factor=1.05), run_time=hop)

    def layout_vectors(self, s):
        self.heading(s.get("heading"))
        items = [i for i in s.get("items", []) if i][:6]
        near = {int(i) for i in s.get("nearest", []) if str(i).isdigit() and int(i) < len(items)} or {0}
        plane = NumberPlane(x_range=[-6, 6, 1], y_range=[-3, 3, 1], x_length=11, y_length=5.2,
                            background_line_style={"stroke_color": ACCENT, "stroke_opacity": 0.18,
                                                   "stroke_width": 1},
                            axis_config={"stroke_opacity": 0.35}).move_to(DOWN * 0.45)
        self.run(Create(plane), run_time=0.8)
        # Vị trí cố định theo nội dung để render lại vẫn ra cùng hình.
        rng = random.Random("|".join(items))
        q_pos = (rng.uniform(-1.0, 1.0), rng.uniform(-0.4, 0.4))
        # Rải đều theo góc để nhãn không đè nhau; tài liệu "gần" nằm sát câu hỏi, còn lại ở xa.
        offset = rng.uniform(0, math.tau)
        points = VGroup()
        for i, text in enumerate(items):
            angle = offset + i * math.tau / len(items)
            radius = 1.7 if i in near else rng.uniform(3.4, 4.2)
            x = max(-5.4, min(5.4, q_pos[0] + radius * 1.25 * math.cos(angle)))
            y = max(-2.6, min(2.6, q_pos[1] + radius * 0.62 * math.sin(angle)))
            dot = Dot(plane.c2p(x, y), color=MUTED, radius=0.1)
            side = UP if math.sin(angle) >= 0 else DOWN
            if abs(math.cos(angle)) > 0.8:
                side = RIGHT if math.cos(angle) > 0 else LEFT
            tag = label(text, 22, MUTED, width_chars=18, max_width=3.2).next_to(dot, side, buff=0.12)
            points.add(VGroup(dot, tag))
        for point, t in zip(points, self.reveal_times(len(points), start=1.4)):
            self.wait_until(min(t, self.duration * 0.45))
            self.run(FadeIn(point, scale=0.6), run_time=0.4)
        query_dot = Dot(plane.c2p(*q_pos), color=GOLD, radius=0.14)
        query_tag = label(s.get("query") or "?", 26, GOLD, bold=True, width_chars=22,
                          max_width=4).next_to(query_dot, UP, buff=0.15)
        self.run(GrowFromCenter(query_dot), Write(query_tag), run_time=0.8)
        lines = [Line(query_dot.get_center(), points[i][0].get_center(), color=GREEN, stroke_width=4)
                 for i in sorted(near) if i < len(points)]
        self.run(*[Create(line) for line in lines], run_time=0.8)
        self.run(*[points[i][0].animate.set_color(GREEN).scale(1.5) for i in sorted(near) if i < len(points)],
                 *[points[i][1].animate.set_color(GREEN) for i in sorted(near) if i < len(points)],
                 run_time=0.6)
        reach = plane.c2p(q_pos[0] + 1.7 * 1.25 + 0.4, q_pos[1])[0] - query_dot.get_center()[0]
        ring = Circle(radius=reach, color=GREEN, stroke_width=2).move_to(query_dot).stretch(0.55, 1)
        self.run(Create(ring), run_time=0.6)

    def layout_compare(self, s):
        self.heading(s.get("heading"))

        def column(title, items, color):
            top = label(title, 36, color, bold=True, width_chars=22, max_width=5.6)
            rows = VGroup(*[label("• " + _nfc(t), 30, INK, width_chars=26, max_width=5.8)
                            for t in items if t][:4]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            return VGroup(top, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.45)

        left = column(s.get("left_title"), s.get("left", []), PINK)
        right = column(s.get("right_title"), s.get("right", []), GREEN)
        for col, x in ((left, -3.3), (right, 3.3)):
            if col.width > 5.8:
                col.scale_to_fit_width(5.8)
            col.move_to(RIGHT * x + DOWN * 0.3)
        top = max(left.get_top()[1], right.get_top()[1])
        left.align_to(UP * top, UP)
        right.align_to(UP * top, UP)
        divider = Line(UP * 2.0, DOWN * 2.8, color=MUTED, stroke_width=2)
        t1, t2 = self.reveal_times(2)
        self.wait_until(t1)
        self.run(FadeIn(left, shift=RIGHT * 0.3), Create(divider), run_time=0.8)
        self.wait_until(t2)
        self.run(FadeIn(right, shift=LEFT * 0.3), run_time=0.8)

    def layout_chat(self, s):
        def bubble(text, color, align):
            txt = label(text, 28, INK, width_chars=38, max_width=8.5)
            box = RoundedRectangle(corner_radius=0.25, width=txt.width + 0.6, height=txt.height + 0.5,
                                   color=color, stroke_width=2).set_fill(color, opacity=0.15)
            group = VGroup(box, txt)
            return group.to_edge(align, buff=1.0)

        q = bubble(s.get("question"), ACCENT, RIGHT)
        a = bubble(s.get("answer"), GREEN, LEFT)
        VGroup(q, a).arrange(DOWN, buff=0.6)
        q.to_edge(RIGHT, buff=1.0)
        a.to_edge(LEFT, buff=1.0)
        self.run(FadeIn(q, shift=LEFT * 0.4), run_time=0.7)
        self.wait_until(self.reveal_times(2)[1])
        self.run(FadeIn(a, shift=RIGHT * 0.4), run_time=0.8)

    def layout_statement(self, s):
        text = label(s.get("text") or s.get("heading") or "", 46, INK, bold=True, width_chars=30)
        cap = label(s.get("caption"), 28, ACCENT, width_chars=50) if s.get("caption") else None
        VGroup(text, *([cap] if cap else [])).arrange(DOWN, buff=0.5).move_to(ORIGIN)
        self.run(Write(text), run_time=min(1.5, 0.3 + 0.04 * len(_nfc(s.get("text") or ""))))
        if cap:
            self.wait_until(self.duration * 0.35)
            self.run(FadeIn(cap, shift=UP * 0.2), run_time=0.6)
