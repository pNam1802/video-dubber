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

import importlib.util
import json
import math
import os
import random
import textwrap
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import manimpango
from manim import (
    BOLD, DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, ArcBetweenPoints, Circle, Create, Dot, FadeIn, FadeOut,
    GrowArrow, GrowFromCenter, GrowFromEdge, Indicate, LaggedStart, Line, NumberPlane, Rectangle,
    RoundedRectangle, Scene, ShowPassingFlash, SurroundingRectangle, Text, VGroup, Write, config,
)

FONT = "Be Vietnam Pro"
MONO = "JetBrains Mono"
FONT_DIR = Path(__file__).parent / "fonts"
BG = "#0f1117"
INK = "#e8eaf0"
MUTED = "#9aa3b5"
ACCENT = "#58c4dd"   # xanh kiểu 3Blue1Brown
GOLD = "#ffd166"
GREEN = "#83c167"
PINK = "#fc6255"
PURPLE = "#b189f5"
ORANGE = "#f19c65"
PALETTE = (ACCENT, GOLD, GREEN, PINK, PURPLE, ORANGE)
FRAME_W = 12.6       # bề ngang dùng được (đơn vị Manim, khung 14.2 x 8)
#: Cảnh viết tay riêng cho từng tập (bố cục "custom"): core/episodes/<module>.py, mỗi cảnh một hàm
#: draw(scene, spec, kit) — kit là các hàm/màu dùng chung của file này.
EPISODES_DIR = Path(__file__).with_name("episodes")

for _font in FONT_DIR.glob("*.ttf"):
    manimpango.register_font(str(_font))
# Nền tối trơn kiểu 3Blue1Brown, không vẽ lưới nền: vài trăm đường kẻ Cairo phải vẽ lại ở MỌI
# khung làm render chậm thêm ~40% (đo 23/11/2026).
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


def episode_module(name):
    """Nạp core/episodes/<name>.py (cảnh vẽ tay của một tập)."""
    spec = importlib.util.spec_from_file_location(f"episode_{name}", EPISODES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kit() -> SimpleNamespace:
    """Màu, font và hàm dùng chung, đưa cho code của từng tập."""
    return SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("_") or k == "_nfc"})


class ThumbnailScene(Scene):
    """Ảnh thumbnail 1280x720 của một tập: hàm thumbnail(sc, k, lang) trong module của tập.
    Render bằng `manim -s` (chỉ khung cuối), xem core/explainer.render_thumbnail."""

    def construct(self):
        spec = json.loads(Path(os.environ["EXPLAINER_SPEC"]).read_text(encoding="utf-8"))
        episode_module(spec["module"]).thumbnail(self, kit(), spec.get("lang", "vi"))


class ExplainerScene(Scene):
    def setup(self):
        self.spec = json.loads(Path(os.environ["EXPLAINER_SPEC"]).read_text(encoding="utf-8"))
        # Không đặt tên self.duration: Manim tự ghi đè thuộc tính đó trong mỗi Scene.play().
        self.scene_sec = max(1.0, float(self.spec.get("duration", 5)))
        self.clock = 0.0
        # Mốc bắt đầu từng câu trong cảnh (giây, tính từ đầu cảnh): phần tử hiện đúng lúc được nói tới.
        self.beats = sorted(float(b) for b in self.spec.get("beats", []) if 0 < float(b) < self.scene_sec)

    # ── Nhịp thời gian: mọi cảnh phải dài đúng bằng đoạn lời đọc của nó ──
    def run(self, *animations, run_time: float = 0.8):
        run_time = max(0.2, min(run_time, self.remaining() - 0.05))
        if run_time < 0.2:
            self.add(*[a.mobject for a in animations])
            return
        self.play(*animations, run_time=run_time)
        self.clock += run_time

    def remaining(self) -> float:
        return self.scene_sec - self.clock

    def wait_until(self, t: float):
        gap = min(t, self.scene_sec) - self.clock
        if gap > 0.04:
            self.wait(gap)
            self.clock += gap

    def reveal_times(self, n: int, start: float = 0.9) -> list[float]:
        """Lúc hiện từng phần tử trong n phần tử. Có đủ mốc câu thì bám theo lời đọc (phần tử
        thứ k hiện khi giọng đọc sang câu tương ứng); không thì rải đều trong ~70% đầu cảnh."""
        if n <= 0:
            return []
        end = max(start + 0.1, self.scene_sec * 0.7)
        even = [start] if n == 1 else [start + i * (end - start) / (n - 1) for i in range(n)]
        # Chỉ dùng mốc câu trong ~70% đầu cảnh: phần tử cuối mà bám câu cuối thì chỉ kịp hiện
        # một hai giây rồi cảnh đã mờ đi (đo 24/11/2026: câu trả lời chat hiện ở giây 19/20).
        marks = [start] + [b for b in self.beats if start + 0.4 < b <= end]
        if n == 1 or len(marks) < n:
            return even
        picked = [marks[round(i * (len(marks) - 1) / (n - 1))] for i in range(n)]
        if any(b - a < 0.45 for a, b in zip(picked, picked[1:])):
            return even
        return picked

    def at(self, k: int) -> float:
        """Lúc câu thứ k của cảnh bắt đầu được đọc (câu 0 = đầu cảnh). Thiếu mốc thì ~70% cảnh."""
        if k <= 0:
            return 0.0
        return self.beats[k - 1] if k - 1 < len(self.beats) else self.scene_sec * 0.7

    def layout_custom(self, s):
        getattr(episode_module(s.get("module")), str(s.get("scene")))(self, s, kit())

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
        # Mờ dần cuối cảnh để nối sang cảnh sau mượt hơn cắt cứng.
        self.wait_until(self.scene_sec - 0.35)
        content = list(self.mobjects)
        if content and self.remaining() > 0.25:
            self.run(*[FadeOut(m) for m in content], run_time=self.remaining() - 0.02)
        self.wait_until(self.scene_sec)

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
            self.wait_until(min(t, self.scene_sec * 0.45))
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
        # Câu trả lời hiện khi lời đọc sang câu thứ hai (thường là lúc kể máy trả lời gì).
        later = [b for b in self.beats if b > self.clock + 0.4]
        self.wait_until(later[0] if later and later[0] <= self.scene_sec * 0.7 else self.reveal_times(2)[1])
        self.run(FadeIn(a, shift=RIGHT * 0.4), run_time=0.8)

    def layout_statement(self, s):
        text = label(s.get("text") or s.get("heading") or "", 46, INK, bold=True, width_chars=30)
        cap = label(s.get("caption"), 28, ACCENT, width_chars=50) if s.get("caption") else None
        VGroup(text, *([cap] if cap else [])).arrange(DOWN, buff=0.5).move_to(ORIGIN)
        self.run(Write(text), run_time=min(1.5, 0.3 + 0.04 * len(_nfc(s.get("text") or ""))))
        if cap:
            self.wait_until(self.scene_sec * 0.35)
            self.run(FadeIn(cap, shift=UP * 0.2), run_time=0.6)

    # ── Bố cục "vẽ": hình chiếm phần lớn khung, chữ chỉ là nhãn ──
    def layout_tokens(self, s):
        """Câu chữ tách thành các ô token màu, kèm mã số token bên dưới."""
        self.heading(s.get("heading"))
        tokens = [str(t) for t in s.get("tokens", []) if str(t).strip()][:12]
        sentence = label(s.get("text") or "".join(tokens), 40, INK, width_chars=40).move_to(UP * 1.4)
        self.run(Write(sentence), run_time=1.0)
        chips = VGroup()
        for i, tok in enumerate(tokens):
            txt = Text(_nfc(tok) or "_", font=MONO, font_size=30, color=INK)
            if txt.width > 3.0:
                txt.scale_to_fit_width(3.0)
            color = PALETTE[i % len(PALETTE)]
            box = RoundedRectangle(corner_radius=0.12, width=txt.width + 0.35, height=0.8, color=color,
                                   stroke_width=2).set_fill(color, opacity=0.18)
            txt.move_to(box)
            chips.add(VGroup(box, txt))
        if not chips:
            return
        chips.arrange(RIGHT, buff=0.15)
        if chips.width > FRAME_W:
            chips.arrange_in_grid(rows=2, buff=(0.15, 0.7))
            if chips.width > FRAME_W:
                chips.scale_to_fit_width(FRAME_W)
        chips.move_to(DOWN * 0.5)
        rng = random.Random("|".join(tokens))
        ids = VGroup()
        for chip in chips:
            tag = Text(str(rng.randint(100, 99999)), font=MONO, font_size=18, color=MUTED)
            if tag.width > chip.width:
                tag.scale_to_fit_width(chip.width * 0.95)
            ids.add(tag.next_to(chip, DOWN, buff=0.12))
        for chip, t in zip(chips, self.reveal_times(len(chips), start=1.6)):
            self.wait_until(min(t, self.scene_sec * 0.6))
            self.run(FadeIn(chip, shift=DOWN * 0.3, scale=0.8), run_time=0.35)
        self.run(LaggedStart(*[FadeIn(i, shift=UP * 0.1) for i in ids], lag_ratio=0.08), run_time=0.8)
        count = label(f"{len(chips)} token", 28, GOLD, bold=True).next_to(chips, DOWN, buff=0.8)
        self.run(FadeIn(count, shift=UP * 0.2), run_time=0.5)

    def layout_probabilities(self, s):
        """Biểu đồ cột xác suất cho từ/token tiếp theo; cột cao nhất được chọn và nối vào câu."""
        self.heading(s.get("heading"))
        options = sorted(s.get("options", []), key=lambda o: -o["p"])[:6]
        if not options:
            return
        context = label(s.get("context") or "", 34, INK, width_chars=40, max_width=10)
        blank = Text("___", font=MONO, font_size=34, color=GOLD)
        VGroup(context, blank).arrange(RIGHT, buff=0.2).move_to(UP * 1.9)
        self.run(Write(context), FadeIn(blank), run_time=0.9)
        max_w, rows = 6.0, VGroup()
        for i, opt in enumerate(options):
            name = label(opt["label"], 28, INK, max_width=3.0)
            bar = Rectangle(width=max(0.05, max_w * opt["p"]), height=0.5, stroke_width=0,
                            fill_color=GOLD if i == 0 else ACCENT, fill_opacity=0.85 if i == 0 else 0.5)
            pct = Text(f"{opt['p'] * 100:.0f}%", font=MONO, font_size=24, color=MUTED)
            rows.add(VGroup(name, bar, pct))
        left_edge = -max_w / 2 - 0.5
        for k, row in enumerate(rows):
            y = -0.2 - k * 0.72
            row[1].move_to([0, y, 0]).align_to([left_edge + 0.3, 0, 0], LEFT)
            row[0].move_to([0, y, 0]).align_to([left_edge, 0, 0], RIGHT)
            row[2].next_to(row[1], RIGHT, buff=0.2)
        rows.move_to(DOWN * 0.9).shift(RIGHT * 0.8)
        self.run(*[FadeIn(r[0]) for r in rows], run_time=0.5)
        self.run(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.15),
                 *[FadeIn(r[2]) for r in rows], run_time=1.4)
        self.wait_until(self.scene_sec * 0.55)
        chosen = label(options[0]["label"], 34, GOLD, bold=True).move_to(blank).align_to(blank, LEFT)
        self.run(Create(SurroundingRectangle(rows[0], color=GOLD, buff=0.12)), run_time=0.5)
        self.run(FadeOut(blank), FadeIn(chosen, shift=UP * 0.3), run_time=0.6)

    def layout_attention(self, s):
        """Một token "nhìn" về các token khác: cung nối dày mỏng theo trọng số chú ý."""
        self.heading(s.get("heading"))
        tokens = [str(t) for t in s.get("tokens", []) if str(t).strip()][:8]
        if len(tokens) < 2:
            return self.layout_statement({"text": " ".join(tokens)})
        focus = s.get("focus", len(tokens) - 1)
        focus = focus if isinstance(focus, int) and 0 <= focus < len(tokens) else len(tokens) - 1
        attends = [i for i in s.get("attends", []) if 0 <= i < len(tokens) and i != focus] or [0 if focus else 1]
        chips = VGroup()
        for i, tok in enumerate(tokens):
            txt = label(tok, 30, INK, bold=i == focus, max_width=2.6)
            box = RoundedRectangle(corner_radius=0.12, width=txt.width + 0.4, height=0.85,
                                   color=GOLD if i == focus else MUTED, stroke_width=2)
            txt.move_to(box)
            chips.add(VGroup(box, txt))
        chips.arrange(RIGHT, buff=0.35)
        if chips.width > FRAME_W:
            chips.scale_to_fit_width(FRAME_W)
        chips.move_to(UP * 1.0)
        self.run(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in chips], lag_ratio=0.1), run_time=1.0)
        self.wait_until(self.reveal_times(2, start=1.2)[0])
        self.run(Indicate(chips[focus], color=GOLD, scale_factor=1.12), run_time=0.7)
        rng = random.Random("|".join(tokens))
        weights = {i: rng.uniform(0.55, 0.95) if i in attends else rng.uniform(0.03, 0.15)
                   for i in range(len(tokens)) if i != focus}
        arcs = VGroup()
        for i, w in sorted(weights.items(), key=lambda kv: kv[1]):
            start, end = chips[focus].get_bottom(), chips[i].get_bottom()
            arc = ArcBetweenPoints(start + DOWN * 0.08, end + DOWN * 0.08, angle=-1.4 if i < focus else 1.4,
                                   color=GOLD if i in attends else MUTED, stroke_width=1 + 12 * w)
            arc.set_stroke(opacity=0.25 + 0.75 * w)
            arcs.add(arc)
        self.run(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.1), run_time=1.4)
        tags = VGroup(*[Text(f"{weights[i]:.2f}", font=MONO, font_size=22, color=GOLD)
                        .next_to(chips[i], UP, buff=0.15) for i in attends])
        self.run(*[FadeIn(t) for t in tags], *[chips[i][0].animate.set_color(GOLD) for i in attends], run_time=0.6)
        if s.get("caption"):
            cap = label(s["caption"], 28, ACCENT, width_chars=50).to_edge(DOWN, buff=0.6)
            self.run(FadeIn(cap, shift=UP * 0.2), run_time=0.5)

    def layout_network(self, s):
        """Mạng nơ-ron: các lớp nút, tín hiệu chạy từ đầu vào tới đầu ra."""
        self.heading(s.get("heading"))
        sizes = [max(1, min(6, int(n))) for n in s.get("layers", [])][:5]
        if len(sizes) < 2:
            sizes = [3, 5, 5, 2]
        layers = VGroup()
        for n in sizes:
            layers.add(VGroup(*[Circle(radius=0.22, color=ACCENT, stroke_width=3).set_fill(BG, opacity=1)
                                for _ in range(n)]).arrange(DOWN, buff=0.35))
        layers.arrange(RIGHT, buff=2.0).move_to(DOWN * 0.3)
        if layers.height > 5.2:
            layers.scale_to_fit_height(5.2)
        edges = VGroup(*[VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1.2, color=MUTED)
                                  .set_stroke(opacity=0.35) for a in left for b in right])
                         for left, right in zip(layers, layers[1:])])
        for layer in layers:
            layer.set_z_index(2)
        labels = VGroup()
        for text, layer, side in ((s.get("input_label"), layers[0], LEFT), (s.get("output_label"), layers[-1], RIGHT)):
            if text:
                labels.add(label(text, 26, MUTED, width_chars=12, max_width=2.2).next_to(layer, side, buff=0.4))
        self.run(LaggedStart(*[LaggedStart(*[GrowFromCenter(n) for n in layer], lag_ratio=0.1) for layer in layers],
                             lag_ratio=0.3), run_time=1.4)
        self.run(Create(edges), *[FadeIn(lbl) for lbl in labels], run_time=1.0)
        # Tín hiệu: các cạnh sáng lên lần lượt từng lớp, lặp tới gần hết cảnh.
        while self.remaining() > 1.0 + 0.8 * len(edges):
            for group, layer in zip(edges, layers[1:]):
                flashes = [ShowPassingFlash(e.copy().set_stroke(GOLD, width=3, opacity=1), time_width=0.6)
                           for e in group]
                self.run(*flashes, *[n.animate.set_fill(GOLD, opacity=0.6) for n in layer], run_time=0.6)
                self.run(*[n.animate.set_fill(BG, opacity=1) for n in layer], run_time=0.2)
            self.wait_until(self.clock + 0.4)

    def layout_code(self, s):
        """Cửa sổ terminal/editor, các dòng hiện lần lượt theo lời đọc."""
        self.heading(s.get("heading"))
        lines = [str(line) for line in s.get("lines", [])][:10]
        if not lines:
            return
        rows = VGroup()
        for line in lines:
            stripped = line.lstrip()
            color = MUTED if stripped.startswith(("#", "//")) else GREEN if stripped.startswith(("$", ">>>")) else INK
            rows.add(Text(stripped or " ", font=MONO, font_size=26, color=color))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        # Manim bỏ khoảng trắng đầu dòng: dịch từng dòng sang phải theo độ thụt, đo bằng bề rộng một ký tự.
        char_w = Text("M", font=MONO, font_size=26).width
        for row, line in zip(rows, lines):
            row.shift(RIGHT * char_w * (len(line) - len(line.lstrip())))
        if rows.width > 10.5:
            rows.scale_to_fit_width(10.5)
        window = RoundedRectangle(corner_radius=0.2, width=max(rows.width + 1.0, 7), height=rows.height + 1.3,
                                  color=MUTED, stroke_width=1.5).set_fill("#161b22", opacity=1)
        window.move_to(DOWN * 0.4)
        dots = VGroup(*[Dot(radius=0.07, color=c) for c in (PINK, GOLD, GREEN)]).arrange(RIGHT, buff=0.15)
        dots.move_to(window.get_corner(UP + LEFT) + RIGHT * 0.45 + DOWN * 0.3)
        chrome = VGroup(window, dots)
        if s.get("language"):
            lang = Text(_nfc(s["language"]), font=MONO, font_size=18, color=MUTED)
            lang.move_to(window.get_corner(UP + RIGHT) + LEFT * (lang.width / 2 + 0.3) + DOWN * 0.3)
            chrome.add(lang)
        rows.move_to(window).shift(DOWN * 0.2).align_to(window.get_left() + RIGHT * 0.5, LEFT)
        self.run(FadeIn(chrome, scale=0.95), run_time=0.6)
        for row, t in zip(rows, self.reveal_times(len(rows), start=1.0)):
            self.wait_until(t)
            self.run(Write(row), run_time=min(1.0, 0.25 + 0.02 * len(row.text)))

    def layout_timeline(self, s):
        """Trục thời gian: các mốc hiện lần lượt, nhãn so le trên/dưới."""
        self.heading(s.get("heading"))
        events = s.get("events", [])[:6]
        if not events:
            return
        axis = Line(LEFT * 6, RIGHT * 6, color=MUTED, stroke_width=3).move_to(DOWN * 0.3)
        self.run(Create(axis), run_time=0.7)
        n = len(events)
        for k, (ev, t) in enumerate(zip(events, self.reveal_times(n, start=1.0))):
            self.wait_until(t)
            color = PALETTE[k % len(PALETTE)]
            dot = Dot(axis.point_from_proportion((k + 0.5) / n), radius=0.12, color=color)
            when = label(ev.get("when", ""), 28, color, bold=True, max_width=2.2)
            what = label(ev.get("label", ""), 24, INK, width_chars=14, max_width=2.2)
            side = UP if k % 2 == 0 else DOWN
            group = VGroup(when, what).arrange(DOWN, buff=0.12).next_to(dot, side, buff=0.35)
            self.run(GrowFromCenter(dot), FadeIn(group, shift=-side * 0.2), run_time=0.6)

    def layout_number(self, s):
        """Một con số lớn bật ra giữa khung, kèm đơn vị và chú thích.

        Không đếm từ 0 lên: dựng lại chữ ở mỗi khung (always_redraw) làm một cảnh 7 giây mất
        hơn 6 phút render, vì mỗi lần dựng chữ Manim lại gọi Pango tạo một file SVG mới."""
        self.heading(s.get("heading"))
        target = float(s.get("value") or 0)
        if float(target).is_integer() or abs(target) >= 100:
            text = f"{target:,.0f}".replace(",", ".")
        else:
            text = f"{target:.1f}".replace(".", ",")
        unit = _nfc(s.get("unit") or "")
        number = Text(text + (" " + unit if unit else ""), font=FONT, weight=BOLD, font_size=96, color=GOLD)
        if number.width > FRAME_W:
            number.scale_to_fit_width(FRAME_W)
        number.move_to(UP * 0.3)
        ring = Circle(radius=max(number.width, number.height) / 2 + 0.6, color=GOLD, stroke_width=3)
        ring.move_to(number).stretch(0.45, 1)
        self.run(GrowFromCenter(number), run_time=0.8)
        self.run(Create(ring), Indicate(number, color=GOLD, scale_factor=1.08), run_time=0.9)
        self.run(FadeOut(ring), run_time=0.3)
        if s.get("caption"):
            self.wait_until(self.reveal_times(2, start=1.2)[1])
            cap = label(s["caption"], 32, INK, width_chars=46).next_to(number, DOWN, buff=0.7)
            self.run(FadeIn(cap, shift=UP * 0.2), run_time=0.6)
