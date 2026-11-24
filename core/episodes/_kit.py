"""
Mảnh ghép vẽ dùng chung cho cảnh viết tay của các tập (từ tập 2 trở đi).
Tập nạp bằng: sys.path.insert(0, thư mục này); from _kit import ...
Tên bắt đầu bằng "_" nên không bao giờ bị coi là module của một tập (CUSTOM_NAME).
"""
from __future__ import annotations

from manim import (
    DOWN, LEFT, RIGHT, UP, Arc, Arrow, Ellipse, Line, Rectangle, RoundedRectangle, Text, VGroup, VMobject, Write,
)

PANEL = "#151925"


def card(k, text, color, size=28, width_chars=18, bold=True, pad=0.5, min_w=0.0, text_color=None):
    txt = k.label(text, size, text_color or k.INK, bold=bold, width_chars=width_chars)
    box = RoundedRectangle(corner_radius=0.2, width=max(min_w, txt.width + pad), height=txt.height + pad * 0.8,
                           color=color, stroke_width=3).set_fill(color, opacity=0.12)
    txt.move_to(box)
    return VGroup(box, txt)


def code_card(k, lines, color, size=26):
    txt = VGroup(*[Text(line.strip() or " ", font=k.MONO, font_size=size, color=k.INK) for line in lines])
    txt.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    # Text bỏ khoảng trắng đầu dòng: tự thụt lại theo bề rộng một ký tự font mono.
    char_w = Text("MMMMMMMMMM", font=k.MONO, font_size=size).width / 10
    for mob, line in zip(txt, lines):
        mob.shift(RIGHT * char_w * (len(line) - len(line.lstrip(" "))))
    box = RoundedRectangle(corner_radius=0.2, width=txt.width + 0.7, height=txt.height + 0.6,
                           color=color, stroke_width=2).set_fill(PANEL, opacity=1)
    txt.move_to(box)
    return VGroup(box, txt)


def arrow(k, a, b, color=None, vertical=False):
    start, end = (a.get_bottom(), b.get_top()) if vertical else (a.get_right(), b.get_left())
    return Arrow(start, end, buff=0.15, color=color or k.MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.2)


def cross(k, mob):
    c = mob.get_center()
    w, h = mob.width / 2 + 0.1, mob.height / 2 + 0.1
    return VGroup(Line(c + LEFT * w + UP * h, c + RIGHT * w + DOWN * h, color=k.PINK, stroke_width=6),
                  Line(c + LEFT * w + DOWN * h, c + RIGHT * w + UP * h, color=k.PINK, stroke_width=6))


def strike(k, mob):
    return Line(mob.get_left() + LEFT * 0.1, mob.get_right() + RIGHT * 0.1, color=k.PINK, stroke_width=6)


def check(k, size=0.5, color=None):
    mark = VMobject(color=color or k.GREEN, stroke_width=8)
    mark.set_points_as_corners([LEFT * size * 0.5, DOWN * size * 0.45, RIGHT * size * 0.7 + UP * size * 0.55])
    return mark


def heading(sc, k, text, size=40):
    head = k.label(text, size, k.INK, bold=True, max_width=k.FRAME_W).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.8)
    return head


def bubble(k, text, color, width_chars=30, size=28):
    txt = k.label(text, size, k.INK, width_chars=width_chars)
    box = RoundedRectangle(corner_radius=0.3, width=txt.width + 0.6, height=txt.height + 0.5,
                           color=color, stroke_width=2).set_fill(color, opacity=0.15)
    txt.move_to(box)
    return VGroup(box, txt)


def database(k, color=None, width=1.6, height=1.4):
    """Hình trụ "cơ sở dữ liệu"."""
    color = color or k.PURPLE
    top = Ellipse(width=width, height=width * 0.3, color=color, stroke_width=3).set_fill(color, opacity=0.2)
    body = Rectangle(width=width, height=height, color=color, stroke_width=0).set_fill(color, opacity=0.12)
    body.next_to(top.get_center(), DOWN, buff=0)
    left = Line(top.get_left(), top.get_left() + DOWN * height, color=color, stroke_width=3)
    right = Line(top.get_right(), top.get_right() + DOWN * height, color=color, stroke_width=3)
    bottom = Arc(radius=1, start_angle=3.1416, angle=3.1416, color=color, stroke_width=3)
    bottom.stretch(width / 2, 0).stretch(width * 0.15, 1).move_to(top.get_center() + DOWN * height, aligned_edge=UP)
    bottom.shift(UP * width * 0.15)
    return VGroup(body, left, right, bottom, top)


def padlock(k, color=None, size=0.6):
    color = color or k.GOLD
    body = RoundedRectangle(corner_radius=0.08, width=size, height=size * 0.8, color=color, stroke_width=4)
    body.set_fill(color, opacity=0.25)
    shackle = Arc(radius=size * 0.3, start_angle=0, angle=3.1416, color=color, stroke_width=5)
    shackle.next_to(body, UP, buff=0)
    return VGroup(shackle, body)
