"""
Series 1 "Bản Chất LLM Agent", tập 2: Kiến trúc tổng quan của một LLM Agent.
Kịch bản: core/episodes/s1e02_vi.txt. Mỗi hàm là một cảnh (sc, spec, k); sc.at(i) = lúc câu thứ i
của cảnh bắt đầu. Mảnh ghép chung ở core/episodes/_kit.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, CurvedArrow, DashedLine, DashedVMobject, Dot, Ellipse, FadeIn,
    FadeOut, GrowArrow, GrowFromCenter, GrowFromEdge, Indicate, Line, MoveAlongPath, Rectangle, RoundedRectangle,
    SurroundingRectangle, Text, Transform, VGroup, Write, linear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import arrow, bubble, card, check, code_card, cross, database, heading, padlock, strike  # noqa: E402

BOARD = [(0, "open_box"), (2, "the_question"), (7, "team_of_four"), (9, "four_parts"), (21, "run_rounds"),
         (38, "state_is_context"), (42, "context_fills"), (46, "who_keeps_rhythm"), (50, "loop_code"),
         (54, "weights_frozen"), (58, "takeaway")]


def _mid(sc, i, frac=0.5):
    """Một điểm giữa câu i và câu i+1 (để chia nhịp trong câu dài)."""
    return sc.at(i) + (sc.at(i + 1) - sc.at(i)) * frac


def _person(k, name, color):
    head = Circle(radius=0.32, color=color, stroke_width=4).set_fill(color, opacity=0.25)
    body = RoundedRectangle(corner_radius=0.35, width=1.2, height=0.75, color=color, stroke_width=4)
    body.set_fill(color, opacity=0.15).next_to(head, DOWN, buff=0.08)
    tag = k.label(name, 30, k.INK, bold=True).next_to(body, DOWN, buff=0.25)
    return VGroup(head, body, tag)


# ── Các cảnh ─────────────────────────────────────────────────
def open_box(sc, s, k):
    steps = VGroup(*[card(k, t, c, 26, 10, min_w=1.6) for t, c in (("Làm", k.ACCENT), ("Nhìn", k.GOLD), ("Sửa", k.GREEN))])
    steps.arrange(RIGHT, buff=0.7).move_to(UP * 0.6)
    links = VGroup(arrow(k, steps[0], steps[1]), arrow(k, steps[1], steps[2]))
    back = CurvedArrow(steps[2].get_bottom() + DOWN * 0.05, steps[0].get_bottom() + DOWN * 0.05, angle=-0.9,
                       color=k.MUTED, stroke_width=4)
    tag = k.label("tập 1: agent = mô hình ngôn ngữ + vòng lặp", 28, k.MUTED).next_to(back, DOWN, buff=0.35)
    sc.run(FadeIn(steps[0]), run_time=0.4)
    sc.run(GrowArrow(links[0]), FadeIn(steps[1]), run_time=0.4)
    sc.run(GrowArrow(links[1]), FadeIn(steps[2]), run_time=0.4)
    sc.run(Create(back), FadeIn(tag), run_time=0.7)

    sc.wait_until(sc.at(1))
    loop = VGroup(steps, links, back)
    sc.run(FadeOut(tag), loop.animate.scale(0.55).to_edge(UP, buff=0.5), run_time=0.7)
    box = RoundedRectangle(corner_radius=0.3, width=7.0, height=3.6, color=k.ACCENT, stroke_width=4)
    box.set_fill(k.ACCENT, opacity=0.06).move_to(DOWN * 1.0)
    lid = Line(box.get_corner(UP + LEFT), box.get_corner(UP + RIGHT), color=k.ACCENT, stroke_width=8)
    name = k.label("bên trong agent", 30, k.ACCENT, bold=True).next_to(box, DOWN, buff=0.25)
    ask = k.label("?", 110, k.GOLD, bold=True).move_to(box)
    sc.run(Create(box), Create(lid), FadeIn(name), run_time=0.7)
    sc.run(FadeIn(ask, scale=0.6), run_time=0.5)
    sc.run(lid.animate.shift(UP * 0.7 + RIGHT * 0.4).rotate(0.12), run_time=0.6)
    dots = VGroup(*[Dot(radius=0.2, color=c) for c in (k.PURPLE, k.GREEN, k.GOLD, k.ACCENT)]).arrange(RIGHT, buff=0.6)
    sc.run(Transform(ask, dots.move_to(box)), run_time=0.8)


def the_question(sc, s, k):
    sc.wait_until(sc.at(1))
    q = bubble(k, "Quý vừa rồi cửa hàng A bán bao nhiêu, B bán bao nhiêu, A hơn B bao nhiêu phần trăm?",
               k.ACCENT, 46).to_edge(UP, buff=0.6)
    sc.run(FadeIn(q, shift=DOWN * 0.2), run_time=0.7)

    sc.wait_until(sc.at(2))
    llm = card(k, "Mô hình ngôn ngữ đứng một mình", k.PURPLE, 28, 16, min_w=3.4).move_to(LEFT * 4.2 + DOWN * 0.6)
    sc.run(FadeIn(llm, scale=0.9), run_time=0.6)

    sc.wait_until(sc.at(3))
    data = card(k, "Doanh thu A, B", k.GOLD, 28, 16).move_to(RIGHT * 1.0 + UP * 0.4)
    nope = cross(k, data)
    no_tag = k.label("không có số liệu của bạn", 26, k.PINK).next_to(data, RIGHT, buff=0.35)
    if no_tag.get_right()[0] > 6.8:
        no_tag.next_to(data, DOWN, buff=0.15)
    sc.run(GrowArrow(arrow(k, llm, data)), FadeIn(data), run_time=0.5)
    sc.run(Create(nope), FadeIn(no_tag), run_time=0.6)

    sc.wait_until(sc.at(4))
    calc = k.label("(50 − 40) ÷ 40 = ?", 36, k.INK, bold=True).move_to(RIGHT * 1.6 + DOWN * 1.4)
    sc.run(GrowArrow(arrow(k, llm, calc)), Write(calc), run_time=0.8)
    guesses = VGroup(*[card(k, g, k.MUTED, 26, 8) for g in ("20%?", "25%?", "40%?")]).arrange(RIGHT, buff=0.4)
    guesses.next_to(calc, DOWN, buff=0.45)
    for g in guesses:
        sc.run(FadeIn(g, shift=UP * 0.15), run_time=0.3)
    risky = k.label("đoán chữ tiếp theo, dễ sai", 28, k.PINK, bold=True).next_to(guesses, DOWN, buff=0.35)
    sc.run(FadeIn(risky), Indicate(guesses, color=k.PINK), run_time=0.7)


def team_of_four(sc, s, k):
    ask = k.label("Vậy agent cần những gì?", 48, k.INK, bold=True)
    sc.run(Write(ask), run_time=0.9)
    sc.wait_until(sc.at(1))
    sc.run(ask.animate.scale(0.75).to_edge(UP, buff=0.6), run_time=0.5)
    team = VGroup(*[_person(k, n, c) for n, c in (("Bộ não", k.PURPLE), ("Kế hoạch", k.GREEN),
                                                  ("Công cụ", k.GOLD), ("Bộ nhớ", k.ACCENT))])
    team.arrange(RIGHT, buff=1.2).move_to(DOWN * 0.3)
    for p in team:
        sc.run(FadeIn(p, shift=UP * 0.3), run_time=0.4)
    note = k.label("bốn người trong một nhóm nhỏ", 28, k.MUTED).next_to(team, DOWN, buff=0.5)
    sc.run(FadeIn(note), run_time=0.4)


def four_parts(sc, s, k):
    brain = card(k, "Bộ não", k.PURPLE, 34, 12, min_w=3.0).move_to(UP * 1.5)
    brain_sub = k.label("mô hình ngôn ngữ", 24, k.MUTED).next_to(brain, DOWN, buff=0.15)
    sc.run(GrowFromCenter(brain), FadeIn(brain_sub), run_time=0.6)
    sc.wait_until(sc.at(1))
    heavy = k.label("không tự làm việc nặng", 24, k.PINK).next_to(brain_sub, DOWN, buff=0.12)
    sc.run(FadeIn(heavy), run_time=0.4)
    sc.wait_until(sc.at(2))
    job = k.label("đọc tình hình → quyết định bước tiếp", 24, k.GOLD).move_to(heavy)
    sc.run(Transform(heavy, job), run_time=0.5)

    def panel(title, color, lines, mono=False):
        head = k.label(title, 30, color, bold=True)
        body = VGroup(*[(Text(t, font=k.MONO, font_size=22, color=k.INK) if mono else k.label(t, 24, k.INK))
                        for t in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        inner = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        box = RoundedRectangle(corner_radius=0.2, width=max(3.4, inner.width + 0.6), height=inner.height + 0.5,
                               color=color, stroke_width=3).set_fill(color, opacity=0.08)
        inner.move_to(box)
        return VGroup(box, head, body)

    plan = panel("Kế hoạch", k.GREEN, ["1. tra doanh thu A", "2. tra doanh thu B", "3. tính phần trăm"])
    plan.move_to(LEFT * 4.9 + UP * 1.3)
    sc.wait_until(sc.at(3))
    sc.run(FadeIn(plan[0]), FadeIn(plan[1]), Create(Line(plan.get_right(), brain.get_left(), color=k.MUTED)), run_time=0.5)
    sc.wait_until(sc.at(4))
    for i, row in enumerate(plan[2]):
        sc.wait_until(_mid(sc, 4, 0.25 + 0.22 * i))
        sc.run(FadeIn(row, shift=RIGHT * 0.2), run_time=0.35)
    sc.wait_until(sc.at(5))
    easy = k.label("dễ làm, dễ kiểm tra", 22, k.GREEN).next_to(plan, DOWN, buff=0.2)
    sc.run(FadeIn(easy), run_time=0.4)

    tools = panel("Công cụ", k.GOLD, ["tra_doanh_thu()", "may_tinh()"], mono=True).move_to(RIGHT * 4.9 + UP * 1.3)
    sc.wait_until(sc.at(6))
    sc.run(FadeIn(tools[0]), FadeIn(tools[1]), Create(Line(brain.get_right(), tools.get_left(), color=k.MUTED)), run_time=0.5)
    sc.wait_until(sc.at(7))
    for row in tools[2]:
        sc.run(FadeIn(row, shift=LEFT * 0.2), run_time=0.35)
    sc.wait_until(sc.at(8))
    ask = k.label("nhờ tính", 22, k.GOLD).next_to(tools, DOWN, buff=0.2)
    sc.run(Indicate(tools[2][1], color=k.GOLD), FadeIn(ask), run_time=0.7)

    sc.wait_until(sc.at(9))
    mem_head = k.label("Bộ nhớ", 30, k.ACCENT, bold=True).move_to(DOWN * 0.9)
    sc.run(FadeIn(mem_head), Create(Line(heavy.get_bottom() + DOWN * 0.1, mem_head.get_top() + UP * 0.1, color=k.MUTED)),
           run_time=0.5)
    sc.wait_until(sc.at(10))
    ctx_lines = VGroup(*[Line(LEFT * 1.1, RIGHT * (1.1 - 0.4 * (i % 2)), color=k.MUTED, stroke_width=4) for i in range(4)])
    ctx_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    ctx_box = RoundedRectangle(corner_radius=0.15, width=2.8, height=1.5, color=k.ACCENT, stroke_width=3)
    ctx = VGroup(ctx_box, ctx_lines.move_to(ctx_box)).move_to(LEFT * 2.2 + DOWN * 2.3)
    ctx_tag = k.label("ngắn hạn: ngữ cảnh", 24, k.ACCENT).next_to(ctx, DOWN, buff=0.15)
    sc.run(FadeIn(ctx), FadeIn(ctx_tag), run_time=0.6)
    sc.wait_until(sc.at(11))
    db = database(k).move_to(RIGHT * 2.4 + DOWN * 2.2)
    db_tag = k.label("dài hạn: kho bên ngoài", 24, k.PURPLE).next_to(db, DOWN, buff=0.15)
    pull = Arrow(db.get_left(), ctx.get_right(), buff=0.2, color=k.PURPLE, stroke_width=4, max_tip_length_to_length_ratio=0.15)
    pull_tag = k.label("lấy ra khi cần", 20, k.PURPLE).next_to(pull, UP, buff=0.08)
    sc.run(FadeIn(db), FadeIn(db_tag), run_time=0.6)
    sc.run(GrowArrow(pull), FadeIn(pull_tag), run_time=0.6)


def run_rounds(sc, s, k):
    col = RoundedRectangle(corner_radius=0.2, width=3.6, height=5.6, color=k.ACCENT, stroke_width=3)
    col.set_fill(k.ACCENT, opacity=0.05).move_to(LEFT * 4.9 + DOWN * 0.5)
    col_head = k.label("Ngữ cảnh", 30, k.ACCENT, bold=True).next_to(col, UP, buff=0.15)
    brain = card(k, "Bộ não", k.PURPLE, 32, 12, min_w=2.6).move_to(UP * 1.9 + RIGHT * 0.2)
    look = card(k, "tra_doanh_thu", k.GOLD, 24, 16, min_w=3.0).move_to(RIGHT * 4.9 + UP * 2.1)
    calc = card(k, "may_tinh", k.GOLD, 24, 16, min_w=3.0).move_to(RIGHT * 4.9 + UP * 0.6)
    plan_rows = VGroup(*[k.label(t, 24, k.INK) for t in ("tra doanh thu A", "tra doanh thu B", "tính phần trăm")])
    boxes = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.32, height=0.32, color=k.GREEN, stroke_width=3)
                     for _ in plan_rows])
    plan = VGroup(*[VGroup(b, r).arrange(RIGHT, buff=0.2) for b, r in zip(boxes, plan_rows)])
    plan.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    plan_head = k.label("Kế hoạch", 26, k.GREEN, bold=True)
    plan_all = VGroup(plan_head, plan).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 4.6 + DOWN * 2.2)
    sc.run(FadeIn(col), FadeIn(col_head), FadeIn(brain), run_time=0.5)
    sc.run(FadeIn(look), FadeIn(calc), FadeIn(plan_all), run_time=0.5)

    slot_y = [col.get_top()[1] - 0.6 - 0.9 * i for i in range(6)]

    def chip(text, color, i):
        c = card(k, text, color, 24, 22, min_w=3.1)
        return c.move_to([col.get_center()[0], slot_y[i], 0])

    round_tag = [None]

    def new_round(text):
        tag = k.label(text, 36, k.INK, bold=True).move_to(UP * 3.45 + RIGHT * 0.2)
        if round_tag[0] is None:
            sc.run(FadeIn(tag, shift=DOWN * 0.2), run_time=0.4)
        else:
            sc.run(Transform(round_tag[0], tag), run_time=0.4)
            return
        round_tag[0] = tag

    def call(text, tool):
        req = Text(text, font=k.MONO, font_size=22, color=k.GOLD).next_to(brain, DOWN, buff=0.35)
        sc.run(Indicate(brain, color=k.PURPLE), FadeIn(req, shift=DOWN * 0.15), run_time=0.6)
        sc.run(req.animate.next_to(tool, LEFT, buff=0.25), run_time=0.7)
        sc.run(Indicate(tool, color=k.GOLD), FadeOut(req), run_time=0.5)

    def answer(text, tool):
        res = card(k, text, k.GREEN, 26, 12).next_to(tool, DOWN, buff=0.15)
        sc.run(FadeIn(res, scale=0.8), run_time=0.4)
        return res

    def store(res, text, i, tick):
        target = chip(text, k.GREEN, i)
        sc.run(Transform(res, target), run_time=0.8)
        sc.run(Create(check(k, 0.28).move_to(boxes[tick])), run_time=0.3)

    sc.wait_until(sc.at(1))
    new_round("Vòng 1")
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(chip("câu hỏi của bạn", k.ACCENT, 0)), run_time=0.4)
    sc.run(FadeIn(chip("mô tả các công cụ", k.ACCENT, 1)), run_time=0.4)
    tokens = k.label("≈ 1.200 token", 24, k.MUTED).next_to(col_head, RIGHT, buff=0.3)
    sc.run(FadeIn(tokens), run_time=0.3)
    sc.wait_until(sc.at(3))
    call('tra_doanh_thu("A")', look)
    sc.wait_until(sc.at(4))
    res = answer("50 triệu", look)
    sc.wait_until(sc.at(5))
    store(res, "A: 50 triệu", 2, 0)

    sc.wait_until(sc.at(6))
    new_round("Vòng 2")
    sc.wait_until(sc.at(7))
    call('tra_doanh_thu("B")', look)
    sc.wait_until(sc.at(8))
    res = answer("40 triệu", look)
    sc.wait_until(sc.at(9))
    store(res, "B: 40 triệu", 3, 1)

    sc.wait_until(sc.at(10))
    new_round("Vòng 3")
    sc.wait_until(sc.at(11))
    both = VGroup(*[m for m in sc.mobjects if isinstance(m, VGroup) and m.get_center()[1] in (slot_y[2], slot_y[3])])
    marks = VGroup(*[SurroundingRectangle(chip("x", k.GREEN, i), color=k.GOLD, buff=0.05) for i in (2, 3)])
    sc.run(Create(marks), run_time=0.5)
    sc.run(FadeOut(marks), run_time=0.4)
    sc.wait_until(sc.at(12))
    call("may_tinh((50-40)/40*100)", calc)
    sc.wait_until(sc.at(13))
    res = answer("25", calc)
    store(res, "kết quả: 25%", 4, 2)

    sc.wait_until(sc.at(14))
    new_round("Vòng 4")
    sc.wait_until(sc.at(15))
    reply = bubble(k, "A bán nhiều hơn B 25%", k.GREEN, 30, 30).next_to(brain, DOWN, buff=0.6)
    sc.run(Indicate(plan_all, color=k.GREEN), run_time=0.6)
    sc.run(FadeIn(reply, shift=DOWN * 0.2), run_time=0.6)
    sc.wait_until(sc.at(16))
    stop = VGroup(check(k, 0.4), k.label("Dừng", 36, k.GREEN, bold=True)).arrange(RIGHT, buff=0.2)
    sc.run(Transform(round_tag[0], stop.move_to(UP * 3.45 + RIGHT * 0.2)), run_time=0.5)


def state_is_context(sc, s, k):
    note = k.label("Bạn để ý một điều", 30, k.MUTED).to_edge(UP, buff=0.6)
    sc.run(FadeIn(note), run_time=0.4)
    sc.wait_until(sc.at(1))
    b1 = card(k, "Bộ não · vòng 1", k.PURPLE, 24, 18).move_to(LEFT * 4.9 + UP * 0.8)
    b2 = card(k, "Bộ não · vòng 2", k.PURPLE, 24, 18).move_to(LEFT * 0.5 + UP * 0.8)
    gap = DashedLine(b1.get_right(), b2.get_left(), color=k.MUTED, stroke_width=4)
    sc.run(FadeIn(b1), FadeIn(b2), Create(gap), run_time=0.6)
    no_mem = cross(k, gap.copy().set_width(0.8))
    tag = k.label("không tự mang trí nhớ sang", 24, k.PINK).next_to(gap, DOWN, buff=0.4)
    sc.run(Create(no_mem), FadeIn(tag), run_time=0.5)

    sc.wait_until(sc.at(2))
    ctx = VGroup(*[card(k, t, c, 24, 20, min_w=3.0) for t, c in (("câu hỏi của bạn", k.ACCENT),
                                                                   ("mô tả các công cụ", k.ACCENT),
                                                                   ("A: 50 triệu", k.GREEN))])
    ctx.arrange(DOWN, buff=0.2)
    frame = SurroundingRectangle(ctx, color=k.ACCENT, buff=0.25, corner_radius=0.2)
    ctx_all = VGroup(frame, ctx).move_to(RIGHT * 4.6 + UP * 0.4)
    head = k.label("Ngữ cảnh", 28, k.ACCENT, bold=True).next_to(frame, UP, buff=0.15)
    feed = Arrow(frame.get_left(), b2.get_right(), buff=0.15, color=k.ACCENT, stroke_width=5,
                 max_tip_length_to_length_ratio=0.15)
    sc.run(FadeIn(ctx_all), FadeIn(head), run_time=0.6)
    sc.run(GrowArrow(feed), Indicate(ctx[2], color=k.GOLD), run_time=0.8)

    sc.wait_until(sc.at(3))
    big = card(k, "Trạng thái của agent = những gì đang có trong ngữ cảnh", k.GOLD, 34, 34, pad=0.7)
    big.move_to(DOWN * 2.4)
    big[0].set_fill("#0f1117", opacity=0.95)
    sc.run(FadeIn(big, shift=UP * 0.2), run_time=0.7)


def context_fills(sc, s, k):
    frame = Rectangle(width=10.0, height=0.9, color=k.MUTED, stroke_width=3).move_to(UP * 0.6)
    head = k.label("Ngữ cảnh (có giới hạn)", 28, k.INK, bold=True).next_to(frame, UP, buff=0.3).align_to(frame, LEFT)
    sc.run(Create(frame), FadeIn(head), run_time=0.5)
    left = frame.get_left()[0]
    blocks = []

    def add_block(width, color):
        x = left + sum(b.width for b in blocks)
        b = Rectangle(width=width, height=0.9, color=color, stroke_width=1).set_fill(color, opacity=0.7)
        b.move_to([x + width / 2, frame.get_center()[1], 0])
        blocks.append(b)
        return b

    for i in range(4):
        sc.run(GrowFromEdge(add_block(0.55, k.ACCENT), LEFT), run_time=0.3)
    four = k.label("4 vòng", 24, k.ACCENT).next_to(VGroup(*blocks), DOWN, buff=0.2)
    sc.run(FadeIn(four), run_time=0.3)

    sc.wait_until(sc.at(1))
    ok = k.label("vẫn còn nhiều chỗ", 26, k.GREEN).next_to(frame, DOWN, buff=0.2).align_to(frame, RIGHT)
    sc.run(FadeIn(ok), run_time=0.4)

    sc.wait_until(sc.at(2))
    sc.run(FadeOut(ok), FadeOut(four), run_time=0.3)
    while sum(b.width for b in blocks) < 10.0 - 0.5:
        sc.run(GrowFromEdge(add_block(0.55, k.ACCENT), LEFT), run_time=0.12)
    last = add_block(10.0 - sum(b.width for b in blocks), k.PINK)
    full = k.label("đầy", 32, k.PINK, bold=True).next_to(frame, RIGHT, buff=0.2).shift(LEFT * 0.0)
    many = k.label("vài chục vòng", 24, k.ACCENT).next_to(frame, DOWN, buff=0.2).align_to(frame, LEFT)
    sc.run(GrowFromEdge(last, LEFT), FadeIn(many), run_time=0.4)
    sc.run(FadeIn(full, scale=1.2), Indicate(frame, color=k.PINK), run_time=0.6)

    sc.wait_until(max(sc.clock, _mid(sc, 2, 0.6)))
    squash = card(k, "tóm tắt bớt", k.GOLD, 28, 14, min_w=3.0).move_to(LEFT * 2.8 + DOWN * 2.0)
    db = database(k, width=1.3, height=1.0).move_to(RIGHT * 2.0 + DOWN * 2.1)
    move = card(k, "chuyển sang bộ nhớ dài hạn", k.PURPLE, 26, 16).next_to(db, RIGHT, buff=0.3)
    sc.run(FadeIn(squash), run_time=0.4)
    summary = Rectangle(width=2.2, height=0.9, color=k.GOLD, stroke_width=1).set_fill(k.GOLD, opacity=0.7)
    summary.move_to([left + 1.1, frame.get_center()[1], 0])
    sc.run(*[FadeOut(b) for b in blocks[:-6]], FadeOut(full), FadeIn(summary), run_time=0.8)
    rest = VGroup(*blocks[-6:])
    sc.run(rest.animate.next_to(summary, RIGHT, buff=0), run_time=0.6)
    sc.run(FadeIn(db), FadeIn(move), run_time=0.5)

    sc.wait_until(sc.at(3))
    later = k.label("nói kỹ ở tập về bộ nhớ", 28, k.MUTED).to_edge(DOWN, buff=0.5)
    sc.run(FadeIn(later), run_time=0.4)


def who_keeps_rhythm(sc, s, k):
    ask = k.label("Ai giữ nhịp?", 56, k.INK, bold=True)
    sc.run(Write(ask), run_time=0.8)
    sc.wait_until(sc.at(1))
    sc.run(ask.animate.scale(0.7).to_edge(UP, buff=0.5), run_time=0.5)
    model = card(k, "Mô hình", k.PURPLE, 34, 12, min_w=2.8).move_to(DOWN * 0.2)
    sc.run(FadeIn(model), run_time=0.4)
    not_it = k.label("không phải mô hình", 28, k.PINK, bold=True).next_to(model, UP, buff=0.4)
    sc.run(FadeIn(not_it), run_time=0.4)

    sc.wait_until(sc.at(2))
    q = card(k, "được hỏi", k.ACCENT, 24, 10).move_to(LEFT * 4.2 + DOWN * 0.2)
    a = card(k, "trả lời", k.GREEN, 24, 10).move_to(RIGHT * 4.2 + DOWN * 0.2)
    sc.run(FadeIn(q), GrowArrow(arrow(k, q, model)), run_time=0.5)
    sc.run(GrowArrow(arrow(k, model, a)), FadeIn(a), run_time=0.5)
    idle = k.label("rồi nằm yên chờ lần hỏi sau", 24, k.MUTED).next_to(model, DOWN, buff=0.35)
    sc.run(FadeIn(idle), run_time=0.4)

    sc.wait_until(sc.at(3))
    everything = VGroup(q, model, a, idle)
    ring = RoundedRectangle(corner_radius=0.5, width=everything.width + 1.2, height=everything.height + 1.6,
                            color=k.GOLD, stroke_width=4).move_to(everything)
    dashed = DashedVMobject(ring, num_dashes=50)
    tag = Text("vòng lặp do lập trình viên viết", font=k.MONO, font_size=26, color=k.GOLD)
    tag.next_to(ring, DOWN, buff=0.2)
    sc.run(FadeOut(not_it), Create(dashed), FadeIn(tag), run_time=0.8)
    dot = Dot(ring.point_from_proportion(0), color=k.GOLD, radius=0.14)
    sc.run(FadeIn(dot), run_time=0.2)
    while sc.remaining() > 2.0:
        sc.run(MoveAlongPath(dot, ring, rate_func=linear), run_time=min(2.5, sc.remaining() - 0.9))


def loop_code(sc, s, k):
    lines = ["for vong in range(SO_VONG_TOI_DA):",
             "    quyet_dinh = mo_hinh(ngu_canh)",
             "    if quyet_dinh.xong:",
             "        break",
             "    ket_qua = chay_cong_cu(quyet_dinh)",
             "    ngu_canh.append(ket_qua)"]
    code = code_card(k, lines, k.ACCENT, size=24).to_edge(LEFT, buff=0.5).shift(UP * 0.4)
    sc.run(FadeIn(code[0]), Write(code[1]), run_time=1.4)

    notes = [(1, "gửi ngữ cảnh, nhận quyết định", k.PURPLE), (4, "chạy công cụ", k.GOLD),
             (5, "ghi kết quả vào ngữ cảnh", k.GREEN)]
    marks = []
    for j, (idx, text, color) in enumerate(notes):
        sc.wait_until(max(sc.clock, _mid(sc, 0, 0.25 + 0.22 * j)))
        line = code[1][idx]
        box = SurroundingRectangle(line, color=color, buff=0.06)
        tag = k.label(text, 24, color, bold=True).next_to(code, RIGHT, buff=0.35)
        tag.set_y(line.get_center()[1])
        sc.run(Create(box), FadeIn(tag), run_time=0.5)
        marks += [box, tag]

    sc.wait_until(sc.at(1))
    stop_box = SurroundingRectangle(VGroup(code[1][2], code[1][3]), color=k.PINK, buff=0.06)
    stop_tag = k.label("xong thì dừng", 24, k.PINK, bold=True).next_to(code, RIGHT, buff=0.35)
    stop_tag.set_y(stop_box.get_center()[1])
    sc.run(Create(stop_box), FadeIn(stop_tag), run_time=0.6)

    sc.wait_until(sc.at(2))
    limit_box = SurroundingRectangle(code[1][0], color=k.GOLD, buff=0.06)
    limit_tag = k.label("giới hạn số vòng", 24, k.GOLD, bold=True).next_to(code, UP, buff=0.25).align_to(code, LEFT)
    sc.run(Create(limit_box), FadeIn(limit_tag), Indicate(code[1][0], color=k.GOLD), run_time=0.8)

    sc.wait_until(sc.at(3))
    sc.run(*[FadeOut(m) for m in [code, stop_box, stop_tag, limit_box, limit_tag, *marks]], run_time=0.5)
    stuck = k.label("Không có giới hạn, agent bị kẹt:", 32, k.INK, bold=True).move_to(UP * 1.8)
    # Bộ đếm nhảy theo bậc: chỉ vẽ chữ ~12 lần (vẽ lại mỗi khung thì Pango làm render chậm hẳn).
    count_tag = k.label("lần gọi công cụ", 30, k.MUTED).move_to(RIGHT * 1.2 + UP * 0.2)
    cost_tag = k.label("bạn trả tiền cho từng lần", 28, k.MUTED).move_to(RIGHT * 1.6 + DOWN * 1.2)
    sc.run(FadeIn(stuck), FadeIn(count_tag), FadeIn(cost_tag), run_time=0.5)
    steps = [1, 2, 3, 5, 8, 13, 21, 40, 80, 150, 300, 480]
    pause = max(0.12, (sc.remaining() - 1.0) / len(steps))
    shown = None
    for n in steps:
        nums = VGroup(k.label(str(n), 90, k.PINK, bold=True).next_to(count_tag, LEFT, buff=0.4),
                      k.label(f"${n * 0.01:.2f}", 60, k.GOLD, bold=True).next_to(cost_tag, LEFT, buff=0.4))
        if shown is not None:
            sc.remove(shown)
        sc.add(nums)
        shown = nums
        sc.wait_until(sc.clock + pause)


def weights_frozen(sc, s, k):
    note = k.label("Một hiểu lầm hay gặp", 30, k.MUTED).to_edge(UP, buff=0.6)
    sc.run(FadeIn(note), run_time=0.4)
    sc.wait_until(sc.at(1))
    nums = ["0.12", "-0.83", "1.07", "0.44", "-0.29", "0.91", "0.05", "-1.20", "0.67", "0.33", "-0.58", "0.76"]
    grid = VGroup(*[Text(n, font=k.MONO, font_size=22, color=k.INK) for n in nums]).arrange_in_grid(3, 4, buff=(0.35, 0.25))
    frame = RoundedRectangle(corner_radius=0.2, width=grid.width + 0.8, height=grid.height + 1.1, color=k.PURPLE,
                             stroke_width=3).set_fill(k.PURPLE, opacity=0.08)
    model = VGroup(frame, grid.move_to(frame).shift(DOWN * 0.2)).move_to(LEFT * 3.3 + DOWN * 0.4)
    model_tag = k.label("trọng số của mô hình", 26, k.PURPLE, bold=True).next_to(frame.get_top(), DOWN, buff=0.18)
    learn = k.label("agent học thêm vào đây trong lúc chạy?", 28, k.INK).move_to(UP * 2.4)
    sc.run(FadeIn(model), FadeIn(model_tag), run_time=0.6)
    sc.run(FadeIn(learn), run_time=0.4)
    no = k.label("Không.", 34, k.PINK, bold=True).next_to(learn, RIGHT, buff=0.3)
    sc.run(FadeIn(no, scale=1.2), run_time=0.4)

    sc.wait_until(sc.at(2))
    lock = padlock(k, size=0.7).next_to(frame, UP, buff=-0.05).align_to(frame, RIGHT).shift(LEFT * 0.3)
    same = k.label("giữ nguyên", 26, k.GOLD, bold=True).next_to(frame, DOWN, buff=0.2)
    sc.run(FadeIn(lock, shift=DOWN * 0.2), FadeIn(same), run_time=0.6)

    sc.wait_until(sc.at(3))
    fact = card(k, "B: 40 triệu", k.GREEN, 26, 12).move_to(RIGHT * 1.9 + DOWN * 0.4)
    ctx = card(k, "ngữ cảnh", k.ACCENT, 26, 12, min_w=2.4).move_to(RIGHT * 5.4 + UP * 0.6)
    db = database(k, width=1.3, height=0.9).move_to(RIGHT * 5.4 + DOWN * 1.7)
    db_tag = k.label("kho bên ngoài", 22, k.PURPLE).next_to(db, DOWN, buff=0.1)
    sc.run(FadeIn(fact, scale=0.8), run_time=0.4)
    to_ctx = Arrow(fact.get_right(), ctx.get_left(), buff=0.15, color=k.ACCENT, stroke_width=4)
    to_db = Arrow(fact.get_right(), db.get_left(), buff=0.15, color=k.PURPLE, stroke_width=4)
    to_w = DashedLine(fact.get_left(), frame.get_right(), color=k.MUTED, stroke_width=4)
    sc.run(FadeIn(ctx), GrowArrow(to_ctx), run_time=0.5)
    sc.run(FadeIn(db), FadeIn(db_tag), GrowArrow(to_db), run_time=0.5)
    sc.run(Create(to_w), run_time=0.4)
    sc.run(Create(cross(k, to_w.copy().set_width(0.7))), run_time=0.4)


def takeaway(sc, s, k):
    remember = k.label("Nếu chỉ nhớ một hình ảnh", 30, k.MUTED).to_edge(UP, buff=0.5)
    sc.run(FadeIn(remember), run_time=0.4)
    ring = Ellipse(width=11.4, height=5.6, color=k.GOLD, stroke_width=4).move_to(DOWN * 0.45)
    ring_tag = Text("vòng lặp: code giữ nhịp", font=k.MONO, font_size=26, color=k.GOLD)
    ring_tag.next_to(ring.get_top(), DOWN, buff=0.4)
    sc.run(Create(ring), FadeIn(ring_tag), run_time=0.8)
    brain = card(k, "Bộ não", k.PURPLE, 34, 12, min_w=2.8).move_to(DOWN * 0.2)
    brain_sub = k.label("đọc ngữ cảnh, ra quyết định", 22, k.MUTED).next_to(brain, DOWN, buff=0.15)
    sc.run(GrowFromCenter(brain), FadeIn(brain_sub), run_time=0.5)
    parts = [card(k, t, c, 26, 12, min_w=2.4) for t, c in (("Kế hoạch", k.GREEN), ("Công cụ", k.GOLD), ("Bộ nhớ", k.ACCENT))]
    parts[0].move_to(LEFT * 3.7 + UP * 0.2)
    parts[1].move_to(RIGHT * 3.7 + UP * 0.2)
    parts[2].move_to(DOWN * 2.3)
    for p in parts:
        link = (Line(p.get_top(), brain_sub.get_bottom(), buff=0.1, color=k.MUTED, stroke_width=2) if p is parts[2]
                else Line(p, brain, buff=0.12, color=k.MUTED, stroke_width=2))
        sc.run(FadeIn(p, scale=0.9), Create(link), run_time=0.4)
    dot = Dot(ring.point_from_proportion(0), color=k.GOLD, radius=0.15)
    sc.run(FadeIn(dot), run_time=0.2)
    sc.wait_until(sc.at(1))
    nxt = k.label("Tập sau: system prompt, lời dặn cho bộ não", 28, k.ACCENT, bold=True).to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(nxt, shift=UP * 0.2), MoveAlongPath(dot, ring), run_time=min(2.5, max(0.6, sc.remaining() - 1.2)))
    while sc.remaining() > 2.0:
        sc.run(MoveAlongPath(dot, ring, rate_func=linear), run_time=min(3.0, sc.remaining() - 0.9))


# ── Thumbnail 1280x720 ───────────────────────────────────────
THUMB_TEXT = {
    "vi": ("TẬP 2  ·  BẢN CHẤT LLM AGENT", "Bên trong", "một LLM Agent", "4 phần + 1 vòng lặp",
           ("Bộ não", "Kế hoạch", "Công cụ", "Bộ nhớ")),
    "en": ("EPISODE 2  ·  LLM AGENTS FROM SCRATCH", "Inside", "an LLM Agent", "4 parts + 1 loop",
           ("Brain", "Plan", "Tools", "Memory")),
}


def thumbnail(sc, k, lang="vi"):
    tag_text, line1, line2, sub_text, parts = THUMB_TEXT.get(lang, THUMB_TEXT["vi"])
    tag = k.label(tag_text, 30, k.MUTED, bold=True).to_corner(UP + LEFT, buff=0.6)
    left = VGroup(k.label(line1, 110, k.INK, bold=True), k.label(line2, 110, k.GOLD, bold=True))
    left.arrange(DOWN, aligned_edge=LEFT, buff=0.2).scale_to_fit_width(6.6).to_edge(LEFT, buff=0.75).shift(DOWN * 0.2)
    sub = k.label(sub_text, 42, k.ACCENT, bold=True).next_to(left, DOWN, aligned_edge=LEFT, buff=0.45)

    center = RIGHT * 4.0 + DOWN * 0.2
    ring = Ellipse(width=5.4, height=4.8, color=k.GOLD, stroke_width=7).move_to(center)
    dot = Dot(ring.point_from_proportion(0.62), color=k.GOLD, radius=0.18)

    def node(text, color, size=40):
        txt = k.label(text, size, k.INK, bold=True)
        base = RoundedRectangle(corner_radius=0.25, width=txt.width + 0.6, height=txt.height + 0.45,
                                stroke_width=0).set_fill(k.BG, opacity=1)  # che đường vòng phía sau
        box = base.copy().set_stroke(color, width=6).set_fill(color, opacity=0.2)
        return VGroup(base, box, txt.move_to(box))

    brain = node(parts[0], k.PURPLE, 48).move_to(center)
    # Ba phần còn lại nằm trên đường vòng: trên cùng, dưới-phải, dưới-trái.
    others = [node(parts[1], k.GREEN).move_to(ring.point_from_proportion(0.25)),
              node(parts[2], k.GOLD).move_to(ring.point_from_proportion(0.875)),
              node(parts[3], k.ACCENT).move_to(ring.point_from_proportion(0.625))]
    for o in others:
        overflow = o.get_right()[0] - 6.8
        if overflow > 0:
            o.shift(LEFT * overflow)
    dot.move_to(ring.point_from_proportion(0.06))
    sc.add(tag, left, sub, ring, dot, brain, *others)
