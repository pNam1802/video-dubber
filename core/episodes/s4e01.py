"""
Series "Machine Learning 2026 Spring" (series id 4), tập 1:
AI có thể tự sửa lỗi không? Tổng hợp phát triển kỹ thuật từ decoding, workflow đến reasoning.
Kể lại bài giảng của GS. Hung-yi Lee (Lý Hoành Nghị, NTU), đã được giáo sư cho phép:
https://www.youtube.com/watch?v=m3i2mk5hs8U

Kịch bản: core/episodes/s4e01_vi.txt (bản đầy đủ có mô tả cảnh: data/sources/hylee/m3i2mk5hs8U/script_vi.md).
Mỗi hàm là một cảnh (sc, spec, k); sc.at(i) = lúc câu thứ i của cảnh bắt đầu. Mảnh ghép chung ở _kit.py.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Circle, Create, CurvedArrow, Dot, FadeIn, FadeOut, GrowArrow,
    GrowFromCenter, GrowFromEdge, Indicate, Line, Polygon, Rectangle, Rotate, RoundedRectangle, Square, SurroundingRectangle,
    Text, Transform, VGroup, Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import arrow, bubble, card, check, cross, strike  # noqa: E402

BOARD = [(0, "two_loops"), (2, "three_routes"), (8, "next_token"), (12, "layers_rep"), (18, "part1_question"),
         (20, "probe"), (29, "yellow_vector"), (35, "classmate"), (40, "contrastive"), (45, "obama"),
         (50, "obama_math"), (58, "dola"), (65, "black_banana"), (73, "other_corruptions"), (75, "mti_forks"),
         (83, "kv_cache"), (92, "part1_recap"), (96, "gen_verify"), (100, "two_intuitions"),
         (109, "internal_external"), (114, "four_cells"), (119, "cl_cs"), (127, "personalities"),
         (129, "three_prompts"), (134, "majority_vote"), (138, "vote_vs_verify"), (144, "part2_recap"),
         (149, "knobs"), (155, "hillary"), (164, "new_errors"), (170, "rlvr"), (176, "rl_aha"), (180, "err_first"),
         (183, "parity_rote"), (187, "parity_steps"), (199, "pass_at_k"), (202, "darts"), (209, "other_camp"),
         (213, "outro")]


# ── Mảnh ghép riêng của tập ──────────────────────────────────
def _bars(k, rows, color, width=3.2, size=26, bar_h=0.34, label_w=1.9):
    """Cột xác suất nằm ngang: rows = [(nhãn, giá trị 0..1)]. Trả VGroup các hàng (nhãn, thanh)."""
    out = VGroup()
    for name, value in rows:
        tag = k.label(name, size, k.INK, max_width=label_w)
        bar = Rectangle(width=max(0.05, width * value), height=bar_h, stroke_width=0).set_fill(color, opacity=0.85)
        row = VGroup(tag, bar)
        tag.move_to(ORIGIN, aligned_edge=RIGHT)
        bar.next_to(ORIGIN, RIGHT, buff=0.2)
        out.add(row)
    out.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    # Căn mép trái của mọi thanh thẳng hàng.
    left = max(r[0].get_right()[0] for r in out) + 0.2
    for r in out:
        r[1].move_to([left + r[1].width / 2, r[1].get_center()[1], 0])
    return out


def _bar_to(k, row, value, color, width=3.2):
    """Thanh mới cùng chỗ với thanh cũ của hàng, dài theo value — dùng với Transform."""
    old = row[1]
    new = Rectangle(width=max(0.05, width * value), height=old.height, stroke_width=0).set_fill(color, opacity=0.85)
    return new.move_to(old.get_left(), aligned_edge=LEFT)


def _panel(k, title, color, body, buff=0.3):
    """Khung có tiêu đề nhỏ ở trên, bao quanh body."""
    head = k.label(title, 24, color, bold=True)
    group = VGroup(head, body).arrange(DOWN, buff=buff)
    box = RoundedRectangle(corner_radius=0.2, width=group.width + 0.6, height=group.height + 0.5,
                           color=color, stroke_width=2).set_fill(color, opacity=0.06)
    box.move_to(group)
    return VGroup(box, head, body)


def _cloud(center, color, n=16, spread=0.9, seed=0, radius=0.09):
    rnd = random.Random(seed)
    dots = VGroup(*[Dot([center[0] + rnd.gauss(0, spread * 0.45), center[1] + rnd.gauss(0, spread * 0.35), 0],
                        radius=radius, color=color) for _ in range(n)])
    return dots


def _duel(k, words, a_vals, b_vals, a_name, b_name, a_color=None, b_color=None, h=3.0, bw=0.6, gap=2.2,
          size=28):
    """Biểu đồ cột đôi lớn: mỗi token một cặp cột (bản thường | bản sai), tên token ở dưới.
    Trả VGroup(trục, nhãn token, cột a, cột b, chú thích) và gắn .h, .bw để _subtract dùng."""
    a_color, b_color = a_color or k.ACCENT, b_color or k.GREEN
    n = len(words)
    width = gap * n
    base = Line(LEFT * width / 2, RIGHT * width / 2, color=k.MUTED, stroke_width=3)
    names, a_bars, b_bars = VGroup(), VGroup(), VGroup()
    for i, (w, av, bv) in enumerate(zip(words, a_vals, b_vals)):
        x = -width / 2 + gap * (i + 0.5)
        a = Rectangle(width=bw, height=max(0.04, h * av), stroke_width=0).set_fill(a_color, 0.9)
        b = Rectangle(width=bw, height=max(0.04, h * bv), stroke_width=0).set_fill(b_color, 0.9)
        a.move_to([x - bw / 2 - 0.04, 0, 0], aligned_edge=DOWN)
        b.move_to([x + bw / 2 + 0.04, 0, 0], aligned_edge=DOWN)
        a_bars.add(a)
        b_bars.add(b)
        names.add(k.label(w, size, k.INK, bold=True, max_width=gap - 0.1).move_to([x, -0.4, 0]))
    legend = VGroup(
        VGroup(Square(0.28, stroke_width=0).set_fill(a_color, 0.9), k.label(a_name, 24, a_color, bold=True)).arrange(RIGHT, buff=0.15),
        VGroup(Square(0.28, stroke_width=0).set_fill(b_color, 0.9), k.label(b_name, 24, b_color, bold=True)).arrange(RIGHT, buff=0.15),
    ).arrange(RIGHT, buff=0.6).move_to(UP * (h + 0.5))
    chart = VGroup(base, names, a_bars, b_bars, legend)
    chart.h, chart.bw = h, bw
    return chart


def _show_duel(sc, chart, both=True):
    """Hiện trục, tên token, cột bản thường; both=True thì hiện luôn cột bản sai."""
    base, names, a_bars, b_bars, legend = chart
    sc.run(Create(base), FadeIn(names), FadeIn(legend[0]), run_time=0.5)
    sc.run(*[GrowFromEdge(b, DOWN) for b in a_bars], run_time=0.7)
    if both:
        _show_b(sc, chart)


def _show_b(sc, chart):
    sc.run(FadeIn(chart[4][1]), *[GrowFromEdge(b, DOWN) for b in chart[3]], run_time=0.7)


def _subtract(sc, k, chart, out_vals, out_name="sau khi trừ", winner=None):
    """Diễn phép trừ: cột bản sai trượt chồng lên cột bản thường, khoét bớt phần chung,
    phần còn lại thành cột vàng; khoanh token thắng."""
    base, names, a_bars, b_bars, legend = chart
    h = chart.h
    # 1) cột sai trượt sang, chồng lên cột thường (nửa trong suốt, viền đỏ)
    sc.run(*[b.animate.move_to(a.get_bottom(), aligned_edge=DOWN).set_fill(opacity=0.45).set_stroke(k.PINK, 3)
             for a, b in zip(a_bars, b_bars)], run_time=0.9)
    minus = VGroup(*[k.label("−", 40, k.PINK, bold=True).next_to(b, UP, buff=0.08) for b in b_bars])
    sc.run(FadeIn(minus), run_time=0.3)
    # 2) cột thường co lại thành kết quả, màu vàng
    outs = VGroup()
    for a, v in zip(a_bars, out_vals):
        outs.add(Rectangle(width=a.width * 2 + 0.08, height=max(0.04, h * v), stroke_width=0)
                 .set_fill(k.GOLD, 0.95).move_to(a.get_bottom() + RIGHT * (a.width / 2 + 0.04), aligned_edge=DOWN))
    new_legend = VGroup(Square(0.28, stroke_width=0).set_fill(k.GOLD, 0.95),
                        k.label(out_name, 26, k.GOLD, bold=True)).arrange(RIGHT, buff=0.15).move_to(legend)
    sc.run(FadeOut(b_bars), FadeOut(minus), *[Transform(a, o) for a, o in zip(a_bars, outs)],
           Transform(legend, new_legend), run_time=1.0)
    if winner is None:
        winner = max(range(len(out_vals)), key=lambda i: out_vals[i])
    ring = RoundedRectangle(corner_radius=0.15, width=max(names[winner].width, a_bars[winner].width) + 0.5,
                            height=a_bars[winner].height + names[winner].height + 0.7, color=k.GOLD, stroke_width=5)
    ring.move_to(VGroup(a_bars[winner], names[winner]))
    sc.run(Create(ring), names[winner].animate.set_color(k.GOLD), run_time=0.6)
    return ring


# Nhãn nhỏ ở góc trên bên trái: người xem luôn biết đang ở hướng nào trong ba hướng.
_PARTS = {1: ("1 · Lúc sinh chữ", "ACCENT"), 2: ("2 · Sinh rồi kiểm", "GREEN"), 3: ("3 · Huấn luyện", "GOLD")}


def _part(n):
    def wrap(fn):
        def scene(sc, s, k):
            text, color = _PARTS[n]
            chip = card(k, text, getattr(k, color), 18, 30, pad=0.3).to_corner(UP + LEFT, buff=0.25)
            sc.add(chip)
            fn(sc, s, k)
        scene.__name__ = fn.__name__
        return scene
    return wrap


def _model(k, text="LLM", color=None, w=2.0, h=1.3):
    color = color or k.PURPLE
    box = RoundedRectangle(corner_radius=0.25, width=w, height=h, color=color, stroke_width=4)
    box.set_fill(color, opacity=0.15)
    return VGroup(box, k.label(text, 34, k.INK, bold=True).move_to(box))


def _tokens(k, words, color=None, size=26):
    color = color or k.ACCENT
    cells = VGroup(*[card(k, w, color, size, 12, pad=0.35) for w in words])
    return cells.arrange(RIGHT, buff=0.12)


# ── MỞ ĐẦU ───────────────────────────────────────────────────
def two_loops(sc, s, k):
    left_t = k.label("Có người nhắc", 30, k.MUTED, bold=True).move_to(LEFT * 3.4 + UP * 2.9)
    right_t = k.label("Không ai nhắc", 30, k.MUTED, bold=True).move_to(RIGHT * 3.4 + UP * 2.9)
    divider = Line(UP * 3.3, DOWN * 3.3, color=k.MUTED, stroke_width=2).set_opacity(0.4)
    sc.run(FadeIn(left_t), FadeIn(right_t), Create(divider), run_time=0.6)

    ai_l = bubble(k, "Đáp án: 42", k.PURPLE, 20, 28).move_to(LEFT * 3.4 + UP * 1.4)
    user = bubble(k, "Sai rồi, sai ở bước 2", k.ACCENT, 22, 28).move_to(LEFT * 3.4 + DOWN * 0.2)
    fixed = bubble(k, "Xin lỗi, đáp án: 40", k.PURPLE, 22, 28).move_to(LEFT * 3.4 + DOWN * 1.8)
    sc.run(FadeIn(ai_l, shift=UP * 0.2), run_time=0.5)
    sc.run(FadeIn(user, shift=UP * 0.2), run_time=0.5)
    sc.run(FadeIn(fixed, shift=UP * 0.2), Create(check(k, 0.5).next_to(fixed, RIGHT, buff=0.2)), run_time=0.6)

    sc.wait_until(sc.at(1))
    ai_r = bubble(k, "Đáp án: 42", k.PURPLE, 20, 28).move_to(RIGHT * 3.4 + UP * 1.4)
    loop = CurvedArrow(ai_r.get_right() + RIGHT * 0.15 + DOWN * 0.1, ai_r.get_left() + LEFT * 0.15 + DOWN * 0.1,
                       angle=-2.6, color=k.GOLD, stroke_width=5)
    ask = k.label("?", 120, k.GOLD, bold=True).move_to(RIGHT * 3.4 + DOWN * 1.3)
    sc.run(FadeIn(ai_r, shift=UP * 0.2), run_time=0.5)
    sc.run(Create(loop), run_time=0.8)
    sc.run(FadeIn(ask, scale=0.5), run_time=0.5)
    sc.run(Indicate(ask, color=k.GOLD), run_time=0.8)


def three_routes(sc, s, k):
    model = _model(k, "LLM", w=2.4, h=1.5).move_to(DOWN * 0.3)
    sc.run(GrowFromCenter(model), run_time=0.6)
    spots = [LEFT * 4.3 + UP * 1.6, RIGHT * 4.3 + UP * 1.6, DOWN * 2.6]
    items = [("1. Inference", "sửa lúc sinh chữ", k.ACCENT), ("2. Workflow", "đổi quy trình bên ngoài", k.GREEN),
             ("3. Tham số", "huấn luyện để reasoning", k.GOLD)]
    for i, ((name, sub, color), spot) in enumerate(zip(items, spots)):
        sc.wait_until(sc.at(i + 1))
        tag = VGroup(k.label(name, 34, color, bold=True), k.label(sub, 24, k.MUTED)).arrange(DOWN, buff=0.12)
        tag.move_to(spot)
        link = Line(model.get_center(), tag.get_center(), color=color, stroke_width=3).set_opacity(0.5)
        link.put_start_and_end_on(model[0].get_boundary_point(tag.get_center() - model.get_center()),
                                  tag.get_center() + (model.get_center() - tag.get_center()) * 0.28)
        sc.run(Create(link), FadeIn(tag, shift=(spot - model.get_center()) * 0.05), run_time=0.6)
        items[i] = tag
    sc.wait_until(sc.at(4))
    hot = card(k, "reasoning", k.GOLD, 30, 12).next_to(items[2], RIGHT, buff=0.5)
    sc.run(FadeIn(hot, scale=1.2), Indicate(items[2], color=k.GOLD), run_time=0.8)


# ── NỀN TẢNG ─────────────────────────────────────────────────
def _prob_rows(k, rows, color, width=3.6, size=30):
    """Bảng xác suất nằm ngang có ghi phần trăm: rows = [(chữ, giá trị 0..1)]."""
    out = _bars(k, rows, color, width=width, size=size, bar_h=0.42, label_w=2.0)
    for (name, v), row in zip(rows, out):
        row.add(k.label(f"{round(v * 100)}%", 24, k.MUTED).next_to(row[1], RIGHT, buff=0.15))
    return out


def next_token(sc, s, k):
    head = k.label("Mô hình viết chữ thế nào?", 38, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.7)

    sc.wait_until(sc.at(1))
    toks = _tokens(k, ["Hôm", "nay", "trời"], size=32).move_to(LEFT * 3.4 + UP * 1.6)
    tok_t = k.label("mỗi ô là một token", 24, k.ACCENT).next_to(toks, DOWN, buff=0.25)
    sc.run(FadeIn(toks, lag_ratio=0.3), run_time=0.8)
    sc.run(FadeIn(tok_t), run_time=0.4)

    sc.wait_until(sc.at(2))
    model = _model(k, "LLM", w=2.2, h=1.1).move_to(LEFT * 3.4 + DOWN * 1.0)
    a1 = Arrow(toks.get_bottom() + DOWN * 0.45, model.get_top(), buff=0.1, color=k.MUTED, stroke_width=4)
    probs = _prob_rows(k, [("đẹp", 0.6), ("mưa", 0.3), ("xanh", 0.1)], k.ACCENT).move_to(RIGHT * 3.0 + DOWN * 1.0)
    p_t = k.label("bảng xác suất cho chữ tiếp theo", 26, k.ACCENT, bold=True).next_to(probs, UP, buff=0.35)
    a2 = Arrow(model.get_right(), probs.get_left(), buff=0.25, color=k.MUTED, stroke_width=4)
    sc.run(FadeOut(tok_t), GrowArrow(a1), FadeIn(model), run_time=0.6)
    sc.run(GrowArrow(a2), FadeIn(p_t), run_time=0.5)
    sc.run(*[GrowFromEdge(r[1], LEFT) for r in probs], *[FadeIn(r[0]) for r in probs],
           *[FadeIn(r[2]) for r in probs], run_time=1.0)

    sc.wait_until(sc.at(3))
    pick = RoundedRectangle(corner_radius=0.12, width=probs[0].width + 0.35, height=probs[0].height + 0.25,
                            color=k.GOLD, stroke_width=5).move_to(probs[0])
    sc.run(Create(pick), run_time=0.4)
    new = card(k, "đẹp", k.GOLD, 32, 12, pad=0.35).move_to(probs[0][0])
    target = card(k, "đẹp", k.GOLD, 32, 12, pad=0.35).next_to(toks, RIGHT, buff=0.12)
    sc.run(Transform(new, target), FadeOut(pick), run_time=0.9)
    again = k.label("gắn vào cuối câu, rồi lặp lại", 26, k.GOLD, bold=True).next_to(VGroup(toks, new), UP, buff=0.3)
    rows2 = [("lắm", 0.45), ("quá", 0.35), ("và", 0.2)]
    probs2 = _prob_rows(k, rows2, k.ACCENT).move_to(probs, aligned_edge=LEFT)
    sc.run(FadeIn(again), FadeOut(probs), FadeIn(probs2), run_time=0.8)
    sc.run(Indicate(model, color=k.GOLD), run_time=0.6)


def layers_rep(sc, s, k):
    head = k.label("Bên trong mô hình", 38, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.6)

    sc.wait_until(sc.at(1))
    layers = VGroup(*[RoundedRectangle(corner_radius=0.1, width=4.2, height=0.5, color=k.PURPLE, stroke_width=3)
                      .set_fill(k.PURPLE, opacity=0.1 + 0.06 * i) for i in range(6)]).arrange(UP, buff=0.14)
    layers.move_to(LEFT * 3.2 + DOWN * 0.5)
    names = VGroup(k.label("tầng 1", 20, k.MUTED).next_to(layers[0], LEFT, buff=0.2),
                   k.label("tầng cuối", 20, k.MUTED).next_to(layers[-1], LEFT, buff=0.2))
    word = card(k, "trời", k.ACCENT, 30, 10, pad=0.35).next_to(layers, DOWN, buff=0.35)
    out = k.label("bảng xác suất", 26, k.ACCENT, bold=True).next_to(layers, UP, buff=0.3)
    sc.run(FadeIn(word), run_time=0.4)
    sc.run(FadeIn(layers, lag_ratio=0.15), FadeIn(names), run_time=1.0)
    sc.run(FadeIn(out, shift=UP * 0.2), run_time=0.4)

    sc.wait_until(sc.at(2))
    rnd = random.Random(4)
    nums = VGroup()
    for i, layer in enumerate(layers):
        vals = " ".join(f"{rnd.uniform(-1, 1):+.1f}" for _ in range(5))
        nums.add(Text(f"[{vals} ...]", font=k.MONO, font_size=18, color=k.GREEN).move_to(layer))
    rep = k.label("representation", 32, k.GREEN, bold=True).move_to(RIGHT * 3.4 + UP * 1.4)
    rep_s = k.label("một dãy số dài ở mỗi tầng", 26, k.GREEN).next_to(rep, DOWN, buff=0.15)
    sc.run(FadeIn(nums, lag_ratio=0.2), run_time=1.2)
    sc.run(FadeIn(rep), FadeIn(rep_s), run_time=0.5)

    sc.wait_until(sc.at(3))
    brain = bubble(k, "“suy nghĩ trong đầu”, viết bằng số", k.GREEN, 20, 28).move_to(RIGHT * 3.4 + DOWN * 0.4)
    sc.run(FadeIn(brain, scale=0.9), run_time=0.6)

    sc.wait_until(sc.at(4))
    deep = Arrow(layers.get_corner(DOWN + RIGHT) + RIGHT * 0.5, layers.get_corner(UP + RIGHT) + RIGHT * 0.5,
                 buff=0, color=k.GOLD, stroke_width=6)
    raw = k.label("non", 24, k.MUTED).next_to(deep.get_start(), RIGHT, buff=0.15)
    ripe = k.label("chín", 24, k.GOLD, bold=True).next_to(deep.get_end(), RIGHT, buff=0.15)
    sc.run(GrowArrow(deep), FadeIn(raw), FadeIn(ripe), run_time=0.8)

    sc.wait_until(sc.at(5))
    keep = VGroup(card(k, "bảng xác suất", k.ACCENT, 26, 20), card(k, "representation", k.GREEN, 26, 20))
    keep.arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=0.35).shift(RIGHT * 2.4)
    sc.run(FadeOut(brain), FadeIn(keep, shift=UP * 0.2), run_time=0.6)


@_part(1)
def part1_question(sc, s, k):
    toks = _tokens(k, ["Hôm", "nay", "trời"]).move_to(LEFT * 3.6 + UP * 2.0)
    tf = _model(k, "LLM", k.PURPLE, w=3.0, h=1.1).move_to(LEFT * 3.0 + DOWN * 0.3)
    reps = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.28, height=1.1, color=k.GREEN, stroke_width=2)
                    .set_fill(k.GREEN, opacity=0.3) for _ in range(3)]).arrange(RIGHT, buff=0.1)
    reps.move_to(RIGHT * 0.5 + DOWN * 0.3)
    probs = _bars(k, [("đẹp", 0.6), ("mưa", 0.3), ("xanh", 0.1)], k.ACCENT, width=2.0, size=28)
    probs.move_to(RIGHT * 4.2 + DOWN * 0.3)
    a1 = Arrow(toks.get_bottom(), tf.get_top(), buff=0.15, color=k.MUTED, stroke_width=4)
    a2, a3 = arrow(k, tf, reps), arrow(k, reps, probs)
    rep_t = k.label("suy nghĩ trong đầu", 24, k.GREEN).next_to(reps, UP, buff=0.25)
    prob_t = k.label("bảng xác suất", 24, k.ACCENT).next_to(probs, UP, buff=0.25)
    sc.run(FadeIn(toks), GrowArrow(a1), FadeIn(tf), run_time=0.6)
    sc.run(GrowArrow(a2), FadeIn(reps), FadeIn(rep_t), GrowArrow(a3), FadeIn(probs), FadeIn(prob_t), run_time=0.8)
    eye1 = k.label("sắp sai?", 32, k.PINK, bold=True).next_to(reps, DOWN, buff=0.5)
    eye2 = k.label("sắp sai?", 32, k.PINK, bold=True).next_to(probs, DOWN, buff=0.5)
    sc.run(FadeIn(eye1, shift=UP * 0.2), Indicate(reps, color=k.PINK), run_time=0.7)
    sc.run(FadeIn(eye2, shift=UP * 0.2), Indicate(probs, color=k.PINK), run_time=0.7)

    sc.wait_until(sc.at(1))
    flow = VGroup(card(k, "phát hiện lỗi", k.PINK, 28, 14), card(k, "tự sửa", k.GREEN, 28, 10)).arrange(RIGHT, buff=1.0)
    flow.to_edge(DOWN, buff=0.5)
    sc.run(FadeIn(flow[0]), run_time=0.4)
    sc.run(GrowArrow(arrow(k, flow[0], flow[1])), FadeIn(flow[1]), run_time=0.5)


def _plane(k):
    frame = RoundedRectangle(corner_radius=0.2, width=6.6, height=5.4, color=k.MUTED, stroke_width=2)
    frame.set_fill(k.INK, opacity=0.03)
    cap = k.label("bản đồ representation: mỗi câu trả lời là một điểm", 22, k.MUTED).next_to(frame, UP, buff=0.15)
    return VGroup(frame, cap)


@_part(1)
def probe(sc, s, k):
    paper = card(k, "paper 2023", k.MUTED, 24, 12).to_corner(UP + RIGHT, buff=0.4)
    sc.run(FadeIn(paper), run_time=0.4)
    plane = _plane(k).move_to(LEFT * 2.8 + DOWN * 0.25)
    center = plane[0].get_center()
    sc.wait_until(sc.at(1))
    sc.run(FadeIn(plane), run_time=0.6)

    sc.wait_until(sc.at(2))
    qs = VGroup(*[card(k, f"câu {i}", k.INK, 22, 8, pad=0.3, bold=False) for i in (1, 2, 3, "...")])
    qs.arrange(DOWN, buff=0.15).move_to(RIGHT * 4.3 + UP * 0.6)
    sc.run(FadeIn(qs, lag_ratio=0.25), run_time=0.8)

    sc.wait_until(sc.at(3))
    g_c, b_c = center + LEFT * 1.4 + UP * 0.9, center + RIGHT * 1.4 + DOWN * 0.9
    good = _cloud(g_c, k.GREEN, seed=1, radius=0.11)
    bad = _cloud(b_c, k.PINK, seed=2, radius=0.11)
    legend = VGroup(VGroup(Dot(radius=0.11, color=k.GREEN), k.label("trả lời đúng", 26, k.GREEN, bold=True)).arrange(RIGHT, buff=0.15),
                    VGroup(Dot(radius=0.11, color=k.PINK), k.label("trả lời sai", 26, k.PINK, bold=True)).arrange(RIGHT, buff=0.15))
    legend.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 4.3 + UP * 0.6)
    # chấm xen kẽ, chưa thấy vùng
    mixed = VGroup(*[d for pair in zip(good[:6], bad[:6]) for d in pair])
    sc.run(FadeOut(qs), FadeIn(legend), run_time=0.4)
    sc.run(FadeIn(mixed, lag_ratio=0.15), run_time=1.0)

    sc.wait_until(sc.at(4))
    sc.run(FadeIn(good[6:], lag_ratio=0.08), FadeIn(bad[6:], lag_ratio=0.08), run_time=1.0)
    zg = Circle(radius=1.0, color=k.GREEN, stroke_width=2).set_fill(k.GREEN, 0.06).move_to(g_c)
    zb = Circle(radius=1.0, color=k.PINK, stroke_width=2).set_fill(k.PINK, 0.06).move_to(b_c)
    sc.run(Create(zg), Create(zb), run_time=0.6)

    sc.wait_until(sc.at(5))
    split = Line(center + LEFT * 2.6 + DOWN * 2.2, center + RIGHT * 2.6 + UP * 2.2, color=k.GOLD, stroke_width=6)
    sc.run(Create(split), run_time=0.9)

    sc.wait_until(sc.at(6))
    clf = card(k, "bộ phân loại", k.GOLD, 28, 14).move_to(RIGHT * 4.3 + DOWN * 0.9)
    sides = k.label("bên trái: đúng · bên phải: sai", 22, k.GOLD).next_to(clf, DOWN, buff=0.15)
    sc.run(FadeIn(clf), FadeIn(sides), run_time=0.6)

    sc.wait_until(sc.at(7))
    new = VGroup(Dot(center + LEFT * 0.9 + UP * 1.7, radius=0.16, color=k.INK),
                 Dot(center + RIGHT * 1.7 + DOWN * 0.2, radius=0.16, color=k.INK))
    q = VGroup(*[k.label("?", 26, k.INK, bold=True).next_to(d, UP, buff=0.08) for d in new])
    sc.run(FadeIn(new, scale=0.5), FadeIn(q), run_time=0.5)
    sc.run(new[0].animate.set_color(k.GREEN), new[1].animate.set_color(k.PINK), FadeOut(q), run_time=0.7)
    unseen = k.label("cả câu hỏi chưa từng thấy", 24, k.INK).next_to(sides, DOWN, buff=0.3)
    sc.run(FadeIn(unseen), run_time=0.4)

    sc.wait_until(sc.at(8))
    msg = k.label("tín hiệu “mình đang nói sai” nằm sẵn trong đầu mô hình", 30, k.GOLD, bold=True)
    msg.to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(msg, shift=UP * 0.2), run_time=0.6)


@_part(1)
def yellow_vector(sc, s, k):
    plane = _plane(k).move_to(LEFT * 2.8 + DOWN * 0.4)
    center = plane[0].get_center()
    g_c, b_c = center + LEFT * 1.4 + UP * 0.9, center + RIGHT * 1.4 + DOWN * 0.9
    good, bad = _cloud(g_c, k.GREEN, seed=1, radius=0.11), _cloud(b_c, k.PINK, seed=2, radius=0.11)
    sc.run(FadeIn(plane), FadeIn(good), FadeIn(bad), run_time=0.6)

    sc.wait_until(sc.at(1))
    gm = Dot(g_c, radius=0.2, color=k.GREEN).set_stroke(k.INK, 3)
    bm = Dot(b_c, radius=0.2, color=k.PINK).set_stroke(k.INK, 3)
    gt = k.label("giữa vùng xanh", 22, k.GREEN, bold=True).next_to(gm, UP, buff=0.2)
    bt = k.label("giữa vùng đỏ", 22, k.PINK, bold=True).next_to(bm, RIGHT, buff=0.2)
    sc.run(good.animate.set_opacity(0.3), bad.animate.set_opacity(0.3), GrowFromCenter(gm), GrowFromCenter(bm),
           FadeIn(gt), FadeIn(bt), run_time=0.8)

    sc.wait_until(sc.at(2))
    vec = Arrow(b_c, g_c, buff=0.22, color=k.GOLD, stroke_width=9)
    vt = k.label("vector vàng", 34, k.GOLD, bold=True).move_to(RIGHT * 4.0 + UP * 1.6)
    vs = k.label("chỉ từ “sai” sang “đúng”", 24, k.GOLD).next_to(vt, DOWN, buff=0.15)
    sc.run(GrowArrow(vec), FadeIn(vt), FadeIn(vs), run_time=0.9)

    sc.wait_until(sc.at(3))
    p = Dot(center + RIGHT * 2.0 + DOWN * 1.8, radius=0.18, color=k.PINK).set_stroke(k.INK, 3)
    p_t = k.label("câu sắp trả lời sai", 22, k.PINK, bold=True).next_to(p, LEFT, buff=0.2)
    sc.run(FadeIn(p, scale=0.5), FadeIn(p_t), run_time=0.5)
    shift = g_c - b_c
    push = Arrow(p.get_center(), p.get_center() + shift, buff=0.12, color=k.GOLD, stroke_width=7)
    sc.run(GrowArrow(push), run_time=0.6)
    sc.run(p.animate.move_to(p.get_center() + shift).set_color(k.GREEN), FadeOut(p_t), run_time=0.9)

    sc.wait_until(sc.at(4))
    ok = VGroup(check(k, 0.5), k.label("trả lời đúng luôn", 28, k.GREEN, bold=True)).arrange(RIGHT, buff=0.2)
    ok.move_to(RIGHT * 4.0 + DOWN * 0.2)
    sc.run(FadeIn(ok, shift=UP * 0.2), run_time=0.5)

    sc.wait_until(sc.at(5))
    cost = card(k, "nhưng phải thu thập trước rất nhiều câu đúng và sai", k.PINK, 24, 22).move_to(RIGHT * 4.0 + DOWN * 2.0)
    sc.run(FadeIn(cost, shift=UP * 0.2), run_time=0.6)


def _student(k, color, label, size=1.0):
    head = Circle(radius=0.3 * size, color=color, stroke_width=4).set_fill(color, 0.25)
    body = RoundedRectangle(corner_radius=0.25 * size, width=0.95 * size, height=0.65 * size, color=color,
                            stroke_width=4).set_fill(color, 0.15)
    body.next_to(head, DOWN, buff=0.05)
    t = k.label(label, 22, color, bold=True).next_to(body, DOWN, buff=0.12)
    return VGroup(head, body, t)


@_part(1)
def classmate(sc, s, k):
    q = k.label("Không cần dữ liệu thì sao?", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(q), run_time=0.5)
    sc.wait_until(sc.at(1))
    title = k.label("Contrastive decoding", 44, k.GOLD, bold=True).move_to(q)
    sc.run(Transform(q, title), run_time=0.7)

    sc.wait_until(sc.at(2))
    me = _student(k, k.ACCENT, "bạn", 1.3).move_to(LEFT * 4.8 + DOWN * 0.2)
    weak = _student(k, k.GREEN, "bạn học kém", 1.3).move_to(RIGHT * 4.8 + DOWN * 0.2)
    sheet = RoundedRectangle(corner_radius=0.12, width=3.4, height=3.4, color=k.INK, stroke_width=3).set_fill("#1b2030", 1)
    sheet.move_to(DOWN * 0.2)
    opts = VGroup(*[k.label(f"{c}.  ________", 28, k.INK) for c in "ABCD"]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    opts.move_to(sheet)
    sc.run(FadeIn(sheet), FadeIn(opts), FadeIn(me), run_time=0.6)
    sc.run(FadeIn(weak, shift=LEFT * 0.3), run_time=0.5)

    sc.wait_until(sc.at(3))
    point = Arrow(weak.get_left(), opts[1].get_right() + RIGHT * 0.2, buff=0.15, color=k.GREEN, stroke_width=5)
    choose = bubble(k, "chắc là B!", k.GREEN, 12, 26).next_to(weak, UP, buff=0.25)
    sc.run(GrowArrow(point), FadeIn(choose), run_time=0.6)
    cut = strike(k, opts[1])
    sc.run(Create(cut), opts[1].animate.set_opacity(0.4), run_time=0.5)
    avoid = bubble(k, "vậy tránh B", k.ACCENT, 12, 26).next_to(me, UP, buff=0.25)
    sc.run(FadeIn(avoid), run_time=0.5)

    sc.wait_until(sc.at(4))
    sc.run(*[FadeOut(m) for m in (sheet, opts, point, choose, avoid, cut)], run_time=0.4)
    m1 = _model(k, "LLM", k.ACCENT, w=2.2, h=1.0).move_to(LEFT * 2.0 + UP * 0.2)
    m2 = _model(k, "LLM", k.GREEN, w=2.2, h=1.0).move_to(RIGHT * 2.0 + UP * 0.2)
    t1 = k.label("mô hình bình thường", 24, k.ACCENT, bold=True).next_to(m1, DOWN, buff=0.2)
    t2 = k.label("bản cố tình trả lời sai", 24, k.GREEN, bold=True).next_to(m2, DOWN, buff=0.2)
    sc.run(FadeIn(m1), FadeIn(t1), run_time=0.5)
    sc.run(FadeIn(m2), FadeIn(t2), run_time=0.5)
    away = k.label("đẩy câu trả lời ra xa những gì bản sai thích", 30, k.GOLD, bold=True).to_edge(DOWN, buff=0.45)
    sc.run(FadeIn(away, shift=UP * 0.2), run_time=0.6)


@_part(1)
def contrastive(sc, s, k):
    title = k.label("Hai bảng xác suất, trừ cho nhau", 36, k.INK, bold=True).to_edge(UP, buff=0.35)
    sc.run(FadeIn(title), run_time=0.5)
    q = bubble(k, "Thủ đô của Úc là ___", k.INK, 24, 28).move_to(LEFT * 4.6 + UP * 1.6)
    normal_m = _model(k, "LLM", k.ACCENT, w=1.8, h=0.9).move_to(LEFT * 5.6 + DOWN * 0.3)
    broken_m = _model(k, "LLM", k.GREEN, w=1.8, h=0.9).move_to(LEFT * 3.4 + DOWN * 0.3)
    hurt = k.label("bản sai", 22, k.GREEN, bold=True).next_to(broken_m, DOWN, buff=0.2)
    norm_t = k.label("bình thường", 22, k.ACCENT, bold=True).next_to(normal_m, DOWN, buff=0.2)
    chart = _duel(k, ["Sydney", "Canberra", "Melbourne"], [0.5, 0.45, 0.12], [0.85, 0.08, 0.25],
                  "bình thường", "bản sai", h=3.0, gap=2.5, size=30).shift(RIGHT * 2.5 + DOWN * 1.6)
    sc.run(FadeIn(q), run_time=0.4)
    sc.run(GrowArrow(Arrow(q.get_bottom(), normal_m.get_top(), buff=0.1, color=k.ACCENT, stroke_width=4)),
           GrowArrow(Arrow(q.get_bottom(), broken_m.get_top(), buff=0.1, color=k.GREEN, stroke_width=4)),
           FadeIn(normal_m), FadeIn(broken_m), FadeIn(norm_t), FadeIn(hurt), run_time=0.6)
    _show_duel(sc, chart)

    sc.wait_until(sc.at(1))
    _subtract(sc, k, chart, [0.35, 0.62, 0.05], "bình thường − một phần bản sai", winner=1)

    sc.wait_until(sc.at(2))
    down = k.label("bản sai rất thích Sydney: bị trừ nhiều, tụt xuống", 24, k.PINK).to_edge(DOWN, buff=0.75)
    sc.run(FadeIn(down), Indicate(chart[1][0], color=k.PINK), run_time=0.7)

    sc.wait_until(sc.at(3))
    up = k.label("chỉ mô hình thường thích Canberra: giữ điểm, lên đầu", 24, k.GOLD, bold=True).next_to(down, DOWN, buff=0.15)
    sc.run(FadeIn(up), run_time=0.5)

    sc.wait_until(sc.at(4))
    formula = Text("(1 + alpha) · bình thường  −  alpha · bản sai,   alpha < 1", font=k.MONO, font_size=26, color=k.INK)
    formula.next_to(title, DOWN, buff=0.25)
    box = RoundedRectangle(corner_radius=0.15, width=formula.width + 0.5, height=formula.height + 0.35,
                           color=k.GOLD, stroke_width=3).move_to(formula)
    sc.run(Write(formula), Create(box), run_time=1.0)


@_part(1)
def obama(sc, s, k):
    era = card(k, "2022 · “thời thượng cổ, chưa có ChatGPT, loài người còn ăn lông ở lỗ”", k.MUTED, 22, 80)
    era.to_edge(UP, buff=0.8)
    sc.run(FadeIn(era, shift=DOWN * 0.2), run_time=0.6)

    sc.wait_until(sc.at(1))
    prompt = VGroup(k.label("Obama sinh ở Honolulu, Hawaii. Ông sinh năm", 34, k.INK, bold=True),
                    card(k, " ? ", k.GOLD, 34, 6, pad=0.3)).arrange(RIGHT, buff=0.2).move_to(UP * 1.6)
    sc.run(Write(prompt[0]), run_time=1.2)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(prompt[1], scale=1.3), run_time=0.5)

    sc.wait_until(sc.at(3))
    big = _model(k, "GPT-2 bản lớn", k.ACCENT, w=4.0, h=1.0).move_to(LEFT * 3.4 + DOWN * 0.6)
    says = bubble(k, "… sinh năm Hawaii", k.ACCENT, 20, 30).next_to(big, DOWN, buff=0.35)
    sc.run(FadeIn(big), run_time=0.5)
    sc.run(FadeIn(says, shift=UP * 0.2), run_time=0.5)
    sc.run(Create(cross(k, says)), run_time=0.4)
    need = k.label("chỗ này phải là một năm", 24, k.PINK).next_to(says, DOWN, buff=0.2)
    sc.run(FadeIn(need), run_time=0.4)

    sc.wait_until(sc.at(4))
    small = _model(k, "GPT-2 Small", k.GREEN, w=3.2, h=0.8).move_to(RIGHT * 3.6 + DOWN * 0.6)
    weak = k.label("bạn học kém", 28, k.GREEN, bold=True).next_to(small, DOWN, buff=0.3)
    sc.run(FadeIn(small, shift=LEFT * 0.3), run_time=0.5)
    sc.run(FadeIn(weak), run_time=0.4)


@_part(1)
def obama_math(sc, s, k):
    prompt = VGroup(k.label("Obama sinh ở Honolulu, Hawaii. Ông sinh năm", 28, k.INK, bold=True),
                    card(k, " ? ", k.GOLD, 28, 6, pad=0.25)).arrange(RIGHT, buff=0.2).to_edge(UP, buff=0.7)
    sc.add(prompt)
    note = k.label("số minh họa", 22, k.MUTED).next_to(prompt, DOWN, buff=0.15)
    sc.run(FadeIn(note), run_time=0.4)

    chart = _duel(k, ["Hawaii", "1961"], [0.4, 0.3], [0.6, 0.05], "GPT-2 bản lớn", "GPT-2 Small",
                  h=3.6, gap=3.4, size=34).shift(LEFT * 2.2 + DOWN * 2.2)
    base, names, a_bars, b_bars, legend = chart

    def pct(bar, text, color):
        return k.label(text, 24, color, bold=True).next_to(bar, UP, buff=0.12)

    sc.wait_until(sc.at(1))
    _show_duel(sc, chart, both=False)
    a_lab = VGroup(pct(a_bars[0], "40%", k.ACCENT), pct(a_bars[1], "30%", k.ACCENT))
    sc.run(FadeIn(a_lab), run_time=0.4)

    sc.wait_until(sc.at(2))
    _show_b(sc, chart)
    b_lab = VGroup(pct(b_bars[0], "60%", k.GREEN), pct(b_bars[1], "5%", k.GREEN))
    sc.run(FadeIn(b_lab), run_time=0.4)

    sc.wait_until(sc.at(3))
    why = k.label("Vì sao?", 32, k.INK, bold=True).move_to(RIGHT * 4.3 + UP * 1.2)
    sc.run(FadeIn(why), run_time=0.4)
    sc.wait_until(sc.at(4))
    rep = bubble(k, "lặp lại chữ vừa thấy: lỗi dễ mắc, mô hình càng kém càng hay mắc", k.GREEN, 20, 24)
    rep.next_to(why, DOWN, buff=0.3)
    sc.run(FadeIn(rep, shift=UP * 0.2), run_time=0.6)

    sc.wait_until(sc.at(5))
    m1 = Text("40 − 60 = −20", font=k.MONO, font_size=30, color=k.PINK).move_to(RIGHT * 4.3 + DOWN * 1.0)
    sc.run(FadeOut(a_lab), FadeOut(b_lab), FadeOut(why), FadeOut(rep), run_time=0.3)
    _subtract(sc, k, chart, [0.02, 0.25], "lớn − nhỏ", winner=1)
    sc.run(Write(m1), names[0].animate.set_color(k.PINK), run_time=0.7)

    sc.wait_until(sc.at(6))
    m2 = Text("30 − 5  =  25", font=k.MONO, font_size=30, color=k.GOLD).next_to(m1, DOWN, buff=0.3)
    sc.run(Write(m2), Indicate(a_bars[1], color=k.GOLD), run_time=0.8)
    sc.run(Transform(prompt[1], card(k, "1961", k.GOLD, 28, 6, pad=0.25).move_to(prompt[1], aligned_edge=LEFT)),
           run_time=0.5)

    sc.wait_until(sc.at(7))
    msg = card(k, "phép trừ khử đi lỗi chung · phần còn lại là thứ chỉ mô hình giỏi mới biết", k.GOLD, 24, 80)
    msg.to_edge(DOWN, buff=0.2)
    sc.run(FadeIn(msg, shift=UP * 0.2), run_time=0.7)


@_part(1)
def dola(sc, s, k):
    head = k.label("Mang theo mô hình nhỏ thì hơi phiền", 34, k.INK, bold=True).to_edge(UP, buff=0.35)
    sc.run(FadeIn(head), run_time=0.4)
    sc.wait_until(sc.at(1))
    tag = card(k, "DoLa · 2023 · bật một cờ trong Hugging Face", k.ACCENT, 26, 60).move_to(head)
    sc.run(Transform(head, tag), run_time=0.6)

    layers = VGroup(*[RoundedRectangle(corner_radius=0.1, width=2.6, height=0.46, color=k.PURPLE, stroke_width=3)
                      .set_fill(k.PURPLE, opacity=0.12 + 0.05 * i) for i in range(8)]).arrange(UP, buff=0.1)
    layers.move_to(LEFT * 5.2 + DOWN * 0.6)
    sh_t = k.label("tầng nông", 24, k.GREEN, bold=True).next_to(layers[1], RIGHT, buff=0.2)
    dp_t = k.label("tầng cuối", 24, k.ACCENT, bold=True).next_to(layers[7], RIGHT, buff=0.2)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(layers, lag_ratio=0.1), run_time=0.7)
    sc.run(Indicate(layers[1], color=k.GREEN), FadeIn(sh_t), run_time=0.5)
    surf = bubble(k, "thấy “Hawaii” thì lặp lại “Hawaii”", k.GREEN, 18, 22).next_to(sh_t, DOWN, buff=0.3)
    surf.align_to(sh_t, LEFT)
    sc.run(FadeIn(surf), run_time=0.5)

    chart = _duel(k, ["Hawaii", "1961"], [0.4, 0.3], [0.6, 0.05], "tầng cuối", "tầng nông",
                  h=3.0, gap=2.8, size=32).shift(RIGHT * 3.4 + DOWN * 1.8)
    sc.wait_until(sc.at(3))
    weak = k.label("tầng nông = bạn học kém", 26, k.GREEN, bold=True).next_to(surf, DOWN, buff=0.3).align_to(sh_t, LEFT)
    sc.run(FadeIn(weak), run_time=0.4)

    sc.wait_until(sc.at(4))
    sc.run(Indicate(layers[7], color=k.ACCENT), FadeIn(dp_t), run_time=0.5)
    _show_duel(sc, chart)
    _subtract(sc, k, chart, [0.02, 0.25], "cuối − nông", winner=1)

    sc.wait_until(sc.at(5))
    same = k.label("y như ví dụ Obama, chỉ đổi người đóng vai học kém", 24, k.GOLD, width_chars=28).next_to(chart, UP, buff=0.3)
    sc.run(FadeIn(same), run_time=0.5)

    sc.wait_until(sc.at(6))
    gain = card(k, "tầng nông vốn đã phải tính · gần như miễn phí", k.GREEN, 26, 60).to_edge(DOWN, buff=0.25)
    sc.run(FadeOut(surf), FadeOut(weak), FadeIn(gain, shift=UP * 0.2), run_time=0.6)


def _photo(k, w=3.0, h=2.2, color=None):
    color = color or k.MUTED
    return RoundedRectangle(corner_radius=0.12, width=w, height=h, color=color, stroke_width=3).set_fill("#1b2030", 1)


def _noise(k, frame, n=12, seed=3):
    """Phủ ô nhiễu xám lên khung ảnh."""
    rnd = random.Random(seed)
    cw, ch = frame.width / n, frame.height / (n * frame.height / frame.width)
    rows = max(1, round(frame.height / cw))
    cells = VGroup()
    for i in range(n):
        for j in range(rows):
            g = rnd.randint(40, 200)
            sq = Square(side_length=cw, stroke_width=0).set_fill(f"#{g:02x}{g:02x}{g:02x}", 0.95)
            sq.move_to(frame.get_corner(UP + LEFT) + RIGHT * (cw * (i + 0.5)) + DOWN * (cw * (j + 0.5)))
            cells.add(sq)
    return cells


def _banana(k, color, w=2.4):
    """Quả chuối: thân cong theo cung tròn, dày ở giữa, thon hai đầu; cuống ở đầu phải."""
    import numpy as np
    from manim import VMobject
    angles = np.linspace(3.75, 5.75, 48)
    u = (angles - angles[0]) / (angles[-1] - angles[0])
    thick = 0.42 * np.sin(np.pi * u) ** 0.6 + 0.03
    outer = [np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    inner = [np.array([(1 - d) * np.cos(a), (1 - d) * np.sin(a), 0]) for a, d in zip(angles, thick)]
    shape = VMobject(stroke_color="#3a3a3a", stroke_width=3).set_points_as_corners(outer + inner[::-1] + [outer[0]])
    shape.set_fill(color, 1)
    tip = (outer[-1] + inner[-1]) / 2
    stem = Polygon(tip + DOWN * 0.04, tip + UP * 0.05, tip + RIGHT * 0.2 + UP * 0.25, tip + RIGHT * 0.27 + UP * 0.17,
                   stroke_width=0).set_fill("#5b4a2e", 1)
    end = Dot((outer[0] + inner[0]) / 2, radius=0.045, color="#3a2a1a")
    group = VGroup(shape, stem, end).rotate(0.2)
    return group.scale_to_fit_width(w)


@_part(1)
def black_banana(sc, s, k):
    note = k.label("Cách làm hỏng còn sáng tạo hơn", 32, k.INK, bold=True).to_edge(UP, buff=0.35)
    sc.run(FadeIn(note), run_time=0.5)
    sc.wait_until(sc.at(1))
    best = k.label("“ví dụ thuyết phục nhất”", 32, k.GOLD, bold=True).move_to(note)
    sc.run(Transform(note, best), run_time=0.5)

    sc.wait_until(sc.at(2))
    photo = _photo(k, 3.4, 2.5).set_fill("#d9d4c7", 1).move_to(LEFT * 4.8 + UP * 1.0)
    ban = _banana(k, "#1f1f1f", 2.5).move_to(photo)
    q = k.label("Chuối màu gì?", 30, k.INK, bold=True).next_to(photo, UP, buff=0.2)
    sc.run(FadeIn(photo), FadeIn(ban), FadeIn(q), run_time=0.8)

    sc.wait_until(sc.at(3))
    web = VGroup(*[_banana(k, k.GOLD, 0.8) for _ in range(5)]).arrange(RIGHT, buff=0.15)
    web_t = k.label("chữ viết trên mạng: chuối gần như luôn vàng", 22, k.GOLD).next_to(web, DOWN, buff=0.15)
    VGroup(web, web_t).move_to(RIGHT * 2.6 + UP * 2.1)
    sc.run(FadeIn(web, lag_ratio=0.2), FadeIn(web_t), run_time=0.9)

    chart = _duel(k, ["vàng", "đen"], [0.55, 0.5], [0.85, 0.1], "ảnh thật", "ảnh nhiễu",
                  h=2.6, gap=2.6, size=32).shift(RIGHT * 2.6 + DOWN * 2.4)
    sc.wait_until(sc.at(4))
    _show_duel(sc, chart, both=False)
    hm = k.label("lưỡng lự", 26, k.MUTED).next_to(chart[2], LEFT, buff=0.6)
    sc.run(FadeIn(hm), run_time=0.3)

    sc.wait_until(sc.at(5))
    noisy = _photo(k, 3.4, 2.5).move_to(LEFT * 4.8 + DOWN * 2.0)
    grain = _noise(k, noisy, n=14)
    sc.run(FadeIn(noisy), FadeIn(grain), FadeOut(hm), run_time=0.6)

    sc.wait_until(sc.at(6))
    blind = k.label("chỉ còn định kiến: “vàng”", 24, k.GREEN, bold=True).next_to(noisy, UP, buff=0.12)
    sc.run(FadeIn(blind), run_time=0.4)
    _show_b(sc, chart)

    sc.wait_until(sc.at(7))
    _subtract(sc, k, chart, [0.12, 0.75], "thật − nhiễu", winner=1)
    left = k.label("trừ đi định kiến, còn lại thứ thật sự nhìn thấy", 24, k.GOLD, bold=True)
    left.to_edge(DOWN, buff=0.2).shift(RIGHT * 2.6)
    sc.run(FadeIn(left), Indicate(ban, color=k.GOLD, scale_factor=1.1), run_time=0.7)


@_part(1)
def other_corruptions(sc, s, k):
    head = k.label("Những cách làm hỏng khác", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.4)
    spell = bubble(k, "Bạn là mô hình rất tệ, hãy trả lời sai.", k.PINK, 18, 24)
    doc = VGroup(*[RoundedRectangle(corner_radius=0.08, width=1.2, height=1.5, color=k.INK, stroke_width=2)
                   .set_fill("#1b2030", 1).shift(RIGHT * 0.12 * i + DOWN * 0.12 * i) for i in range(3)])
    lines = VGroup(*[Line(LEFT * 0.35, RIGHT * 0.35, color=k.MUTED, stroke_width=3) for _ in range(4)])
    lines.arrange(DOWN, buff=0.2).move_to(doc[2])
    doc.add(lines)
    wave = VGroup(*[Line(UP * h, DOWN * h, color=k.ACCENT, stroke_width=7)
                    for h in (0.2, 0.5, 0.7, 0.35, 0.6, 0.25, 0.45)]).arrange(RIGHT, buff=0.18)
    items = [(spell, "chèn câu “làm ngu”"), (doc, "bỏ tài liệu tra cứu"), (wave, "âm thanh thành khoảng lặng")]
    cols = VGroup()
    for pic, text in items:
        holder = RoundedRectangle(corner_radius=0.2, width=3.8, height=2.8, color=k.MUTED, stroke_width=2)
        holder.set_fill(k.INK, 0.03)
        pic.move_to(holder)
        cols.add(VGroup(holder, pic, k.label(text, 26, k.INK, bold=True).next_to(holder, DOWN, buff=0.25)))
    cols.arrange(RIGHT, buff=0.35).move_to(UP * 0.2)
    for c in cols:
        sc.run(FadeIn(c), run_time=0.5)
    sc.run(Create(cross(k, doc)), Transform(wave, Line(LEFT * 1.1, RIGHT * 1.1, color=k.ACCENT, stroke_width=5)
                                            .move_to(cols[2][0])), run_time=0.9)

    sc.wait_until(sc.at(1))
    same = card(k, "cùng một ý: tạo bạn học kém rồi tránh xa nó", k.GOLD, 30, 60).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(same, shift=UP * 0.2), run_time=0.6)


@_part(1)
def mti_forks(sc, s, k):
    cost = VGroup(_model(k, "LLM", k.ACCENT, w=1.8, h=0.9), k.label("+", 44, k.INK, bold=True),
                  _model(k, "bản sai", k.GREEN, w=1.8, h=0.9), k.label("= gấp đôi công mỗi chữ", 30, k.PINK, bold=True))
    cost.arrange(RIGHT, buff=0.3).to_edge(UP, buff=0.8)
    sc.run(FadeIn(cost, lag_ratio=0.2), run_time=0.8)

    sc.wait_until(sc.at(1))
    tag = card(k, "MTI · cuối 2025", k.ACCENT, 24, 30).to_edge(UP, buff=0.4)
    sc.run(FadeOut(cost), FadeIn(tag), run_time=0.5)
    road = Line(LEFT * 6.2, RIGHT * 6.2, color=k.MUTED, stroke_width=5).shift(UP * 1.8)
    sc.run(Create(road), run_time=0.6)
    small = [(-5.0, "ăn sáng gì"), (-1.0, "mặc áo gì"), (4.4, "chiều đi đâu")]
    big = [(-3.2, "chọn trường nào"), (1.8, "cưới ai")]
    for x, t in small:
        br = Line([x, 1.8, 0], [x + 0.5, 2.3, 0], color=k.MUTED, stroke_width=2)
        sc.run(Create(br), FadeIn(k.label(t, 22, k.MUTED).move_to([x + 0.4, 2.6, 0])), run_time=0.25)
    for x, t in big:
        br = Line([x, 1.8, 0], [x + 1.0, 0.8, 0], color=k.GOLD, stroke_width=7)
        sc.run(Create(br), FadeIn(k.label(t, 28, k.GOLD, bold=True).move_to([x + 1.3, 0.45, 0])), run_time=0.4)

    sc.wait_until(sc.at(2))
    words = ["Đáp", "án", "là", "42", "vì", "ta", "có", "x", "=", "6"]
    hot = {3, 7}
    toks = VGroup(*[card(k, w, k.GOLD if i in hot else k.MUTED, 26, 6, pad=0.3) for i, w in enumerate(words)])
    toks.arrange(RIGHT, buff=0.1).move_to(DOWN * 0.5)
    sc.run(FadeIn(toks, lag_ratio=0.08), run_time=0.9)

    sc.wait_until(sc.at(3))
    how = k.label("Chữ nào then chốt? Nhìn bảng xác suất:", 28, k.INK, bold=True).move_to(DOWN * 1.4)
    sc.run(FadeIn(how), run_time=0.4)

    def mini(vals, color):
        bars = VGroup(*[Rectangle(width=0.32, height=max(0.05, 1.4 * v), stroke_width=0).set_fill(color, 0.9)
                        for v in vals]).arrange(RIGHT, buff=0.08, aligned_edge=DOWN)
        return bars

    sc.wait_until(sc.at(5))
    sure = mini([0.95, 0.03, 0.02, 0.0], k.MUTED).move_to(LEFT * 3.4 + DOWN * 2.8)
    sure_t = k.label("95%: chắc rồi, khỏi lo", 24, k.MUTED).next_to(sure, RIGHT, buff=0.3)
    sc.run(GrowFromEdge(sure, DOWN), FadeIn(sure_t), run_time=0.6)

    sc.wait_until(sc.at(6))
    flat = mini([0.27, 0.25, 0.25, 0.23], k.GOLD).move_to(RIGHT * 2.0 + DOWN * 3.1)
    flat_t = k.label("dàn đều: ngã rẽ!", 24, k.GOLD, bold=True).next_to(flat, RIGHT, buff=0.3)
    sc.run(GrowFromEdge(flat, DOWN), FadeIn(flat_t), run_time=0.6)

    sc.wait_until(sc.at(7))
    on = VGroup(*[k.label("bật CD", 20, k.GOLD, bold=True).next_to(toks[i], UP, buff=0.12) for i in hot])
    sc.run(FadeIn(on, shift=DOWN * 0.1), *[Indicate(toks[i], color=k.GOLD) for i in hot], run_time=0.8)


@_part(1)
def kv_cache(sc, s, k):
    head = k.label("Mẹo thứ hai của MTI", 36, k.INK, bold=True).to_edge(UP, buff=0.4)
    sc.run(FadeIn(head), run_time=0.4)

    sc.wait_until(sc.at(1))
    words = ["Chào", "mọi", "người,", "mình", "là", "Đại", "Kim"]
    row = VGroup(*[card(k, w, k.ACCENT, 26, 10, pad=0.32) for w in words]).arrange(RIGHT, buff=0.12).move_to(UP * 1.4)
    notes = VGroup(*[RoundedRectangle(corner_radius=0.05, width=c.width * 0.8, height=0.32, stroke_width=0)
                     .set_fill(k.GOLD, 0.6).next_to(c, DOWN, buff=0.1) for c in row])
    kv = k.label("KV cache: ghi nhớ phần đã tính cho từng chữ", 26, k.GOLD, bold=True).next_to(row, UP, buff=0.25)
    sc.run(FadeIn(row, lag_ratio=0.1), run_time=0.7)
    sc.run(FadeIn(notes, lag_ratio=0.1), FadeIn(kv), run_time=0.8)

    sc.wait_until(sc.at(2))
    book = RoundedRectangle(corner_radius=0.08, width=2.2, height=1.6, color=k.INK, stroke_width=3).set_fill("#1b2030", 1)
    blines = VGroup(*[Line(LEFT * 0.7, RIGHT * 0.5, color=k.MUTED, stroke_width=3) for _ in range(5)]).arrange(DOWN, buff=0.18)
    blines.move_to(book).shift(LEFT * 0.2)
    margin = VGroup(*[Line(LEFT * 0.12, RIGHT * 0.12, color=k.GOLD, stroke_width=4) for _ in range(4)]).arrange(DOWN, buff=0.22)
    margin.next_to(blines, RIGHT, buff=0.15)
    bk = VGroup(book, blines, margin).move_to(RIGHT * 5.4 + UP * 1.0)
    bt = k.label("như ghi chú bên lề", 22, k.GOLD).next_to(bk, DOWN, buff=0.15)
    sc.run(FadeIn(bk), FadeIn(bt), run_time=0.6)

    sc.wait_until(sc.at(3))
    words2 = ["Chào", "mọi", "người,", "mình", "là", "Tiểu", "Kim"]
    row2 = VGroup(*[card(k, w, k.GREEN if i < 5 else k.PINK, 26, 10, pad=0.32) for i, w in enumerate(words2)])
    row2.arrange(RIGHT, buff=0.12).move_to(DOWN * 0.6)
    sc.run(FadeIn(row2, lag_ratio=0.1), run_time=0.7)
    reuse = VGroup(*[Arrow(notes[i].get_bottom(), row2[i].get_top(), buff=0.05, color=k.GREEN, stroke_width=3,
                           max_tip_length_to_length_ratio=0.25) for i in range(5)])
    rt = k.label("giống hệt phần đầu: dùng lại", 24, k.GREEN, bold=True).next_to(row2[:5], DOWN, buff=0.2)
    sc.run(*[GrowArrow(a) for a in reuse], FadeIn(rt), run_time=0.7)

    sc.wait_until(sc.at(4))
    lost = k.label("từ chữ khác đi trở về sau: phải tính lại", 24, k.PINK, bold=True).next_to(row2[5:], DOWN, buff=0.2)
    lost.align_to(row2[5], LEFT).shift(DOWN * 0.5)
    sc.run(FadeIn(lost), Indicate(row2[5:], color=k.PINK), run_time=0.7)

    sc.wait_until(sc.at(5))
    sc.run(*[FadeOut(m) for m in (row2, reuse, rt, lost, bk, bt)], run_time=0.4)
    q = VGroup(*[card(k, w, k.ACCENT, 24, 10, pad=0.3) for w in ["Câu", "hỏi", "…", "của", "bạn"]]).arrange(RIGHT, buff=0.1)
    tail = VGroup(card(k, "Output", k.PINK, 26, 10, pad=0.32), card(k, "Error", k.PINK, 26, 10, pad=0.32)).arrange(RIGHT, buff=0.1)
    line2 = VGroup(q, tail).arrange(RIGHT, buff=0.2).move_to(DOWN * 0.6)
    sc.run(FadeIn(q), run_time=0.4)
    sc.run(FadeIn(tail, shift=LEFT * 0.4), run_time=0.6)
    mean = k.label("“trả lời sai đi”", 24, k.PINK).next_to(tail, UP, buff=0.15)
    sc.run(FadeIn(mean), run_time=0.3)

    sc.wait_until(sc.at(6))
    a = k.label("dùng lại ghi chú", 24, k.GREEN, bold=True).next_to(q, DOWN, buff=0.25)
    b = k.label("chỉ tính 2 chữ", 24, k.PINK, bold=True).next_to(tail, DOWN, buff=0.25)
    sc.run(FadeIn(a), FadeIn(b), run_time=0.5)

    sc.wait_until(sc.at(7))
    acc = VGroup(Text("62%", font=k.MONO, font_size=44, color=k.MUTED), Text("→", font=k.MONO, font_size=44, color=k.INK),
                 Text("72%", font=k.MONO, font_size=54, color=k.GOLD, weight="BOLD")).arrange(RIGHT, buff=0.3)
    acc.move_to(DOWN * 2.4 + LEFT * 2.8)
    sc.run(FadeIn(acc, lag_ratio=0.3), run_time=0.8)

    sc.wait_until(sc.at(8))
    junk = VGroup(Text("Monkey", font=k.MONO, font_size=28, color=k.MUTED), Text("Apple", font=k.MONO, font_size=28, color=k.MUTED))
    junk.arrange(RIGHT, buff=0.4).move_to(DOWN * 2.4 + RIGHT * 3.4)
    sc.run(FadeIn(junk), run_time=0.4)
    sc.run(Create(cross(k, junk)), run_time=0.4)


@_part(1)
def part1_recap(sc, s, k):
    head = k.label("Tóm lại phần một", 40, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.6)

    sc.wait_until(sc.at(1))
    good = _model(k, "bản tốt", k.ACCENT, w=2.6, h=1.0)
    minus = k.label("−", 60, k.INK, bold=True)
    bad = _model(k, "bản sai", k.GREEN, w=2.6, h=1.0)
    eq = k.label("=", 60, k.INK, bold=True)
    res = card(k, "khử lỗi chung", k.GOLD, 32, 16)
    core = VGroup(good, minus, bad, eq, res).arrange(RIGHT, buff=0.35).move_to(UP * 1.2)
    sc.run(FadeIn(core, lag_ratio=0.25), run_time=1.2)

    sc.wait_until(sc.at(2))
    srcs = VGroup(*[card(k, t, k.GREEN, 24, 16, bold=False) for t in
                    ("mô hình nhỏ", "tầng nông", "ảnh nhiễu", "câu “làm ngu”")]).arrange(RIGHT, buff=0.3)
    srcs.move_to(DOWN * 0.8)
    st = k.label("bản sai lấy từ đâu?", 24, k.GREEN).next_to(srcs, UP, buff=0.2)
    links = VGroup(*[Line(bad.get_bottom(), c.get_top(), color=k.GREEN, stroke_width=2).set_opacity(0.5) for c in srcs])
    sc.run(FadeIn(st), Create(links), FadeIn(srcs, lag_ratio=0.2), run_time=1.0)

    sc.wait_until(sc.at(3))
    joke = card(k, "không cần huấn luyện lại · bí ý tưởng thì cứ chạy thử", k.GOLD, 26, 60).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(joke, shift=UP * 0.2), run_time=0.6)


# ── PHẦN 2: SINH RỒI KIỂM ────────────────────────────────────
def _chart(k, w=5.0, h=3.0, xlabel="", ylabel=""):
    """Trục toạ độ đơn giản: trả (nhóm trục, hàm đổi (x 0..1, y 0..1) sang điểm)."""
    x_axis = Line(ORIGIN, RIGHT * w, color=k.MUTED, stroke_width=3)
    y_axis = Line(ORIGIN, UP * h, color=k.MUTED, stroke_width=3)
    group = VGroup(x_axis, y_axis)
    if xlabel:
        group.add(k.label(xlabel, 22, k.MUTED).next_to(x_axis, DOWN, buff=0.15))
    if ylabel:
        group.add(k.label(ylabel, 22, k.MUTED).next_to(y_axis, UP, buff=0.12))

    def at(x, y):
        return x_axis.get_start() + RIGHT * (w * x) + UP * (h * y)
    return group, at


def _polyline(k, at, pts, color, width=5):
    p = [at(x, y) for x, y in pts]
    return VGroup(*[Line(p[i], p[i + 1], color=color, stroke_width=width) for i in range(len(p) - 1)])


@_part(2)
def gen_verify(sc, s, k):
    head = k.label("Phần 2 · Sinh rồi kiểm", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.8)

    sc.wait_until(sc.at(1))
    gen = card(k, "Sinh", k.ACCENT, 32, 10, min_w=2.2)
    check_box = card(k, "“Kiểm tra lại đi”", k.GOLD, 30, 20)
    regen = card(k, "Viết tiếp", k.GREEN, 32, 10, min_w=2.2)
    VGroup(gen, check_box, regen).arrange(RIGHT, buff=1.0).move_to(UP * 0.4)
    sc.run(FadeIn(gen), run_time=0.4)

    sc.wait_until(sc.at(2))
    sc.run(GrowArrow(arrow(k, gen, check_box)), FadeIn(check_box), run_time=0.6)
    sc.run(GrowArrow(arrow(k, check_box, regen)), FadeIn(regen), run_time=0.6)
    gear = Text("</>", font=k.MONO, font_size=40, color=k.GOLD, weight="BOLD").next_to(check_box, DOWN, buff=0.45)
    auto = k.label("chương trình tự chèn", 24, k.GOLD).next_to(gear, DOWN, buff=0.15)
    sc.run(FadeIn(gear), FadeIn(auto), run_time=0.5)

    sc.wait_until(sc.at(3))
    notes = VGroup(k.label("không dính gì tới câu hỏi", 26, k.MUTED), k.label("không có người gõ", 26, k.MUTED))
    notes.arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.9)
    verdict = k.label("vẫn tính là tự sửa", 30, k.GREEN, bold=True).next_to(notes, DOWN, buff=0.2)
    sc.run(FadeIn(notes), run_time=0.5)
    sc.run(FadeIn(verdict, shift=UP * 0.2), run_time=0.5)


@_part(2)
def two_intuitions(sc, s, k):
    q = k.label("Vì sao có thể có ích?", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(q), run_time=0.7)
    sc.wait_until(sc.at(1))
    left_h = k.label("1. Chê thì dễ hơn làm", 30, k.ACCENT, bold=True).move_to(LEFT * 3.5 + UP * 2.0)
    right_h = k.label("2. Viết rồi không xoá được", 30, k.PINK, bold=True).move_to(RIGHT * 3.4 + UP * 2.0)
    divider = Line(UP * 2.5, DOWN * 3.3, color=k.MUTED, stroke_width=2).set_opacity(0.4)
    sc.run(Create(divider), run_time=0.4)

    sc.wait_until(sc.at(2))
    sc.run(FadeIn(left_h), run_time=0.5)
    sc.wait_until(sc.at(3))
    book = RoundedRectangle(corner_radius=0.08, width=2.2, height=2.6, color=k.GOLD, stroke_width=4).set_fill(k.GOLD, 0.15)
    book.move_to(LEFT * 3.5 + UP * 0.1)
    spine = Line(book.get_corner(UP + LEFT) + RIGHT * 0.2, book.get_corner(DOWN + LEFT) + RIGHT * 0.2, color=k.GOLD, stroke_width=3)
    title = k.label("Kim Dung", 28, k.GOLD, bold=True).move_to(book).shift(RIGHT * 0.1)
    cant = k.label("không viết nổi", 26, k.PINK).next_to(book, DOWN, buff=0.25)
    can = k.label("nhưng đọc thì biết là hay", 26, k.GREEN, bold=True).next_to(cant, DOWN, buff=0.12)
    sc.run(FadeIn(book), Create(spine), Write(title), run_time=0.7)
    sc.run(FadeIn(cant), run_time=0.4)
    sc.run(FadeIn(can), run_time=0.4)

    sc.wait_until(sc.at(4))
    sc.run(FadeIn(right_h), run_time=0.5)
    sc.wait_until(sc.at(5))
    toks = VGroup(card(k, "sai", k.PINK, 26, 6, pad=0.3), *[card(k, "...", k.PINK, 26, 6, pad=0.3) for _ in range(4)])
    toks.arrange(RIGHT, buff=0.12).move_to(RIGHT * 3.4 + UP * 0.9)
    sc.run(FadeIn(toks[0], scale=0.8), run_time=0.4)
    sc.run(FadeIn(toks[1:], lag_ratio=0.3), run_time=0.9)
    stuck = k.label("cãi cố tới hết câu", 26, k.PINK).next_to(toks, DOWN, buff=0.2)
    sc.run(FadeIn(stuck), run_time=0.4)

    sc.wait_until(sc.at(6))
    door = card(k, "“Kiểm tra lại đi”", k.GOLD, 28, 20).move_to(RIGHT * 3.4 + DOWN * 0.6)
    exit_t = k.label("mở một lối thoát", 24, k.GOLD).next_to(door, DOWN, buff=0.15)
    sc.run(FadeIn(door, shift=UP * 0.2), FadeIn(exit_t), run_time=0.6)

    sc.wait_until(sc.at(7))
    only = k.label("mô hình chỉ biết nối chữ", 24, k.MUTED).move_to(RIGHT * 3.4 + DOWN * 2.0)
    sc.run(FadeIn(only), run_time=0.4)
    sc.wait_until(sc.at(8))
    web = k.label("trên mạng: sau câu đó thường là đoạn sửa sai", 24, k.GREEN, bold=True, width_chars=26)
    web.next_to(only, DOWN, buff=0.2)
    sc.run(FadeIn(web), run_time=0.5)


@_part(2)
def internal_external(sc, s, k):
    head = k.label("Thực nghiệm nói gì?", 38, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    tag = card(k, "2 paper · cuối 2025 · rất nhiều mô hình", k.MUTED, 24, 60).next_to(head, DOWN, buff=0.2)
    sc.run(FadeIn(tag), run_time=0.5)

    def panel(values, color, title, x):
        base = Line(LEFT * 2.6, RIGHT * 2.6, color=k.MUTED, stroke_width=3).move_to([x, -0.5, 0])
        bars = VGroup()
        for i, v in enumerate(values):
            bar = Rectangle(width=0.36, height=abs(v), stroke_width=0).set_fill(color if v >= 0 else k.PINK, 0.85)
            bar.move_to(base.get_left() + RIGHT * (0.45 + i * 0.52), aligned_edge=DOWN if v >= 0 else UP)
            bar.grow_from = DOWN if v >= 0 else UP
            bars.add(bar)
        t = k.label(title, 28, color, bold=True).next_to(base, UP, buff=1.8)
        cap = k.label("cột: điểm thay đổi sau khi sửa", 18, k.MUTED).next_to(t, DOWN, buff=0.1)
        return VGroup(base, bars, t, cap)

    sc.wait_until(sc.at(2))
    inner = panel([0.9, -0.7, 0.4, -1.1, 1.2, 0.2, -0.5, 0.7, -0.9], k.ACCENT, "tự nghĩ lại", -3.4)
    sc.run(Create(inner[0]), FadeIn(inner[2]), FadeIn(inner[3]), run_time=0.4)
    sc.run(*[GrowFromEdge(b, b.grow_from) for b in inner[1]], run_time=1.0)
    shaky = k.label("lúc được lúc không", 26, k.PINK, bold=True).next_to(inner[0], DOWN, buff=1.3)
    sc.run(FadeIn(shaky), run_time=0.4)

    sc.wait_until(sc.at(3))
    outer = panel([1.0, 1.3, 0.8, 1.5, 0.2, 1.1, 0.9, 1.4, 0.6], k.GREEN, "phản hồi bên ngoài", 3.4)
    sc.run(Create(outer[0]), FadeIn(outer[2]), FadeIn(outer[3]), run_time=0.4)
    sc.run(*[GrowFromEdge(b, b.grow_from) for b in outer[1]], run_time=1.0)
    steady = k.label("chạy code, tiêu chí chấm: ổn định", 26, k.GREEN, bold=True).next_to(outer[0], DOWN, buff=1.3)
    sc.run(FadeIn(steady), run_time=0.4)

    sc.wait_until(sc.at(4))
    why = card(k, "bên ngoài mang thông tin mới · tự nghĩ lại vẫn là cái đầu cũ", k.GOLD, 24, 70).to_edge(DOWN, buff=0.2)
    sc.run(FadeIn(why, shift=UP * 0.2), run_time=0.6)


def _cell(k, text, sub, color, w=4.6, h=1.6):
    box = RoundedRectangle(corner_radius=0.2, width=w, height=h, color=color, stroke_width=4).set_fill(color, 0.12)
    t = k.label(text, 34, k.INK, bold=True)
    st = k.label(sub, 24, color)
    VGroup(t, st).arrange(DOWN, buff=0.15).move_to(box)
    return VGroup(box, t, st)


@_part(2)
def four_cells(sc, s, k):
    head = k.label("Vì sao tự kiểm lại chập chờn?", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.7)
    names = [("Đúng, vẫn Đúng", "chỉ tốn công", k.MUTED), ("Sai, thành Đúng", "mong nhất", k.GREEN),
             ("Sai, vẫn Sai", "chỉ tốn công", k.MUTED), ("Đúng, thành Sai", "nghĩ nhiều quá hoá sai", k.PINK)]
    cells = VGroup(*[_cell(k, a, b, c) for a, b, c in names]).arrange_in_grid(2, 2, buff=0.35).move_to(DOWN * 0.5)
    sc.wait_until(sc.at(1))
    four = k.label("bốn trường hợp", 28, k.MUTED).next_to(head, DOWN, buff=0.15)
    sc.run(FadeIn(four), *[Create(c[0]) for c in cells], run_time=0.7)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(cells[1][1:], scale=0.9), run_time=0.6)
    sc.wait_until(sc.at(3))
    sc.run(FadeIn(cells[0][1:], scale=0.9), FadeIn(cells[2][1:], scale=0.9), run_time=0.6)
    sc.wait_until(sc.at(4))
    sc.run(FadeIn(cells[3][1:], scale=0.9), run_time=0.6)
    sc.run(Indicate(cells[3], color=k.PINK, scale_factor=1.05), run_time=0.8)


def _dot_grid(k, n_good, color_good, color_bad, r=0.11, gap=0.33):
    dots = VGroup(*[Dot(radius=r, color=color_good if i < n_good else color_bad) for i in range(100)])
    dots.arrange_in_grid(10, 10, buff=gap - 2 * r)
    return dots


@_part(2)
def cl_cs(sc, s, k):
    head = k.label("Hai con số: CL và CS", 38, k.INK, bold=True).to_edge(UP, buff=0.4)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    cl = VGroup(k.label("CL", 40, k.GREEN, bold=True), k.label("đang đúng, giữ được đúng", 24, k.GREEN))
    cl.arrange(DOWN, buff=0.1).move_to(RIGHT * 4.0 + UP * 1.8)
    sc.run(FadeIn(cl, shift=DOWN * 0.2), run_time=0.6)
    sc.wait_until(sc.at(2))
    cs = VGroup(k.label("CS", 40, k.GOLD, bold=True), k.label("đang sai, sửa được thành đúng", 24, k.GOLD))
    cs.arrange(DOWN, buff=0.1).next_to(cl, DOWN, buff=0.4)
    sc.run(FadeIn(cs, shift=DOWN * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    grid = _dot_grid(k, 60, k.GREEN, k.PINK).move_to(LEFT * 3.0 + DOWN * 0.4)
    cap = k.label("100 câu: 60 đúng, 40 sai", 28, k.INK, bold=True).next_to(grid, UP, buff=0.3)
    sc.run(FadeIn(grid, lag_ratio=0.01), FadeIn(cap), run_time=1.2)

    sc.wait_until(sc.at(4))
    lost = [grid[i] for i in (5, 17, 28, 39, 44, 52)]
    t1 = k.label("CL 90%: giữ 54, làm hỏng 6", 26, k.GREEN, bold=True).move_to(RIGHT * 4.0 + DOWN * 0.7)
    sc.run(*[d.animate.set_color(k.PINK).scale(1.3) for d in lost], FadeIn(t1), run_time=0.8)

    sc.wait_until(sc.at(5))
    fixed = [grid[i] for i in (61, 68, 73, 79, 84, 90, 95, 99)]
    t2 = k.label("CS 20%: sửa được 8", 26, k.GOLD, bold=True).next_to(t1, DOWN, buff=0.25)
    sc.run(*[d.animate.set_color(k.GOLD).scale(1.3) for d in fixed], FadeIn(t2), run_time=0.8)

    sc.wait_until(sc.at(6))
    total = VGroup(Text("54 + 8 = 62", font=k.MONO, font_size=34, color=k.INK),
                   k.label("chỉ nhích hơn 60 một chút", 24, k.MUTED)).arrange(DOWN, buff=0.1)
    total.next_to(t2, DOWN, buff=0.35)
    sc.run(Write(total[0]), FadeIn(total[1]), run_time=0.8)

    sc.wait_until(sc.at(7))
    worse = VGroup(Text("CL 80%: 48 + 8 = 56", font=k.MONO, font_size=28, color=k.PINK),
                   k.label("tự sửa xong còn tệ hơn lúc đầu", 24, k.PINK, bold=True)).arrange(DOWN, buff=0.1)
    worse.move_to(total)
    sc.run(FadeOut(total), FadeIn(worse), run_time=0.6)


@_part(2)
def personalities(sc, s, k):
    head = k.label("Phần lớn mô hình: cứng đầu", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)
    def meter(name, value, color, y):
        t = k.label(name, 30, color, bold=True).move_to([-4.6, y, 0])
        track = RoundedRectangle(corner_radius=0.15, width=6.0, height=0.45, color=k.MUTED, stroke_width=2).move_to([0.6, y, 0])
        fill = RoundedRectangle(corner_radius=0.15, width=6.0 * value, height=0.45, stroke_width=0).set_fill(color, 0.9)
        fill.move_to(track.get_left(), aligned_edge=LEFT)
        num = k.label(f"{round(value * 100)}%", 28, color, bold=True).next_to(track, RIGHT, buff=0.3)
        return VGroup(t, track, fill, num)
    cl = meter("CL", 0.9, k.GREEN, 1.2)
    cs = meter("CS", 0.15, k.GOLD, 0.0)
    sc.run(FadeIn(cl[:2]), GrowFromEdge(cl[2], LEFT), FadeIn(cl[3]), run_time=0.8)
    sc.run(FadeIn(cs[:2]), GrowFromEdge(cs[2], LEFT), FadeIn(cs[3]), run_time=0.8)
    a = k.label("ít khi làm hỏng câu đúng", 24, k.GREEN).next_to(cl[1], DOWN, buff=0.08)
    b = k.label("ít khi sửa được câu sai", 24, k.GOLD).next_to(cs[1], DOWN, buff=0.08)
    sc.run(FadeIn(a), FadeIn(b), run_time=0.5)

    sc.wait_until(sc.at(1))
    easy = _student(k, k.ACCENT, "dễ nghe góp ý").move_to(LEFT * 5.6 + DOWN * 2.3)
    hard = _student(k, k.ORANGE, "cứng đầu").move_to(RIGHT * 0.6 + DOWN * 2.3)
    pull = VGroup(easy, hard)
    p1 = k.label("sửa được nhiều lỗi, nhưng dễ lung lay cả chỗ đúng", 22, k.ACCENT, width_chars=24)
    p1.next_to(easy, RIGHT, buff=0.25)
    p2 = k.label("giữ chắc chỗ đúng, nhưng ít khi sửa", 22, k.ORANGE, width_chars=22).next_to(hard, RIGHT, buff=0.25)
    sc.run(FadeIn(pull[0]), FadeIn(p1), run_time=0.6)
    sc.run(FadeIn(pull[1]), FadeIn(p2), run_time=0.6)


@_part(2)
def three_prompts(sc, s, k):
    head = k.label("Câu chèn vào đổi cả tính cách", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)

    def row(text, color, cl_up):
        b = bubble(k, text, color, 26, 28)
        cl = VGroup(k.label("CL", 28, k.GREEN, bold=True), k.label("tăng" if cl_up else "giảm", 26, k.GREEN))
        cs = VGroup(k.label("CS", 28, k.GOLD, bold=True), k.label("giảm" if cl_up else "tăng", 26, k.GOLD))
        cl.arrange(RIGHT, buff=0.1)
        cs.arrange(RIGHT, buff=0.1)
        return VGroup(b, VGroup(cl, cs).arrange(RIGHT, buff=0.5))

    praise = row("Chắc bạn đúng rồi.", k.GREEN, True)
    doubt = row("Bạn chắc chưa?", k.PINK, False)
    for r, y in ((praise, 1.0), (doubt, -0.8)):
        r[0].move_to([-4.0, y, 0])
        r[1].move_to([0.6, y, 0])
    pt = k.label("tự tin hơn, cố chấp hơn", 24, k.GREEN).next_to(praise[1], RIGHT, buff=0.4)
    dt = k.label("dễ đổi ý hơn", 24, k.PINK).next_to(doubt[1], RIGHT, buff=0.4)

    sc.wait_until(sc.at(1))
    sc.run(FadeIn(praise[0], shift=RIGHT * 0.2), run_time=0.5)
    sc.run(FadeIn(praise[1]), FadeIn(pt), run_time=0.5)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(doubt[0], shift=RIGHT * 0.2), run_time=0.5)
    sc.run(FadeIn(doubt[1]), FadeIn(dt), run_time=0.5)

    sc.wait_until(sc.at(3))
    straw = k.label("chỉ một câu do chương trình tự chèn", 28, k.GOLD, bold=True).to_edge(DOWN, buff=1.0)
    sc.run(FadeIn(straw), run_time=0.5)
    sc.wait_until(sc.at(4))
    mix = k.label("mỗi paper chèn một câu khác, nên kết luận lẫn lộn", 26, k.MUTED).next_to(straw, DOWN, buff=0.2)
    sc.run(FadeIn(mix), run_time=0.5)


@_part(2)
def majority_vote(sc, s, k):
    head = k.label("Còn chuyện chi phí", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)
    sc.wait_until(sc.at(1))
    title = k.label("Bỏ phiếu đa số: giải nhiều lần, chọn đáp án hay gặp nhất", 30, k.ACCENT, bold=True)
    title.next_to(head, DOWN, buff=0.3)
    sc.run(FadeIn(title), run_time=0.6)

    sc.wait_until(sc.at(2))
    answers = ["42", "42", "40", "42", "38"]
    cards = VGroup(*[card(k, a, k.ACCENT if a == "42" else k.PINK, 36, 4, pad=0.5) for a in answers])
    cards.arrange(RIGHT, buff=0.35).move_to(UP * 0.3)
    nums = VGroup(*[k.label(f"lần {i + 1}", 20, k.MUTED).next_to(c, UP, buff=0.12) for i, c in enumerate(cards)])
    for c, n in zip(cards, nums):
        sc.run(FadeIn(c, shift=DOWN * 0.2), FadeIn(n), run_time=0.3)
    tally = VGroup(
        VGroup(Text("42", font=k.MONO, font_size=34, color=k.ACCENT), k.label("× 3", 30, k.ACCENT, bold=True)).arrange(RIGHT, buff=0.2),
        VGroup(Text("40", font=k.MONO, font_size=34, color=k.PINK), k.label("× 1", 30, k.PINK)).arrange(RIGHT, buff=0.2),
        VGroup(Text("38", font=k.MONO, font_size=34, color=k.PINK), k.label("× 1", 30, k.PINK)).arrange(RIGHT, buff=0.2),
    ).arrange(RIGHT, buff=0.8).move_to(DOWN * 1.3)
    sc.run(FadeIn(tally, lag_ratio=0.3), run_time=0.8)
    win = SurroundingRectangle(tally[0], color=k.GOLD, buff=0.15, corner_radius=0.1)
    pick = k.label("chọn 42", 30, k.GOLD, bold=True).next_to(win, DOWN, buff=0.2)
    sc.run(Create(win), FadeIn(pick), run_time=0.6)

    sc.wait_until(sc.at(3))
    why = card(k, "sai thì mỗi lần một kiểu · đúng thì hay lặp lại", k.GOLD, 26, 60).to_edge(DOWN, buff=0.3)
    sc.run(Indicate(cards[2], color=k.PINK), Indicate(cards[4], color=k.PINK), run_time=0.6)
    sc.run(FadeIn(why, shift=UP * 0.2), run_time=0.6)


@_part(2)
def vote_vs_verify(sc, s, k):
    head = k.label("Cùng một lượng tính toán", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)

    sc.wait_until(sc.at(1))
    one_check = card(k, "kiểm 1 lời giải", k.PINK, 26, 16)
    eq = k.label("≈", 44, k.INK, bold=True)
    sols = VGroup(*[card(k, "giải", k.ACCENT, 24, 6, pad=0.3) for _ in range(3)]).arrange(RIGHT, buff=0.1)
    scale = VGroup(one_check, eq, sols).arrange(RIGHT, buff=0.4).next_to(head, DOWN, buff=0.4)
    sc.run(FadeIn(one_check), run_time=0.4)
    sc.run(FadeIn(eq), FadeIn(sols, lag_ratio=0.3), run_time=0.7)

    sc.wait_until(sc.at(2))
    axes, at = _chart(k, 6.0, 3.0, "lượng tính toán", "điểm")
    axes.move_to(LEFT * 2.4 + DOWN * 1.4)
    vote = _polyline(k, at, [(0, 0.2), (0.15, 0.5), (0.3, 0.63), (0.55, 0.7), (1, 0.71)], k.ACCENT)
    ver = _polyline(k, at, [(0.12, 0.12), (0.3, 0.33), (0.55, 0.55), (0.8, 0.68), (1, 0.76)], k.PINK)
    tv = k.label("bỏ phiếu", 24, k.ACCENT, bold=True).next_to(vote[-1].get_end(), RIGHT, buff=0.15).shift(DOWN * 0.2)
    tr = k.label("+ kiểm tra", 24, k.PINK, bold=True).next_to(ver[-1].get_end(), RIGHT, buff=0.15).shift(UP * 0.2)
    sc.run(FadeIn(axes), run_time=0.4)
    sc.run(Create(vote), FadeIn(tv), run_time=0.8)
    sc.run(Create(ver), FadeIn(tr), run_time=0.8)
    budget = Rectangle(width=6.0 * 0.5, height=3.0, stroke_width=0).set_fill(k.ACCENT, 0.1).move_to(at(0.25, 0.5))
    bt = k.label("ngân sách có hạn: bỏ phiếu thắng", 22, k.ACCENT, bold=True, width_chars=18).move_to(at(0.25, 0.92))
    sc.run(FadeIn(budget), FadeIn(bt), run_time=0.6)

    sc.wait_until(sc.at(3))
    cost = card(k, "+3,8% cần hơn 100 lần tính toán", k.PINK, 24, 18).move_to(RIGHT * 4.6 + DOWN * 0.6)
    sc.run(FadeIn(cost, shift=LEFT * 0.2), run_time=0.6)
    sc.wait_until(sc.at(4))
    lux = card(k, "kiểm tra = món xa xỉ", k.GOLD, 28, 24).next_to(cost, DOWN, buff=0.3)
    sc.run(GrowFromCenter(lux), run_time=0.6)
    sc.wait_until(sc.at(5))
    rule = k.label("quy trình mới: nhớ so với bỏ phiếu đa số, cùng lượng tính toán", 26, k.GOLD, bold=True)
    rule.to_edge(DOWN, buff=0.2)
    sc.run(FadeIn(rule, shift=UP * 0.2), run_time=0.6)


@_part(2)
def part2_recap(sc, s, k):
    head = k.label("Tóm lại phần hai", 40, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    pts = VGroup(card(k, "tự kiểm: chập chờn", k.PINK, 26, 20), card(k, "phản hồi ngoài: chắc ăn", k.GREEN, 26, 24),
                 card(k, "kiểm tra: tốn kém", k.GOLD, 26, 20)).arrange(RIGHT, buff=0.3).move_to(UP * 1.6)
    sc.run(FadeIn(pts, lag_ratio=0.4), run_time=1.2)

    sc.wait_until(sc.at(2))
    always = VGroup(*[VGroup(card(k, a, k.GREEN if a == "đúng" else k.PINK, 24, 6, pad=0.3),
                             k.label("rồi vẫn nghĩ thêm", 24, k.MUTED)).arrange(RIGHT, buff=0.2) for a in ("đúng", "sai", "đúng")])
    always.arrange(DOWN, buff=0.2).move_to(LEFT * 3.4 + DOWN * 1.2)
    sc.run(FadeIn(always, lag_ratio=0.3), run_time=0.8)
    waste = k.label("đúng rồi vẫn bị ép nghĩ thêm", 24, k.PINK).next_to(always, DOWN, buff=0.2)
    sc.run(FadeIn(waste), run_time=0.4)

    sc.wait_until(sc.at(3))
    model = _model(k, "LLM", w=2.4, h=1.2).move_to(RIGHT * 3.4 + DOWN * 1.2)
    loop = CurvedArrow(model.get_corner(UP + RIGHT) + LEFT * 0.3, model.get_corner(UP + LEFT) + RIGHT * 0.3,
                       angle=2.2, color=k.GOLD, stroke_width=5)
    inner = k.label("tự biết lúc nào cần sửa", 26, k.GOLD, bold=True).next_to(loop, UP, buff=0.12)
    sc.run(GrowFromCenter(model), Create(loop), FadeIn(inner), run_time=0.9)

    sc.wait_until(sc.at(4))
    nxt = card(k, "Phần 3 · Huấn luyện", k.GOLD, 30, 30).to_edge(DOWN, buff=0.3).shift(RIGHT * 3.4)
    sc.run(FadeIn(nxt, shift=UP * 0.2), run_time=0.5)


# ── PHẦN 3: HUẤN LUYỆN ĐỂ TỰ SỬA ─────────────────────────────
def _knob(k, color, r=0.42, angle=0.0):
    dial = Circle(radius=r, color=color, stroke_width=4).set_fill(color, 0.15)
    tick = Line(ORIGIN, UP * r * 0.85, color=k.INK, stroke_width=5).rotate(angle, about_point=ORIGIN)
    return VGroup(dial, tick.shift(dial.get_center()))


@_part(3)
def knobs(sc, s, k):
    head = k.label("Huấn luyện là gì?", 38, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.6)

    sc.wait_until(sc.at(1))
    rnd = random.Random(9)
    grid = VGroup(*[_knob(k, k.PURPLE, angle=rnd.uniform(-2.5, 2.5)) for _ in range(24)]).arrange_in_grid(4, 6, buff=0.3)
    grid.move_to(LEFT * 2.6 + DOWN * 0.4)
    frame = RoundedRectangle(corner_radius=0.3, width=grid.width + 0.8, height=grid.height + 0.8, color=k.PURPLE,
                             stroke_width=3).move_to(grid)
    lab = k.label("tham số: hàng tỉ núm vặn", 28, k.PURPLE, bold=True).next_to(frame, UP, buff=0.2)
    sc.run(Create(frame), FadeIn(grid, lag_ratio=0.03), FadeIn(lab), run_time=1.2)

    sc.wait_until(sc.at(2))
    ans = bubble(k, "câu trả lời", k.ACCENT, 14, 28).move_to(RIGHT * 4.4 + UP * 0.8)
    a = Arrow(frame.get_right(), ans.get_left(), buff=0.15, color=k.MUTED, stroke_width=4)
    sc.run(GrowArrow(a), FadeIn(ans), run_time=0.6)
    sub = k.label("vị trí các núm quyết định câu trả lời", 22, k.MUTED, width_chars=18).next_to(ans, DOWN, buff=0.2)
    sc.run(FadeIn(sub), run_time=0.4)

    sc.wait_until(sc.at(3))
    loop = VGroup(card(k, "làm bài", k.ACCENT, 24, 10), card(k, "xem kết quả", k.GOLD, 24, 12),
                  card(k, "vặn nhẹ núm", k.GREEN, 24, 12)).arrange(DOWN, buff=0.35).move_to(RIGHT * 4.4 + DOWN * 1.6)
    sc.run(FadeOut(sub), FadeIn(loop, lag_ratio=0.3), run_time=0.8)
    sc.run(*[Rotate(kn[1], angle=rnd.uniform(-0.6, 0.6), about_point=kn[0].get_center()) for kn in grid], run_time=1.0)

    sc.wait_until(sc.at(4))
    p12 = k.label("phần 1, 2: không đụng núm nào", 26, k.MUTED).to_edge(DOWN, buff=0.3).shift(LEFT * 2.6)
    sc.run(FadeIn(p12), run_time=0.5)
    sc.wait_until(sc.at(5))
    p3 = k.label("phần 3: vặn núm", 28, k.GOLD, bold=True).move_to(p12)
    sc.run(Transform(p12, p3), *[Indicate(kn, color=k.GOLD, scale_factor=1.1) for kn in grid[::3]], run_time=0.8)


@_part(3)
def hillary(sc, s, k):
    naive = k.label("dạy thêm kiến thức thì tự sửa giỏi hơn?", 30, k.MUTED, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(naive), run_time=0.5)
    sc.wait_until(sc.at(1))
    sc.run(Create(cross(k, naive)), run_time=0.5)

    sc.wait_until(sc.at(2))
    q1 = bubble(k, "Kể tên một chính trị gia sinh ở New York.", k.ACCENT, 24, 26).move_to(LEFT * 3.4 + UP * 1.3)
    sc.run(FadeIn(q1, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(3))
    a1 = bubble(k, "Hillary Clinton.", k.PURPLE, 20, 30).next_to(q1, DOWN, buff=0.3).align_to(q1, RIGHT)
    sc.run(FadeIn(a1, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(4))
    sc.run(Create(cross(k, a1)), run_time=0.4)

    sc.wait_until(sc.at(5))
    q2 = bubble(k, "Hillary Clinton sinh ở đâu?", k.ACCENT, 24, 26).move_to(RIGHT * 3.4 + UP * 1.3)
    sc.run(FadeIn(q2, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(6))
    a2 = bubble(k, "Chicago.", k.PURPLE, 20, 30).next_to(q2, DOWN, buff=0.3).align_to(q2, RIGHT)
    tick = check(k, 0.45).next_to(a2, RIGHT, buff=0.2)
    sc.run(FadeIn(a2, shift=UP * 0.2), Create(tick), run_time=0.6)

    sc.wait_until(sc.at(7))
    mid = k.label("có kiến thức, mà lúc nói sai không giật mình", 30, k.GOLD, bold=True).move_to(DOWN * 1.2)
    sc.run(FadeIn(mid), run_time=0.5)
    sc.wait_until(sc.at(8))
    skill = VGroup(card(k, "biết", k.ACCENT, 30, 8), Text("≠", font=k.MONO, font_size=44, color=k.INK),
                   card(k, "tự sửa: kỹ năng riêng", k.GREEN, 30, 24)).arrange(RIGHT, buff=0.4).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(skill, shift=UP * 0.2), run_time=0.6)


def _token(k, text, color, size=26):
    return card(k, text, color, size, 24, pad=0.35)


@_part(3)
def new_errors(sc, s, k):
    head = k.label("Dạy thẳng cách sửa?", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)
    ex = VGroup(_token(k, "câu trả lời sai", k.PINK), _token(k, "bản sửa đúng", k.GREEN)).arrange(RIGHT, buff=1.0)
    ex.next_to(head, DOWN, buff=0.4)
    sc.run(FadeIn(ex[0]), run_time=0.4)
    ex_a = arrow(k, ex[0], ex[1])
    sc.run(GrowArrow(ex_a), FadeIn(ex[1]), run_time=0.6)

    sc.wait_until(sc.at(1))
    trap = k.label("Nhưng có một cái bẫy", 32, k.PINK, bold=True).move_to(head)
    sc.run(Transform(head, trap), FadeOut(ex), FadeOut(ex_a), run_time=0.5)

    sc.wait_until(sc.at(2))
    before = _model(k, "mô hình cũ", k.PURPLE, w=3.4, h=1.0).move_to(LEFT * 4.0 + UP * 0.6)
    err_g = _token(k, "lỗi màu xanh", k.ACCENT).next_to(before, DOWN, buff=0.5)
    taught = k.label("được dạy sửa", 24, k.ACCENT).next_to(err_g, DOWN, buff=0.15)
    sc.run(FadeIn(before), run_time=0.4)
    sc.run(FadeIn(err_g), FadeIn(taught), run_time=0.5)

    sc.wait_until(sc.at(3))
    after = _model(k, "mô hình mới", k.ORANGE, w=3.4, h=1.0).move_to(RIGHT * 3.6 + UP * 0.6)
    change = Arrow(before.get_right(), after.get_left(), buff=0.2, color=k.GOLD, stroke_width=5)
    ct = k.label("núm vặn đã đổi", 24, k.GOLD).next_to(change, UP, buff=0.15)
    sc.run(GrowArrow(change), FadeIn(ct), FadeIn(after), run_time=0.8)

    sc.wait_until(sc.at(4))
    err_r = _token(k, "lỗi màu đỏ", k.PINK).next_to(after, DOWN, buff=0.5)
    never = k.label("chưa từng được dạy sửa", 24, k.PINK).next_to(err_r, DOWN, buff=0.15)
    sc.run(FadeIn(err_r, scale=0.8), FadeIn(never), run_time=0.6)
    sc.wait_until(sc.at(5))
    exam = card(k, "như luyện kỹ đề cũ cho một đề thi đã đổi", k.MUTED, 26, 50).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(exam), run_time=0.5)


@_part(3)
def rlvr(sc, s, k):
    head = k.label("Học tăng cường · reinforcement learning (RL)", 30, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.8)

    sc.wait_until(sc.at(1))
    pet = k.label("như dạy thú cưng: không chỉ từng động tác, chỉ thưởng khi làm đúng", 26, k.MUTED)
    pet.next_to(head, DOWN, buff=0.25)
    sc.run(FadeIn(pet), run_time=0.5)

    sc.wait_until(sc.at(2))
    inp = _token(k, "bài toán", k.ACCENT).move_to(LEFT * 5.4 + UP * 0.4)
    think = VGroup(*[Line(LEFT * 0.0, RIGHT * w, color=k.MUTED, stroke_width=4) for w in (3.6, 3.0, 3.4, 2.4, 3.2)])
    think.arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(inp, RIGHT, buff=0.6)
    free = k.label("nghĩ gì cũng được, nháp dài bao nhiêu cũng được", 22, k.MUTED).next_to(think, UP, buff=0.15)
    sc.run(FadeIn(inp), run_time=0.3)
    sc.run(Create(think, lag_ratio=0.3), FadeIn(free), run_time=1.2)

    sc.wait_until(sc.at(3))
    ans = _token(k, "đáp án", k.PURPLE).next_to(think, RIGHT, buff=0.6)
    plus = k.label("đúng: +1", 32, k.GREEN, bold=True).next_to(ans, RIGHT, buff=0.4).shift(UP * 0.35)
    zero = k.label("sai: 0", 32, k.PINK, bold=True).next_to(ans, RIGHT, buff=0.4).shift(DOWN * 0.35)
    only = k.label("chỉ chấm đáp án cuối", 22, k.GOLD).next_to(ans, DOWN, buff=0.3)
    sc.run(FadeIn(ans), run_time=0.3)
    sc.run(FadeIn(plus), FadeIn(zero), FadeIn(only), run_time=0.6)

    sc.wait_until(sc.at(4))
    rnd = random.Random(3)
    ks = VGroup(*[_knob(k, k.PURPLE, r=0.3, angle=rnd.uniform(-2, 2)) for _ in range(6)]).arrange(RIGHT, buff=0.25)
    ks.move_to(DOWN * 1.8)
    kt = k.label("vặn núm để cách nghĩ được thưởng dễ xuất hiện hơn", 22, k.GREEN).next_to(ks, DOWN, buff=0.2)
    sc.run(FadeIn(ks), FadeIn(kt), run_time=0.5)
    sc.run(*[Rotate(kn[1], angle=0.7, about_point=kn[0].get_center()) for kn in ks], run_time=0.8)

    sc.wait_until(sc.at(5))
    name = card(k, "RLVR: phần thưởng kiểm chứng được (toán, code)", k.GOLD, 26, 60).to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(name, shift=UP * 0.2), run_time=0.6)


@_part(3)
def rl_aha(sc, s, k):
    head = k.label("Vì sao RL tránh được cái bẫy?", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)
    now = _model(k, "mô hình hiện tại", k.ORANGE, w=4.6, h=1.0).move_to(LEFT * 3.6 + UP * 1.5)
    err = _token(k, "lỗi nó đang mắc", k.PINK).next_to(now, DOWN, buff=0.4)
    grade = card(k, "được chấm ngay trên lỗi đó", k.GREEN, 24, 30).next_to(err, DOWN, buff=0.35)
    sc.run(FadeIn(now), FadeIn(err), run_time=0.6)
    sc.run(FadeIn(grade), run_time=0.5)

    sc.wait_until(sc.at(1))
    lines = VGroup(bubble(k, "Đáp án là 42.", k.PURPLE, 20, 26),
                   bubble(k, "Khoan, đúng không nhỉ?", k.GOLD, 24, 26),
                   bubble(k, "À, sai bước 2. Đáp án là 40.", k.GREEN, 26, 26)).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
    lines.move_to(RIGHT * 3.4 + UP * 0.6)
    nobody = k.label("không ai dạy", 24, k.MUTED).next_to(lines, UP, buff=0.2)
    sc.run(FadeIn(nobody), run_time=0.3)
    for b in lines:
        sc.run(FadeIn(b, shift=UP * 0.2), run_time=0.5)

    sc.wait_until(sc.at(2))
    bars = VGroup()
    for name, v, c in (("không kiểm lại", 0.35, k.MUTED), ("chịu kiểm lại", 0.8, k.GREEN)):
        bar = Rectangle(width=1.2, height=1.7 * v, stroke_width=0).set_fill(c, 0.9)
        bars.add(VGroup(bar, k.label(name, 22, c, bold=True).next_to(bar, DOWN, buff=0.12)))
    bars.arrange(RIGHT, buff=0.8, aligned_edge=DOWN).move_to(RIGHT * 2.8 + DOWN * 3.7, aligned_edge=DOWN)
    bl = k.label("phần thưởng", 22, k.MUTED).next_to(bars[1][0], UP, buff=0.12)
    sc.run(*[GrowFromEdge(b[0], DOWN) for b in bars], FadeIn(VGroup(*[b[1] for b in bars])), FadeIn(bl), run_time=0.8)
    kept = k.label("thói quen kiểm lại được giữ", 26, k.GREEN, bold=True).move_to(LEFT * 3.6 + DOWN * 1.5)
    sc.run(FadeIn(kept), run_time=0.4)

    sc.wait_until(sc.at(3))
    rs = card(k, "mô hình reasoning bây giờ", k.GOLD, 26, 30).next_to(kept, DOWN, buff=0.3)
    sc.run(FadeIn(rs, shift=UP * 0.2), run_time=0.5)


@_part(3)
def err_first(sc, s, k):
    q = k.label("Sao phải sai rồi mới đúng?", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(q), run_time=0.5)
    sc.wait_until(sc.at(1))
    senior = _student(k, k.ORANGE, "đàn anh", 1.4).move_to(LEFT * 2.2 + DOWN * 0.5)
    junior = _student(k, k.ACCENT, "đàn em", 1.4).move_to(RIGHT * 2.2 + DOWN * 0.5)
    scold = bubble(k, "Sao làm thí nghiệm lại mắc lỗi này?", k.ORANGE, 22, 26).next_to(senior, UP, buff=0.3)
    sc.run(FadeIn(senior), FadeIn(junior), run_time=0.5)
    sc.run(FadeIn(scold), run_time=0.5)
    year = k.label("năm sau", 26, k.GOLD, bold=True).move_to(DOWN * 2.5)
    swap = CurvedArrow(junior.get_bottom() + DOWN * 0.1, senior.get_bottom() + DOWN * 0.1, angle=-1.0,
                       color=k.GOLD, stroke_width=4)
    sc.run(Create(swap), FadeIn(year), run_time=0.7)
    newbie = _student(k, k.GREEN, "đàn em mới", 1.4).move_to(RIGHT * 5.4 + DOWN * 0.5)
    scold2 = bubble(k, "Sao lại mắc lỗi này?", k.ACCENT, 20, 26).next_to(junior, UP, buff=0.3)
    sc.run(FadeIn(newbie), FadeIn(scold2), run_time=0.7)

    sc.wait_until(sc.at(2))
    mine = card(k, "có những lỗi phải tự giẫm một lần mới gỡ được", k.GOLD, 28, 50).to_edge(DOWN, buff=0.35)
    sc.run(FadeIn(mine, shift=UP * 0.2), run_time=0.6)


def _bits(k, bits, color=None, size=40):
    return VGroup(*[card(k, str(b), color or k.ACCENT, size, 3, pad=0.45) for b in bits]).arrange(RIGHT, buff=0.15)


@_part(3)
def parity_rote(sc, s, k):
    head = k.label("Parity check", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    cells = _bits(k, [0, 1, 1, 0, 1, 0]).next_to(head, DOWN, buff=0.4)
    q = k.label("có chẵn hay lẻ số chữ 1?", 28, k.MUTED).next_to(cells, DOWN, buff=0.25)
    sc.run(FadeIn(cells, lag_ratio=0.1), FadeIn(q), run_time=0.8)

    sc.wait_until(sc.at(2))
    rote = k.label("học thuộc: nhìn cả dãy, đoán ngay", 28, k.PINK, bold=True).move_to(DOWN * 0.4)
    sc.run(FadeIn(rote), run_time=0.5)
    sc.wait_until(sc.at(3))
    per = VGroup(*[k.label("×2", 26, k.PINK, bold=True).next_to(c, UP, buff=0.1) for c in cells])
    sc.run(FadeIn(per, lag_ratio=0.15), run_time=0.8)
    # Một khối Text duy nhất: 64 Text rời làm cảnh render quá 10 phút.
    rows = ["  ".join(f"{i:06b}" for i in range(r * 8, r * 8 + 8)) for r in range(8)]
    table = Text("\n".join(rows), font=k.MONO, font_size=18, color=k.MUTED, line_spacing=0.8)
    table.scale_to_fit_height(2.6).move_to(DOWN * 2.2)
    count = k.label("2 mũ 6 = 64 trường hợp", 30, k.PINK, bold=True).next_to(table, UP, buff=0.2)
    sc.run(FadeOut(rote), FadeIn(table), FadeIn(count), run_time=1.2)


@_part(3)
def parity_steps(sc, s, k):
    head = k.label("Làm từng bước", 38, k.GREEN, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)
    sc.wait_until(sc.at(1))
    bits = [0, 1, 1, 0, 1, 0]
    cells = _bits(k, bits, size=40).next_to(head, DOWN, buff=0.4)
    sc.run(FadeIn(cells, lag_ratio=0.1), run_time=0.6)

    states = ["lẻ", "chẵn", "chẵn", "lẻ", "lẻ"]
    slots = VGroup()
    for i in range(5):
        slots.add(card(k, states[i], k.GOLD, 28, 6, pad=0.35))
    slots.arrange(RIGHT, buff=0.55).next_to(cells, DOWN, buff=0.8)
    marks = [2] + list(range(3, 7))
    for i, st in enumerate(slots):
        sc.wait_until(sc.at(marks[i]))
        span = cells[: i + 2]
        ring = SurroundingRectangle(cells[i + 1] if i else span, color=k.GOLD, buff=0.08, corner_radius=0.1)
        sc.run(Create(ring), FadeIn(st, shift=DOWN * 0.2), run_time=0.6)
        if i:
            sc.run(GrowArrow(Arrow(slots[i - 1].get_right(), st.get_left(), buff=0.08, color=k.GOLD, stroke_width=4,
                                   max_tip_length_to_length_ratio=0.3)), FadeOut(ring), run_time=0.4)
        else:
            sc.run(FadeOut(ring), run_time=0.2)

    sc.wait_until(sc.at(7))
    ans = card(k, "Đáp án: lẻ", k.GOLD, 32, 14).next_to(slots, DOWN, buff=0.4)
    sc.run(GrowFromCenter(ans), run_time=0.5)

    sc.wait_until(sc.at(8))
    sc.run(FadeOut(ans), run_time=0.3)
    rules = VGroup(*[Text(t, font=k.MONO, font_size=26, color=k.GREEN) for t in
                     ("chẵn + 0 = chẵn", "chẵn + 1 = lẻ", "lẻ + 0 = lẻ", "lẻ + 1 = chẵn")])
    rules.arrange_in_grid(2, 2, buff=(0.8, 0.2)).next_to(slots, DOWN, buff=0.45)
    rt = k.label("mỗi bước chỉ 1 quy tắc nhỏ, 4 khả năng", 26, k.GREEN, bold=True).next_to(rules, DOWN, buff=0.2)
    sc.run(FadeIn(rules, lag_ratio=0.2), FadeIn(rt), run_time=0.9)

    sc.wait_until(sc.at(9))
    cmp = VGroup(k.label("từng bước: 5 × 4 = 20 ví dụ", 28, k.GREEN, bold=True),
                 k.label("học thuộc: 64", 28, k.PINK, bold=True)).arrange(RIGHT, buff=0.8)
    cmp.to_edge(DOWN, buff=0.35)
    sc.run(FadeOut(rt), FadeIn(cmp), run_time=0.6)

    sc.wait_until(sc.at(10))
    big = VGroup(k.label("dãy 20 chữ số:", 26, k.INK, bold=True), k.label("học thuộc > 1 triệu", 26, k.PINK, bold=True),
                 k.label("từng bước: vài chục", 26, k.GREEN, bold=True)).arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=0.35)
    sc.run(FadeOut(cmp), FadeIn(big), run_time=0.6)

    sc.wait_until(sc.at(11))
    gen = card(k, "muốn làm được bài chưa từng gặp: phải suy luận từng bước", k.GOLD, 26, 60).to_edge(DOWN, buff=0.3)
    sc.run(FadeOut(big), FadeIn(gen, shift=UP * 0.2), run_time=0.6)


def _target(k, r=1.5):
    rings = VGroup(*[Circle(radius=r * f, color=k.MUTED, stroke_width=3) for f in (1.0, 0.66, 0.33)])
    bull = Dot(radius=r * 0.1, color=k.GOLD)
    return VGroup(rings, bull)


@_part(3)
def pass_at_k(sc, s, k):
    q = k.label("RL dạy điều mới, hay làm lộ điều có sẵn?", 34, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(q), run_time=0.8)
    sc.wait_until(sc.at(1))
    name = k.label("pass@k: giải k lần, đúng một lần là đậu", 30, k.GOLD, bold=True).next_to(q, DOWN, buff=0.4)
    sc.run(FadeIn(name), run_time=0.5)
    sc.wait_until(sc.at(2))
    tgt = _target(k, 1.8).move_to(DOWN * 0.8)
    sc.run(FadeIn(tgt), run_time=0.5)
    rnd = random.Random(5)
    darts = VGroup(*[Dot(tgt.get_center() + RIGHT * rnd.uniform(-1.8, 1.8) + UP * rnd.uniform(-1.6, 1.6),
                         radius=0.09, color=k.PINK) for _ in range(5)])
    hit = Dot(tgt.get_center() + RIGHT * 0.05, radius=0.11, color=k.GREEN)
    sc.run(FadeIn(darts, lag_ratio=0.3), run_time=0.8)
    sc.run(FadeIn(hit, scale=2), run_time=0.4)
    ok = k.label("trúng một phát: đậu", 26, k.GREEN, bold=True).next_to(tgt, RIGHT, buff=0.5)
    sc.run(FadeIn(ok), run_time=0.4)


@_part(3)
def darts(sc, s, k):
    tag = card(k, "paper 04/2025", k.MUTED, 22, 20).to_edge(UP, buff=0.35)
    sc.run(FadeIn(tag), run_time=0.4)
    t_rl = _target(k, 1.5).move_to(LEFT * 3.4 + UP * 0.3)
    t_base = _target(k, 1.5).move_to(RIGHT * 3.4 + UP * 0.3)
    l_rl = k.label("qua RL", 30, k.GOLD, bold=True).next_to(t_rl, UP, buff=0.25)
    l_base = k.label("mô hình gốc", 30, k.INK, bold=True).next_to(t_base, UP, buff=0.25)
    sc.run(FadeIn(t_rl), FadeIn(t_base), FadeIn(l_rl), FadeIn(l_base), run_time=0.6)

    first_rl = Dot(t_rl.get_center() + RIGHT * 0.08, radius=0.11, color=k.GREEN)
    first_base = Dot(t_base.get_center() + RIGHT * 0.9 + UP * 0.7, radius=0.11, color=k.PINK)
    k1 = k.label("ném 1 lần", 30, k.INK, bold=True).move_to(UP * 2.4)
    sc.run(FadeIn(k1), FadeIn(first_rl, scale=2), FadeIn(first_base, scale=2), run_time=0.6)
    w1 = k.label("RL thắng xa", 26, k.GOLD, bold=True).next_to(t_rl, DOWN, buff=0.3)
    sc.run(FadeIn(w1), run_time=0.4)

    sc.wait_until(sc.at(1))
    rnd = random.Random(8)
    # RL: dồn quanh một chỗ quen tay, lệch khỏi hồng tâm một chút
    spot = t_rl.get_center() + RIGHT * 0.5 + DOWN * 0.4
    rl_more = VGroup(*[Dot(spot + RIGHT * rnd.gauss(0, 0.15) + UP * rnd.gauss(0, 0.15), radius=0.07, color=k.GOLD)
                       for _ in range(30)])
    base_more = VGroup(*[Dot(t_base.get_center() + RIGHT * rnd.uniform(-1.5, 1.5) + UP * rnd.uniform(-1.4, 1.4),
                             radius=0.07, color=k.MUTED) for _ in range(30)])
    lucky = Dot(t_base.get_center() + LEFT * 0.06 + UP * 0.04, radius=0.11, color=k.GREEN)
    k256 = k.label("ném 256 lần", 30, k.INK, bold=True).move_to(k1)
    sc.run(Transform(k1, k256), FadeOut(w1), FadeIn(rl_more, lag_ratio=0.03), FadeIn(base_more, lag_ratio=0.03),
           run_time=1.2)
    sc.run(FadeIn(lucky, scale=2), run_time=0.4)
    tie = k.label("mô hình gốc ngang ngửa", 26, k.INK, bold=True).next_to(t_base, DOWN, buff=0.3)
    sc.run(FadeIn(tie), run_time=0.4)

    sc.wait_until(sc.at(2))
    why = k.label("Vì sao?", 30, k.INK, bold=True).to_edge(DOWN, buff=0.9)
    sc.run(FadeIn(why), run_time=0.3)
    sc.wait_until(sc.at(3))
    coach = k.label("luôn ném về một chỗ quen tay", 24, k.GOLD).next_to(t_rl, DOWN, buff=0.3)
    sc.run(Indicate(rl_more, color=k.GOLD), FadeIn(coach), run_time=0.7)
    sc.wait_until(sc.at(4))
    sc.run(Indicate(base_more, color=k.INK), run_time=0.7)
    spread = k.label("ném tản mát, đủ nhiều thì cũng trúng", 24, k.MUTED).next_to(tie, DOWN, buff=0.1)
    sc.run(FadeIn(spread), run_time=0.4)

    sc.wait_until(sc.at(5))
    msg = k.label("đáp án đúng có sẵn trong mô hình gốc · RL chỉ làm nó dễ được chọn hơn", 26, k.GOLD, bold=True)
    msg.to_edge(DOWN, buff=0.35)
    sc.run(FadeOut(why), FadeIn(msg), run_time=0.5)
    sc.wait_until(sc.at(6))
    sutra = card(k, "Hoa Nghiêm: ai cũng sẵn trí tuệ Như Lai, chỉ bị vọng tưởng che lấp", k.GOLD, 24, 70)
    sutra.move_to(msg)
    sc.run(FadeOut(msg), FadeIn(sutra, shift=UP * 0.2), run_time=0.6)


@_part(3)
def other_camp(sc, s, k):
    head = k.label("Phe kia phản bác", 36, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(head), run_time=0.5)
    nums = VGroup(*[card(k, str(i), k.MUTED, 24, 3, pad=0.35) for i in range(1, 11)]).arrange(RIGHT, buff=0.1)
    nums.next_to(head, DOWN, buff=0.35)
    guess = k.label("đáp án chỉ là số nhỏ: thử 256 lần, đoán bừa cũng trúng", 24, k.PINK).next_to(nums, DOWN, buff=0.15)
    sc.run(FadeIn(nums, lag_ratio=0.08), FadeIn(guess), run_time=0.9)

    sc.wait_until(sc.at(1))
    axes, at = _chart(k, 4.4, 2.4, "số lần thử k", "đậu: lời giải cũng phải đúng")
    axes.move_to(LEFT * 3.6 + DOWN * 1.1)
    rl = _polyline(k, at, [(0, 0.5), (0.5, 0.7), (1, 0.82)], k.GOLD)
    base = _polyline(k, at, [(0, 0.2), (0.5, 0.42), (1, 0.55)], k.MUTED)
    tag = k.label("paper 06/2025", 22, k.MUTED).next_to(axes, UP, buff=0.4)
    sc.run(FadeIn(axes), FadeIn(tag), run_time=0.5)
    sc.run(Create(rl), Create(base), run_time=0.9)
    win = k.label("qua RL hơn hẳn", 24, k.GOLD, bold=True).next_to(at(1, 0.82), UP, buff=0.15).shift(LEFT * 0.6)
    sc.run(FadeIn(win), run_time=0.4)

    sc.wait_until(sc.at(2))
    axis = Arrow(LEFT * 2.4, RIGHT * 2.4, buff=0, color=k.MUTED, stroke_width=4).move_to(RIGHT * 4.3 + DOWN * 1.8)
    at_t = k.label("thời gian huấn luyện RL", 22, k.MUTED).next_to(axis, DOWN, buff=0.15)
    early = card(k, "đầu: khai thác cái có sẵn", k.ACCENT, 20, 14)
    late = card(k, "sau: mọc năng lực mới", k.GREEN, 20, 14)
    VGroup(early, late).arrange(RIGHT, buff=0.25).next_to(axis, UP, buff=0.3)
    both = k.label("cả hai phe đều đúng", 26, k.GOLD, bold=True).next_to(VGroup(early, late), UP, buff=0.25)
    sc.run(GrowArrow(axis), FadeIn(at_t), run_time=0.5)
    sc.run(FadeIn(early), FadeIn(late), FadeIn(both), run_time=0.7)

    sc.wait_until(sc.at(3))
    still = card(k, "đến giờ vẫn đang tranh luận", k.MUTED, 26, 40).to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(still, shift=UP * 0.2), run_time=0.5)


def outro(sc, s, k):
    """Tóm tắt: ba cột, mỗi cột một hướng, liệt kê những gì video đã kể."""
    head = k.label("Video này đã đi qua", 40, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.7)
    cols = [("1 · Lúc sinh chữ", "bản sai rồi trừ đi", k.ACCENT,
             ["probe + vector vàng", "bạn học kém: trừ đi", "Obama: 40 − 60", "DoLa: tầng nông",
              "quả chuối đen", "MTI: “Output Error”"]),
            ("2 · Sinh rồi kiểm", "đổi quy trình", k.GREEN,
             ["chê dễ hơn làm?", "tự kiểm: chập chờn", "phản hồi ngoài: chắc ăn", "CL và CS",
              "khen hay hỏi vặn", "so với bỏ phiếu đa số"]),
            ("3 · Huấn luyện", "học tăng cường", k.GOLD,
             ["núm vặn, Hillary/Chicago", "lỗi xanh, lỗi đỏ", "RLVR tự biết kiểm lại", "parity: 64 hay 20",
              "phi tiêu pass@k", "hai phe đều đúng"])]
    panels = VGroup()
    for title, sub, color, items in cols:
        top = VGroup(k.label(title, 32, color, bold=True), k.label(sub, 24, k.MUTED)).arrange(DOWN, buff=0.08)
        rows = VGroup(*[k.label(t, 25, k.INK, max_width=3.6) for t in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        body = VGroup(top, rows).arrange(DOWN, buff=0.4)
        box = RoundedRectangle(corner_radius=0.2, width=4.1, height=6.1, color=color, stroke_width=3)
        box.set_fill(color, opacity=0.07).move_to(body)
        body.align_to(box, UP).shift(DOWN * 0.3)
        rows.align_to(box, LEFT).shift(RIGHT * 0.3)
        panels.add(VGroup(box, top, rows))
    panels.arrange(RIGHT, buff=0.25).next_to(head, DOWN, buff=0.3).set_x(0)

    for i, (box, top, rows) in enumerate(panels):
        sc.wait_until(sc.at(i + 1))
        sc.run(Create(box), FadeIn(top, shift=DOWN * 0.15), run_time=0.5)
        dots = VGroup(*[Dot(radius=0.05, color=cols[i][2]).next_to(r, LEFT, buff=0.12) for r in rows])
        sc.run(FadeIn(rows, lag_ratio=0.35), FadeIn(dots, lag_ratio=0.35), run_time=2.2)


# ── Thumbnail 1280x720 ───────────────────────────────────────
THUMB_TEXT = {
    "vi": ("MACHINE LEARNING 2026 SPRING", "AI có tự", "sửa lỗi?", "decoding · workflow · reasoning",
           "Đáp án: 42", "Kiểm tra lại..."),
    "en": ("MACHINE LEARNING 2026 SPRING", "Can AI fix", "itself?", "decoding · workflow · reasoning",
           "Answer: 42", "Let me re-check..."),
}


def thumbnail(sc, k, lang="vi"):
    tag_text, line1, line2, sub_text, first, second = THUMB_TEXT.get(lang, THUMB_TEXT["vi"])
    tag = k.label(tag_text, 28, k.MUTED, bold=True).to_corner(UP + LEFT, buff=0.6)
    left = VGroup(k.label(line1, 100, k.INK, bold=True), k.label(line2, 100, k.GOLD, bold=True))
    left.arrange(DOWN, aligned_edge=LEFT, buff=0.2).scale_to_fit_width(6.4).to_edge(LEFT, buff=0.75).shift(DOWN * 0.1)
    sub = k.label(sub_text, 34, k.ACCENT, bold=True).next_to(left, DOWN, aligned_edge=LEFT, buff=0.45)
    a = bubble(k, first, k.PURPLE, 18, 36).move_to(RIGHT * 4.0 + UP * 1.5)
    loop = CurvedArrow(a.get_right() + RIGHT * 0.2, a.get_bottom() + DOWN * 1.2 + RIGHT * 0.6, angle=-1.6,
                       color=k.GOLD, stroke_width=10)
    b = bubble(k, second, k.GOLD, 18, 34).move_to(RIGHT * 3.8 + DOWN * 1.6)
    mark = k.label("?", 140, k.PINK, bold=True).next_to(b, LEFT, buff=0.3).shift(UP * 1.3)
    sc.add(tag, left, sub, a, loop, b, mark)
