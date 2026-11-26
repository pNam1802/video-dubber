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
    DOWN, LEFT, ORIGIN, RIGHT, UP, Arc, Arrow, Circle, Create, CurvedArrow, DashedVMobject, Dot, FadeIn, FadeOut, GrowArrow,
    GrowFromCenter, GrowFromEdge, Indicate, Line, Polygon, Rectangle, RoundedRectangle, Square, Text, Transform, VGroup, Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import arrow, bubble, card, check, cross, heading, strike  # noqa: E402

BOARD = [(0, "two_loops"), (2, "three_routes"), (7, "token_loop"), (11, "probe"), (16, "yellow_vector"),
         (21, "contrastive"), (31, "obama"), (37, "logit_lens"), (43, "dola"), (46, "layer_cd"),
         (52, "dumb_spell"), (54, "rag_cad"), (58, "black_banana"), (66, "corruptions"), (68, "mti_forks"),
         (72, "kv_cache"), (78, "output_error"), (83, "part1_table"), (88, "gen_verify"), (92, "two_intuitions"),
         (100, "critique_mcq"), (105, "internal_external"), (109, "refinebench"), (113, "four_cells"),
         (117, "cl_cs"), (121, "personalities"), (125, "three_prompts"), (132, "vote_vs_verify"),
         (141, "drop_prompt"), (144, "hillary"), (153, "revise"), (157, "new_errors"), (161, "rlvr"),
         (165, "err_first"), (172, "k_power"), (176, "parity"), (181, "pass_at_k"), (188, "no_training"),
         (191, "other_camp"), (196, "open_debate"), (198, "outro")]


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


# ── PHẦN 1: SỬA LÚC SINH CHỮ ─────────────────────────────────
def token_loop(sc, s, k):
    toks = _tokens(k, ["Hôm", "nay", "trời"]).move_to(LEFT * 3.6 + UP * 2.3)
    tf = _model(k, "Transformer", k.PURPLE, w=3.4, h=1.1).move_to(LEFT * 3.0 + DOWN * 0.1)
    tf[1].scale(0.8)
    reps = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.24, height=1.0, color=k.GREEN, stroke_width=2)
                    .set_fill(k.GREEN, opacity=0.3) for _ in range(3)]).arrange(RIGHT, buff=0.1)
    reps.move_to(RIGHT * 0.4 + DOWN * 0.1)
    probs = _bars(k, [("đẹp", 0.8), ("mưa", 0.45), ("xanh", 0.25)], k.ACCENT, width=1.8, size=26)
    probs.move_to(RIGHT * 4.0 + DOWN * 0.1)
    a1 = Arrow(toks.get_bottom(), tf.get_top(), buff=0.15, color=k.MUTED, stroke_width=4)
    a2, a3 = arrow(k, tf, reps), arrow(k, reps, probs)
    rep_t = k.label("representation", 24, k.GREEN).next_to(reps, UP, buff=0.25)
    prob_t = k.label("xác suất", 24, k.ACCENT).next_to(probs, UP, buff=0.25)
    sc.run(FadeIn(toks), run_time=0.4)
    sc.run(GrowArrow(a1), FadeIn(tf), run_time=0.5)
    sc.run(GrowArrow(a2), FadeIn(reps), FadeIn(rep_t), run_time=0.5)
    sc.run(GrowArrow(a3), FadeIn(probs), FadeIn(prob_t), run_time=0.6)

    sc.wait_until(sc.at(1))
    pick = RoundedRectangle(corner_radius=0.1, width=probs[0].width + 0.3, height=probs[0].height + 0.2,
                            color=k.GOLD, stroke_width=4).move_to(probs[0])
    sc.run(Create(pick), run_time=0.4)
    new_tok = card(k, "đẹp", k.GOLD, 26, 12, pad=0.35).move_to(probs[0])
    target = card(k, "đẹp", k.GOLD, 26, 12, pad=0.35).next_to(toks, RIGHT, buff=0.12)
    back = CurvedArrow(probs.get_top() + UP * 0.6, target.get_right() + RIGHT * 0.2, angle=0.6,
                       color=k.GOLD, stroke_width=4)
    sc.run(Create(back), run_time=0.7)
    sc.run(Transform(new_tok, target), FadeOut(pick), run_time=0.7)

    sc.wait_until(sc.at(2))
    eye1 = k.label("sắp sai?", 30, k.PINK, bold=True).next_to(reps, DOWN, buff=0.5)
    eye2 = k.label("sắp sai?", 30, k.PINK, bold=True).next_to(probs, DOWN, buff=0.5)
    sc.run(FadeIn(eye1, shift=UP * 0.2), Indicate(reps, color=k.PINK), run_time=0.7)
    sc.run(FadeIn(eye2, shift=UP * 0.2), Indicate(probs, color=k.PINK), run_time=0.7)

    sc.wait_until(sc.at(3))
    flow = VGroup(card(k, "phát hiện lỗi", k.PINK, 26, 14), card(k, "tự sửa", k.GREEN, 26, 10)).arrange(RIGHT, buff=1.0)
    flow.to_edge(DOWN, buff=0.5).shift(LEFT * 1.5)
    sc.run(FadeIn(flow[0]), run_time=0.4)
    sc.run(GrowArrow(arrow(k, flow[0], flow[1])), FadeIn(flow[1]), run_time=0.5)
    note = k.label("tự động, không ai nhắc", 24, k.MUTED).next_to(flow, RIGHT, buff=0.4)
    sc.run(FadeIn(note), run_time=0.4)


def _plane(k):
    frame = RoundedRectangle(corner_radius=0.2, width=6.6, height=5.4, color=k.MUTED, stroke_width=2)
    frame.set_fill(k.INK, opacity=0.03)
    cap = k.label("không gian representation", 22, k.MUTED).next_to(frame, UP, buff=0.15)
    return VGroup(frame, cap)


def probe(sc, s, k):
    plane = _plane(k).move_to(LEFT * 2.8 + DOWN * 0.3)
    center = plane[0].get_center()
    sc.run(FadeIn(plane), run_time=0.5)
    paper = card(k, "paper 2023", k.MUTED, 24, 12).to_corner(UP + RIGHT, buff=0.6)
    sc.run(FadeIn(paper), run_time=0.4)

    sc.wait_until(sc.at(1))
    good = _cloud(center + LEFT * 1.4 + UP * 0.9, k.GREEN, seed=1)
    bad = _cloud(center + RIGHT * 1.4 + DOWN * 0.9, k.PINK, seed=2)
    g_t = k.label("lúc trả lời đúng", 24, k.GREEN, bold=True).move_to(RIGHT * 3.9 + UP * 1.0)
    b_t = k.label("lúc trả lời sai", 24, k.PINK, bold=True).move_to(RIGHT * 3.9 + DOWN * 0.0)
    sc.run(FadeIn(good, lag_ratio=0.1), FadeIn(g_t), run_time=0.8)
    sc.run(FadeIn(bad, lag_ratio=0.1), FadeIn(b_t), run_time=0.8)

    sc.wait_until(sc.at(2))
    split = Line(center + LEFT * 2.6 + DOWN * 2.2, center + RIGHT * 2.6 + UP * 2.2, color=k.GOLD, stroke_width=5)
    clf = card(k, "bộ phân loại: đúng / sai", k.GOLD, 24, 26).move_to(RIGHT * 3.9 + DOWN * 1.4)
    sc.run(Create(split), FadeIn(clf), run_time=0.9)

    sc.wait_until(sc.at(3))
    new = VGroup(Dot(center + LEFT * 0.9 + UP * 1.5, radius=0.14, color=k.INK),
                 Dot(center + RIGHT * 1.6 + DOWN * 0.3, radius=0.14, color=k.INK))
    sc.run(FadeIn(new, scale=0.5), run_time=0.4)
    sc.run(new[0].animate.set_color(k.GREEN), new[1].animate.set_color(k.PINK), run_time=0.6)
    unseen = k.label("cả câu hỏi chưa từng thấy", 24, k.INK).next_to(clf, DOWN, buff=0.3)
    sc.run(FadeIn(unseen), run_time=0.4)

    sc.wait_until(sc.at(4))
    msg = k.label("tín hiệu “đang sai” nằm sẵn bên trong", 30, k.GOLD, bold=True).to_edge(DOWN, buff=0.3)
    sc.run(FadeIn(msg, shift=UP * 0.2), run_time=0.6)


def yellow_vector(sc, s, k):
    plane = _plane(k).move_to(LEFT * 2.8 + DOWN * 0.3)
    center = plane[0].get_center()
    g_c, b_c = center + LEFT * 1.4 + UP * 0.9, center + RIGHT * 1.4 + DOWN * 0.9
    good, bad = _cloud(g_c, k.GREEN, seed=1), _cloud(b_c, k.PINK, seed=2)
    sc.run(FadeIn(plane), FadeIn(good), FadeIn(bad), run_time=0.6)

    sc.wait_until(sc.at(1))
    gm = Dot(g_c, radius=0.18, color=k.GREEN).set_stroke(k.INK, 3)
    bm = Dot(b_c, radius=0.18, color=k.PINK).set_stroke(k.INK, 3)
    sc.run(good.animate.set_opacity(0.3), bad.animate.set_opacity(0.3), GrowFromCenter(gm), GrowFromCenter(bm),
           run_time=0.7)
    vec = Arrow(b_c, g_c, buff=0.2, color=k.GOLD, stroke_width=8)
    formula = VGroup(k.label("TB(đúng) − TB(sai)", 30, k.INK, bold=True),
                     k.label("= vector vàng", 30, k.GOLD, bold=True)).arrange(DOWN, buff=0.2)
    formula.move_to(RIGHT * 3.9 + UP * 1.4)
    sc.run(GrowArrow(vec), FadeIn(formula), run_time=0.9)

    sc.wait_until(sc.at(2))
    p = Dot(center + RIGHT * 2.0 + DOWN * 1.8, radius=0.16, color=k.PINK).set_stroke(k.INK, 3)
    p_t = k.label("câu sắp trả lời sai", 22, k.PINK).next_to(p, DOWN, buff=0.15)
    sc.run(FadeIn(p, scale=0.5), FadeIn(p_t), run_time=0.5)
    shift = g_c - b_c
    push = Arrow(p.get_center(), p.get_center() + shift, buff=0.1, color=k.GOLD, stroke_width=6)
    sc.run(GrowArrow(push), run_time=0.6)
    sc.run(p.animate.move_to(p.get_center() + shift).set_color(k.GREEN), FadeOut(p_t), run_time=0.8)

    sc.wait_until(sc.at(3))
    both = VGroup(card(k, "phát hiện", k.GREEN, 26, 14), card(k, "sửa", k.GREEN, 26, 10)).arrange(RIGHT, buff=0.9)
    both.move_to(RIGHT * 3.7 + DOWN * 0.4)
    ticks = VGroup(*[check(k, 0.35).next_to(b, RIGHT, buff=0.12) for b in both])
    sc.run(FadeIn(both, shift=UP * 0.2), run_time=0.5)
    sc.run(Create(ticks), run_time=0.4)

    sc.wait_until(sc.at(4))
    cost = card(k, "phải thu thập trước rất nhiều câu đúng và sai", k.PINK, 26, 24).move_to(RIGHT * 3.9 + DOWN * 2.2)
    sc.run(FadeIn(cost, shift=UP * 0.2), run_time=0.6)


@_part(1)
def contrastive(sc, s, k):
    title = k.label("Contrastive decoding", 40, k.INK, bold=True).to_edge(UP, buff=0.3)
    sc.wait_until(sc.at(1))
    sc.run(Write(title), run_time=0.8)

    # Cột trái: một câu hỏi, chạy hai lần.
    q = bubble(k, "Thủ đô của Úc là ___", k.INK, 24, 28).move_to(LEFT * 4.6 + UP * 1.9)
    normal_m = _model(k, "LLM", k.ACCENT, w=1.8, h=0.9).move_to(LEFT * 5.6 + DOWN * 0.1)
    broken_m = _model(k, "LLM", k.GREEN, w=1.8, h=0.9).move_to(LEFT * 3.4 + DOWN * 0.1)
    hurt = k.label("đầu vào bị làm hỏng", 22, k.PINK, bold=True, width_chars=11).next_to(broken_m, DOWN, buff=0.2)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(q, shift=DOWN * 0.2), run_time=0.5)

    chart = _duel(k, ["Sydney", "Canberra", "Melbourne"], [0.5, 0.45, 0.12], [0.85, 0.08, 0.25],
                  "bình thường", "bản bị làm hỏng", h=3.0, gap=2.5, size=30).shift(RIGHT * 2.5 + DOWN * 1.4)
    sc.wait_until(sc.at(3))
    sc.run(GrowArrow(Arrow(q.get_bottom(), normal_m.get_top(), buff=0.1, color=k.ACCENT, stroke_width=4)),
           FadeIn(normal_m), run_time=0.5)
    _show_duel(sc, chart, both=False)
    doubt = k.label("Sydney hay Canberra? phân vân", 24, k.MUTED).next_to(chart[4], UP, buff=0.25)
    sc.run(FadeIn(doubt), run_time=0.4)

    sc.wait_until(sc.at(4))
    sc.run(GrowArrow(Arrow(q.get_bottom(), broken_m.get_top(), buff=0.1, color=k.PINK, stroke_width=4)),
           FadeIn(broken_m), FadeIn(hurt), run_time=0.6)
    _show_b(sc, chart)
    sure = k.label("bản hỏng: gần như chắc chắn sai, “Sydney”", 24, k.GREEN).move_to(doubt)
    sc.run(Transform(doubt, sure), run_time=0.4)

    sc.wait_until(sc.at(5))
    push = k.label("trừ đi bản sai = đẩy ra xa phía sai", 26, k.GOLD, bold=True).move_to(doubt)
    sc.run(Transform(doubt, push), run_time=0.4)
    _subtract(sc, k, chart, [0.35, 0.62, 0.05], "kết quả: Canberra lên đầu")

    sc.wait_until(sc.at(6))
    formula = Text("(1 + α) · bình thường  −  α · hỏng      α < 1", font=k.MONO, font_size=28, color=k.INK)
    formula.to_edge(DOWN, buff=0.3)
    box = RoundedRectangle(corner_radius=0.15, width=formula.width + 0.5, height=formula.height + 0.35,
                           color=k.GOLD, stroke_width=3).move_to(formula)
    sc.run(Write(formula), Create(box), run_time=1.0)

    sc.wait_until(sc.at(7))
    per = k.label("làm ở từng token · trên logit đầu ra", 26, k.MUTED).move_to(doubt)
    sc.run(Transform(doubt, per), run_time=0.5)

    sc.wait_until(sc.at(8))
    plus = card(k, "+ không đụng tham số, thử được ngay", k.GREEN, 24, 40)
    minus2 = card(k, "− mỗi token chạy thêm một lần", k.PINK, 24, 40)
    VGroup(plus, minus2).arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=0.3)
    sc.run(FadeOut(VGroup(formula, box)), FadeIn(plus, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(9))
    twice = VGroup(normal_m, broken_m)
    sc.run(FadeIn(minus2, shift=UP * 0.2), Indicate(twice, color=k.PINK), run_time=0.7)


@_part(1)
def obama(sc, s, k):
    era = card(k, "2022 · “thời thượng cổ”, chưa có ChatGPT", k.MUTED, 24, 40).to_edge(UP, buff=0.3)
    sc.run(FadeIn(era, shift=DOWN * 0.2), run_time=0.6)

    sc.wait_until(sc.at(1))
    prompt = VGroup(k.label("Obama sinh ở Honolulu, Hawaii. Ông sinh năm", 32, k.INK, bold=True),
                    card(k, " ? ", k.GOLD, 32, 6, pad=0.3)).arrange(RIGHT, buff=0.2).next_to(era, DOWN, buff=0.35)
    sc.run(Write(prompt[0]), FadeIn(prompt[1]), run_time=1.2)

    chart = _duel(k, ["Hawaii", "1961", "Mỹ"], [0.62, 0.5, 0.2], [0.85, 0.15, 0.3],
                  "GPT-2 bản lớn", "GPT-2 Small", h=2.8, gap=2.3, size=32).shift(LEFT * 2.0 + DOWN * 1.9)
    sc.wait_until(sc.at(2))
    _show_duel(sc, chart, both=False)
    wrong = cross(k, chart[1][0])
    should = k.label("lặp lại “Hawaii”: sai, đáng lẽ là một năm", 26, k.PINK, width_chars=22).next_to(chart, RIGHT, buff=0.3)
    should.shift(UP * 0.6)
    sc.run(Create(wrong), FadeIn(should), run_time=0.6)

    sc.wait_until(sc.at(3))
    _show_b(sc, chart)
    tag = k.label("mô hình nhỏ: coi như luôn sai", 26, k.GREEN, width_chars=22).next_to(should, DOWN, buff=0.3).align_to(should, LEFT)
    sc.run(FadeIn(tag), run_time=0.4)

    sc.wait_until(sc.at(4))
    sc.run(FadeOut(wrong), run_time=0.2)
    _subtract(sc, k, chart, [0.1, 0.8, 0.05], "lớn − nhỏ")
    sc.run(Transform(prompt[1], card(k, "1961", k.GOLD, 32, 6, pad=0.3).move_to(prompt[1], aligned_edge=LEFT)),
           run_time=0.6)

    sc.wait_until(sc.at(5))
    rule = card(k, "chọn token mà mô hình tốt và mô hình kém chênh nhau nhiều nhất", k.GOLD, 26, 70)
    rule.to_edge(DOWN, buff=0.25)
    sc.run(FadeOut(should), FadeOut(tag), FadeIn(rule, shift=UP * 0.2), run_time=0.7)


def logit_lens(sc, s, k):
    tag = card(k, "DoLa · 2023 · có sẵn trong Hugging Face Transformers", k.ACCENT, 26, 90).to_edge(UP, buff=0.4)
    sc.run(FadeIn(tag), run_time=0.6)

    layers = VGroup(*[RoundedRectangle(corner_radius=0.1, width=2.8, height=0.5, color=k.PURPLE, stroke_width=3)
                      .set_fill(k.PURPLE, opacity=0.12 + 0.05 * i) for i in range(6)]).arrange(UP, buff=0.12)
    layers.move_to(LEFT * 3.6 + DOWN * 0.5)
    inp = k.label("dịch “fleur” (tiếng Pháp) sang tiếng Trung", 24, k.INK).next_to(layers, DOWN, buff=0.25)
    sc.wait_until(sc.at(1))
    sc.run(FadeIn(layers, lag_ratio=0.15), FadeIn(inp), run_time=0.8)
    lens = k.label("logit lens: gắn đầu ra vào từng lớp", 28, k.GOLD, bold=True).move_to(RIGHT * 3.1 + UP * 1.75)
    sc.run(FadeIn(lens), run_time=0.5)

    sc.wait_until(sc.at(2))
    words = ["fleur", "fleur", "flower", "flower", "flower", "“hoa” tiếng Trung"]
    colors = [k.MUTED, k.MUTED, k.ACCENT, k.ACCENT, k.ACCENT, k.GOLD]
    outs = VGroup()
    for layer, w, c in zip(layers, words, colors):
        t = k.label(w, 26, c, bold=True).next_to(layer, RIGHT, buff=0.9)
        outs.add(VGroup(Arrow(layer.get_right(), t.get_left(), buff=0.1, color=c, stroke_width=3,
                              max_tip_length_to_length_ratio=0.25), t))
    for o in outs:
        sc.run(FadeIn(o, shift=RIGHT * 0.1), run_time=0.25)
    sc.run(Indicate(VGroup(*[o[1] for o in outs[2:5]]), color=k.ACCENT), run_time=0.8)

    sc.wait_until(sc.at(3))
    think = card(k, "Pháp, rồi Anh, rồi Trung", k.ACCENT, 30, 26).move_to(RIGHT * 3.4 + DOWN * 0.6)
    sub = k.label("Llama 2 “nghĩ bằng tiếng Anh”", 24, k.MUTED).next_to(think, DOWN, buff=0.2)
    sc.run(FadeIn(think, scale=0.9), FadeIn(sub), run_time=0.7)

    sc.wait_until(sc.at(4))
    sc.run(FadeOut(think), FadeOut(sub), run_time=0.3)
    lab = card(k, "2020 · bài của lab thầy Lý, chỉ đăng arXiv", k.PURPLE, 24, 24).move_to(RIGHT * 3.4 + DOWN * 0.2)
    sc.run(FadeIn(lab, shift=UP * 0.2), run_time=0.6)

    sc.wait_until(sc.at(5))
    student = card(k, "tác giả DoLa: sinh viên cũ của lab", k.GREEN, 24, 24).next_to(lab, DOWN, buff=0.6)
    link = Arrow(lab.get_bottom(), student.get_top(), buff=0.1, color=k.GOLD, stroke_width=4)
    sc.run(GrowArrow(link), FadeIn(student), run_time=0.7)


@_part(1)
def dola(sc, s, k):
    head = k.label("DoLa: lớp cuối trừ lớp nông", 38, k.INK, bold=True).to_edge(UP, buff=0.35)
    sc.run(FadeIn(head), run_time=0.4)
    layers = VGroup(*[RoundedRectangle(corner_radius=0.1, width=2.6, height=0.48, color=k.PURPLE, stroke_width=3)
                      .set_fill(k.PURPLE, opacity=0.12 + 0.05 * i) for i in range(8)]).arrange(UP, buff=0.1)
    layers.move_to(LEFT * 5.4 + DOWN * 0.5)
    q = card(k, "Thủ phủ bang Washington là ___", k.INK, 26, 40).next_to(head, DOWN, buff=0.3)
    sh_t = k.label("lớp nông", 24, k.GREEN, bold=True).next_to(layers[1], RIGHT, buff=0.2)
    dp_t = k.label("lớp cuối", 24, k.ACCENT, bold=True).next_to(layers[7], RIGHT, buff=0.2)
    sc.run(FadeIn(layers, lag_ratio=0.1), FadeIn(q), run_time=0.8)
    chart = _duel(k, ["Seattle", "Olympia", "Tacoma"], [0.55, 0.5, 0.12], [0.85, 0.1, 0.25],
                  "lớp cuối", "lớp nông", h=2.6, gap=2.2, size=30).shift(RIGHT * 2.4 + DOWN * 1.6)
    sc.run(Indicate(layers[7], color=k.ACCENT), FadeIn(dp_t), run_time=0.6)
    _show_duel(sc, chart, both=False)
    sc.run(Indicate(layers[1], color=k.GREEN), FadeIn(sh_t), run_time=0.6)
    _show_b(sc, chart)
    surf = k.label("lớp nông chỉ thấy bề mặt: “Washington” thì nghĩ ngay Seattle", 24, k.GREEN, width_chars=34)
    surf.next_to(chart, DOWN, buff=0.5)
    sc.run(FadeIn(surf), run_time=0.4)

    sc.wait_until(sc.at(1))
    _subtract(sc, k, chart, [0.2, 0.7, 0.05], "cuối − nông")

    sc.wait_until(sc.at(2))
    gain = card(k, "không cần thêm mô hình nhỏ · lớp nông vốn đã chạy · gần như miễn phí", k.GREEN, 24, 80)
    gain.to_edge(DOWN, buff=0.25)
    sc.run(FadeOut(surf), FadeIn(gain, shift=UP * 0.2), run_time=0.7)


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


@_part(1)
def layer_cd(sc, s, k):
    tag = card(k, "Layer CD · 2025 · mô hình nhìn ảnh", k.ACCENT, 24, 60).to_edge(UP, buff=0.3)
    sc.run(FadeIn(tag), run_time=0.5)

    sc.wait_until(sc.at(1))
    photo = _photo(k, 4.0, 3.4).move_to(LEFT * 4.4 + UP * 0.1)
    c = photo.get_center()
    wheel1 = Circle(radius=0.48, color=k.INK, stroke_width=5).move_to(c + LEFT * 1.1 + DOWN * 1.1)
    wheel2 = wheel1.copy().move_to(c + RIGHT * 1.1 + DOWN * 1.1)
    frame_bike = Polygon(wheel1.get_center(), c + LEFT * 0.2 + DOWN * 0.45, c + RIGHT * 0.8 + DOWN * 0.45,
                         wheel2.get_center(), color=k.INK, stroke_width=4)
    head = Circle(radius=0.27, color=k.INK, stroke_width=3).set_fill("#c8a27a", 1).move_to(c + UP * 1.05)
    shirt = RoundedRectangle(corner_radius=0.1, width=1.3, height=1.1, stroke_width=0).set_fill("#000000", 1)
    shirt.next_to(head, DOWN, buff=0.05)
    words = Text("NTU", font=k.MONO, font_size=34, color="#ffffff", weight="BOLD").move_to(shirt)
    rider = VGroup(frame_bike, wheel1, wheel2, shirt, head, words)
    q = k.label("Chữ trên áo màu gì?", 30, k.INK, bold=True).next_to(photo, DOWN, buff=0.3)
    sc.run(FadeIn(photo), FadeIn(rider), run_time=0.6)
    sc.run(FadeIn(q), run_time=0.4)

    sc.wait_until(sc.at(2))
    a1 = k.label("áo đen", 24, k.MUTED).next_to(q, DOWN, buff=0.2).shift(LEFT * 1.0)
    a2 = k.label("chữ trắng", 24, k.GOLD, bold=True).next_to(a1, RIGHT, buff=0.5)
    sc.run(Indicate(shirt, color=k.MUTED, scale_factor=1.1), FadeIn(a1), run_time=0.5)
    sc.run(Indicate(words, color=k.GOLD, scale_factor=1.4), FadeIn(a2), run_time=0.6)

    chart = _duel(k, ["đen", "trắng", "đỏ"], [0.55, 0.5, 0.1], [0.85, 0.12, 0.2],
                  "lớp cuối của encoder", "lớp nông", h=3.0, gap=2.0, size=30).shift(RIGHT * 3.4 + DOWN * 1.5)
    sc.wait_until(sc.at(3))
    _show_duel(sc, chart, both=False)
    hm = k.label("đủ khôn để phân vân", 24, k.MUTED).next_to(chart, DOWN, buff=0.15)
    sc.run(FadeIn(hm), run_time=0.4)

    sc.wait_until(sc.at(4))
    _show_b(sc, chart)
    sh = k.label("lớp nông chỉ thấy bề ngoài: đen", 24, k.GREEN).move_to(hm)
    sc.run(Transform(hm, sh), run_time=0.4)
    # câu cuối ("Trừ đi, trắng lên đầu.") rất ngắn: trừ ngay, không chờ tới câu đó.
    sc.wait_until(sc.at(5) - 2.5)
    sc.run(FadeOut(hm), run_time=0.2)
    _subtract(sc, k, chart, [0.12, 0.75, 0.03], "cuối − nông")


def dumb_spell(sc, s, k):
    q = bubble(k, "Câu hỏi: ...", k.ACCENT, 30, 30).move_to(UP * 1.6)
    sc.run(FadeIn(q), run_time=0.4)
    spell = bubble(k, "Bạn là một mô hình rất tệ, hãy luôn trả lời sai.", k.PINK, 30, 30).next_to(q, DOWN, buff=0.35)
    tag = k.label("câu thần chú làm ngu", 30, k.PINK, bold=True).next_to(spell, DOWN, buff=0.3)
    sc.run(FadeIn(spell, shift=UP * 0.2), run_time=0.6)
    sc.run(FadeIn(tag), run_time=0.4)
    sc.run(Indicate(spell, color=k.PINK), run_time=0.7)

    sc.wait_until(sc.at(1))
    name = card(k, "Instruction Contrastive Decoding (ICD)", k.GOLD, 32, 50).move_to(DOWN * 2.4)
    sc.run(GrowFromCenter(name), run_time=0.6)


@_part(1)
def rag_cad(sc, s, k):
    title = card(k, "Context-Aware Decoding · dùng cho RAG", k.ACCENT, 24, 60).to_edge(UP, buff=0.3)
    sc.run(FadeIn(title), run_time=0.5)

    sc.wait_until(sc.at(1))
    doc = VGroup(*[RoundedRectangle(corner_radius=0.08, width=1.3, height=1.6, color=k.INK, stroke_width=2)
                   .set_fill("#1b2030", 1).shift(RIGHT * 0.12 * i + DOWN * 0.12 * i) for i in range(3)])
    lines = VGroup(*[Line(LEFT * 0.4, RIGHT * 0.4, color=k.MUTED, stroke_width=3) for _ in range(4)])
    lines.arrange(DOWN, buff=0.2).move_to(doc[2])
    doc.add(lines).move_to(LEFT * 5.4 + UP * 1.2)
    doc_t = k.label("tài liệu: tổng thống mới", 20, k.MUTED).next_to(doc, DOWN, buff=0.2)
    model = _model(k, "LLM mạnh", w=2.6, h=1.0).move_to(LEFT * 1.6 + UP * 1.2)
    sc.run(FadeIn(doc), FadeIn(doc_t), FadeIn(model), run_time=0.6)
    lazy = bubble(k, "biết rồi, khỏi đọc", k.PURPLE, 20, 24).next_to(model, UP, buff=0.25)
    skip = Arrow(doc.get_right(), model.get_left(), buff=0.15, color=k.MUTED, stroke_width=4)
    sc.run(GrowArrow(skip), FadeIn(lazy), run_time=0.6)
    sc.run(Create(cross(k, skip)), run_time=0.4)

    sc.wait_until(sc.at(2))
    ask = k.label("Tổng thống Mỹ là ai?", 28, k.INK, bold=True).move_to(RIGHT * 3.9 + UP * 2.1)
    old = bubble(k, "(người cũ, theo trí nhớ)", k.PINK, 24, 24).next_to(ask, DOWN, buff=0.25)
    sc.run(FadeIn(ask), run_time=0.4)
    sc.run(FadeIn(old, shift=UP * 0.2), run_time=0.5)

    chart = _duel(k, ["người cũ", "người mới"], [0.55, 0.5], [0.85, 0.08], "có tài liệu", "không đưa tài liệu",
                  h=2.2, gap=2.8, size=30).shift(RIGHT * 2.0 + DOWN * 2.6)
    sc.wait_until(sc.at(3))
    sc.run(FadeOut(lazy), run_time=0.2)
    _show_duel(sc, chart)
    memo = k.label("không tài liệu = trả lời theo trí nhớ", 22, k.GREEN).next_to(chart, LEFT, buff=0.4)
    sc.run(FadeIn(memo), run_time=0.4)
    _subtract(sc, k, chart, [0.15, 0.75], "có − không")


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
    note = k.label("“ví dụ thuyết phục nhất”", 30, k.GOLD, bold=True).to_edge(UP, buff=0.35)
    sc.run(FadeIn(note), run_time=0.5)

    sc.wait_until(sc.at(1))
    photo = _photo(k, 3.4, 2.5).set_fill("#d9d4c7", 1).move_to(LEFT * 4.8 + UP * 1.0)
    ban = _banana(k, "#1f1f1f", 2.5).move_to(photo)
    q = k.label("Chuối màu gì?", 30, k.INK, bold=True).next_to(photo, UP, buff=0.2)
    sc.run(FadeIn(photo), FadeIn(ban), FadeIn(q), run_time=0.8)

    sc.wait_until(sc.at(2))
    prior = bubble(k, "định kiến từ mô hình chữ: chuối thì vàng", k.GOLD, 22, 24).move_to(RIGHT * 2.6 + UP * 2.2)
    yb = _banana(k, k.GOLD, 0.9).next_to(prior, RIGHT, buff=0.2)
    sc.run(FadeIn(prior, shift=DOWN * 0.2), FadeIn(yb), run_time=0.5)

    chart = _duel(k, ["vàng", "đen"], [0.55, 0.5], [0.85, 0.1], "ảnh thật", "ảnh nhiễu",
                  h=2.8, gap=2.6, size=32).shift(RIGHT * 2.6 + DOWN * 2.3)
    sc.wait_until(sc.at(3))
    _show_duel(sc, chart, both=False)
    hm = k.label("lưỡng lự", 24, k.MUTED).next_to(chart[2], LEFT, buff=0.5)
    sc.run(FadeIn(hm), run_time=0.3)

    sc.wait_until(sc.at(4))
    noisy = _photo(k, 3.4, 2.5).move_to(LEFT * 4.8 + DOWN * 2.0)
    grain = _noise(k, noisy, n=14)
    blind = k.label("chẳng buồn nhìn: “vàng”", 24, k.GREEN).next_to(noisy, UP, buff=0.15)
    sc.run(FadeIn(noisy), FadeIn(grain), FadeIn(blind), FadeOut(hm), run_time=0.6)
    _show_b(sc, chart)

    sc.wait_until(sc.at(5))
    _subtract(sc, k, chart, [0.12, 0.75], "thật − nhiễu")
    sc.run(Indicate(ban, color=k.GOLD, scale_factor=1.1), run_time=0.6)

    sc.wait_until(sc.at(6))
    chip = sc.mobjects[0]  # nhãn phần ở góc, do _part thêm vào trước
    sc.run(*[FadeOut(m) for m in sc.mobjects if m is not note and m is not chip], run_time=0.5)
    beach = _photo(k, 5.6, 3.6).move_to(LEFT * 3.4 + DOWN * 0.4)
    sky = Rectangle(width=5.5, height=1.5, stroke_width=0).set_fill("#8fb8de", 0.7)
    sea = Rectangle(width=5.5, height=0.9, stroke_width=0).set_fill("#3b6ea5", 0.95)
    sand = Rectangle(width=5.5, height=1.1, stroke_width=0).set_fill("#c9a86a", 0.95)
    VGroup(sky, sea, sand).arrange(DOWN, buff=0).move_to(beach)
    sun = Circle(radius=0.32, stroke_width=0).set_fill(k.GOLD, 1).move_to(sky.get_corner(UP + RIGHT) + LEFT * 0.7 + DOWN * 0.5)
    umbrella = VGroup(Arc(radius=0.7, start_angle=0, angle=3.1416, color=k.PINK, stroke_width=4).set_fill(k.PINK, 0.8),
                      Line(UP * 0, DOWN * 1.0, color=k.INK, stroke_width=4))
    umbrella[1].next_to(umbrella[0], DOWN, buff=0)
    umbrella.move_to(sand.get_center() + LEFT * 1.4 + UP * 0.6)
    people = VGroup()
    for dx in (0.6, 1.3):
        people.add(VGroup(Circle(radius=0.14, stroke_width=0).set_fill(k.INK, 1),
                          Line(UP * 0, DOWN * 0.55, color=k.INK, stroke_width=5)).arrange(DOWN, buff=0)
                   .move_to(sand.get_center() + RIGHT * dx + UP * 0.35))
    sc.run(FadeIn(VGroup(beach, sky, sea, sand, sun, umbrella, people)), run_time=0.7)
    seen = VGroup(*[k.label(t, 30, k.INK, bold=True) for t in ("người", "ô che nắng", "ván lướt sóng")])
    seen.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.6 + UP * 0.3)
    st = k.label("mô hình liệt kê:", 24, k.MUTED).next_to(seen, UP, buff=0.35).align_to(seen, LEFT)
    sc.run(FadeIn(st), FadeIn(seen, lag_ratio=0.4), run_time=1.0)
    board = RoundedRectangle(corner_radius=0.25, width=0.4, height=1.4, color=k.PINK, stroke_width=4)
    board = DashedVMobject(board.rotate(0.4), num_dashes=18).move_to(sea.get_center() + RIGHT * 0.4)
    none = k.label("ảnh không có ván!", 24, k.PINK, bold=True).next_to(beach, DOWN, buff=0.2)
    sc.run(Create(board), seen[2].animate.set_color(k.PINK), FadeIn(none), run_time=0.6)
    why = k.label("biển thì hay có ván", 22, k.PINK).next_to(seen[2], DOWN, buff=0.15).align_to(seen, LEFT)
    sc.run(FadeIn(why), run_time=0.4)

    sc.wait_until(sc.at(7))
    blur = k.label("ảnh mờ: nói “ván” còn to hơn", 24, k.GREEN).next_to(why, DOWN, buff=0.3).align_to(seen, LEFT)
    sc.run(FadeIn(blur), seen[2].animate.scale(1.15), run_time=0.6)
    gone = strike(k, seen[2])
    sc.run(Create(gone), seen[2].animate.set_opacity(0.35), FadeOut(board), run_time=0.7)
    ok = k.label("trừ đi: ván biến mất", 26, k.GOLD, bold=True).next_to(blur, DOWN, buff=0.3).align_to(seen, LEFT)
    sc.run(FadeIn(ok), run_time=0.4)


def corruptions(sc, s, k):
    items = []
    base = _photo(k, 2.4, 1.8)
    # 1. thêm nhiễu
    a = base.copy()
    items.append(VGroup(a, _noise(k, a, n=8, seed=5), k.label("thêm nhiễu", 24, k.MUTED)))
    # 2. cắt mảnh rồi xáo
    b = base.copy()
    tiles = VGroup()
    cols = [k.ACCENT, k.GOLD, k.GREEN, k.PURPLE, k.PINK, k.ORANGE]
    order = [4, 0, 5, 2, 1, 3]
    for idx, o in enumerate(order):
        t = Rectangle(width=0.78, height=0.88, stroke_width=1, stroke_color="#0b0b0b").set_fill(cols[o], 0.7)
        t.move_to(b.get_corner(UP + LEFT) + RIGHT * (0.42 + 0.8 * (idx % 3)) + DOWN * (0.47 + 0.9 * (idx // 3)))
        tiles.add(t)
    items.append(VGroup(b, tiles, k.label("cắt mảnh rồi xáo", 24, k.MUTED)))
    # 3. xoá vật quan trọng
    c = base.copy()
    obj = Circle(radius=0.5, color=k.GOLD, stroke_width=4).set_fill(k.GOLD, 0.5).move_to(c)
    eraser = Rectangle(width=1.2, height=1.2, stroke_width=0).set_fill("#1b2030", 1).move_to(c)
    items.append(VGroup(c, obj, k.label("xoá vật quan trọng", 24, k.MUTED)))
    # 4. âm thanh
    d = base.copy()
    wave = VGroup(*[Line(UP * h, DOWN * h, color=k.ACCENT, stroke_width=6)
                    for h in (0.2, 0.5, 0.7, 0.35, 0.6, 0.25, 0.45)]).arrange(RIGHT, buff=0.18).move_to(d)
    items.append(VGroup(d, wave, k.label("âm thanh thành khoảng lặng", 24, k.MUTED)))
    for it in items:
        it[2].next_to(it[0], DOWN, buff=0.2)
    grid = VGroup(*items).arrange(RIGHT, buff=0.5)
    grid.scale_to_fit_width(min(k.FRAME_W, grid.width * 1.15)).move_to(UP * 0.5)
    for it in items[:3]:
        sc.run(FadeIn(it[0]), FadeIn(it[2]), FadeIn(it[1]), run_time=0.45)
    eraser.match_width(items[2][0]).stretch_to_fit_height(items[2][0].height * 0.8).move_to(items[2][0])
    sc.run(Create(cross(k, items[2][1])), run_time=0.4)
    sc.run(FadeIn(eraser), run_time=0.5)

    sc.wait_until(sc.at(1))
    lab = k.label("Audio-Aware Decoding · lab của thầy Lý", 28, k.GOLD, bold=True).to_edge(DOWN, buff=0.6)
    sc.run(FadeIn(items[3][0]), FadeIn(items[3][1]), FadeIn(items[3][2]), FadeIn(lab), run_time=0.6)
    flat = Line(LEFT * 1.0, RIGHT * 1.0, color=k.ACCENT, stroke_width=4).move_to(items[3][0])
    sc.run(Transform(items[3][1], flat), run_time=0.8)


def mti_forks(sc, s, k):
    tag = card(k, "MTI · Minimal Test-Time Intervention · cuối 2025", k.ACCENT, 26, 70).to_edge(UP, buff=0.4)
    sc.wait_until(sc.at(1))
    sc.run(FadeIn(tag), run_time=0.5)
    road = Line(LEFT * 6.2, RIGHT * 6.2, color=k.MUTED, stroke_width=5).shift(UP * 0.8)
    sc.run(Create(road), run_time=0.6)
    big = [(-3.2, "chọn trường nào"), (1.8, "cưới ai")]
    small = [(-5.0, "ăn sáng gì"), (-1.0, "mặc áo gì"), (4.4, "chiều đi đâu")]
    for x, t in small:
        br = Line([x, 0.8, 0], [x + 0.5, 1.3, 0], color=k.MUTED, stroke_width=2)
        lb = k.label(t, 20, k.MUTED).move_to([x + 0.4, 1.6, 0])
        sc.run(Create(br), FadeIn(lb), run_time=0.25)
    for x, t in big:
        br = Line([x, 0.8, 0], [x + 1.0, -0.4, 0], color=k.GOLD, stroke_width=6)
        lb = k.label(t, 26, k.GOLD, bold=True).move_to([x + 1.2, -0.75, 0])
        sc.run(Create(br), FadeIn(lb), run_time=0.4)

    sc.wait_until(sc.at(2))
    toks = VGroup()
    rnd = random.Random(7)
    hot = {3, 8}
    for i in range(12):
        c = k.GOLD if i in hot else k.MUTED
        toks.add(RoundedRectangle(corner_radius=0.08, width=0.75, height=0.5, color=c, stroke_width=3)
                 .set_fill(c, 0.35 if i in hot else 0.08))
    toks.arrange(RIGHT, buff=0.12).move_to(DOWN * 2.0)
    sc.run(FadeIn(toks, lag_ratio=0.05), run_time=0.7)
    for i in hot:
        flat = VGroup(*[Rectangle(width=0.12, height=0.28, stroke_width=0).set_fill(k.GOLD, 0.9)
                        for _ in range(4)]).arrange(RIGHT, buff=0.04, aligned_edge=DOWN).next_to(toks[i], DOWN, buff=0.12)
        sc.run(FadeIn(flat), run_time=0.3)
    ent = k.label("phân bố dàn đều = entropy cao", 24, k.GOLD).next_to(toks, DOWN, buff=0.65)
    sc.run(FadeIn(ent), run_time=0.4)

    sc.wait_until(sc.at(3))
    on = VGroup(*[k.label("CD", 20, k.GOLD, bold=True).next_to(toks[i], UP, buff=0.12) for i in hot])
    sc.run(FadeIn(on, shift=DOWN * 0.1), *[Indicate(toks[i], color=k.GOLD) for i in hot], run_time=0.7)


def kv_cache(sc, s, k):
    worry = k.label("Vẫn phải chạy bản hỏng từ đầu tới chỗ đó", 30, k.PINK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(worry), run_time=0.5)
    sc.wait_until(sc.at(1))
    full = VGroup(*[RoundedRectangle(corner_radius=0.08, width=0.6, height=0.45, color=k.PINK, stroke_width=3)
                    .set_fill(k.PINK, 0.2) for _ in range(10)]).arrange(RIGHT, buff=0.08).move_to(UP * 1.6)
    sc.run(FadeIn(full, lag_ratio=0.1), run_time=0.9)

    sc.wait_until(sc.at(2))
    sc.run(FadeOut(full), FadeOut(worry), run_time=0.4)
    head = k.label("KV cache", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.6)

    sc.wait_until(sc.at(3))
    def sent(words, colors):
        return VGroup(*[card(k, w, c, 26, 14, pad=0.35) for w, c in zip(words, colors)]).arrange(RIGHT, buff=0.12)
    a = sent(["Chào", "mọi", "người,", "mình", "là", "Đại", "Kim"], [k.ACCENT] * 7).move_to(UP * 1.3)
    b = sent(["Chào", "mọi", "người,", "mình", "là", "Tiểu", "Kim"], [k.GREEN] * 5 + [k.GOLD, k.GOLD]).move_to(DOWN * 0.1)
    sc.run(FadeIn(a), run_time=0.5)
    sc.run(FadeIn(b), run_time=0.5)
    copies = VGroup(*[Arrow(a[i].get_bottom(), b[i].get_top(), buff=0.06, color=k.GREEN, stroke_width=3,
                            max_tip_length_to_length_ratio=0.3) for i in range(5)])
    reuse = k.label("dùng lại, khỏi tính", 24, k.GREEN, bold=True).next_to(b[:5], DOWN, buff=0.25)
    sc.run(GrowArrow(copies[0]), *[GrowArrow(c) for c in copies[1:]], FadeIn(reuse), run_time=0.7)

    sc.wait_until(sc.at(4))
    rule = k.label("chỉ khi giống nhau từ đầu", 28, k.GOLD, bold=True).move_to(DOWN * 1.5)
    sc.run(FadeIn(rule), run_time=0.5)

    sc.wait_until(sc.at(5))
    c = sent(["Chào", "các", "bạn,", "mình", "là", "Đại", "Kim"], [k.ACCENT, k.PINK, k.PINK] + [k.PINK] * 4)
    c.move_to(DOWN * 2.5)
    sc.run(FadeIn(c), run_time=0.5)
    lost = k.label("đổi một chữ ở giữa: phần sau không dùng lại được", 24, k.PINK).next_to(c, DOWN, buff=0.3)
    sc.run(Create(cross(k, c[1:])), FadeIn(lost), run_time=0.6)


def output_error(sc, s, k):
    cache = VGroup(*[RoundedRectangle(corner_radius=0.08, width=0.55, height=0.45, color=k.ACCENT, stroke_width=3)
                     .set_fill(k.ACCENT, 0.25) for _ in range(9)]).arrange(RIGHT, buff=0.06)
    tail = VGroup(card(k, "Output", k.PINK, 24, 10, pad=0.3), card(k, "Error", k.PINK, 24, 10, pad=0.3)).arrange(RIGHT, buff=0.08)
    row = VGroup(cache, tail).arrange(RIGHT, buff=0.15).move_to(UP * 2.3)
    sc.run(FadeIn(cache), run_time=0.5)
    sc.run(FadeIn(tail, shift=LEFT * 0.3), run_time=0.6)

    sc.wait_until(sc.at(1))
    reuse = k.label("dùng lại cache", 24, k.ACCENT).next_to(cache, DOWN, buff=0.2)
    only = k.label("chỉ tính 2 token", 24, k.PINK, bold=True).next_to(tail, DOWN, buff=0.2)
    sc.run(FadeIn(reuse), FadeIn(only), run_time=0.6)

    sc.wait_until(sc.at(2))
    axis = Line(LEFT * 2.6, RIGHT * 2.6, color=k.MUTED, stroke_width=3).move_to(LEFT * 3.2 + DOWN * 2.3)
    ys = [62, 66, 70, 72, 69, 55]
    pts = [axis.get_left() + RIGHT * (0.4 + i * 0.9) + UP * ((y - 50) * 0.11) for i, y in enumerate(ys)]
    curve = VGroup(*[Line(pts[i], pts[i + 1], color=k.GOLD, stroke_width=5) for i in range(len(pts) - 1)])
    dots = VGroup(*[Dot(p, radius=0.08, color=k.GOLD) for p in pts])
    xlab = k.label("omega tăng dần", 22, k.MUTED).next_to(axis, DOWN, buff=0.15)
    l62 = k.label("62%", 26, k.INK, bold=True).next_to(pts[0], LEFT, buff=0.15)
    l72 = k.label("72%", 30, k.GOLD, bold=True).next_to(pts[3], UP, buff=0.2)
    sc.run(Create(axis), FadeIn(xlab), FadeIn(dots[0]), FadeIn(l62), run_time=0.6)
    sc.run(Create(curve[:3]), FadeIn(dots[1:4]), FadeIn(l72), run_time=0.9)
    sc.run(Create(curve[3:]), FadeIn(dots[4:]), run_time=0.6)
    broke = k.label("trừ quá tay", 22, k.PINK).next_to(pts[5], RIGHT, buff=0.15)
    sc.run(FadeIn(broke), run_time=0.3)

    sc.wait_until(sc.at(3))
    table = [("Output Error", True), ("Status Error", True), ("Output Correct", None), ("Monkey · Apple · !", False)]
    rows = VGroup()
    for text, good in table:
        color = k.GREEN if good else (k.MUTED if good is None else k.PINK)
        mark = check(k, 0.3, k.GREEN) if good else (k.label("kém hơn", 20, k.MUTED) if good is None else
                                                      cross(k, Square(side_length=0.25)))
        rows.add(VGroup(Text(text, font=k.MONO, font_size=26, color=color), mark).arrange(RIGHT, buff=0.3))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.6 + DOWN * 1.2)
    for r in rows:
        sc.run(FadeIn(r, shift=LEFT * 0.2), run_time=0.35)

    sc.wait_until(sc.at(4))
    sc.run(Indicate(rows[0], color=k.GOLD), Indicate(rows[1], color=k.GOLD), run_time=0.8)


def part1_table(sc, s, k):
    head = k.label("Phần 1 gói lại", 40, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.6)

    sc.wait_until(sc.at(1))
    left_h = k.label("Bản sai lấy từ đâu?", 30, k.ACCENT, bold=True)
    right_h = k.label("Trừ ở đâu?", 30, k.GOLD, bold=True)
    sources = ["mô hình nhỏ", "lớp nông", "lớp nông encoder ảnh", "câu thần chú làm ngu", "bỏ tài liệu (RAG)",
               "làm hỏng ảnh", "làm hỏng âm thanh", "thần chú + KV cache (MTI)"]
    left = VGroup(*[card(k, t, k.ACCENT, 22, 30, bold=False, pad=0.3) for t in sources]).arrange_in_grid(4, 2, buff=0.18)
    places = VGroup(*[card(k, t, k.GOLD, 26, 20, pad=0.35) for t in ("logit đầu ra", "representation ẩn", "trọng số attention")])
    places.arrange(DOWN, buff=0.3)
    lcol = VGroup(left_h, left).arrange(DOWN, buff=0.3).move_to(LEFT * 2.6 + UP * 0.1)
    rcol = VGroup(right_h, places).arrange(DOWN, buff=0.3).move_to(RIGHT * 3.9 + UP * 0.1)
    sc.run(FadeIn(left_h), FadeIn(right_h), run_time=0.5)
    sc.run(FadeIn(left, lag_ratio=0.12), run_time=1.4)

    sc.wait_until(sc.at(2))
    for p in places:
        sc.run(FadeIn(p, shift=LEFT * 0.2), run_time=0.35)
    acg = k.label("ACG: tốt hơn", 20, k.GREEN, bold=True).next_to(places[2], DOWN, buff=0.12)
    sc.run(FadeIn(acg), run_time=0.4)

    sc.wait_until(sc.at(3))
    open_q = k.label("tốt nhất ở đâu? còn đang nghiên cứu", 22, k.MUTED).next_to(acg, DOWN, buff=0.2)
    sc.run(FadeIn(open_q), run_time=0.4)

    sc.wait_until(sc.at(4))
    joke = card(k, "không cần huấn luyện: bí ý tưởng thì cứ chạy thử", k.GREEN, 26, 60).to_edge(DOWN, buff=0.3)
    joke.align_to(lcol, LEFT)
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


def gen_verify(sc, s, k):
    head = k.label("Phần 2 · Sinh rồi kiểm", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.8)

    sc.wait_until(sc.at(1))
    gen = card(k, "Sinh", k.ACCENT, 32, 10, min_w=2.2)
    check_box = card(k, "“Kiểm tra lại đi”", k.GOLD, 30, 20)
    regen = card(k, "Sinh lại", k.GREEN, 32, 10, min_w=2.2)
    row = VGroup(gen, check_box, regen).arrange(RIGHT, buff=1.0).move_to(UP * 0.4)
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


def two_intuitions(sc, s, k):
    q = k.label("Vì sao có thể có ích?", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(q), run_time=0.7)
    sc.wait_until(sc.at(1))
    left_h = k.label("1. Chê thì dễ hơn làm", 30, k.ACCENT, bold=True).move_to(LEFT * 3.4 + UP * 1.9)
    right_h = k.label("2. Sinh chữ không quay lại được", 30, k.PINK, bold=True).move_to(RIGHT * 3.1 + UP * 1.9)
    divider = Line(UP * 2.4, DOWN * 3.2, color=k.MUTED, stroke_width=2).set_opacity(0.4)
    sc.run(Create(divider), run_time=0.4)

    sc.wait_until(sc.at(2))
    sc.run(FadeIn(left_h), run_time=0.5)
    sc.wait_until(sc.at(3))
    book = RoundedRectangle(corner_radius=0.08, width=2.4, height=2.6, color=k.GOLD, stroke_width=4).set_fill(k.GOLD, 0.15)
    book.move_to(LEFT * 3.4 + UP * 0.1)
    spine = Line(book.get_corner(UP + LEFT) + RIGHT * 0.2, book.get_corner(DOWN + LEFT) + RIGHT * 0.2, color=k.GOLD, stroke_width=3)
    title = k.label("Kim Dung", 28, k.GOLD, bold=True).move_to(book).shift(RIGHT * 0.1)
    cant = k.label("không viết nổi", 24, k.PINK).next_to(book, DOWN, buff=0.25)
    can = k.label("nhưng biết là hay", 24, k.GREEN, bold=True).next_to(cant, DOWN, buff=0.12)
    sc.run(FadeIn(book), Create(spine), Write(title), run_time=0.7)
    sc.run(FadeIn(cant), run_time=0.4)
    sc.run(FadeIn(can), run_time=0.4)

    sc.wait_until(sc.at(4))
    sc.run(FadeIn(right_h), run_time=0.5)
    die = RoundedRectangle(corner_radius=0.15, width=0.9, height=0.9, color=k.INK, stroke_width=4).move_to(RIGHT * 1.3 + UP * 0.6)
    pips = VGroup(*[Dot(die.get_center() + d * 0.22, radius=0.07, color=k.INK) for d in (UP + LEFT, ORIGIN, DOWN + RIGHT)])
    sc.wait_until(sc.at(5))
    toks = VGroup(card(k, "sai", k.PINK, 26, 6, pad=0.3), *[card(k, "...", k.PINK, 26, 6, pad=0.3) for _ in range(3)])
    toks.arrange(RIGHT, buff=0.12).next_to(die, RIGHT, buff=0.4)
    sc.run(FadeIn(die), FadeIn(pips), run_time=0.4)
    sc.run(FadeIn(toks[0], scale=0.8), run_time=0.4)
    sc.run(FadeIn(toks[1:], lag_ratio=0.3), run_time=0.8)
    stuck = k.label("cãi cố tới hết câu", 24, k.PINK).next_to(toks, DOWN, buff=0.2)
    sc.run(FadeIn(stuck), run_time=0.4)

    sc.wait_until(sc.at(6))
    only = k.label("mô hình chỉ biết nối chữ", 24, k.MUTED).move_to(RIGHT * 3.3 + DOWN * 1.0)
    sc.run(FadeIn(only), run_time=0.4)
    sc.wait_until(sc.at(7))
    door = card(k, "sau “kiểm tra lại đi” thường là đoạn sửa sai", k.GREEN, 24, 30)
    door.move_to(RIGHT * 3.3 + DOWN * 2.2)
    sc.run(FadeIn(door, shift=UP * 0.2), run_time=0.6)


def critique_mcq(sc, s, k):
    head = k.label("Nhưng đó là trực giác của người", 34, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)

    sc.wait_until(sc.at(1))
    opts = VGroup(*[card(k, f"{c}. đáp án tự sinh", k.PURPLE, 24, 20, bold=False) for c in "ABCD"])
    opts.arrange(DOWN, buff=0.2).move_to(LEFT * 3.2 + DOWN * 0.3)
    ask = k.label("Cái nào đúng?", 28, k.ACCENT, bold=True).next_to(opts, UP, buff=0.3)
    sc.run(FadeIn(ask), FadeIn(opts, lag_ratio=0.2), run_time=0.9)
    pick = RoundedRectangle(corner_radius=0.15, width=opts[2].width + 0.2, height=opts[2].height + 0.15,
                            color=k.PINK, stroke_width=4).move_to(opts[2])
    sc.run(Create(pick), run_time=0.4)
    fail = k.label("chọn sai, trừ khi đúng sai chênh rất xa", 24, k.PINK).next_to(opts, DOWN, buff=0.3)
    sc.run(FadeIn(fail), run_time=0.4)

    sc.wait_until(sc.at(2))
    bubble_t = bubble(k, "paper biến việc chấm thành trắc nghiệm", k.GOLD, 22, 26).move_to(RIGHT * 3.4 + UP * 1.0)
    who = k.label("thầy Lý phản biện", 24, k.GOLD, bold=True).next_to(bubble_t, UP, buff=0.15)
    sc.run(FadeIn(who), FadeIn(bubble_t, shift=LEFT * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    two = VGroup(card(k, "không biết chấm?", k.MUTED, 26, 20), card(k, "không biết làm trắc nghiệm?", k.GOLD, 26, 30))
    two.arrange(DOWN, buff=0.3).move_to(RIGHT * 3.4 + DOWN * 1.0)
    sc.run(FadeIn(two[0]), run_time=0.4)
    sc.run(FadeIn(two[1]), run_time=0.4)

    sc.wait_until(sc.at(4))
    open_q = k.label("còn bỏ ngỏ", 30, k.INK, bold=True).to_edge(DOWN, buff=0.4).shift(RIGHT * 3.4)
    sc.run(FadeIn(open_q), run_time=0.5)


def internal_external(sc, s, k):
    head = k.label("Thực nghiệm", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    tag = card(k, "CorrectBench · 10/2025 · nhiều mô hình, nhiều benchmark", k.MUTED, 24, 70).next_to(head, DOWN, buff=0.25)
    sc.run(FadeIn(tag), run_time=0.5)

    def panel(values, color, title, x):
        base = Line(LEFT * 2.6, RIGHT * 2.6, color=k.MUTED, stroke_width=3).move_to([x, -0.6, 0])
        bars = VGroup()
        for i, v in enumerate(values):
            bar = Rectangle(width=0.36, height=abs(v), stroke_width=0).set_fill(color if v >= 0 else k.PINK, 0.85)
            bar.move_to(base.get_left() + RIGHT * (0.45 + i * 0.52), aligned_edge=DOWN if v >= 0 else UP)
            bar.grow_from = DOWN if v >= 0 else UP
            bars.add(bar)
        zero = k.label("0", 20, k.MUTED).next_to(base, LEFT, buff=0.1)
        t = k.label(title, 28, color, bold=True).next_to(base, UP, buff=2.0)
        return VGroup(base, zero, bars, t)

    sc.wait_until(sc.at(2))
    inner = panel([0.9, -0.7, 0.4, -1.1, 1.2, 0.2, -0.5, 0.7, -0.9], k.ACCENT, "tự phản tư", -3.4)
    sc.run(Create(inner[0]), FadeIn(inner[1]), FadeIn(inner[3]), run_time=0.4)
    sc.run(*[GrowFromEdge(b, b.grow_from) for b in inner[2]], run_time=1.0)
    shaky = k.label("không ổn định", 26, k.PINK, bold=True).next_to(inner[0], DOWN, buff=1.4)
    sc.run(FadeIn(shaky), run_time=0.4)

    sc.wait_until(sc.at(3))
    outer = panel([1.0, 1.3, 0.8, 1.5, -0.2, 1.1, 0.9, 1.4, 0.6], k.GREEN, "phản hồi bên ngoài", 3.4)
    sc.run(Create(outer[0]), FadeIn(outer[1]), FadeIn(outer[3]), run_time=0.4)
    sc.run(*[GrowFromEdge(b, b.grow_from) for b in outer[2]], run_time=1.0)
    steady = k.label("chạy code, tìm thêm: ổn định", 26, k.GREEN, bold=True).next_to(outer[0], DOWN, buff=1.4)
    sc.run(FadeIn(steady), run_time=0.4)


def refinebench(sc, s, k):
    head = k.label("RefineBench · một tháng sau", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)

    sc.wait_until(sc.at(1))
    axes, at = _chart(k, 6.4, 3.6, "lượt sửa 1 đến 5", "điểm")
    axes.move_to(LEFT * 1.6 + DOWN * 0.4)
    sc.run(FadeIn(axes), run_time=0.5)
    xs = [0.05, 0.27, 0.5, 0.73, 0.95]
    lines = [([0.30, 0.31, 0.32, 0.32, 0.33], k.MUTED, "tự sửa"),
             ([0.30, 0.45, 0.55, 0.60, 0.63], k.ACCENT, "xem nửa checklist"),
             ([0.30, 0.55, 0.66, 0.72, 0.75], k.ORANGE, "biết sẽ bị chấm gì"),
             ([0.30, 0.70, 0.85, 0.92, 0.95], k.GOLD, "phản hồi đầy đủ")]
    drawn = []
    for i, (ys, color, name) in enumerate(lines):
        poly = _polyline(k, at, list(zip(xs, ys)), color)
        tag = k.label(name, 22, color, bold=True, max_width=4.2).next_to(poly[-1].get_end(), RIGHT, buff=0.2)
        drawn.append((poly, tag))
    models = k.label("Claude 3.5 Sonnet · Gemini 1.5 Pro", 24, k.MUTED).next_to(head, DOWN, buff=0.2)
    sc.run(Create(drawn[0][0]), FadeIn(drawn[0][1]), FadeIn(models), run_time=1.0)
    small = k.label("chỉ nhích chút xíu", 24, k.MUTED).next_to(drawn[0][0], DOWN, buff=0.2)
    sc.run(FadeIn(small), run_time=0.4)

    sc.wait_until(sc.at(2))
    for poly, tag in drawn[1:]:
        sc.run(Create(poly), FadeIn(tag), run_time=0.7)

    sc.wait_until(sc.at(3))
    msg = k.label("phản hồi từ bên ngoài mới làm mô hình tiến bộ", 30, k.GOLD, bold=True).to_edge(DOWN, buff=0.3)
    sc.run(FadeIn(msg, shift=UP * 0.2), run_time=0.6)


def _cell(k, text, sub, color, w=4.4, h=1.6):
    box = RoundedRectangle(corner_radius=0.2, width=w, height=h, color=color, stroke_width=4).set_fill(color, 0.12)
    t = k.label(text, 34, k.INK, bold=True)
    st = k.label(sub, 24, color)
    VGroup(t, st).arrange(DOWN, buff=0.15).move_to(box)
    return VGroup(box, t, st)


def four_cells(sc, s, k):
    head = k.label("Bốn trường hợp khi tự sửa", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.7)
    names = [("Đúng, vẫn Đúng", "chỉ tốn tính toán", k.MUTED), ("Sai, thành Đúng", "mong nhất", k.GREEN),
             ("Sai, vẫn Sai", "chỉ tốn tính toán", k.MUTED), ("Đúng, thành Sai", "nghĩ nhiều quá hoá sai", k.PINK)]
    cells = VGroup(*[_cell(k, a, b, c) for a, b, c in names]).arrange_in_grid(2, 2, buff=0.35).move_to(DOWN * 0.5)
    sc.wait_until(sc.at(1))
    sc.run(FadeIn(cells[1], scale=0.9), run_time=0.6)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(cells[0], scale=0.9), run_time=0.5)
    sc.run(FadeIn(cells[2], scale=0.9), run_time=0.5)
    sc.wait_until(sc.at(3))
    sc.run(FadeIn(cells[3], scale=0.9), run_time=0.6)
    sc.run(Indicate(cells[3], color=k.PINK, scale_factor=1.05), run_time=0.8)


def _gauge(k, name, sub, value, color, r=1.6):
    track = Arc(radius=r, start_angle=3.1416, angle=-3.1416, color=k.MUTED, stroke_width=14).set_opacity(0.35)
    fill = Arc(radius=r, start_angle=3.1416, angle=-3.1416 * value, color=color, stroke_width=14)
    num = k.label(f"{value:.1f}".replace(".", ","), 44, k.INK, bold=True).move_to(track.get_center() + DOWN * 0.1)
    t = k.label(name, 30, color, bold=True).next_to(track, DOWN, buff=0.15)
    st = k.label(sub, 22, k.MUTED).next_to(t, DOWN, buff=0.1)
    return VGroup(track, fill, num, t, st)


def cl_cs(sc, s, k):
    head = k.label("Hai con số", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    cl = _gauge(k, "CL · độ tự tin", "đang đúng, giữ được đúng", 0.9, k.GREEN).move_to(LEFT * 3.2 + UP * 0.6)
    sc.run(Create(cl[0]), Create(cl[1]), FadeIn(cl[2:]), run_time=0.9)
    sc.wait_until(sc.at(2))
    cs = _gauge(k, "CS · độ tiếp thu", "đang sai, sửa được thành đúng", 0.2, k.GOLD).move_to(RIGHT * 3.2 + UP * 0.6)
    sc.run(Create(cs[0]), Create(cs[1]), FadeIn(cs[2:]), run_time=0.9)
    sc.wait_until(sc.at(3))
    formula = Text("Acc2 = Acc1 × CL + (1 − Acc1) × CS", font=k.MONO, font_size=34, color=k.INK).to_edge(DOWN, buff=0.6)
    box = RoundedRectangle(corner_radius=0.15, width=formula.width + 0.6, height=formula.height + 0.4,
                           color=k.ACCENT, stroke_width=3).move_to(formula)
    sc.run(Write(formula), Create(box), run_time=1.2)


def personalities(sc, s, k):
    axes, at = _chart(k, 6.0, 4.2, "CS: hay sửa câu sai", "CL: giữ câu đúng")
    axes.move_to(LEFT * 1.4 + DOWN * 0.2)
    sc.run(FadeIn(axes), run_time=0.5)
    sc.wait_until(sc.at(0))
    high = Line(at(0, 0.8), at(1, 0.8), color=k.GREEN, stroke_width=2).set_opacity(0.5)
    sc.run(Create(high), run_time=0.5)
    sc.wait_until(sc.at(1))
    low = Rectangle(width=6.0 * 0.3, height=4.2, stroke_width=0).set_fill(k.PINK, 0.08).move_to(at(0.15, 0.5))
    stub = k.label("cứng đầu", 26, k.PINK, bold=True).move_to(at(0.15, 0.35))
    sc.run(FadeIn(low), FadeIn(stub), run_time=0.6)

    sc.wait_until(sc.at(2))
    pts = [("DeepSeek", 0.08, 0.93, k.PINK, RIGHT), ("Mixtral", 0.16, 0.84, k.PINK, RIGHT),
           ("Llama 3 70B", 0.62, 0.74, k.ACCENT, UP), ("GPT-3.5", 0.48, 0.66, k.ACCENT, LEFT),
           ("GPT-4", 0.75, 0.62, k.ACCENT, DOWN)]
    for name, x, y, c, side in pts:
        d = Dot(at(x, y), radius=0.12, color=c)
        t = k.label(name, 22, c, bold=True).next_to(d, side, buff=0.12)
        sc.run(FadeIn(d, scale=0.5), FadeIn(t), run_time=0.35)

    sc.wait_until(sc.at(3))
    trade = Arrow(at(0.3, 0.55), at(0.95, 0.3), buff=0.1, color=k.GOLD, stroke_width=5)
    note = k.label("kéo nhau: dễ nghe góp ý thì khó giữ câu đúng", 26, k.GOLD, bold=True).to_edge(DOWN, buff=0.3)
    sc.run(GrowArrow(trade), FadeIn(note), run_time=0.8)


def three_prompts(sc, s, k):
    head = k.label("Câu chèn đổi tính cách", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)
    sc.wait_until(sc.at(1))
    tag = card(k, "Llama 3 · ba bộ dữ liệu", k.MUTED, 24, 30).next_to(head, DOWN, buff=0.2)
    sc.run(FadeIn(tag), run_time=0.4)
    prompts = [("Làm lại lần nữa.", "trung tính", k.MUTED, 0.85, 0.30),
               ("Chắc bạn đúng rồi, cho mình đáp án lần nữa.", "khen", k.GREEN, 0.95, 0.15),
               ("Bạn chắc chưa? Nghĩ lại đi.", "hỏi vặn", k.PINK, 0.70, 0.50)]
    rows = VGroup()
    for text, kind, color, cl, cs in prompts:
        b = bubble(k, text, color, 26, 24)
        kd = k.label(kind, 22, color, bold=True)
        cl_bar = Rectangle(width=2.2 * cl, height=0.26, stroke_width=0).set_fill(k.GREEN, 0.85)
        cs_bar = Rectangle(width=2.2 * cs, height=0.26, stroke_width=0).set_fill(k.GOLD, 0.85)
        bars = VGroup(VGroup(k.label("CL", 20, k.GREEN, bold=True), cl_bar).arrange(RIGHT, buff=0.15),
                      VGroup(k.label("CS", 20, k.GOLD, bold=True), cs_bar).arrange(RIGHT, buff=0.15))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        rows.add(VGroup(kd, b, bars))
    for r in rows:
        r[0].move_to(LEFT * 5.6)
        r[1].next_to(r[0], RIGHT, buff=0.3)
        r[2].move_to(RIGHT * 4.6, aligned_edge=LEFT).shift(LEFT * 1.4)
    rows.arrange(DOWN, buff=0.35).move_to(DOWN * 0.6)
    for r in rows:
        r[0].align_to(rows, LEFT)
        r[1].next_to(r[0], RIGHT, buff=0.3)
        r[2].move_to([3.6, r[1].get_center()[1], 0], aligned_edge=LEFT)
    sc.run(FadeIn(rows[0]), run_time=0.5)
    sc.wait_until(sc.at(2))
    sc.run(FadeIn(rows[1]), run_time=0.5)
    sc.wait_until(sc.at(3))
    sc.run(FadeIn(rows[2]), run_time=0.5)
    sc.wait_until(sc.at(4))
    straw = k.label("chỉ là “người rơm” do chương trình chèn vào", 26, k.GOLD, bold=True).to_edge(DOWN, buff=0.35)
    sc.run(FadeIn(straw), run_time=0.5)
    sc.wait_until(sc.at(6))
    tip = k.label("tự tin quá thì hỏi vặn, thiếu tự tin thì khen", 26, k.ACCENT, bold=True).to_edge(DOWN, buff=0.35)
    sc.run(FadeOut(straw), FadeIn(tip), run_time=0.5)


def vote_vs_verify(sc, s, k):
    head = k.label("Kiểm tra có đáng tiền không?", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)
    sc.wait_until(sc.at(1))
    votes = VGroup(*[card(k, t, k.ACCENT, 24, 4, pad=0.3) for t in "AABAC"]).arrange(RIGHT, buff=0.1)
    votes.next_to(head, DOWN, buff=0.3)
    win = k.label("bỏ phiếu đa số: A", 24, k.ACCENT, bold=True).next_to(votes, RIGHT, buff=0.3)
    sc.run(FadeIn(votes, lag_ratio=0.2), FadeIn(win), run_time=0.8)

    sc.wait_until(sc.at(2))
    axes, at = _chart(k, 5.2, 3.2, "số lần lấy mẫu (2 đến 128)", "điểm")
    axes.move_to(LEFT * 2.8 + DOWN * 0.9)
    vote = _polyline(k, at, [(0, 0.2), (0.25, 0.42), (0.5, 0.58), (0.75, 0.66), (1, 0.7)], k.MUTED)
    ver = _polyline(k, at, [(0, 0.35), (0.25, 0.6), (0.5, 0.74), (0.75, 0.82), (1, 0.86)], k.PINK)
    tv = k.label("bỏ phiếu", 22, k.MUTED, bold=True).next_to(vote[-1].get_end(), RIGHT, buff=0.15)
    tr = k.label("+ 32 lần kiểm tra", 22, k.PINK, bold=True).next_to(ver[-1].get_end(), RIGHT, buff=0.15)
    sc.run(FadeIn(axes), run_time=0.4)
    sc.run(Create(vote), FadeIn(tv), run_time=0.7)
    sc.run(Create(ver), FadeIn(tr), run_time=0.7)

    sc.wait_until(sc.at(3))
    good = k.label("trông như kiểm tra tốt hơn hẳn", 24, k.PINK).next_to(axes, UP, buff=0.05).shift(RIGHT * 1.0)
    sc.run(FadeIn(good), run_time=0.4)

    sc.wait_until(sc.at(4))
    xlab = axes[2]
    new_x = k.label("lượng tính toán", 22, k.GOLD, bold=True).move_to(xlab)
    sc.run(Transform(xlab, new_x), FadeOut(good), Indicate(axes[0], color=k.GOLD), run_time=0.8)

    sc.wait_until(sc.at(5))
    vote2 = _polyline(k, at, [(0, 0.2), (0.15, 0.48), (0.3, 0.62), (0.55, 0.69), (1, 0.71)], k.MUTED)
    ver2 = _polyline(k, at, [(0.12, 0.15), (0.3, 0.35), (0.55, 0.55), (0.8, 0.68), (1, 0.75)], k.PINK)
    sc.run(Transform(vote, vote2), Transform(ver, ver2),
           tv.animate.next_to(vote2[-1].get_end(), RIGHT, buff=0.15).shift(DOWN * 0.2),
           tr.animate.next_to(ver2[-1].get_end(), RIGHT, buff=0.15).shift(UP * 0.2), run_time=1.2)
    budget = Rectangle(width=5.2 * 0.5, height=3.2, stroke_width=0).set_fill(k.ACCENT, 0.08).move_to(at(0.25, 0.5))
    btxt = k.label("ngân sách có hạn: bỏ phiếu thắng", 22, k.ACCENT, bold=True, width_chars=16)
    btxt.move_to(at(0.25, 0.88))
    sc.run(FadeIn(budget), FadeIn(btxt), run_time=0.6)

    sc.wait_until(sc.at(6))
    cost = card(k, "hơn 3,8% cần hơn 100 lần tính toán", k.PINK, 24, 22).move_to(RIGHT * 4.6 + DOWN * 1.2)
    sc.run(FadeIn(cost, shift=LEFT * 0.2), run_time=0.6)
    sc.wait_until(sc.at(7))
    lux = card(k, "kiểm tra = món xa xỉ", k.GOLD, 28, 24).next_to(cost, DOWN, buff=0.3)
    sc.run(GrowFromCenter(lux), run_time=0.6)
    sc.wait_until(sc.at(8))
    rule = k.label("workflow mới phải so với bỏ phiếu đa số, cùng lượng tính toán", 26, k.GOLD, bold=True)
    rule.to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(rule, shift=UP * 0.2), run_time=0.6)



# ── PHẦN 3: HUẤN LUYỆN ĐỂ TỰ SỬA ─────────────────────────────
def drop_prompt(sc, s, k):
    head = k.label("Phần 3 · Huấn luyện để tự sửa", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.8)
    gen = card(k, "Sinh", k.ACCENT, 32, 10, min_w=2.2)
    chk = card(k, "“Kiểm tra lại đi”", k.GOLD, 30, 20)
    regen = card(k, "Sinh lại", k.GREEN, 32, 10, min_w=2.2)
    row = VGroup(gen, chk, regen).arrange(RIGHT, buff=1.0).move_to(UP * 0.8)
    a1, a2 = arrow(k, gen, chk), arrow(k, chk, regen)
    always = k.label("lần nào cũng ép nghĩ thêm, dù đúng hay sai", 26, k.PINK).next_to(row, DOWN, buff=0.4)
    sc.run(FadeIn(row), GrowArrow(a1), GrowArrow(a2), run_time=0.7)
    sc.run(FadeIn(always), run_time=0.5)

    sc.wait_until(sc.at(1))
    sc.run(FadeOut(VGroup(row, a1, a2, always)), run_time=0.4)
    model = _model(k, "LLM", w=3.0, h=1.6).move_to(DOWN * 0.3)
    loop = CurvedArrow(model.get_corner(UP + RIGHT) + LEFT * 0.4, model.get_corner(UP + LEFT) + RIGHT * 0.4,
                       angle=2.2, color=k.GOLD, stroke_width=5)
    inner = k.label("tự biết lúc nào cần sửa", 28, k.GOLD, bold=True).next_to(loop, UP, buff=0.15)
    sc.run(GrowFromCenter(model), run_time=0.5)
    sc.run(Create(loop), FadeIn(inner), run_time=0.8)

    sc.wait_until(sc.at(2))
    gains = VGroup(card(k, "đỡ tốn", k.GREEN, 28, 10), card(k, "thông minh hơn", k.GREEN, 28, 16)).arrange(RIGHT, buff=0.5)
    gains.to_edge(DOWN, buff=0.6)
    sc.run(FadeIn(gains, shift=UP * 0.2), run_time=0.6)


def hillary(sc, s, k):
    naive = k.label("dạy thêm kiến thức thì tự sửa giỏi hơn?", 30, k.MUTED, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(naive), run_time=0.5)
    sc.wait_until(sc.at(1))
    sc.run(Create(cross(k, naive)), run_time=0.5)

    sc.wait_until(sc.at(2))
    q1 = bubble(k, "Kể tên một chính trị gia sinh ở New York.", k.ACCENT, 24, 26).move_to(LEFT * 3.4 + UP * 1.3)
    sc.run(FadeIn(q1, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(3))
    a1 = bubble(k, "Hillary Clinton.", k.PURPLE, 20, 28).next_to(q1, DOWN, buff=0.3).align_to(q1, RIGHT)
    sc.run(FadeIn(a1, shift=UP * 0.2), run_time=0.5)
    sc.run(Create(cross(k, a1)), run_time=0.4)

    sc.wait_until(sc.at(4))
    q2 = bubble(k, "Hillary Clinton sinh ở đâu?", k.ACCENT, 24, 26).move_to(RIGHT * 3.4 + UP * 1.3)
    sc.run(FadeIn(q2, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(5))
    a2 = bubble(k, "Chicago.", k.PURPLE, 20, 28).next_to(q2, DOWN, buff=0.3).align_to(q2, RIGHT)
    tick = check(k, 0.45).next_to(a2, RIGHT, buff=0.2)
    sc.run(FadeIn(a2, shift=UP * 0.2), Create(tick), run_time=0.6)

    sc.wait_until(sc.at(6))
    mid = k.label("có kiến thức, mà không giật mình", 30, k.GOLD, bold=True).move_to(DOWN * 1.0)
    sc.run(FadeIn(mid), run_time=0.5)

    sc.wait_until(sc.at(7))
    vec = Arrow(LEFT * 1.6, RIGHT * 1.6, buff=0, color=k.GOLD, stroke_width=8).move_to(DOWN * 2.1 + LEFT * 2.6)
    vt = k.label("steering vector “tự sửa”", 24, k.GOLD).next_to(vec, DOWN, buff=0.15)
    eff = k.label("cộng vào: sửa cả lúc không cần", 24, k.PINK).move_to(DOWN * 2.1 + RIGHT * 3.0)
    sc.run(GrowArrow(vec), FadeIn(vt), run_time=0.6)
    sc.run(FadeIn(eff), run_time=0.4)

    sc.wait_until(sc.at(8))
    skill = card(k, "tự sửa là một kỹ năng riêng", k.GREEN, 28, 30).to_edge(DOWN, buff=0.2)
    sc.run(FadeOut(VGroup(vec, vt, eff)), FadeIn(skill, shift=UP * 0.2), run_time=0.6)


def _token(k, text, color, size=26):
    return card(k, text, color, size, 14, pad=0.35)


def revise(sc, s, k):
    head = k.label("ReVISE", 44, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    split = VGroup(card(k, "phát hiện", k.PINK, 28, 12), card(k, "sửa", k.GREEN, 28, 8)).arrange(RIGHT, buff=0.9)
    split.next_to(head, DOWN, buff=0.4)
    apart = k.label("tách ra học thì dễ hơn", 24, k.MUTED).next_to(split, DOWN, buff=0.15)
    sc.run(FadeIn(split), FadeIn(apart), run_time=0.6)

    sc.wait_until(sc.at(2))
    s1 = k.label("Bước 1", 28, k.PINK, bold=True).move_to(LEFT * 5.8 + DOWN * 0.4)
    wrong = _token(k, "đáp án sai", k.PINK)
    refine = _token(k, "[Refine]", k.GOLD)
    right = _token(k, "đáp án đúng", k.GREEN)
    eos = _token(k, "[Kết thúc]", k.MUTED)
    r1 = VGroup(wrong, refine).arrange(RIGHT, buff=0.6).next_to(s1, RIGHT, buff=0.4)
    r2 = VGroup(right, eos).arrange(RIGHT, buff=0.6).next_to(r1, RIGHT, buff=0.6)
    sc.run(FadeIn(s1), FadeIn(wrong), run_time=0.4)
    sc.run(GrowArrow(arrow(k, wrong, refine)), FadeIn(refine), run_time=0.5)
    sc.run(FadeIn(right), run_time=0.3)
    sc.run(GrowArrow(arrow(k, right, eos)), FadeIn(eos), run_time=0.5)

    sc.wait_until(sc.at(3))
    s2 = k.label("Bước 2", 28, k.GREEN, bold=True).move_to(LEFT * 5.8 + DOWN * 2.2)
    w2, rf2, ok2 = _token(k, "đáp án sai", k.PINK), _token(k, "[Refine]", k.GOLD), _token(k, "đáp án đúng", k.GREEN)
    VGroup(w2, rf2).arrange(RIGHT, buff=0.12).next_to(s2, RIGHT, buff=0.5)
    ok2.next_to(rf2, RIGHT, buff=0.8)
    sc.run(FadeIn(s2), FadeIn(w2), FadeIn(rf2), run_time=0.5)
    sc.run(GrowArrow(arrow(k, rf2, ok2, color=k.GREEN)), FadeIn(ok2), run_time=0.6)


def new_errors(sc, s, k):
    head = k.label("Cái bẫy (paper 2024)", 38, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)
    sc.wait_until(sc.at(1))
    before = _model(k, "mô hình cũ", k.PURPLE, w=3.4, h=1.0).move_to(LEFT * 4.2 + UP * 0.6)
    err_g = _token(k, "lỗi xanh", k.GREEN).next_to(before, DOWN, buff=0.5)
    taught = k.label("được dạy sửa", 22, k.GREEN).next_to(err_g, DOWN, buff=0.15)
    sc.run(FadeIn(before), run_time=0.4)
    sc.run(FadeIn(err_g), FadeIn(taught), run_time=0.5)

    sc.wait_until(sc.at(2))
    after = _model(k, "mô hình mới", k.ORANGE, w=3.4, h=1.0).move_to(RIGHT * 3.6 + UP * 0.6)
    change = Arrow(before.get_right(), after.get_left(), buff=0.2, color=k.GOLD, stroke_width=5)
    ct = k.label("tham số đổi", 24, k.GOLD).next_to(change, UP, buff=0.15)
    sc.run(GrowArrow(change), FadeIn(ct), FadeIn(after), run_time=0.8)

    sc.wait_until(sc.at(3))
    err_r = _token(k, "lỗi đỏ", k.PINK).next_to(after, DOWN, buff=0.5)
    never = k.label("chưa từng được dạy sửa", 22, k.PINK).next_to(err_r, DOWN, buff=0.15)
    sc.run(FadeIn(err_r, scale=0.8), FadeIn(never), run_time=0.6)
    exam = k.label("như luyện đề cũ cho một đề thi đã đổi", 26, k.MUTED).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(exam), run_time=0.5)


def rlvr(sc, s, k):
    head = k.label("Học tăng cường", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    inp = _token(k, "đầu vào", k.ACCENT).move_to(LEFT * 5.4 + UP * 0.6)
    think = VGroup(*[Line(LEFT * 0.0, RIGHT * w, color=k.MUTED, stroke_width=4) for w in (3.6, 3.0, 3.4, 2.4)])
    think.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(inp, RIGHT, buff=0.6)
    free = k.label("muốn nghĩ gì thì nghĩ", 22, k.MUTED).next_to(think, UP, buff=0.15)
    ans = _token(k, "đáp án", k.PURPLE).next_to(think, RIGHT, buff=0.6)
    sc.run(FadeIn(inp), run_time=0.3)
    sc.run(Create(think, lag_ratio=0.3), FadeIn(free), run_time=1.0)
    sc.run(FadeIn(ans), run_time=0.3)
    plus = k.label("+1", 36, k.GREEN, bold=True).next_to(ans, RIGHT, buff=0.4).shift(UP * 0.3)
    zero = k.label("0", 36, k.PINK, bold=True).next_to(ans, RIGHT, buff=0.4).shift(DOWN * 0.4)
    sc.run(FadeIn(plus), FadeIn(zero), run_time=0.5)

    sc.wait_until(sc.at(2))
    name = card(k, "RLVR: phần thưởng kiểm chứng được (toán, code)", k.GOLD, 26, 60).move_to(DOWN * 1.9)
    sc.run(FadeIn(name, shift=UP * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    aha = bubble(k, "Khoan, chỗ này hình như sai...", k.GOLD, 34, 26).next_to(think, DOWN, buff=0.35)
    sc.run(FadeIn(aha, scale=0.9), run_time=0.6)
    nobody = k.label("không ai dạy, tự nó biết kiểm lại", 26, k.GREEN, bold=True).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(nobody), run_time=0.5)


def _student(k, color, label):
    head = Circle(radius=0.28, color=color, stroke_width=4).set_fill(color, 0.25)
    body = RoundedRectangle(corner_radius=0.25, width=0.9, height=0.6, color=color, stroke_width=4).set_fill(color, 0.15)
    body.next_to(head, DOWN, buff=0.05)
    t = k.label(label, 20, color, bold=True).next_to(body, DOWN, buff=0.1)
    return VGroup(head, body, t)


def err_first(sc, s, k):
    q = k.label("Sao phải sai rồi mới đúng?", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(q), run_time=0.5)
    sc.wait_until(sc.at(1))
    senior = _student(k, k.ORANGE, "đàn anh").move_to(LEFT * 5.0 + UP * 0.6)
    junior = _student(k, k.ACCENT, "đàn em").move_to(LEFT * 2.6 + UP * 0.6)
    scold = bubble(k, "Sao lại mắc lỗi này?", k.ORANGE, 20, 22).next_to(VGroup(senior, junior), UP, buff=0.2)
    sc.run(FadeIn(senior), FadeIn(junior), run_time=0.5)
    sc.run(FadeIn(scold), run_time=0.5)
    loop = CurvedArrow(junior.get_bottom() + DOWN * 0.1, senior.get_bottom() + DOWN * 0.1, angle=-1.2,
                       color=k.MUTED, stroke_width=3)
    year = k.label("năm sau, lại mắng đàn em mới", 22, k.MUTED).next_to(loop, DOWN, buff=0.15)
    sc.run(Create(loop), FadeIn(year), run_time=0.6)

    sc.wait_until(sc.at(2))
    mine = k.label("có lỗi phải tự giẫm một lần", 26, k.GOLD, bold=True).move_to(LEFT * 3.8 + DOWN * 2.4)
    sc.run(FadeIn(mine), run_time=0.5)

    sc.wait_until(sc.at(3))
    axes, at = _chart(k, 3.6, 2.6, "token của mô hình", "thời gian người giải")
    axes.move_to(RIGHT * 3.6 + UP * 0.2)
    rnd = random.Random(11)
    dots = VGroup(*[Dot(at(x, min(0.95, max(0.05, x * 0.85 + rnd.uniform(-0.1, 0.1)))), radius=0.07, color=k.ACCENT)
                    for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
    tag = k.label("“Cost of Thinking” · MIT News", 22, k.MUTED).next_to(axes, UP, buff=0.3)
    sc.run(FadeIn(axes), FadeIn(tag), run_time=0.5)
    sc.run(FadeIn(dots, lag_ratio=0.1), run_time=0.8)

    sc.wait_until(sc.at(4))
    both = k.label("carbon hay silicon, nghĩ đều tốn công", 24, k.GOLD, bold=True).next_to(axes, DOWN, buff=0.7)
    sc.run(FadeIn(both), run_time=0.5)
    sc.wait_until(sc.at(5))
    crit = k.label("token so với thời gian?", 24, k.PINK).next_to(both, DOWN, buff=0.2)
    sc.run(FadeIn(crit), run_time=0.4)
    sc.wait_until(sc.at(6))
    better = k.label("nên so với số chữ viết nháp", 24, k.GREEN).next_to(crit, DOWN, buff=0.15)
    sc.run(FadeIn(better), run_time=0.4)


def k_power(sc, s, k):
    tag = card(k, "3 paper · 02/2025", k.MUTED, 24, 20).to_edge(UP, buff=0.4)
    sc.run(FadeIn(tag), run_time=0.4)
    sc.wait_until(sc.at(1))
    steps = VGroup(*[RoundedRectangle(corner_radius=0.1, width=1.2, height=0.8, color=k.ACCENT, stroke_width=3)
                     .set_fill(k.ACCENT, 0.12) for _ in range(5)]).arrange(RIGHT, buff=0.35).move_to(UP * 1.3)
    kk = VGroup(*[k.label("K", 30, k.INK, bold=True).move_to(b) for b in steps])
    links = VGroup(*[arrow(k, steps[i], steps[i + 1]) for i in range(4)])
    under = k.label("T + 1 bước, mỗi bước K biến thể", 24, k.MUTED).next_to(steps, DOWN, buff=0.2)
    sc.run(FadeIn(steps, lag_ratio=0.15), FadeIn(kk, lag_ratio=0.15), run_time=0.8)
    sc.run(*[GrowArrow(l) for l in links], FadeIn(under), run_time=0.5)

    sc.wait_until(sc.at(2))
    step_way = VGroup(k.label("học từng bước", 28, k.GREEN, bold=True),
                      Text("K × (T + 1)", font=k.MONO, font_size=40, color=k.GREEN)).arrange(DOWN, buff=0.2)
    step_way.move_to(LEFT * 3.4 + DOWN * 1.6)
    sc.run(FadeIn(step_way, shift=UP * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    direct = VGroup(k.label("học thẳng ra đáp án", 28, k.PINK, bold=True),
                    Text("K ^ (T + 1)", font=k.MONO, font_size=40, color=k.PINK)).arrange(DOWN, buff=0.2)
    direct.move_to(RIGHT * 3.4 + DOWN * 1.6)
    expo = k.label("tăng theo hàm mũ", 24, k.PINK).next_to(direct, DOWN, buff=0.15)
    sc.run(FadeIn(direct, shift=UP * 0.2), run_time=0.6)
    sc.run(FadeIn(expo), run_time=0.4)


def parity(sc, s, k):
    head = k.label("Parity check", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.wait_until(sc.at(0))
    sc.run(Write(head), run_time=0.6)
    sc.wait_until(sc.at(1))
    bits = [0, 1, 1, 0, 1, 0]
    cells = VGroup(*[card(k, str(b), k.ACCENT, 34, 3, pad=0.45) for b in bits]).arrange(RIGHT, buff=0.15)
    cells.next_to(head, DOWN, buff=0.4)
    q = k.label("số bit 1 chẵn hay lẻ?", 26, k.MUTED).next_to(cells, RIGHT, buff=0.4)
    sc.run(FadeIn(cells, lag_ratio=0.1), FadeIn(q), run_time=0.8)

    sc.wait_until(sc.at(2))
    table = VGroup(*[Text(f"{i:06b} → {bin(i).count('1') % 2}", font=k.MONO, font_size=18, color=k.MUTED)
                     for i in range(10)]).arrange(DOWN, buff=0.06)
    dots = Text("...", font=k.MONO, font_size=22, color=k.MUTED)
    rote = VGroup(table, dots).arrange(DOWN, buff=0.05).scale(1.2).to_edge(LEFT, buff=1.0).shift(DOWN * 1.4)
    count = k.label("học thuộc: 64 trường hợp", 26, k.PINK, bold=True).next_to(rote, UP, buff=0.2)
    count.align_to(rote, LEFT).shift(LEFT * 0.4)
    sc.run(FadeIn(rote, lag_ratio=0.05), FadeIn(count), run_time=0.9)

    sc.wait_until(sc.at(3))
    chain = VGroup()
    acc = bits[0]
    prev = card(k, str(acc), k.GOLD, 28, 3, pad=0.4)
    chain.add(prev)
    for b in bits[1:]:
        acc ^= b
        op = k.label(f"XOR {b}", 22, k.MUTED)
        nxt = card(k, str(acc), k.GOLD, 28, 3, pad=0.4)
        chain.add(op, nxt)
    chain.arrange(RIGHT, buff=0.16).scale(0.92).move_to(RIGHT * 2.3 + DOWN * 0.6)
    rule = VGroup(*[Text(t, font=k.MONO, font_size=24, color=k.GREEN) for t in ("00 → 0", "01 → 1", "10 → 1", "11 → 0")])
    rule.arrange(RIGHT, buff=0.4).next_to(chain, DOWN, buff=0.4)
    few = k.label("mỗi bước 4 trường hợp, 5 bước: 20 ví dụ", 26, k.GREEN, bold=True).next_to(rule, DOWN, buff=0.3)
    for m in chain:
        sc.run(FadeIn(m), run_time=0.15)
    sc.run(FadeIn(rule), run_time=0.5)
    sc.run(FadeIn(few), run_time=0.5)

    sc.wait_until(sc.at(4))
    gen = card(k, "muốn tổng quát sang bài mới thì phải dạy suy luận", k.GOLD, 26, 60).to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(gen, shift=UP * 0.2), run_time=0.6)


def _tree(k, depth=4, width=6.0, height=3.2):
    nodes, edges = [], VGroup()
    level = [ORIGIN + UP * height / 2]
    nodes.append(level)
    for d in range(1, depth):
        n = 2 ** d
        y = height / 2 - d * height / (depth - 1)
        xs = [(-width / 2 + (i + 0.5) * width / n) for i in range(n)]
        cur = [RIGHT * x + UP * y for x in xs]
        for i, p in enumerate(cur):
            edges.add(Line(level[i // 2], p, color=k.MUTED, stroke_width=3))
        nodes.append(cur)
        level = cur
    return nodes, edges


def pass_at_k(sc, s, k):
    q = k.label("RL dạy cái mới, hay làm lộ cái có sẵn?", 32, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(FadeIn(q), run_time=0.5)
    sc.wait_until(sc.at(1))
    nodes, edges = _tree(k, 4, 5.4, 3.0)
    tree = VGroup(edges).move_to(LEFT * 3.6 + DOWN * 0.4)
    shift = tree.get_center() - VGroup(*[Dot(p) for lvl in nodes for p in lvl]).get_center()
    path = VGroup()
    pts = [nodes[0][0], nodes[1][0], nodes[2][1], nodes[3][2]]
    for a, b in zip(pts, pts[1:]):
        path.add(Line(a + shift, b + shift, color=k.GREEN, stroke_width=3))
    sc.run(Create(edges), run_time=0.8)
    sc.run(Create(path), run_time=0.6)
    thin = k.label("đường đúng có sẵn, nhưng mảnh", 22, k.GREEN).next_to(tree, DOWN, buff=0.2)
    sc.run(FadeIn(thin), run_time=0.4)
    sc.run(path.animate.set_stroke(width=10), run_time=0.6)
    thick = k.label("sau RL: dày lên, dễ được chọn", 22, k.GREEN, bold=True).move_to(thin)
    sc.run(Transform(thin, thick), run_time=0.4)

    sc.wait_until(sc.at(2))
    axes, at = _chart(k, 4.6, 3.0, "k (1 đến 256)", "pass@k")
    axes.move_to(RIGHT * 3.4 + DOWN * 0.3)
    tag = k.label("paper 04/2025", 22, k.MUTED).next_to(axes, UP, buff=0.35)
    sc.run(FadeIn(axes), FadeIn(tag), run_time=0.5)
    sc.wait_until(sc.at(3))
    rl = _polyline(k, at, [(0, 0.55), (0.25, 0.68), (0.5, 0.75), (0.75, 0.79), (1, 0.81)], k.GOLD)
    base = _polyline(k, at, [(0, 0.25), (0.25, 0.48), (0.5, 0.66), (0.75, 0.78), (1, 0.84)], k.MUTED)
    tr = k.label("qua RL", 22, k.GOLD, bold=True).move_to(at(0.15, 0.82))
    sc.run(Create(rl), FadeIn(tr), run_time=0.8)
    sc.wait_until(sc.at(4))
    tb = k.label("mô hình gốc", 22, k.INK, bold=True).next_to(at(1, 0.84), UP, buff=0.12).shift(LEFT * 0.4)
    sc.run(Create(base), FadeIn(tb), run_time=1.0)

    sc.wait_until(sc.at(5))
    sutra = card(k, "Hoa Nghiêm: ai cũng sẵn trí tuệ Như Lai, chỉ bị vọng tưởng che lấp", k.GOLD, 24, 70)
    sutra.to_edge(DOWN, buff=0.25)
    sc.run(FadeIn(sutra, shift=UP * 0.2), run_time=0.7)


def no_training(sc, s, k):
    q = k.label("Đã có sẵn thì cần gì RL?", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(q), run_time=0.5)
    sc.wait_until(sc.at(1))
    zen = card(k, "Thiền tông đốn ngộ: buông được là ngộ", k.GOLD, 26, 50).next_to(q, DOWN, buff=0.3)
    sc.run(FadeIn(zen), run_time=0.6)
    sc.wait_until(sc.at(2))
    base = Line(LEFT * 3.4, RIGHT * 3.4, color=k.MUTED, stroke_width=3).move_to(DOWN * 2.4)
    vals = [("mô hình gốc", 1.4, k.MUTED), ("qua RL", 2.8, k.GOLD), ("gốc + cách lấy mẫu mới", 2.7, k.GREEN)]
    bars = VGroup()
    for i, (name, h, c) in enumerate(vals):
        bar = Rectangle(width=1.3, height=h, stroke_width=0).set_fill(c, 0.85)
        bar.move_to(base.get_left() + RIGHT * (1.1 + i * 2.3), aligned_edge=DOWN)
        t = k.label(name, 22, c, bold=True, width_chars=14).next_to(bar, DOWN, buff=0.15)
        bars.add(VGroup(bar, t))
    sc.run(Create(base), run_time=0.3)
    for b in bars:
        sc.run(GrowFromEdge(b[0], DOWN), FadeIn(b[1]), run_time=0.45)
    note = k.label("không huấn luyện gì · 10/2025", 22, k.GREEN).next_to(bars[2][0], UP, buff=0.15)
    sc.run(FadeIn(note), run_time=0.4)


def other_camp(sc, s, k):
    head = k.label("Phe còn lại phản bác", 36, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeIn(head), run_time=0.5)
    nums = VGroup(*[card(k, str(i), k.MUTED, 24, 3, pad=0.35) for i in range(1, 11)]).arrange(RIGHT, buff=0.1)
    nums.next_to(head, DOWN, buff=0.35)
    guess = k.label("thử 1024 lần, đoán bừa cũng trúng", 24, k.PINK).next_to(nums, DOWN, buff=0.15)
    sc.run(FadeIn(nums, lag_ratio=0.08), FadeIn(guess), run_time=0.8)

    sc.wait_until(sc.at(1))
    axes, at = _chart(k, 4.6, 2.6, "k", "CoT-pass@k")
    axes.move_to(LEFT * 3.0 + DOWN * 1.3)
    rl = _polyline(k, at, [(0, 0.5), (0.5, 0.7), (1, 0.82)], k.GOLD)
    base = _polyline(k, at, [(0, 0.2), (0.5, 0.42), (1, 0.55)], k.MUTED)
    tag = k.label("lời giải cũng phải đúng · 06/2025", 22, k.MUTED).next_to(axes, UP, buff=0.3)
    sc.run(FadeIn(axes), FadeIn(tag), run_time=0.5)
    sc.run(Create(rl), Create(base), run_time=0.9)
    win = k.label("qua RL hơn hẳn", 24, k.GOLD, bold=True).next_to(at(1, 0.82), RIGHT, buff=0.15)
    sc.run(FadeIn(win), run_time=0.4)

    sc.wait_until(sc.at(2))
    who = bubble(k, "1024 lời giải, ai chấm?", k.GOLD, 24, 26).move_to(RIGHT * 3.8 + DOWN * 1.3)
    sc.run(FadeIn(who), run_time=0.5)
    sc.wait_until(sc.at(3))
    judge = _model(k, "LLM chấm", k.PURPLE, w=2.8, h=0.9).next_to(who, DOWN, buff=0.3)
    sc.run(FadeIn(judge), run_time=0.5)
    sc.wait_until(sc.at(4))
    hmm = k.label("?", 60, k.PINK, bold=True).next_to(judge, RIGHT, buff=0.3)
    sc.run(FadeIn(hmm, scale=0.6), run_time=0.4)


def open_debate(sc, s, k):
    title = card(k, "The Debate on RLVR Reasoning Capability Boundary", k.MUTED, 24, 60).to_edge(UP, buff=0.5)
    sc.run(FadeIn(title), run_time=0.5)
    axis = Arrow(LEFT * 5.5, RIGHT * 5.5, buff=0, color=k.MUTED, stroke_width=4).move_to(DOWN * 0.2)
    at = k.label("thời gian huấn luyện RL", 24, k.MUTED).next_to(axis, DOWN, buff=0.2)
    sc.run(GrowArrow(axis), FadeIn(at), run_time=0.6)
    early = card(k, "đầu: khai thác đường có sẵn", k.ACCENT, 26, 30).move_to(LEFT * 3.0 + UP * 1.2)
    late = card(k, "sau: mọc năng lực mới", k.GREEN, 26, 30).move_to(RIGHT * 3.0 + UP * 1.2)
    sc.run(FadeIn(early, shift=DOWN * 0.2), run_time=0.5)
    sc.run(FadeIn(late, shift=DOWN * 0.2), run_time=0.5)
    sc.wait_until(sc.at(1))
    q = k.label("RL kiểu nào, phần thưởng kiểu nào? vẫn đang tranh luận", 26, k.GOLD, bold=True).to_edge(DOWN, buff=0.8)
    sc.run(FadeIn(q), run_time=0.6)


def outro(sc, s, k):
    """Tóm tắt: ba cột, mỗi cột một hướng, liệt kê những gì video đã kể."""
    head = k.label("Video này đã đi qua", 40, k.INK, bold=True).to_edge(UP, buff=0.45)
    sc.run(Write(head), run_time=0.7)
    cols = [("1 · Lúc sinh chữ", "bản sai rồi trừ đi", k.ACCENT,
             ["probe + vector vàng", "GPT-2 lớn − Small", "DoLa, Layer CD", "câu thần chú, RAG",
              "quả chuối đen", "MTI: “Output Error”"]),
            ("2 · Sinh rồi kiểm", "đổi quy trình", k.GREEN,
             ["chê dễ hơn làm?", "tự kiểm: chập chờn", "phản hồi ngoài: chắc ăn", "CL và CS",
              "khen hay hỏi vặn", "so với bỏ phiếu đa số"]),
            ("3 · Huấn luyện", "học tăng cường", k.GOLD,
             ["Hillary sinh ở đâu?", "ReVISE, lỗi mới", "RLVR tự biết kiểm lại", "parity: 64 hay 20",
              "pass@k: có sẵn hay mới?", "hai phe đều đúng"])]
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
