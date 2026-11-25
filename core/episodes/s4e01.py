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
    GrowFromCenter, GrowFromEdge, Indicate, Line, Rectangle, RoundedRectangle, Text, Transform, VGroup, Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import arrow, bubble, card, check, cross, heading  # noqa: E402

BOARD = [(0, "two_loops"), (2, "three_routes"), (7, "token_loop"), (11, "probe"), (16, "yellow_vector"),
         (21, "contrastive"), (31, "obama"), (37, "logit_lens"), (43, "dola")]


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


def contrastive(sc, s, k):
    title = k.label("Contrastive decoding", 44, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.wait_until(sc.at(1))
    sc.run(Write(title), run_time=0.8)

    sc.wait_until(sc.at(2))
    q = card(k, "Câu hỏi", k.ACCENT, 28, 10).move_to(LEFT * 5.2 + DOWN * 0.2)
    sc.run(FadeIn(q), run_time=0.4)
    normal_m = _model(k, "LLM", k.ACCENT, w=1.6, h=0.9).move_to(LEFT * 2.4 + UP * 0.8)
    broken_m = _model(k, "LLM", k.GREEN, w=1.6, h=0.9).move_to(LEFT * 2.4 + DOWN * 1.5)

    sc.wait_until(sc.at(3))
    n_bars = _bars(k, [("A", 0.55), ("B", 0.5), ("C", 0.15)], k.ACCENT, width=1.8, size=26, label_w=0.5)
    n_bars.next_to(normal_m, RIGHT, buff=0.7)
    n_t = k.label("bình thường", 24, k.ACCENT, bold=True).next_to(n_bars, UP, buff=0.15)
    sc.run(GrowArrow(Arrow(q.get_right(), normal_m.get_left(), buff=0.1, color=k.MUTED, stroke_width=4)),
           FadeIn(normal_m), run_time=0.5)
    sc.run(GrowArrow(arrow(k, normal_m, n_bars)), FadeIn(n_bars), FadeIn(n_t), run_time=0.6)

    sc.wait_until(sc.at(4))
    b_bars = _bars(k, [("A", 0.15), ("B", 0.7), ("C", 0.2)], k.GREEN, width=1.8, size=26, label_w=0.5)
    b_bars.next_to(broken_m, RIGHT, buff=0.7)
    b_t = k.label("cố tình làm hỏng", 24, k.GREEN, bold=True).next_to(b_bars, UP, buff=0.15)
    hurt = k.label("đầu vào bị sửa", 20, k.PINK).next_to(broken_m, DOWN, buff=0.15)
    sc.run(GrowArrow(Arrow(q.get_right(), broken_m.get_left(), buff=0.1, color=k.PINK, stroke_width=4)),
           FadeIn(broken_m), FadeIn(hurt), run_time=0.5)
    sc.run(GrowArrow(arrow(k, broken_m, b_bars)), FadeIn(b_bars), FadeIn(b_t), run_time=0.6)

    sc.wait_until(sc.at(5))
    out = _bars(k, [("A", 0.75), ("B", 0.2), ("C", 0.1)], k.GOLD, width=1.8, size=26, label_w=0.5)
    out.move_to(RIGHT * 4.6 + DOWN * 0.2)
    minus = k.label("−", 60, k.INK, bold=True).move_to(RIGHT * 2.0 + DOWN * 0.2)
    sc.run(FadeIn(minus), run_time=0.3)
    sc.run(GrowArrow(Arrow(minus.get_right(), out.get_left(), buff=0.2, color=k.GOLD, stroke_width=5)),
           FadeIn(out), run_time=0.7)
    win = k.label("A thắng", 26, k.GOLD, bold=True).next_to(out, DOWN, buff=0.25)
    sc.run(FadeIn(win), run_time=0.4)

    sc.wait_until(sc.at(6))
    formula = Text("(1 + alpha) · bình thường  −  alpha · hỏng", font=k.MONO, font_size=30, color=k.INK)
    formula.to_edge(DOWN, buff=0.35)
    box = RoundedRectangle(corner_radius=0.15, width=formula.width + 0.5, height=formula.height + 0.35,
                           color=k.GOLD, stroke_width=3).move_to(formula)
    sc.run(Write(formula), Create(box), run_time=1.0)
    alpha = Text("alpha < 1", font=k.MONO, font_size=28, color=k.GOLD).next_to(box, RIGHT, buff=0.3)
    sc.run(FadeIn(alpha), run_time=0.4)

    sc.wait_until(sc.at(7))
    per = k.label("mỗi token một lần · làm trên logit", 24, k.MUTED).next_to(title, DOWN, buff=0.15)
    sc.run(FadeIn(per), run_time=0.5)

    sc.wait_until(sc.at(8))
    plus = card(k, "+ không đụng tham số, thử được ngay", k.GREEN, 26, 40)
    minus2 = card(k, "− mỗi token chạy thêm một lần", k.PINK, 26, 40)
    VGroup(plus, minus2).arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=0.35)
    sc.run(FadeOut(VGroup(formula, box, alpha)), run_time=0.3)
    sc.run(FadeIn(plus, shift=UP * 0.2), run_time=0.5)
    sc.wait_until(sc.at(9))
    sc.run(FadeIn(minus2, shift=UP * 0.2), run_time=0.5)


def obama(sc, s, k):
    era = card(k, "2022 · “thời thượng cổ”, chưa có ChatGPT", k.MUTED, 26, 40).to_edge(UP, buff=0.5)
    sc.run(FadeIn(era, shift=DOWN * 0.2), run_time=0.6)

    sc.wait_until(sc.at(1))
    prompt = k.label("Obama sinh ở Honolulu, Hawaii. Ông sinh năm ___", 34, k.INK, bold=True)
    prompt.next_to(era, DOWN, buff=0.5)
    sc.run(Write(prompt), run_time=1.2)

    rows_big = [("Hawaii", 0.62), ("1961", 0.5), ("ở", 0.2)]
    rows_small = [("Hawaii", 0.85), ("1961", 0.2), ("ở", 0.3)]
    rows_out = [("Hawaii", 0.12), ("1961", 0.8), ("ở", 0.05)]
    big = _panel(k, "GPT-2 bản lớn", k.ACCENT, _bars(k, rows_big, k.ACCENT, 2.2, 26, label_w=1.3))
    small = _panel(k, "GPT-2 Small", k.GREEN, _bars(k, rows_small, k.GREEN, 2.2, 26, label_w=1.3))
    out = _panel(k, "lớn − nhỏ", k.GOLD, _bars(k, rows_out, k.GOLD, 2.2, 26, label_w=1.3))
    VGroup(big, small, out).arrange(RIGHT, buff=0.5).move_to(DOWN * 1.1)

    sc.wait_until(sc.at(2))
    sc.run(FadeIn(big, shift=UP * 0.2), run_time=0.6)
    wrong = cross(k, big[2][0][0])
    sc.run(Create(wrong), run_time=0.4)
    should = k.label("cần một năm", 24, k.PINK).next_to(big, DOWN, buff=0.2)
    sc.run(FadeIn(should), run_time=0.4)

    sc.wait_until(sc.at(3))
    sc.run(FadeIn(small, shift=UP * 0.2), run_time=0.6)
    tag = k.label("coi như luôn sai", 24, k.GREEN).next_to(small, DOWN, buff=0.2)
    sc.run(FadeIn(tag), run_time=0.4)

    sc.wait_until(sc.at(4))
    sc.run(FadeIn(out, shift=UP * 0.2), run_time=0.7)
    ring = RoundedRectangle(corner_radius=0.1, width=out[2][1].width + 0.3, height=out[2][1].height + 0.2,
                            color=k.GOLD, stroke_width=4).move_to(out[2][1])
    sc.run(Create(ring), run_time=0.5)

    sc.wait_until(sc.at(5))
    rule = k.label("chọn token mà mô hình tốt và mô hình kém chênh nhau nhiều nhất", 28, k.GOLD, bold=True)
    rule.to_edge(DOWN, buff=0.3)
    sc.run(FadeIn(rule, shift=UP * 0.2), run_time=0.7)


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


def dola(sc, s, k):
    layers = VGroup(*[RoundedRectangle(corner_radius=0.1, width=2.6, height=0.5, color=k.PURPLE, stroke_width=3)
                      .set_fill(k.PURPLE, opacity=0.12 + 0.05 * i) for i in range(6)]).arrange(UP, buff=0.12)
    layers.move_to(LEFT * 5.0 + UP * 0.2)
    sc.run(FadeIn(layers), run_time=0.5)
    shallow = _bars(k, [("A", 0.25), ("B", 0.75), ("C", 0.3)], k.GREEN, 1.7, 26, label_w=0.5)
    deep = _bars(k, [("A", 0.6), ("B", 0.55), ("C", 0.2)], k.ACCENT, 1.7, 26, label_w=0.5)
    p_sh = _panel(k, "lớp nông: chỉ thấy bề mặt", k.GREEN, shallow).move_to(LEFT * 1.0 + DOWN * 1.3)
    p_dp = _panel(k, "lớp cuối", k.ACCENT, deep).move_to(LEFT * 1.0 + UP * 2.0)
    sc.run(GrowArrow(Arrow(layers[1].get_right(), p_sh.get_left(), buff=0.1, color=k.GREEN, stroke_width=4)),
           FadeIn(p_sh), run_time=0.7)
    sc.run(GrowArrow(Arrow(layers[5].get_right(), p_dp.get_left(), buff=0.1, color=k.ACCENT, stroke_width=4)),
           FadeIn(p_dp), run_time=0.7)

    sc.wait_until(sc.at(1))
    out = _panel(k, "lớp cuối − lớp nông", k.GOLD,
                 _bars(k, [("A", 0.8), ("B", 0.15), ("C", 0.1)], k.GOLD, 1.7, 26, label_w=0.5))
    out.move_to(RIGHT * 4.2 + UP * 0.4)
    sc.run(GrowArrow(Arrow(VGroup(p_sh, p_dp).get_right(), out.get_left(), buff=0.2, color=k.GOLD,
                           stroke_width=5)), FadeIn(out), run_time=0.8)

    sc.wait_until(sc.at(2))
    gain = card(k, "không cần mô hình nhỏ · lớp nông vốn đã chạy · gần như miễn phí", k.GREEN, 26, 80)
    gain.to_edge(DOWN, buff=0.3)
    sc.run(FadeIn(gain, shift=UP * 0.2), run_time=0.7)
