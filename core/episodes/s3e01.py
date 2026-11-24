"""
Series "Kiểm chứng AI" (series id 3), tập 1: Vì sao AI tự kiểm chính nó là chưa đủ?
Kịch bản: core/episodes/s3e01_vi.txt. Mỗi hàm là một cảnh (sc, spec, k); sc.at(i) = lúc câu thứ i
của cảnh bắt đầu. Mảnh ghép chung ở core/episodes/_kit.py.
Nguồn: Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" (Google DeepMind, ICLR 2024).
"""
from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, CurvedArrow, FadeIn, FadeOut, GrowArrow, GrowFromCenter,
    GrowFromEdge, Indicate, Line, Rectangle, RoundedRectangle, SurroundingRectangle, Text, Transform, Triangle,
    VGroup, Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import arrow, bubble, card, check, code_card, cross, heading, strike  # noqa: E402

BOARD = [(0, "series_open"), (2, "apples"), (6, "are_you_sure"), (9, "three_paths"), (12, "the_paper"),
         (18, "results"), (20, "oracle_loop"), (26, "exam_room"), (33, "same_brain"), (37, "hard_part"),
         (39, "form_vs_logic"), (45, "outside_signal"), (51, "apple_tool"), (54, "warning"), (56, "takeaway")]


def _person(k, color, size=1.0):
    head = Circle(radius=0.3 * size, color=color, stroke_width=4).set_fill(color, opacity=0.25)
    body = RoundedRectangle(corner_radius=0.3 * size, width=1.0 * size, height=0.65 * size, color=color,
                            stroke_width=4).set_fill(color, opacity=0.15).next_to(head, DOWN, buff=0.06)
    return VGroup(head, body)


def _who(k, text, color, mob):
    return k.label(text, 22, color, bold=True).next_to(mob, UP, buff=0.12).align_to(mob, LEFT)


def _flow(k, a_text, a_color, b_text, b_color, caption, width=2.0):
    a = card(k, a_text, a_color, 28, 10, min_w=width)
    b = card(k, b_text, b_color, 28, 10, min_w=width)
    VGroup(a, b).arrange(RIGHT, buff=1.4)
    link = arrow(k, a, b)
    cap = k.label(caption, 26, k.MUTED).next_to(b, RIGHT, buff=0.5)
    return VGroup(a, link, b, cap)


def _calculator(k, color=None):
    color = color or k.GREEN
    body = RoundedRectangle(corner_radius=0.12, width=1.1, height=1.5, color=color, stroke_width=4)
    body.set_fill(color, opacity=0.15)
    screen = Rectangle(width=0.8, height=0.32, color=color, stroke_width=3).next_to(body.get_top(), DOWN, buff=0.15)
    keys = VGroup(*[Rectangle(width=0.18, height=0.14, color=color, stroke_width=2) for _ in range(9)])
    keys.arrange_in_grid(3, 3, buff=0.08).next_to(screen, DOWN, buff=0.14)
    return VGroup(body, screen, keys)


def _warning_sign(k, size=1.2):
    tri = Triangle(color=k.GOLD, stroke_width=6).set_fill(k.GOLD, opacity=0.15).scale(size * 0.6)
    mark = k.label("!", 56 * size, k.GOLD, bold=True).move_to(tri.get_center() + DOWN * 0.12 * size)
    return VGroup(tri, mark)


# ── Các cảnh ─────────────────────────────────────────────────
def series_open(sc, s, k):
    title = k.label("Kiểm chứng AI", 60, k.GOLD, bold=True).move_to(UP * 2.5)
    sub = k.label("làm sao biết câu trả lời của AI là đúng, trước khi tin nó?", 30, k.MUTED).next_to(title, DOWN, buff=0.3)
    sc.run(Write(title), run_time=0.8)
    sc.run(FadeIn(sub), run_time=0.5)
    answer = card(k, "Câu trả lời của AI", k.ACCENT, 30, 18).move_to(LEFT * 3.8 + DOWN * 1.0)
    use = card(k, "Tin và dùng", k.GREEN, 30, 12).move_to(RIGHT * 3.8 + DOWN * 1.0)
    ask = k.label("?", 90, k.GOLD, bold=True).move_to(DOWN * 1.0)
    sc.run(FadeIn(answer, scale=0.9), run_time=0.5)
    sc.run(GrowArrow(arrow(k, answer, ask)), FadeIn(ask, scale=0.6), run_time=0.5)
    sc.run(GrowArrow(arrow(k, ask, use)), FadeIn(use), run_time=0.5)

    sc.wait_until(sc.at(1))
    loop = CurvedArrow(answer.get_corner(DOWN + RIGHT) + LEFT * 0.4, answer.get_corner(DOWN + LEFT) + RIGHT * 0.4,
                       angle=-2.2, color=k.PINK, stroke_width=5)
    self_check = k.label("bảo AI tự kiểm tra lại", 28, k.PINK, bold=True).next_to(loop, DOWN, buff=0.2)
    sc.run(Create(loop), FadeIn(self_check), run_time=0.8)
    sc.run(Indicate(ask, color=k.PINK), run_time=0.8)


def apples(sc, s, k):
    sc.wait_until(sc.at(1))
    q = bubble(k, "Cửa hàng có 23 quả táo, bán 20 quả, rồi nhập thêm 6 quả. Còn bao nhiêu?", k.ACCENT, 40)
    q.to_edge(UP, buff=0.9).to_edge(LEFT, buff=0.8)
    you = _who(k, "Bạn", k.ACCENT, q)
    sc.run(FadeIn(you), FadeIn(q, shift=DOWN * 0.2), run_time=0.7)
    math = Text("23 − 20 + 6 = ?", font=k.MONO, font_size=44, color=k.INK).move_to(UP * 0.1)
    sc.run(Write(math), run_time=0.8)

    sc.wait_until(sc.at(2))
    a = bubble(k, "Còn 9 quả.", k.PURPLE, 30, 34).move_to(RIGHT * 3.2 + DOWN * 1.7)
    model = _who(k, "Mô hình", k.PURPLE, a)
    sc.run(FadeIn(model), FadeIn(a, shift=UP * 0.2), run_time=0.6)
    nine = Text("23 − 20 + 6 = 9", font=k.MONO, font_size=44, color=k.INK).move_to(math)
    sc.run(Transform(math, nine), run_time=0.5)

    sc.wait_until(sc.at(3))
    ok = check(k, 0.7).next_to(a, RIGHT, buff=0.35)
    right = k.label("Đúng", 34, k.GREEN, bold=True).next_to(ok, DOWN, buff=0.2)
    sc.run(Create(ok), FadeIn(right), run_time=0.6)


def are_you_sure(sc, s, k):
    first = bubble(k, "Còn 9 quả.", k.PURPLE, 30, 28).to_corner(UP + LEFT, buff=0.9)
    ok = check(k, 0.45).next_to(first, RIGHT, buff=0.25)
    sc.run(FadeIn(first), Create(ok), run_time=0.4)
    q = bubble(k, "Bạn chắc chưa? Kiểm tra lại đi.", k.ACCENT, 34, 30).to_corner(UP + RIGHT, buff=0.9).shift(DOWN * 0.9)
    you = _who(k, "Bạn", k.ACCENT, q)
    sc.run(FadeIn(you), FadeIn(q, shift=LEFT * 0.2), run_time=0.6)

    sc.wait_until(sc.at(1))
    a = bubble(k, "Xin lỗi, mình đã nhầm. Còn 3 quả.", k.PURPLE, 36, 30).move_to(LEFT * 2.6 + DOWN * 1.2)
    model = _who(k, "Mô hình", k.PURPLE, a)
    sc.run(FadeIn(model), FadeIn(a, shift=UP * 0.2), run_time=0.6)
    big = k.label("9", 120, k.GREEN, bold=True).move_to(RIGHT * 4.0 + DOWN * 1.2)
    sc.run(FadeIn(big, scale=0.6), run_time=0.4)
    sc.run(Transform(big, k.label("3", 120, k.PINK, bold=True).move_to(big)), run_time=0.6)

    sc.wait_until(sc.at(2))
    flow = _flow(k, "Đúng", k.GREEN, "Sai", k.PINK, "", width=1.8)[:3].move_to(DOWN * 3.2)
    sc.run(FadeIn(flow[0]), GrowArrow(flow[1]), FadeIn(flow[2]), run_time=0.7)


def three_paths(sc, s, k):
    head = heading(sc, k, "Bảo kiểm tra lại thì...")
    rows = [_flow(k, "Đúng", k.GREEN, "Sai", k.PINK, "như vừa thấy"),
            _flow(k, "Sai", k.PINK, "Sai", k.PINK, "sai theo kiểu khác"),
            _flow(k, "Sai", k.PINK, "Đúng", k.GREEN, "có lúc, nhưng không đều")]
    group = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.7).scale(1.3)
    group.move_to(DOWN * 0.5)
    for i, row in enumerate(rows):
        sc.wait_until(sc.at(i))
        sc.run(FadeIn(row[0]), GrowArrow(row[1]), FadeIn(row[2]), FadeIn(row[3]), run_time=0.6)
    sc.run(Indicate(rows[2][2], color=k.GOLD), run_time=0.8)


def the_paper(sc, s, k):
    sc.wait_until(sc.at(1))
    page = RoundedRectangle(corner_radius=0.12, width=3.8, height=4.6, color=k.INK, stroke_width=3)
    page.set_fill(k.INK, opacity=0.05).move_to(LEFT * 4.4 + DOWN * 0.2)
    title = k.label("Large Language Models Cannot Self-Correct Reasoning Yet", 26, k.INK, bold=True,
                    width_chars=18, max_width=page.width - 0.5).next_to(page.get_top(), DOWN, buff=0.35)
    who = k.label("Google DeepMind · 2023", 22, k.ACCENT).next_to(title, DOWN, buff=0.25)
    lines = VGroup(*[Line(LEFT * 1.4, RIGHT * (1.4 - 0.5 * (i % 3 == 2)), color=k.MUTED, stroke_width=2)
                     for i in range(6)]).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(who, DOWN, buff=0.35)
    vi = k.label("chưa thể tự sửa lỗi suy luận", 28, k.GOLD, bold=True).next_to(page, RIGHT, buff=0.6).shift(UP * 1.4)
    sc.run(FadeIn(page), Write(title), run_time=0.9)
    sc.run(FadeIn(who), Create(lines), run_time=0.6)
    sc.run(FadeIn(vi, shift=LEFT * 0.2), run_time=0.5)

    sc.wait_until(sc.at(2))
    badge = card(k, "ICLR 2024", k.GOLD, 26, 10).next_to(page, DOWN, buff=0.2)
    sc.run(GrowFromCenter(badge), run_time=0.5)

    sc.wait_until(sc.at(3))
    tasks = VGroup(*[card(k, t, k.ACCENT, 24, 12, min_w=2.3) for t in ("Bài toán", "Thường thức", "Suy luận nhiều bước")])
    tasks.arrange(RIGHT, buff=0.3).move_to(RIGHT * 2.6 + UP * 1.5)
    sc.run(FadeOut(vi), run_time=0.3)
    for t in tasks:
        sc.run(FadeIn(t, shift=DOWN * 0.2), run_time=0.35)

    sc.wait_until(sc.at(4))
    steps = VGroup(*[card(k, t, c, 24, 10, min_w=1.9) for t, c in
                     (("Trả lời", k.PURPLE), ("Tự xem lại", k.PURPLE), ("Sửa", k.PURPLE))])
    steps.arrange(RIGHT, buff=0.6).move_to(RIGHT * 2.6 + DOWN * 0.4)
    links = VGroup(arrow(k, steps[0], steps[1]), arrow(k, steps[1], steps[2]))
    sc.run(FadeIn(steps[0]), run_time=0.3)
    sc.run(GrowArrow(links[0]), FadeIn(steps[1]), run_time=0.4)
    sc.run(GrowArrow(links[1]), FadeIn(steps[2]), run_time=0.4)
    info = card(k, "thông tin mới", k.GREEN, 22, 14).next_to(steps[1], DOWN, buff=1.0)
    feed = Arrow(info.get_top(), steps[1].get_bottom(), buff=0.1, color=k.GREEN, stroke_width=4)
    sc.run(FadeIn(info), GrowArrow(feed), run_time=0.4)
    sc.run(Create(cross(k, VGroup(info, feed))), run_time=0.4)

    sc.wait_until(sc.at(5))
    tag = k.label("tự sửa nội tại", 34, k.GOLD, bold=True).next_to(steps, UP, buff=0.35)
    sc.run(FadeIn(tag, scale=1.1), run_time=0.5)


def results(sc, s, k):
    base = Line(LEFT * 6.0, LEFT * 1.4, color=k.MUTED, stroke_width=3).move_to(LEFT * 3.7 + DOWN * 2.2)
    before = Rectangle(width=1.3, height=3.4, color=k.ACCENT, stroke_width=0).set_fill(k.ACCENT, opacity=0.8)
    after = Rectangle(width=1.3, height=3.0, color=k.PINK, stroke_width=0).set_fill(k.PINK, opacity=0.8)
    before.move_to(base.get_center() + LEFT * 1.2, aligned_edge=DOWN)
    after.move_to(base.get_center() + RIGHT * 1.2, aligned_edge=DOWN)
    t1 = k.label("trước khi\ntự sửa", 24, k.MUTED).next_to(before, DOWN, buff=0.25)
    t2 = k.label("sau khi\ntự sửa", 24, k.MUTED).next_to(after, DOWN, buff=0.25)
    title = k.label("độ chính xác", 30, k.INK, bold=True).next_to(VGroup(before, after), UP, buff=0.6)
    note = k.label("hình minh hoạ xu hướng, không phải số liệu", 20, k.MUTED).to_edge(DOWN, buff=0.25).shift(LEFT * 3.0)
    sc.run(Create(base), FadeIn(title), run_time=0.4)
    sc.run(GrowFromEdge(before, DOWN), FadeIn(t1), run_time=0.6)
    sc.run(GrowFromEdge(after, DOWN), FadeIn(t2), FadeIn(note), run_time=0.6)

    sc.wait_until(sc.at(1))
    a = _flow(k, "Đúng", k.GREEN, "Sai", k.PINK, "", width=1.6)[:3]
    b = _flow(k, "Sai", k.PINK, "Đúng", k.GREEN, "", width=1.6)[:3]
    a.move_to(RIGHT * 3.6 + UP * 0.9)
    b.move_to(RIGHT * 3.6 + DOWN * 1.2)
    a[1].set_stroke(width=14)
    b[1].set_stroke(width=3)
    ta = k.label("hay gặp hơn", 26, k.PINK, bold=True).next_to(a, UP, buff=0.25)
    tb = k.label("ít hơn", 26, k.MUTED).next_to(b, UP, buff=0.25)
    sc.run(FadeIn(a[0]), GrowArrow(a[1]), FadeIn(a[2]), FadeIn(ta), run_time=0.7)
    sc.run(FadeIn(b[0]), GrowArrow(b[1]), FadeIn(b[2]), FadeIn(tb), run_time=0.7)


def oracle_loop(sc, s, k):
    claim = k.label("bài báo trước: tự sửa giúp tăng điểm", 30, k.INK).to_edge(UP, buff=0.6)
    gain = Arrow(DOWN * 0.5, UP * 0.5, buff=0, color=k.GREEN, stroke_width=8).next_to(claim, RIGHT, buff=0.3)
    sc.run(FadeIn(claim), GrowArrow(gain), run_time=0.7)

    sc.wait_until(sc.at(1))
    fix = card(k, "Sửa", k.PURPLE, 30, 10, min_w=2.0).move_to(LEFT * 4.6 + DOWN * 0.4)
    key = card(k, "So với đáp án đúng", k.GOLD, 28, 12, min_w=3.2).move_to(LEFT * 0.6 + DOWN * 0.4)
    sc.run(FadeIn(fix), run_time=0.4)
    sc.run(GrowArrow(arrow(k, fix, key)), FadeIn(key), run_time=0.5)

    sc.wait_until(sc.at(2))
    stop = card(k, "Đúng: dừng", k.GREEN, 26, 12, min_w=2.6).move_to(RIGHT * 4.4 + UP * 0.6)
    again = card(k, "Sai: sửa tiếp", k.PINK, 26, 12, min_w=2.6).move_to(RIGHT * 4.4 + DOWN * 1.4)
    to_stop = Arrow(key.get_right(), stop.get_left(), buff=0.15, color=k.GREEN, stroke_width=4)
    to_again = Arrow(key.get_right(), again.get_left(), buff=0.15, color=k.PINK, stroke_width=4)
    sc.run(GrowArrow(to_stop), FadeIn(stop), run_time=0.5)

    sc.wait_until(sc.at(3))
    back = CurvedArrow(again.get_bottom() + DOWN * 0.05, fix.get_bottom() + DOWN * 0.05, angle=-0.7,
                       color=k.PINK, stroke_width=4)
    sc.run(GrowArrow(to_again), FadeIn(again), run_time=0.5)
    sc.run(Create(back), run_time=0.7)

    sc.wait_until(sc.at(4))
    ring = SurroundingRectangle(key, color=k.GOLD, buff=0.15, stroke_width=5)
    leak = k.label("đáp án đúng đã lọt vào vòng lặp", 30, k.GOLD, bold=True).next_to(key, UP, buff=0.55)
    sc.run(Create(ring), FadeIn(leak), run_time=0.7)

    sc.wait_until(sc.at(5))
    sc.run(Create(cross(k, key)), run_time=0.5)
    gone = k.label("bỏ đáp án ra: phần lớn cái lợi biến mất", 30, k.PINK, bold=True).move_to(claim)
    sc.run(FadeOut(gain), Transform(claim, gone), run_time=0.7)


def exam_room(sc, s, k):
    head = k.label("Vì sao?", 44, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.5)

    sc.wait_until(sc.at(1))
    room = RoundedRectangle(corner_radius=0.2, width=7.0, height=4.6, color=k.MUTED, stroke_width=3)
    room.move_to(LEFT * 2.8 + DOWN * 0.5)
    desk = Line(LEFT * 1.3, RIGHT * 1.3, color=k.MUTED, stroke_width=6).move_to(room.get_center() + LEFT * 1.2 + DOWN * 1.1)
    student = _person(k, k.ACCENT).next_to(desk, UP, buff=0.05)
    paper = card(k, "9", k.INK, 26, 4, pad=0.3).next_to(desk, UP, buff=0.05).shift(RIGHT * 0.9)
    sc.run(Create(room), run_time=0.5)
    sc.run(FadeIn(student), Create(desk), FadeIn(paper), run_time=0.5)
    nots = VGroup(*[k.label(t, 22, k.MUTED) for t in ("sách", "máy tính")]).arrange(RIGHT, buff=0.6)
    nots.next_to(room, DOWN, buff=0.25)
    no_tag = k.label("không có:", 22, k.MUTED).next_to(nots, LEFT, buff=0.3)
    nots = VGroup(no_tag, *nots)
    sc.run(FadeIn(nots), run_time=0.3)
    sc.run(*[Create(strike(k, n)) for n in nots[1:]], run_time=0.4)

    sc.wait_until(sc.at(2))
    proctor = _person(k, k.GOLD, 1.1).move_to(room.get_center() + RIGHT * 1.9 + DOWN * 0.6)
    ask = bubble(k, "Em chắc chưa?", k.GOLD, 16, 26).next_to(proctor, UP, buff=0.3)
    ask.shift(LEFT * max(0, ask.get_right()[0] - room.get_right()[0] + 0.2))
    sc.run(FadeIn(proctor, shift=LEFT * 0.4), run_time=0.5)
    sc.run(FadeIn(ask, scale=0.9), run_time=0.4)

    col = RIGHT * 4.6
    sc.wait_until(sc.at(3))
    info = card(k, "thông tin mới: 0", k.PINK, 26, 18, min_w=3.6).move_to(col + UP * 1.4)
    sc.run(FadeIn(info, shift=LEFT * 0.2), run_time=0.5)

    sc.wait_until(sc.at(4))
    know = card(k, "kiến thức: y như cũ", k.MUTED, 26, 20, min_w=3.6).next_to(info, DOWN, buff=0.35)
    sc.run(FadeIn(know, shift=LEFT * 0.2), run_time=0.5)

    sc.wait_until(sc.at(5))
    feel = bubble(k, "chắc mình sai rồi...", k.PINK, 20, 24).next_to(student, UP, buff=0.35)
    feel.shift(LEFT * max(0, feel.get_right()[0] - ask.get_left()[0] + 0.2))
    sc.run(FadeIn(feel, scale=0.8), run_time=0.5)

    sc.wait_until(sc.at(6))
    keep = card(k, "giữ nguyên", k.MUTED, 24, 12, min_w=1.9)
    swap = card(k, "đổi theo cảm giác", k.PINK, 24, 12, min_w=1.9)
    VGroup(keep, swap).arrange(DOWN, buff=0.25).next_to(know, DOWN, buff=0.6)
    sc.run(FadeIn(keep), FadeIn(swap), run_time=0.5)


def same_brain(sc, s, k):
    sc.wait_until(sc.at(1))
    writer = card(k, "Mô hình", k.PURPLE, 32, 10, min_w=2.6).move_to(LEFT * 4.2 + UP * 1.0)
    w_tag = k.label("viết câu trả lời", 24, k.MUTED).next_to(writer, DOWN, buff=0.2)
    checker = card(k, "Mô hình", k.PURPLE, 32, 10, min_w=2.6).move_to(RIGHT * 4.2 + UP * 1.0)
    c_tag = k.label("kiểm tra lại", 24, k.MUTED).next_to(checker, DOWN, buff=0.2)
    out = card(k, "Còn 9 quả", k.ACCENT, 26, 12).move_to(UP * 1.0)
    sc.run(FadeIn(writer), FadeIn(w_tag), run_time=0.5)
    sc.run(GrowArrow(arrow(k, writer, out)), FadeIn(out), run_time=0.5)
    sc.run(GrowArrow(arrow(k, out, checker)), FadeIn(checker), FadeIn(c_tag), run_time=0.5)
    same = k.label("cùng một bộ não", 32, k.GOLD, bold=True).to_edge(UP, buff=0.5)
    link = Line(writer.get_top(), checker.get_top(), path_arc=-0.5, color=k.GOLD, stroke_width=3)
    sc.run(Create(link), FadeIn(same), run_time=0.6)

    sc.wait_until(sc.at(2))
    logic = k.label("đủ hiểu để thấy lỗi thì ngay từ đầu đã ít mắc lỗi", 28, k.INK).move_to(DOWN * 0.8)
    sc.run(FadeIn(logic, shift=UP * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    nudge = bubble(k, "chắc chưa?", k.ACCENT, 14, 26).move_to(LEFT * 3.0 + DOWN * 2.6)
    change = card(k, "đổi đáp án", k.PINK, 26, 12).move_to(RIGHT * 1.6 + DOWN * 2.6)
    please = k.label("mô hình hay chiều theo người hỏi", 24, k.PINK).next_to(change, RIGHT, buff=0.3)
    if please.get_right()[0] > 6.9:
        please.shift(LEFT * (please.get_right()[0] - 6.9))
        please.next_to(change, DOWN, buff=0.15)
    sc.run(FadeIn(nudge), run_time=0.4)
    sc.run(GrowArrow(arrow(k, nudge, change, color=k.PINK)), FadeIn(change), run_time=0.5)
    sc.run(FadeIn(please), run_time=0.4)


def hard_part(sc, s, k):
    fix = card(k, "Sửa", k.GREEN, 30, 8, min_w=2.2)
    know = card(k, "Biết sai ở đâu", k.PINK, 44, 16, min_w=5.4)
    VGroup(fix, know).arrange(RIGHT, buff=1.2).move_to(UP * 0.6)
    easy = k.label("phần dễ", 26, k.MUTED).next_to(fix, DOWN, buff=0.3)
    hard = k.label("phần khó nhất", 30, k.PINK, bold=True).next_to(know, DOWN, buff=0.3)
    sc.run(FadeIn(fix), FadeIn(easy), run_time=0.5)
    sc.run(GrowFromCenter(know), FadeIn(hard), run_time=0.7)

    sc.wait_until(sc.at(1))
    alone = k.label("mô hình tự làm việc này: không đáng tin", 32, k.GOLD, bold=True).move_to(DOWN * 2.2)
    sc.run(FadeIn(alone, shift=UP * 0.2), Indicate(know, color=k.PINK), run_time=0.8)


def form_vs_logic(sc, s, k):
    left_x, right_x = LEFT * 3.4, RIGHT * 3.4
    lh = card(k, "Lỗi hình thức", k.GREEN, 32, 14, min_w=4.6).move_to(left_x + UP * 2.4)
    rh = card(k, "Lỗi suy luận", k.PINK, 32, 14, min_w=4.6).move_to(right_x + UP * 2.4)
    sc.run(FadeIn(lh), run_time=0.5)

    sc.wait_until(sc.at(1))
    items = VGroup(*[k.label("• " + t, 26, k.INK) for t in ("quá dài", "giọng chưa hợp", "thiếu một mục được yêu cầu")])
    items.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(lh, DOWN, buff=0.4)
    for it in items:
        sc.run(FadeIn(it, shift=RIGHT * 0.2), run_time=0.3)
    ok = VGroup(check(k, 0.45), k.label("tự xem lại thường giúp được", 24, k.GREEN, bold=True)).arrange(RIGHT, buff=0.2)
    ok.next_to(items, DOWN, buff=0.4).align_to(items, LEFT)
    sc.run(FadeIn(ok), run_time=0.4)

    sc.wait_until(sc.at(2))
    paper = k.label("bài báo cũng ghi nhận: chỉnh giọng văn", 22, k.MUTED).next_to(ok, DOWN, buff=0.3).align_to(items, LEFT)
    sc.run(FadeIn(paper), run_time=0.4)

    sc.wait_until(sc.at(3))
    sep = Line(UP * 3.0, DOWN * 3.4, color=k.MUTED, stroke_width=2)
    sc.run(Create(sep), FadeIn(rh), run_time=0.5)
    r_items = VGroup(*[k.label("• " + t, 26, k.INK) for t in ("tính toán", "bài đố", "suy luận nhiều bước")])
    r_items.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(rh, DOWN, buff=0.4)
    sc.run(FadeIn(r_items), run_time=0.4)

    sc.wait_until(sc.at(4))
    see = k.label("nhìn vào là thấy", 28, k.GREEN, bold=True).to_edge(DOWN, buff=0.4).set_x(left_x[0])
    sc.run(FadeIn(see, shift=UP * 0.2), run_time=0.5)

    sc.wait_until(sc.at(5))
    need = k.label("cần thứ bên ngoài để so", 28, k.PINK, bold=True).to_edge(DOWN, buff=0.4).set_x(right_x[0])
    sc.run(FadeIn(need, shift=UP * 0.2), run_time=0.5)


def outside_signal(sc, s, k):
    q = k.label("Thứ bên ngoài đó là gì?", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(Write(q), run_time=0.6)

    sc.wait_until(sc.at(1))
    student = _person(k, k.ACCENT).move_to(LEFT * 3.5 + DOWN * 0.2)
    paper = card(k, "9", k.INK, 26, 4, pad=0.3).next_to(student, RIGHT, buff=0.3)
    sc.run(FadeIn(student), FadeIn(paper), run_time=0.5)

    sc.wait_until(sc.at(2))
    calc = _calculator(k).move_to(RIGHT * 0.6 + DOWN * 0.2)
    give = Arrow(calc.get_left(), paper.get_right(), buff=0.2, color=k.GREEN, stroke_width=4)
    sub = k.label("hoặc thế ngược đáp án vào đề", 26, k.MUTED).next_to(calc, DOWN, buff=0.4)
    sc.run(FadeIn(calc, shift=LEFT * 0.3), GrowArrow(give), run_time=0.6)
    sc.run(FadeIn(sub), run_time=0.4)

    sc.wait_until(sc.at(3))
    new = card(k, "tín hiệu mới thật sự", k.GREEN, 28, 18).move_to(RIGHT * 4.6 + DOWN * 0.2)
    to_new = arrow(k, calc, new, color=k.GREEN)
    sc.run(GrowArrow(to_new), FadeIn(new), run_time=0.6)

    sc.wait_until(sc.at(4))
    old = VGroup(student, paper, calc, give, sub, new, to_new)
    sc.run(FadeOut(old), FadeOut(q), run_time=0.5)
    model = card(k, "Agent", k.PURPLE, 34, 10, min_w=2.6).move_to(DOWN * 0.1)
    sc.run(GrowFromCenter(model), run_time=0.4)
    spots = (LEFT * 4.6 + UP * 1.9, RIGHT * 4.6 + UP * 1.9, LEFT * 4.6 + DOWN * 2.1, RIGHT * 4.6 + DOWN * 2.1)
    names = ("Chạy code: có lỗi không?", "Test: qua hay trượt", "Dữ liệu thật", "Máy tính")
    for text, spot in zip(names, spots):
        c = card(k, text, k.GREEN, 26, 14, min_w=3.4).move_to(spot)
        link = Arrow(c.get_center(), model.get_center(), buff=1.3, color=k.GREEN, stroke_width=4)
        sc.run(FadeIn(c, scale=0.9), GrowArrow(link), run_time=0.45)

    sc.wait_until(sc.at(5))
    tag = k.label("không phụ thuộc mô hình nghĩ gì về chính nó", 28, k.GOLD, bold=True).to_edge(UP, buff=0.4)
    sc.run(FadeIn(tag, shift=DOWN * 0.2), run_time=0.6)


def apple_tool(sc, s, k):
    agent = card(k, "Agent", k.PURPLE, 32, 10, min_w=2.4).move_to(LEFT * 4.8 + UP * 0.6)
    call = code_card(k, ["calc(23 - 20 + 6)"], k.GOLD, 30).move_to(UP * 0.6)
    tool = k.label("gọi công cụ tính", 24, k.GOLD).next_to(call, UP, buff=0.25)
    sc.run(FadeIn(agent), run_time=0.4)
    sc.run(GrowArrow(arrow(k, agent, call)), FadeIn(call), FadeIn(tool), run_time=0.6)

    sc.wait_until(sc.at(1))
    result = card(k, "9", k.GREEN, 44, 4, min_w=1.4).move_to(RIGHT * 4.8 + UP * 0.6)
    ok = check(k, 0.6).next_to(result, DOWN, buff=0.3)
    sc.run(GrowArrow(arrow(k, call, result, color=k.GREEN)), FadeIn(result, scale=0.7), run_time=0.6)
    sc.run(Create(ok), run_time=0.4)

    sc.wait_until(sc.at(2))
    vs = VGroup(bubble(k, "chắc chưa?", k.MUTED, 14, 24), k.label("cảm giác", 24, k.MUTED))
    vs.arrange(DOWN, buff=0.15).move_to(LEFT * 3.0 + DOWN * 2.2)
    sure = VGroup(card(k, "kết quả công cụ", k.GREEN, 24, 16), k.label("căn cứ", 24, k.GREEN, bold=True))
    sure.arrange(DOWN, buff=0.15).move_to(RIGHT * 3.0 + DOWN * 2.2)
    sc.run(FadeIn(vs), run_time=0.4)
    sc.run(Create(strike(k, vs[0])), FadeIn(sure, shift=LEFT * 0.2), run_time=0.6)


def warning(sc, s, k):
    prompt = code_card(k, ["# bước kiểm tra duy nhất", '"Hãy kiểm tra lại câu trả lời của bạn."'], k.MUTED, 26)
    prompt.move_to(UP * 1.0)
    shield = k.label("lớp bảo vệ?", 34, k.INK, bold=True).next_to(prompt, DOWN, buff=0.5)
    sc.run(FadeIn(prompt, shift=DOWN * 0.2), run_time=0.6)
    sc.run(FadeIn(shield), run_time=0.4)
    sc.run(Create(cross(k, shield)), run_time=0.5)

    sc.wait_until(sc.at(1))
    sign = _warning_sign(k, 1.1).move_to(LEFT * 2.2 + DOWN * 2.3)
    worry = k.label("dấu hiệu nên lo", 34, k.GOLD, bold=True).next_to(sign, RIGHT, buff=0.4)
    sc.run(GrowFromCenter(sign), FadeIn(worry), run_time=0.6)


def takeaway(sc, s, k):
    remember = k.label("Tóm lại", 32, k.MUTED).to_edge(UP, buff=0.5)
    sc.run(FadeIn(remember), run_time=0.3)
    ask = bubble(k, "chắc chưa?", k.ACCENT, 14, 30).move_to(LEFT * 3.6 + UP * 1.2)
    zero = card(k, "0 thông tin mới", k.PINK, 30, 16).move_to(RIGHT * 2.8 + UP * 1.2)
    sc.run(FadeIn(ask), run_time=0.4)
    sc.run(GrowArrow(arrow(k, ask, zero, color=k.PINK)), FadeIn(zero), run_time=0.6)

    sc.wait_until(sc.at(1))
    agent = card(k, "Kiểm chứng thật", k.PURPLE, 30, 16).move_to(LEFT * 3.6 + DOWN * 0.9)
    outside = card(k, "tín hiệu từ bên ngoài mô hình", k.GREEN, 30, 30).move_to(RIGHT * 2.8 + DOWN * 0.9)
    sc.run(FadeIn(agent), run_time=0.4)
    sc.run(GrowArrow(Arrow(outside.get_left(), agent.get_right(), buff=0.15, color=k.GREEN, stroke_width=5)),
           FadeIn(outside), run_time=0.6)

    sc.wait_until(sc.at(2))
    nxt = k.label("Tập sau: test, công cụ, dữ liệu thật, nối vào vòng lặp agent", 28, k.ACCENT,
                  bold=True).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(nxt, shift=UP * 0.2), run_time=0.6)


# ── Thumbnail 1280x720 ───────────────────────────────────────
THUMB_TEXT = {
    "vi": ("TẬP 1  ·  KIỂM CHỨNG AI", "Hỏi AI", "“Chắc chưa?”", "đúng thành sai", "Còn 9 quả.", "Xin lỗi... 3 quả."),
    "en": ("EPISODE 1  ·  VERIFYING AI", "Ask AI", "“Are you sure?”", "right turns wrong", "9 apples left.",
           "Sorry... 3 apples."),
}


def thumbnail(sc, k, lang="vi"):
    tag_text, line1, line2, sub_text, first, second = THUMB_TEXT.get(lang, THUMB_TEXT["vi"])
    tag = k.label(tag_text, 30, k.MUTED, bold=True).to_corner(UP + LEFT, buff=0.6)
    left = VGroup(k.label(line1, 100, k.INK, bold=True), k.label(line2, 100, k.GOLD, bold=True))
    left.arrange(DOWN, aligned_edge=LEFT, buff=0.2).scale_to_fit_width(6.6).to_edge(LEFT, buff=0.75).shift(DOWN * 0.1)
    sub = k.label(sub_text, 46, k.PINK, bold=True).next_to(left, DOWN, aligned_edge=LEFT, buff=0.45)

    center_x = 4.1
    good = bubble(k, first, k.GREEN, 18, 34).move_to(RIGHT * center_x + UP * 1.6)
    ok = check(k, 0.6).next_to(good, RIGHT, buff=0.2)
    bad = bubble(k, second, k.PINK, 18, 34).move_to(RIGHT * center_x + DOWN * 1.6)
    no = cross(k, VGroup(Rectangle(width=0.5, height=0.5))).next_to(bad, RIGHT, buff=0.25)
    flip = Arrow(good.get_bottom(), bad.get_top(), buff=0.2, color=k.GOLD, stroke_width=10)
    for mob in (ok, no):
        over = mob.get_right()[0] - 6.9
        if over > 0:
            VGroup(good, bad, ok, no, flip).shift(LEFT * over)
    sc.add(tag, left, sub, good, ok, flip, bad, no)
