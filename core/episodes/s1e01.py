"""
Series 1 "Bản Chất LLM Agent", tập 1: LLM Agent là gì?
Cảnh vẽ tay cho từng đoạn của kịch bản (core/episodes/s1e01_vi.txt). Mỗi hàm là một cảnh,
nhận (sc, spec, k): sc là ExplainerScene (sc.at(i) = lúc câu thứ i của cảnh bắt đầu),
k là bộ màu/hàm dùng chung của core/explainer_scene.py.

BOARD: (chỉ số câu bắt đầu cảnh, tên hàm) — đưa vào bảng phân cảnh dạng
{"start": ..., "layout": "custom", "module": "s1e01", "scene": ...}.
"""
from __future__ import annotations

from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, ArcBetweenPoints, Circle, Create, CurvedArrow, DashedLine,
    DashedVMobject, Dot, FadeIn, FadeOut, GrowArrow, GrowFromCenter, GrowFromEdge, Indicate, Line,
    MoveAlongPath, Rectangle, RoundedRectangle, Text, Transform, VGroup, VMobject, Write,
)

BOARD = [(0, "request_and_reply"), (5, "where_numbers_come_from"), (10, "open_loop"),
         (16, "assistant_budget"), (25, "replan"), (29, "act_loop"), (33, "agent_loop"),
         (39, "tool_call"), (43, "loop_fixes_guess"), (46, "no_consciousness"),
         (50, "four_parts"), (56, "takeaway")]


# ── Mảnh ghép dùng chung ─────────────────────────────────────
def card(k, text, color, size=28, width_chars=18, bold=True, pad=0.5, min_w=0.0, text_color=None):
    txt = k.label(text, size, text_color or k.INK, bold=bold, width_chars=width_chars)
    box = RoundedRectangle(corner_radius=0.2, width=max(min_w, txt.width + pad), height=txt.height + pad * 0.8,
                           color=color, stroke_width=3).set_fill(color, opacity=0.12)
    txt.move_to(box)
    return VGroup(box, txt)


def code_card(k, lines, color, size=26):
    txt = VGroup(*[Text(line, font=k.MONO, font_size=size, color=k.INK) for line in lines])
    txt.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    box = RoundedRectangle(corner_radius=0.2, width=txt.width + 0.7, height=txt.height + 0.6,
                           color=color, stroke_width=2).set_fill("#151925", opacity=1)
    txt.move_to(box)
    return VGroup(box, txt)


def arrow(k, a, b, color=None):
    return Arrow(a.get_right(), b.get_left(), buff=0.15, color=color or k.MUTED, stroke_width=4,
                 max_tip_length_to_length_ratio=0.2)


def cross(k, mob):
    c = mob.get_center()
    w, h = mob.width / 2 + 0.1, mob.height / 2 + 0.1
    return VGroup(Line(c + LEFT * w + UP * h, c + RIGHT * w + DOWN * h, color=k.PINK, stroke_width=6),
                  Line(c + LEFT * w + DOWN * h, c + RIGHT * w + UP * h, color=k.PINK, stroke_width=6))


def strike(k, mob):
    return Line(mob.get_left() + LEFT * 0.1, mob.get_right() + RIGHT * 0.1, color=k.PINK, stroke_width=6)


def check(k, size=0.5):
    mark = VMobject(color=k.GREEN, stroke_width=8)
    mark.set_points_as_corners([LEFT * size * 0.5, DOWN * size * 0.45, RIGHT * size * 0.7 + UP * size * 0.55])
    return mark


def heading(sc, k, text, size=40):
    head = k.label(text, size, k.INK, bold=True, max_width=k.FRAME_W).to_edge(UP, buff=0.5)
    sc.run(Write(head), run_time=0.8)
    return head


def bubble(k, text, color, width_chars=30):
    txt = k.label(text, 28, k.INK, width_chars=width_chars)
    box = RoundedRectangle(corner_radius=0.3, width=txt.width + 0.6, height=txt.height + 0.5,
                           color=color, stroke_width=2).set_fill(color, opacity=0.15)
    txt.move_to(box)
    return VGroup(box, txt)


def budget_bar(k, y=-1.9, unit=1.0, ten_buff=0.25):
    """Thanh ngân sách 10 triệu: 1 triệu = `unit` đơn vị Manim, mép trái ở x = -5."""
    left = -5.0
    frame = Rectangle(width=10 * unit, height=0.8, color=k.MUTED, stroke_width=3).move_to(
        [left + 5 * unit, y, 0])
    zero = k.label("0", 24, k.MUTED).next_to(frame, LEFT, buff=0.2)
    ten = k.label("10 triệu", 24, k.MUTED).next_to(frame, RIGHT, buff=ten_buff)
    return VGroup(frame, zero, ten), left, y


def segment(k, left, y, start, length, color, text, above=True):
    seg = Rectangle(width=length, height=0.8, color=color, stroke_width=0).set_fill(color, opacity=0.75)
    seg.move_to([left + start + length / 2, y, 0])
    tag = k.label(text, 24, color, bold=True)
    tag.next_to(seg, UP if above else DOWN, buff=0.2)
    return seg, tag


# ── Các cảnh ─────────────────────────────────────────────────
def request_and_reply(sc, s, k):
    phone = RoundedRectangle(corner_radius=0.4, width=8.4, height=6.8, color=k.MUTED, stroke_width=2)
    phone.set_fill("#151925", opacity=1)
    title = k.label("Chatbot", 24, k.MUTED, bold=True).next_to(phone.get_top(), DOWN, buff=0.3)
    sc.run(FadeIn(phone), FadeIn(title), run_time=0.6)
    q = bubble(k, "Đặt chuyến Đà Nẵng 3 ngày 2 đêm cho 2 người, tổng không quá 10 triệu", k.ACCENT, 30)
    q.next_to(title, DOWN, buff=0.35).align_to(phone, RIGHT).shift(LEFT * 0.35)
    sc.run(FadeIn(q, shift=UP * 0.3), run_time=0.8)

    rows = VGroup(k.label("Bay chuyến sáng thứ bảy: 1,2 triệu", 28, k.INK),
                  k.label("Khách sạn 4 sao: 800 nghìn/đêm", 28, k.INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
    box = RoundedRectangle(corner_radius=0.3, width=rows.width + 0.7, height=rows.height + 0.6,
                           color=k.GREEN, stroke_width=2).set_fill(k.GREEN, opacity=0.12)
    box.next_to(q, DOWN, buff=0.4).align_to(phone, LEFT).shift(RIGHT * 0.35)
    rows.move_to(box)
    dots = VGroup(*[Dot(radius=0.08, color=k.MUTED) for _ in range(3)]).arrange(RIGHT, buff=0.15).move_to(box)
    sc.wait_until(sc.at(1))
    sc.run(FadeIn(box), FadeIn(dots), run_time=0.5)
    sc.run(Indicate(dots, color=k.INK), run_time=0.8)
    sc.wait_until(sc.at(2))
    sc.run(FadeOut(dots), FadeIn(rows[0], shift=RIGHT * 0.2), run_time=0.6)
    sc.wait_until(sc.at(3))
    sc.run(FadeIn(rows[1], shift=RIGHT * 0.2), run_time=0.6)
    sc.wait_until(sc.at(4))
    ok = VGroup(check(k, 0.45), k.label("Nghe rất ổn", 28, k.GREEN, bold=True)).arrange(RIGHT, buff=0.25)
    ok.next_to(box, DOWN, buff=0.35).align_to(box, LEFT)
    sc.run(FadeIn(ok, scale=0.8), run_time=0.6)


def where_numbers_come_from(sc, s, k):
    ticket = card(k, "Vé: 1,2 triệu", k.GOLD, 30, 20)
    room = card(k, "Phòng: 800 nghìn", k.GOLD, 30, 20)
    chips = VGroup(ticket, room).arrange(RIGHT, buff=0.6).move_to(UP * 2.6)
    ask = k.label("?", 90, k.GOLD, bold=True).next_to(chips, RIGHT, buff=0.5)
    sc.run(FadeIn(chips, shift=DOWN * 0.2), run_time=0.7)
    sc.run(Write(ask), run_time=0.6)

    sc.wait_until(sc.at(1))
    board = VGroup(k.label("Lịch bay hôm nay", 26, k.INK, bold=True),
                   *[Line(LEFT * 1.6, RIGHT * 1.6, color=k.MUTED, stroke_width=4) for _ in range(3)])
    board.arrange(DOWN, buff=0.3)
    frame = RoundedRectangle(corner_radius=0.2, width=board.width + 0.8, height=board.height + 0.6,
                             color=k.MUTED, stroke_width=2)
    flights = VGroup(frame, board.move_to(frame)).move_to(DOWN * 0.4)
    sc.run(FadeIn(flights), run_time=0.6)
    no_look = cross(k, flights)
    sc.run(Create(no_look), run_time=0.6)

    sc.wait_until(sc.at(2))
    data = VGroup(*[RoundedRectangle(corner_radius=0.1, width=1.8, height=1.1, color=k.PURPLE, stroke_width=2)
                    .set_fill(k.PURPLE, opacity=0.15).shift(UP * 0.12 * i + RIGHT * 0.12 * i) for i in range(3)])
    data_tag = k.label("Dữ liệu lúc huấn luyện", 22, k.PURPLE).next_to(data, DOWN, buff=0.2)
    model = card(k, "Mô hình", k.ACCENT, 30, 12, min_w=2.4)
    guess = card(k, "1,2 triệu?", k.GOLD, 30, 12)
    flow = VGroup(VGroup(data, data_tag), model, guess).arrange(RIGHT, buff=1.2).move_to(DOWN * 0.6)
    a1, a2 = arrow(k, flow[0], model), arrow(k, model, guess)
    sc.run(FadeOut(flights), FadeOut(no_look), run_time=0.4)
    sc.run(FadeIn(flow[0]), run_time=0.5)
    sc.run(GrowArrow(a1), FadeIn(model), run_time=0.6)
    sc.run(GrowArrow(a2), FadeIn(guess, shift=RIGHT * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    real = card(k, "Thực tế: 1,8 triệu", k.GREEN, 30, 20).next_to(chips, DOWN, buff=0.5).align_to(chips, LEFT)
    sc.run(Create(strike(k, ticket)), run_time=0.5)
    sc.run(FadeIn(real, shift=UP * 0.2), run_time=0.6)

    sc.wait_until(sc.at(4))
    stamp = card(k, "Ảo giác  ·  hallucination", k.PINK, 44, 30, pad=0.7, text_color=k.PINK)
    stamp.rotate(0.08).move_to(DOWN * 0.6)
    sc.run(FadeOut(flow), FadeOut(a1), FadeOut(a2), run_time=0.4)
    sc.run(GrowFromCenter(stamp), run_time=0.6)


def open_loop(sc, s, k):
    q = card(k, "Câu hỏi", k.ACCENT, 30, 12, min_w=2.6)
    m = card(k, "Mô hình", k.PURPLE, 30, 12, min_w=2.6)
    a = card(k, "Câu trả lời", k.GREEN, 30, 12, min_w=2.6)
    row = VGroup(q, m, a).arrange(RIGHT, buff=1.4).move_to(UP * 0.6)
    a1, a2 = arrow(k, q, m), arrow(k, m, a)
    sc.run(FadeIn(q), run_time=0.4)
    sc.run(GrowArrow(a1), FadeIn(m), run_time=0.5)
    sc.run(GrowArrow(a2), FadeIn(a), run_time=0.5)

    sc.wait_until(sc.at(1))
    fine = k.label("mô hình không tệ", 24, k.GREEN).next_to(m, UP, buff=0.3)
    sc.run(Indicate(m, color=k.GREEN), FadeIn(fine), run_time=0.8)

    sc.wait_until(sc.at(2))
    head = k.label("Vấn đề nằm ở cách làm việc", 40, k.INK, bold=True).to_edge(UP, buff=0.5)
    sc.run(FadeOut(fine), Write(head), run_time=0.8)

    sc.wait_until(sc.at(3))
    dot = Dot(q.get_bottom() + DOWN * 0.35, color=k.GOLD, radius=0.13)
    sc.run(FadeIn(dot), run_time=0.3)
    sc.run(dot.animate.move_to(m.get_bottom() + DOWN * 0.35), run_time=0.7)
    sc.run(dot.animate.move_to(a.get_bottom() + DOWN * 0.35), run_time=0.7)
    done = k.label("xong", 26, k.MUTED).next_to(dot, RIGHT, buff=0.2)
    sc.run(FadeIn(done), run_time=0.4)

    sc.wait_until(sc.at(4))
    one_way = Arrow(q.get_left() + DOWN * 1.6, a.get_right() + DOWN * 1.6, buff=0, color=k.GOLD, stroke_width=5,
                    max_tip_length_to_length_ratio=0.04)
    one_tag = k.label("một chiều", 28, k.GOLD, bold=True).next_to(one_way, DOWN, buff=0.2)
    sc.run(FadeOut(dot), FadeOut(done), GrowArrow(one_way), FadeIn(one_tag), run_time=0.8)

    sc.wait_until(sc.at(5))
    world = card(k, "Thực tế", k.MUTED, 26, 12, min_w=2.2).move_to([a.get_center()[0], -2.9, 0])
    back = DashedVMobject(CurvedArrow(world.get_left(), m.get_bottom() + DOWN * 0.1, angle=-0.9,
                                      color=k.MUTED, stroke_width=4), num_dashes=24)
    sc.run(FadeOut(one_way), FadeOut(one_tag), run_time=0.3)
    sc.run(FadeIn(world), Create(back), run_time=0.8)
    sc.run(Create(cross(k, VGroup(back).copy().scale(0.35))), run_time=0.5)


def assistant_budget(sc, s, k):
    head = heading(sc, k, "Một người trợ lý giỏi sẽ làm gì?")
    sc.wait_until(sc.at(2))
    glass = VGroup(Circle(radius=0.16, color=k.INK, stroke_width=4),
                   Line(ORIGIN, RIGHT * 0.22 + DOWN * 0.22, color=k.INK, stroke_width=4).shift(RIGHT * 0.13 + DOWN * 0.13))
    query = k.label("tra giá vé Đà Nẵng, thứ bảy", 28, k.INK)
    search = VGroup(glass, query).arrange(RIGHT, buff=0.3)
    sbox = RoundedRectangle(corner_radius=0.35, width=search.width + 0.8, height=0.8, color=k.ACCENT,
                            stroke_width=3).set_fill(k.ACCENT, opacity=0.1)
    search = VGroup(sbox, search.move_to(sbox)).move_to(UP * 1.9 + LEFT * 2.5)
    sc.run(FadeIn(search, shift=DOWN * 0.2), run_time=0.6)

    sc.wait_until(sc.at(3))
    result = card(k, "1,8 triệu / chiều / người", k.GREEN, 28, 30).next_to(search, RIGHT, buff=0.5)
    sc.run(FadeIn(result, shift=LEFT * 0.2), run_time=0.6)

    sc.wait_until(sc.at(4))
    calc = k.label("1,8 × 2 người × 2 chiều = 7,2 triệu", 32, k.INK, bold=True).move_to(UP * 0.6)
    bar, left, y = budget_bar(k)
    sc.run(Write(calc), run_time=0.9)
    sc.run(FadeIn(bar), run_time=0.4)
    ve, ve_tag = segment(k, left, y, 0, 7.2, k.ACCENT, "vé 7,2")
    sc.run(GrowFromEdge(ve, LEFT), FadeIn(ve_tag), run_time=0.8)

    sc.wait_until(sc.at(6))
    rest = DashedLine([left + 7.2, y - 0.65, 0], [left + 10, y - 0.65, 0], color=k.GREEN, stroke_width=4)
    rest_tag = k.label("còn 2,8", 24, k.GREEN, bold=True).next_to(rest, DOWN, buff=0.15)
    sc.run(Create(rest), FadeIn(rest_tag), run_time=0.6)

    sc.wait_until(sc.at(7))
    hotel, hotel_tag = segment(k, left, y, 7.2, 3.6, k.ORANGE, "khách sạn 4 sao 3,6", above=False)
    hotel_tag.align_to(hotel, RIGHT)
    limit = Line([left + 10, y + 0.7, 0], [left + 10, y - 0.7, 0], color=k.INK, stroke_width=5)
    limit_tag = k.label("10 triệu", 24, k.INK, bold=True).next_to(limit, UP, buff=0.1)
    sc.run(FadeOut(rest), FadeOut(rest_tag), FadeOut(bar[2]), run_time=0.3)
    sc.run(GrowFromEdge(hotel, LEFT), FadeIn(hotel_tag), run_time=0.8)
    sc.run(Create(limit), FadeIn(limit_tag), run_time=0.4)

    sc.wait_until(sc.at(8))
    over, _ = segment(k, left, y, 10, 0.8, k.PINK, "")
    warn = k.label("Vượt ngân sách", 34, k.PINK, bold=True).next_to(calc, DOWN, buff=0.45)
    sc.run(FadeIn(over), FadeIn(warn, scale=1.2), Indicate(hotel_tag, color=k.PINK), run_time=0.8)


def replan(sc, s, k):
    head = heading(sc, k, "Đổi kế hoạch")
    bar, left, y = budget_bar(k)
    bar[2].set_z_index(5)
    ve, ve_tag = segment(k, left, y, 0, 7.2, k.ACCENT, "vé 7,2")
    hotel, hotel_tag = segment(k, left, y, 7.2, 3.6, k.PINK, "khách sạn 4 sao 3,6", above=False)
    hotel_tag.align_to(hotel, RIGHT)
    sc.run(FadeIn(bar), FadeIn(ve), FadeIn(ve_tag), FadeIn(hotel), FadeIn(hotel_tag), run_time=0.6)

    sc.wait_until(sc.at(1))
    hotel3, hotel3_tag = segment(k, left, y, 7.2, 1.8, k.GREEN, "3 sao: 1,8")
    sc.run(Transform(hotel, hotel3), FadeOut(hotel_tag), run_time=0.9)
    sc.run(FadeIn(hotel3_tag), run_time=0.4)

    sc.wait_until(sc.at(2))
    mark = Line([left + 9, y + 0.6, 0], [left + 9, y - 0.6, 0], color=k.INK, stroke_width=4)
    total = k.label("9 triệu", 26, k.INK, bold=True).next_to(mark, DOWN, buff=0.15)
    spare = Rectangle(width=1.0, height=0.8, color=k.GOLD, stroke_width=3).move_to([left + 9.5, y, 0])
    spare_tag = k.label("dư 1 triệu: ăn uống, đi lại", 26, k.GOLD).next_to(spare, UP, buff=1.2)
    spare_tag.align_to(spare, RIGHT)
    pointer = Arrow(spare_tag.get_corner(DOWN + RIGHT) + LEFT * 0.25, spare.get_top(), buff=0.1, color=k.GOLD, stroke_width=3,
                    max_tip_length_to_length_ratio=0.3)
    sc.run(Create(mark), FadeIn(total), run_time=0.5)
    sc.run(Create(spare), FadeIn(spare_tag), GrowArrow(pointer), run_time=0.7)

    sc.wait_until(sc.at(3))
    button = card(k, "Đặt", k.GREEN, 34, 8, min_w=2.4).move_to(UP * 1.9)
    tick = check(k, 0.5).next_to(button, RIGHT, buff=0.3)
    sc.run(FadeIn(button, scale=0.8), run_time=0.5)
    sc.run(Indicate(button, color=k.GREEN), Create(tick), run_time=0.7)


def _loop_nodes(k, names, radius=2.2, center=ORIGIN, size=28):
    colors = (k.ACCENT, k.GOLD, k.GREEN)
    angles = (90, -30, 210)
    import math
    nodes = VGroup()
    for name, color, deg in zip(names, colors, angles):
        pos = center + radius * (RIGHT * math.cos(math.radians(deg)) * 1.4 + UP * math.sin(math.radians(deg)))
        nodes.add(card(k, name, color, size, 14, min_w=2.6).move_to(pos))
    arrows = VGroup(*[CurvedArrow(nodes[i].get_center(), nodes[(i + 1) % 3].get_center(), angle=-1.0,
                                  color=k.MUTED, stroke_width=4) for i in range(3)])
    for arr, i in zip(arrows, range(3)):
        arr.scale(0.55, about_point=(nodes[i].get_center() + nodes[(i + 1) % 3].get_center()) / 2)
    return nodes, arrows


def act_loop(sc, s, k):
    ask = k.label("Khác nhau ở đâu?", 44, k.INK, bold=True)
    sc.run(Write(ask), run_time=0.8)
    sc.wait_until(sc.at(1))
    note = k.label("không phải vì biết nhiều hơn", 30, k.MUTED).next_to(ask, DOWN, buff=0.4)
    sc.run(FadeIn(note), run_time=0.5)

    sc.wait_until(sc.at(2))
    sc.run(FadeOut(note), ask.animate.scale(0.8).to_edge(UP, buff=0.5), run_time=0.5)
    nodes, arrows = _loop_nodes(k, ["Hành động", "Nhìn kết quả", "Điều chỉnh"], center=DOWN * 0.3)
    for i in range(3):
        sc.run(FadeIn(nodes[i], scale=0.8), run_time=0.45)
        sc.run(Create(arrows[i]), run_time=0.35)

    sc.wait_until(sc.at(3))
    tag = k.label("lặp lại đến khi xong việc", 30, k.GOLD, bold=True).to_edge(DOWN, buff=0.5)
    sc.run(FadeIn(tag), run_time=0.4)
    dot = Dot(arrows[0].get_start(), color=k.GOLD, radius=0.14)
    sc.run(FadeIn(dot), run_time=0.2)
    i = 0
    while sc.remaining() > 1.6:
        sc.run(MoveAlongPath(dot, arrows[i % 3]), Indicate(nodes[(i + 1) % 3], color=k.GOLD, scale_factor=1.08),
               run_time=min(1.0, sc.remaining() - 0.8))
        i += 1


def agent_loop(sc, s, k):
    title = k.label("LLM agent", 72, k.INK, bold=True)
    sc.run(Write(title), run_time=1.0)
    sc.wait_until(sc.at(1))
    short = k.label("gọi tắt: agent", 30, k.ACCENT).next_to(title, DOWN, buff=0.35)
    sc.run(FadeIn(short), run_time=0.5)

    sc.wait_until(sc.at(2))
    sc.run(FadeOut(short), title.animate.scale(0.6).to_edge(UP, buff=0.4), run_time=0.7)
    llm = card(k, "Mô hình ngôn ngữ lớn", k.PURPLE, 30, 22, min_w=3.4).move_to(LEFT * 3.2 + DOWN * 0.3)
    same = k.label("giống hệt chatbot", 24, k.MUTED).next_to(llm, DOWN, buff=0.25)
    sc.run(FadeIn(llm, scale=0.9), FadeIn(same), run_time=0.6)

    sc.wait_until(sc.at(3))
    prog = card(k, "Chương trình", k.ACCENT, 30, 12, min_w=3.4).move_to(RIGHT * 3.2 + DOWN * 0.3)
    top = CurvedArrow(llm.get_top() + UP * 0.1, prog.get_top() + UP * 0.1, angle=-0.9, color=k.MUTED, stroke_width=4)
    bottom = CurvedArrow(prog.get_bottom() + DOWN * 0.1, llm.get_bottom() + DOWN * 0.55, angle=-0.9,
                         color=k.MUTED, stroke_width=4)
    loop_tag = k.label("vòng lặp", 26, k.GOLD, bold=True).move_to(DOWN * 0.3)
    sc.run(FadeOut(same), FadeIn(prog), Create(top), Create(bottom), FadeIn(loop_tag), run_time=0.9)

    sc.wait_until(sc.at(4))
    ask = card(k, "tra giá vé", k.GOLD, 24, 14).move_to(top.point_from_proportion(0.05))
    sc.run(FadeIn(ask, scale=0.7), run_time=0.3)
    sc.run(MoveAlongPath(ask, top), run_time=1.4)
    sc.run(Indicate(prog, color=k.GOLD), FadeOut(ask), run_time=0.5)

    sc.wait_until(sc.at(5))
    ans = card(k, "1,8 triệu", k.GREEN, 24, 14).move_to(bottom.point_from_proportion(0.0))
    sc.run(FadeIn(ans, scale=0.7), run_time=0.3)
    sc.run(MoveAlongPath(ans, bottom), run_time=1.4)
    sc.run(Indicate(llm, color=k.GREEN), run_time=0.6)


def tool_call(sc, s, k):
    heading(sc, k, "Tool use  ·  function calling")
    llm = card(k, "Mô hình", k.PURPLE, 30, 12, min_w=2.6).move_to(LEFT * 4.6 + DOWN * 0.2)
    sc.run(FadeIn(llm), run_time=0.4)

    sc.wait_until(sc.at(1))
    bar = Rectangle(width=4.2, height=0.45, color=k.MUTED, stroke_width=2).set_fill(k.MUTED, opacity=0.25)
    page = Rectangle(width=4.2, height=2.4, color=k.MUTED, stroke_width=2)
    VGroup(bar, page).arrange(DOWN, buff=0)
    site = VGroup(bar, page, k.label("trang hãng bay", 24, k.MUTED).move_to(page)).move_to(RIGHT * 3.0 + DOWN * 0.2)
    reach = DashedLine(llm.get_right(), site.get_left(), color=k.MUTED, stroke_width=4)
    sc.run(FadeIn(site), Create(reach), run_time=0.6)
    no_go = cross(k, site)
    sc.run(Create(no_go), run_time=0.5)

    sc.wait_until(sc.at(2))
    sc.run(FadeOut(site), FadeOut(reach), FadeOut(no_go), run_time=0.4)
    call = code_card(k, ['tra_ve(', '  diem_den="Da Nang",', '  ngay="thu bay"', ')'], k.GOLD)
    call.move_to(RIGHT * 0.2 + DOWN * 0.2)
    sc.run(GrowArrow(arrow(k, llm, call)), run_time=0.4)
    sc.run(FadeIn(call[0]), Write(call[1]), run_time=1.4)
    writes = k.label("mô hình chỉ viết ra yêu cầu", 24, k.GOLD).next_to(call, DOWN, buff=0.3)
    sc.run(FadeIn(writes), run_time=0.4)

    sc.wait_until(sc.at(3))
    prog = card(k, "Chương trình gọi API thật", k.ACCENT, 26, 14, min_w=3.0).move_to(RIGHT * 4.9 + DOWN * 0.2)
    sc.run(GrowArrow(arrow(k, call, prog)), FadeIn(prog), run_time=0.7)
    result = card(k, "1,8 triệu", k.GREEN, 26, 12).next_to(prog, DOWN, buff=0.5)
    sc.run(FadeIn(result, shift=DOWN * 0.2), run_time=0.5)


def loop_fixes_guess(sc, s, k):
    heading(sc, k, "Vì sao vòng lặp giúp bớt ảo giác?")
    sc.wait_until(sc.at(1))
    r1 = k.label("Vòng 1", 28, k.MUTED, bold=True)
    g1 = card(k, "đoán: vé 1,2 triệu", k.GOLD, 26, 22)
    t1 = card(k, "công cụ: 1,8 triệu", k.GREEN, 26, 22)
    row1 = VGroup(r1, g1, t1).arrange(RIGHT, buff=0.8).move_to(UP * 0.8)
    a1 = arrow(k, g1, t1)
    sc.run(FadeIn(r1), FadeIn(g1), run_time=0.5)
    mid = sc.at(1) + (sc.at(2) - sc.at(1)) * 0.45
    sc.wait_until(mid)
    sc.run(GrowArrow(a1), FadeIn(t1, shift=LEFT * 0.2), run_time=0.6)
    sc.run(Create(strike(k, g1[1])), run_time=0.4)

    sc.wait_until(sc.at(2) - 0.6 if sc.at(2) - 0.6 > sc.clock else sc.clock)
    r2 = k.label("Vòng 2", 28, k.MUTED, bold=True)
    g2 = card(k, "dùng giá thật: 1,8 triệu", k.GREEN, 26, 26)
    row2 = VGroup(r2, g2).arrange(RIGHT, buff=0.8).next_to(row1, DOWN, buff=0.9).align_to(row1, LEFT)
    carry = CurvedArrow(t1.get_bottom(), g2.get_right() + RIGHT * 0.1, angle=-1.2, color=k.GREEN, stroke_width=4)
    sc.run(Create(carry), FadeIn(row2), run_time=0.8)
    sc.wait_until(sc.at(2))
    note = k.label("không còn đoán mò", 32, k.GREEN, bold=True).next_to(row2, DOWN, buff=0.7)
    sc.run(FadeIn(note, scale=0.9), run_time=0.6)


def no_consciousness(sc, s, k):
    small = k.label("hay bị hiểu nhầm", 28, k.MUTED).to_edge(UP, buff=0.6)
    sc.run(FadeIn(small), run_time=0.5)
    sc.wait_until(sc.at(1))
    big = k.label("Agent không có ý thức hay ý chí", 48, k.INK, bold=True)
    sc.run(Write(big), run_time=1.0)

    sc.wait_until(sc.at(2))
    sc.run(FadeOut(small), big.animate.scale(0.6).to_edge(UP, buff=0.5), run_time=0.6)
    seen = card(k, "những gì đang thấy", k.ACCENT, 26, 12, min_w=2.8).move_to(LEFT * 4.6 + DOWN * 0.4)
    llm = card(k, "Mô hình", k.PURPLE, 30, 12, min_w=2.4).move_to(LEFT * 1.0 + DOWN * 0.4)
    options = [("tra giá vé", 0.7), ("đặt phòng", 0.2), ("hỏi lại người dùng", 0.1)]
    bars = VGroup()
    for name, p in options:
        tag = k.label(name, 24, k.INK)
        rect = Rectangle(width=max(0.08, 2.6 * p), height=0.36, color=k.GOLD, stroke_width=0).set_fill(k.GOLD, opacity=0.85)
        pct = k.label(f"{round(p * 100)}%", 22, k.GOLD)
        bars.add(VGroup(tag, rect, pct).arrange(RIGHT, buff=0.2))
    bars.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.7 + DOWN * 0.4)
    sc.run(FadeIn(seen), run_time=0.4)
    sc.run(GrowArrow(arrow(k, seen, llm)), FadeIn(llm), run_time=0.5)
    sc.run(GrowArrow(arrow(k, llm, bars)), run_time=0.4)
    for row in bars:
        sc.run(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), FadeIn(row[2]), run_time=0.45)
    pick = k.label("chọn bước hợp lý nhất", 24, k.GOLD).next_to(bars, DOWN, buff=0.35)
    sc.run(FadeIn(pick), run_time=0.4)

    sc.wait_until(sc.at(3))
    everything = VGroup(seen, llm, bars, pick)
    frame = DashedVMobject(RoundedRectangle(corner_radius=0.3, width=everything.width + 1.0,
                                            height=everything.height + 1.6, color=k.ACCENT, stroke_width=3),
                           num_dashes=60).move_to(everything).shift(DOWN * 0.2)
    code = Text("while chua_xong:", font=k.MONO, font_size=26, color=k.ACCENT)
    code.next_to(frame, UP, buff=0.1).align_to(frame, LEFT).shift(RIGHT * 0.2)
    back = CurvedArrow(pick.get_bottom() + DOWN * 0.15, seen.get_bottom() + DOWN * 0.1, angle=-0.4,
                       color=k.GREEN, stroke_width=4)
    back_tag = k.label("kết quả thật", 24, k.GREEN).next_to(back, DOWN, buff=0.1)
    sc.run(Create(frame), Write(code), run_time=0.9)
    sc.run(Create(back), FadeIn(back_tag), run_time=0.7)


def four_parts(sc, s, k):
    heading(sc, k, "Bốn phần của một agent")
    sc.wait_until(sc.at(1))
    core = card(k, "Mô hình ngôn ngữ", k.PURPLE, 30, 22, min_w=3.6).move_to(UP * 0.6)
    core_tag = k.label("suy nghĩ, quyết định", 22, k.MUTED).next_to(core, DOWN, buff=0.15)
    sc.run(GrowFromCenter(core), FadeIn(core_tag), run_time=0.6)

    def part(title, lines, color, pos, mono=False):
        head = k.label(title, 28, color, bold=True)
        body = VGroup(*[(Text(t, font=k.MONO, font_size=22, color=k.INK) if mono else k.label(t, 22, k.INK))
                        for t in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        inner = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        box = RoundedRectangle(corner_radius=0.2, width=inner.width + 0.6, height=inner.height + 0.5,
                               color=color, stroke_width=3).set_fill(color, opacity=0.1)
        return VGroup(box, inner.move_to(box)).move_to(pos)

    memory = part("Bộ nhớ", ["ngân sách: 10 triệu", "vé: 7,2 triệu"], k.ACCENT, LEFT * 4.6 + UP * 0.4)
    tools = part("Công cụ", ["tra_ve()", "dat_phong()"], k.GOLD, RIGHT * 4.6 + UP * 0.4, mono=True)
    planning = part("Lập kế hoạch", ["1. tra vé", "2. tính ngân sách", "3. đặt phòng"], k.GREEN, DOWN * 1.9)
    links = [Line(memory.get_right(), core.get_left(), color=k.MUTED, stroke_width=3),
             Line(core.get_right(), tools.get_left(), color=k.MUTED, stroke_width=3),
             Line(core_tag.get_bottom(), planning.get_top(), color=k.MUTED, stroke_width=3)]
    for i, box in ((2, memory), (3, tools), (4, planning)):
        sc.wait_until(sc.at(i))
        sc.run(FadeIn(box, scale=0.9), Create(links[i - 2]), run_time=0.6)

    sc.wait_until(sc.at(5))
    whole = VGroup(core, core_tag, memory, tools, planning, *links)
    later = card(k, "mỗi phần sẽ có một tập riêng", k.INK, 32, 40).move_to(DOWN * 0.2)
    later[0].set_fill("#0f1117", opacity=0.92)
    sc.run(whole.animate.set_opacity(0.3), run_time=0.5)
    sc.run(FadeIn(later), run_time=0.5)


def takeaway(sc, s, k):
    remember = k.label("Nếu chỉ nhớ một điều", 30, k.MUTED).to_edge(UP, buff=0.6)
    sc.run(FadeIn(remember), run_time=0.4)
    bot = k.label("Chatbot", 32, k.PINK, bold=True)
    q, a = card(k, "câu hỏi", k.MUTED, 24, 10, min_w=1.9), card(k, "trả lời", k.MUTED, 24, 10, min_w=1.9)
    stop = k.label("hết", 26, k.MUTED)
    row1 = VGroup(bot, q, a, stop).arrange(RIGHT, buff=0.8).move_to(UP * 1.3)
    sc.run(FadeIn(bot), FadeIn(q), run_time=0.4)
    sc.run(GrowArrow(arrow(k, q, a)), FadeIn(a), run_time=0.5)
    sc.run(GrowArrow(arrow(k, a, stop)), FadeIn(stop), run_time=0.4)

    agent = k.label("Agent", 32, k.GREEN, bold=True)
    steps = [card(k, t, c, 24, 12, min_w=1.9) for t, c in (("làm", k.ACCENT), ("nhìn kết quả", k.GOLD), ("sửa", k.GREEN))]
    row2 = VGroup(agent, *steps).arrange(RIGHT, buff=0.8).move_to(DOWN * 0.9).align_to(row1, LEFT)
    back = CurvedArrow(steps[2].get_bottom() + DOWN * 0.05, steps[0].get_bottom() + DOWN * 0.05, angle=-1.0,
                       color=k.GREEN, stroke_width=4)
    sc.run(FadeIn(agent), FadeIn(steps[0]), run_time=0.4)
    sc.run(GrowArrow(arrow(k, steps[0], steps[1])), FadeIn(steps[1]), run_time=0.4)
    sc.run(GrowArrow(arrow(k, steps[1], steps[2])), FadeIn(steps[2]), run_time=0.4)
    sc.run(Create(back), run_time=0.6)
    again = k.label("lặp lại đến khi xong", 24, k.GREEN).next_to(back, DOWN, buff=0.1)
    sc.run(FadeIn(again), run_time=0.4)

    sc.wait_until(sc.at(1))
    nxt = k.label("Tập sau: kiến trúc của một agent", 28, k.ACCENT, bold=True).to_edge(DOWN, buff=0.4)
    sc.run(FadeIn(nxt, shift=UP * 0.2), run_time=0.6)


# ── Thumbnail 1280x720 (core/explainer.render_thumbnail) ─────
THUMB_TEXT = {
    "vi": ("TẬP 1  ·  BẢN CHẤT LLM AGENT", "LLM Agent", "là gì?", "Khác chatbot ở đâu?", ("Làm", "Nhìn", "Sửa")),
    "en": ("EPISODE 1  ·  LLM AGENTS FROM SCRATCH", "LLM Agents", "explained", "How are they not chatbots?",
           ("Act", "Check", "Fix")),
}


def thumbnail(sc, k, lang="vi"):
    tag_text, line1, line2, sub_text, steps = THUMB_TEXT.get(lang, THUMB_TEXT["vi"])

    def node(text, color):
        txt = k.label(text, 48, k.INK, bold=True)
        box = RoundedRectangle(corner_radius=0.3, width=txt.width + 0.8, height=txt.height + 0.6, color=color,
                               stroke_width=6).set_fill(color, opacity=0.18)
        return VGroup(box, txt.move_to(box))

    tag = k.label(tag_text, 30, k.MUTED, bold=True).to_corner(UP + LEFT, buff=0.6)
    left = VGroup(k.label(line1, 120, k.INK, bold=True), k.label(line2, 120, k.GOLD, bold=True))
    left.arrange(DOWN, aligned_edge=LEFT, buff=0.25).scale_to_fit_width(6.9).to_edge(LEFT, buff=0.75).shift(DOWN * 0.2)
    sub = k.label(sub_text, 40, k.ACCENT, bold=True, max_width=6.9).next_to(left, DOWN, aligned_edge=LEFT, buff=0.45)

    center = RIGHT * 4.1 + DOWN * 0.1
    a = node(steps[0], k.ACCENT).move_to(center + UP * 2.0)
    b = node(steps[1], k.GOLD).move_to(center + RIGHT * 1.6 + DOWN * 1.3)
    c = node(steps[2], k.GREEN).move_to(center + LEFT * 1.6 + DOWN * 1.3)
    overflow = b.get_right()[0] - 6.75  # chữ dài (bản tiếng Anh) thì lùi nút vào trong khung
    if overflow > 0:
        b.shift(LEFT * overflow)
    arrows = VGroup(
        CurvedArrow(a.get_right() + RIGHT * 0.05, b.get_top() + UP * 0.05, angle=-0.9, color=k.MUTED, stroke_width=7),
        CurvedArrow(b.get_bottom() + LEFT * 0.4 + DOWN * 0.05, c.get_bottom() + RIGHT * 0.4 + DOWN * 0.05,
                    angle=-0.9, color=k.MUTED, stroke_width=7),
        CurvedArrow(c.get_top() + UP * 0.05, a.get_left() + LEFT * 0.05, angle=-0.9, color=k.MUTED, stroke_width=7),
    )
    dot = Dot(arrows[0].point_from_proportion(0.5), color=k.GOLD, radius=0.16)
    sc.add(tag, left, sub, arrows, a, b, c, dot)
