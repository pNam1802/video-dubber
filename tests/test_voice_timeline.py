"""Xếp giọng đọc liền mạch — xem core/voice_timeline.py.

Bug thật (job #32): mỗi câu đặt đúng thời điểm câu gốc, tiếng Việt đọc nhanh
hơn nên sau mỗi câu là khoảng lặng — 34 chỗ nghỉ ≥ 0.5s, dài nhất 3.1s.
"""
from __future__ import annotations

import pytest

from core.voice_timeline import MAX_LAG, MAX_LEAD, MAX_SPEED, PAUSE, plan_timeline


def _ends(slots, durations, plan):
    return [p.start + d / p.speed for p, d in zip(plan, durations)]


def test_short_voice_closes_gap_instead_of_leaving_silence():
    # Câu 1 đọc 3s trong khe 5s — câu 2 phải vào sớm chứ không đợi tới giây 5.
    slots = [(0.0, 5.0), (5.0, 9.0)]
    plan = plan_timeline(slots, [3.0, 3.0])
    assert plan[1].start == max(3.0 + PAUSE, 5.0 - MAX_LEAD)
    assert plan[1].start < 5.0


def test_never_starts_more_than_max_lead_early():
    plan = plan_timeline([(0.0, 2.0), (10.0, 12.0)], [1.0, 1.0])
    assert plan[1].start == 10.0 - MAX_LEAD


def test_never_overlaps_previous_sentence():
    slots = [(i * 3.0, i * 3.0 + 3.0) for i in range(10)]
    durations = [3.4] * 10  # dài hơn khe một chút
    plan = plan_timeline(slots, durations)
    ends = _ends(slots, durations, plan)
    for i in range(1, len(plan)):
        assert plan[i].start >= ends[i - 1] + PAUSE - 1e-6


def test_speeds_up_only_when_next_sentence_would_lag_too_much():
    slots = [(0.0, 4.0), (4.0, 8.0)]
    relaxed = plan_timeline(slots, [4.5, 1.0])  # chỉ trễ 0.75s < MAX_LAG -> giữ tốc độ thường
    assert relaxed[0].speed == 1.0
    tight = plan_timeline(slots, [7.0, 1.0])
    assert 1.0 < tight[0].speed <= MAX_SPEED
    assert tight[1].start <= 4.0 + MAX_LAG + 1e-6


def test_speed_capped():
    plan = plan_timeline([(0.0, 1.0), (1.0, 2.0)], [20.0, 1.0])
    assert plan[0].speed == MAX_SPEED


def test_fixed_speed_plan_only_shifts_starts():
    plan = plan_timeline([(0.0, 1.0), (1.0, 2.0)], [5.0, 1.0], max_speed=1.0)
    assert all(p.speed == 1.0 for p in plan)
    assert plan[1].start == 5.0 + PAUSE


# ── Tốc độ chung + lấp khoảng hở (job #33: giọng Việt = 0.92 lần bản gốc) ──
from core.voice_timeline import MIN_BASE_SPEED, MIN_FILL_FACTOR, base_speed, fill_gaps  # noqa: E402


def test_base_speed_slows_down_when_vietnamese_is_shorter():
    assert base_speed([(0, 10), (10, 20)], [9.2, 9.2]) == pytest.approx(0.92)


def test_base_speed_never_speeds_up_and_is_bounded():
    assert base_speed([(0, 10)], [12.0]) == 1.0
    assert base_speed([(0, 10)], [5.0]) == MIN_BASE_SPEED


def test_fill_gaps_slows_sentence_followed_by_gap():
    slots = [(0.0, 5.0), (6.0, 9.0)]
    durations = [3.0, 2.0]
    plan = plan_timeline(slots, durations)  # câu 2 vào ở 4.0 -> hở 1.0s sau câu 1
    factors = fill_gaps(durations, plan)
    assert MIN_FILL_FACTOR <= factors[0] < 1.0
    assert factors[1] == 1.0  # câu cuối không có gì phía sau để lấp


def test_fill_gaps_never_closes_below_pause():
    slots = [(0.0, 5.0), (6.0, 9.0)]
    durations = [3.6, 2.0]
    plan = plan_timeline(slots, durations)
    factors = fill_gaps(durations, plan)
    end = plan[0].start + durations[0] / factors[0]
    assert plan[1].start - end >= PAUSE - 1e-6


def test_fill_gaps_leaves_sped_up_sentences_alone():
    plan = plan_timeline([(0.0, 2.0), (2.0, 3.0)], [6.0, 0.5])
    assert plan[0].speed > 1.0
    assert fill_gaps([6.0, 0.5], plan)[0] == 1.0
