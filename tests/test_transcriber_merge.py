"""Gộp đoạn Whisper thành câu — không được ra "câu" dài cả phút.

Bug thật (job #29, video review truyện giọng kể liền mạch): không có khoảng
lặng, Whisper lại cắt giữa câu nên đoạn ghép không bao giờ kết thúc bằng dấu
chấm — một "câu" kéo dài tới 112 giây. Phụ đề dài cả trang, giọng đọc lệch
dần so với hình, và bản dịch vượt giới hạn token.
"""
from __future__ import annotations

from core.transcriber import Segment, Transcriber


def _chunks(texts, step=3.0):
    """Đoạn Whisper liền nhau, không có khoảng lặng giữa các đoạn."""
    return [Segment(start=i * step, end=(i + 1) * step, text=t) for i, t in enumerate(texts)]


def merge(segments):
    return Transcriber._merge_segments_into_sentences(segments, silence_threshold=0.45)


def test_short_sentences_still_merge_at_sentence_end():
    merged = merge(_chunks(["Minjun opens the door", "and steps outside.", "Yena follows."]))
    assert [s.text for s in merged] == ["Minjun opens the door and steps outside.", "Yena follows."]


def test_continuous_narration_is_capped_at_hard_limit():
    merged = merge(_chunks([f"part {i} of a sentence that never ends" for i in range(40)]))
    assert len(merged) > 1
    assert max(s.end - s.start for s in merged) <= Transcriber._HARD_MAX_DURATION


def test_prefers_clause_boundary_after_soft_limit():
    texts = ["they walk past the cars", "and the empty shops", "until night falls,", "then they rest", "in a basement."]
    merged = merge(_chunks(texts))
    # 9 giây (quá SOFT 8s) và kết thúc bằng dấu phẩy -> cắt ở đó.
    assert merged[0].text.endswith("until night falls,")
    assert merged[1].text == "then they rest in a basement."


def test_nothing_is_lost_when_splitting():
    texts = [f"chunk{i}" for i in range(30)]
    merged = merge(_chunks(texts))
    assert " ".join(s.text for s in merged).split() == texts
    assert merged[0].start == 0 and merged[-1].end == 90.0


# ── Tách câu theo mốc thời gian từng từ ──────────────────────
def _words(text, step=0.4, start=0.0):
    out, t = [], start
    for token in text.split():
        out.append((t, t + step - 0.05, " " + token))
        t += step
    return out


def words_to_sentences(words):
    return Transcriber._sentences_from_words(words)


def test_word_level_cuts_exactly_at_sentence_end_inside_whisper_segment():
    """Bug thật (job #32): đoạn Whisper kết thúc giữa câu ("...is only under"
    | "caution, and ...") nên gộp theo đoạn buộc phải cắt giữa câu."""
    words = _words("Their neighborhood is only under caution. Nearby districts are marked safe.")
    sentences = words_to_sentences(words)
    assert [s.text for s in sentences] == [
        "Their neighborhood is only under caution.",
        "Nearby districts are marked safe.",
    ]
    assert sentences[1].start == words[6][0]


def test_word_level_ignores_abbreviation_periods():
    sentences = words_to_sentences(_words("Mr. Kim waves at them. They run."))
    assert [s.text for s in sentences] == ["Mr. Kim waves at them.", "They run."]


def test_word_level_long_sentence_splits_at_comma_after_soft_limit():
    long = " ".join(["word"] * 32) + ", " + " ".join(["more"] * 5) + "."
    sentences = words_to_sentences(_words(long))  # 0.4s/từ -> quá 12s tại dấu phẩy
    assert sentences[0].text.endswith(",")
    assert sentences[1].text.endswith("more.")


def test_word_level_hard_limit():
    sentences = words_to_sentences(_words(" ".join(["word"] * 80)))
    assert max(s.end - s.start for s in sentences) <= Transcriber._WORD_HARD_MAX


def test_word_level_keeps_every_word():
    text = "One two three. Four five, six seven eight nine. Ten."
    sentences = words_to_sentences(_words(text))
    assert " ".join(s.text for s in sentences) == text
