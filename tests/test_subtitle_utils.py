"""segments_to_srt(): chưa từng có test riêng, và vừa đổi tên chế độ từ
'vi'/'en' sang 'target'/'source' (chuẩn bị cho ngôn ngữ đích khác tiếng
Việt) — khoá lại hành vi để một chuỗi gõ sai không âm thầm rơi về nhánh
'bilingual' mà không ai biết."""
from __future__ import annotations

from core.transcriber import Segment
from utils.subtitle_utils import segments_to_srt

SEGMENTS = [
    Segment(start=0.0, end=1.5, text="Hello.", translated="Xin chào."),
    Segment(start=1.5, end=3.0, text="Goodbye.", translated="Tạm biệt."),
]


def test_source_mode_shows_only_original_text():
    srt = segments_to_srt(SEGMENTS, mode="source")
    assert "Hello." in srt
    assert "Xin chào." not in srt


def test_target_mode_shows_only_translated_text():
    srt = segments_to_srt(SEGMENTS, mode="target")
    assert "Xin chào." in srt
    assert "Hello." not in srt


def test_target_mode_falls_back_to_source_when_untranslated():
    """Câu dịch thất bại (translated rỗng) thì vẫn phải hiện được gì đó,
    không để trống hẳn dòng phụ đề."""
    untranslated = [Segment(start=0.0, end=1.0, text="Hello.", translated="")]
    srt = segments_to_srt(untranslated, mode="target")
    assert "Hello." in srt


def test_bilingual_mode_shows_both_lines():
    srt = segments_to_srt(SEGMENTS, mode="bilingual")
    assert "Hello." in srt
    assert "Xin chào." in srt


def test_none_mode_is_same_as_bilingual():
    """mode="none" chỉ có nghĩa ở tầng pipeline (không xuất file .srt) —
    bản thân hàm này không có nhánh riêng cho "none", nên coi như bilingual."""
    assert segments_to_srt(SEGMENTS, mode="none") == segments_to_srt(SEGMENTS, mode="bilingual")


def test_output_written_to_file(tmp_path):
    out = tmp_path / "sub.srt"
    content = segments_to_srt(SEGMENTS, mode="bilingual", output_path=out)
    assert out.read_text(encoding="utf-8") == content
    assert content.startswith("1\n00:00:00,000 --> 00:00:01,500\n")
