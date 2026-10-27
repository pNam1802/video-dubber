"""Tua nhanh hình ở khoảng lặng cho khớp giọng đọc."""
from __future__ import annotations

import subprocess

import pytest

from core.transcriber import Segment
from core.video_retime import is_identity, map_time, plan_retime
from tests.test_jobs import _awaiting_review_job, _upload_and_capture_config


def test_short_voice_speeds_window_up():
    # Câu 0 nói trong khe 0-10s nhưng giọng Việt chỉ 4s -> tua cửa sổ 10s.
    windows = plan_retime([(0, 9), (10, 14)], [6.0, 3.75], video_duration=14)
    assert windows[0].speed == pytest.approx(10 / 6.25)
    assert windows[1].speed == pytest.approx(1.0)


def test_speed_capped_and_never_slower():
    windows = plan_retime([(0, 2), (2, 30)], [5.0, 3.0], video_duration=30)
    assert windows[0].speed == 1.0      # giọng dài hơn khe: hình KHÔNG chậm lại
    assert windows[1].speed == 2.0      # khe 28s cho giọng 3s: chặn ở 2×


def test_intro_kept_and_windows_cover_video():
    windows = plan_retime([(5, 8), (8, 12)], [2.0, 2.0], video_duration=12)
    assert (windows[0].src_start, windows[0].src_end, windows[0].speed) == (0, 5, 1.0)
    assert windows[-1].src_end == 12
    for a, b in zip(windows, windows[1:]):
        assert a.src_end == b.src_start
        assert a.out_end == pytest.approx(b.out_start)


def test_map_time_is_piecewise_linear():
    windows = plan_retime([(0, 4), (8, 10)], [1.75, 1.75], video_duration=12)
    # Cửa sổ 0-8 tua 2× -> 8s gốc thành 4s.
    assert map_time(windows, 0) == 0
    assert map_time(windows, 8) == pytest.approx(4)
    assert map_time(windows, 4) == pytest.approx(2)


def test_identity_when_voice_fills_slots():
    windows = plan_retime([(0, 5), (5, 10)], [4.75, 4.75], video_duration=10)
    assert is_identity(windows)


def _make_video(path, seconds=6):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=500:duration={seconds}",
                    str(path)], check=True)


def _probe(path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout)


def test_composer_really_shortens_video(tmp_path, monkeypatch):
    """Chạy ffmpeg thật: bộ lọc setpts từng khúc + atempo phải hợp lệ."""
    import core.video_composer as vc

    monkeypatch.setattr(vc, "OUTPUT_DIR", tmp_path / "out")
    video = tmp_path / "in.mp4"
    _make_video(video, 6)
    voices = []
    for i in range(2):
        p = tmp_path / f"v{i}.mp3"
        _tone(p, 1.0)
        voices.append(p)
    segs = [Segment(0, 2, "a", "x"), Segment(3, 5, "b", "y")]
    windows = plan_retime([(0, 2), (3, 5)], [1.0, 1.0], video_duration=6)
    starts = [map_time(windows, 0), map_time(windows, 3)]
    out = vc.VideoComposer().compose(video, segs, voices, start_times=starts, retime=windows)
    # 0-3s tua 3/1.25 -> chặn 2× = 1.5s; 3-6s cũng 1.5s -> ~3s.
    assert _probe(out) == pytest.approx(windows[-1].out_end, abs=0.3)
    assert _probe(out) < 4.0


# ── Lưu lựa chọn từ form tới giai đoạn 2 ────────────────────────
def test_upload_retime_checked_by_default_form(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch, retime_video="on")
    assert response.status_code == 202
    assert config.max_video_speed == 2.0


def test_upload_retime_unchecked(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch)
    assert config.max_video_speed == 0.0


def test_continue_passes_retime(app, as_user, user, monkeypatch, tmp_path):
    import app.api as api_mod

    captured = {}
    monkeypatch.setattr(api_mod, "continue_job", lambda flask_app, job_id, config: captured.update(config=config))
    job_id, _, _ = _awaiting_review_job(app, user, tmp_path, max_video_speed=2.0)
    as_user.post(f"/api/jobs/{job_id}/continue")
    assert captured["config"].max_video_speed == 2.0


def test_index_has_retime_checkbox_checked(as_user):
    html = as_user.get("/").get_data(as_text=True)
    assert 'name="retime_video"' in html and 'id="retime_video" checked' in html
