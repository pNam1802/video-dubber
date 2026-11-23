"""Giọng tự thu: cắt bản thu từng đoạn thành từng câu tại chỗ ngắt hơi; API thu ở Xưởng kịch bản."""
from __future__ import annotations

import io
import shutil
import subprocess

import pytest

from core import recorded_voice as rv

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")


def _take(path, parts):
    """File wav giả: tiếng (sine) và lặng xen kẽ, [(giây, có_tiếng)]."""
    inputs, labels = [], []
    for k, (sec, voiced) in enumerate(parts):
        src = f"sine=frequency=220:duration={sec}" if voiced else f"anullsrc=r=24000:cl=mono:d={sec}"
        inputs += ["-f", "lavfi", "-i", src]
        labels.append(f"[{k}:a]")
    graph = "".join(labels) + f"concat=n={len(parts)}:v=0:a=1,aresample=24000,aformat=channel_layouts=mono"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, str(path)], check=True)
    return path


def test_sentence_cuts_prefer_nearby_silences():
    cuts = rv.sentence_cuts(["a" * 10, "b" * 10, "c" * 10], 9.0, [(2.6, 3.2), (5.0, 5.2), (6.0, 6.5)])
    assert cuts == [pytest.approx(2.9), pytest.approx(6.25)]
    # Không có chỗ lặng nào gần: cắt theo tỉ lệ số ký tự.
    assert rv.sentence_cuts(["a" * 10, "b" * 30], 8.0, []) == [pytest.approx(2.0)]


@needs_ffmpeg
def test_split_take_cuts_inside_pauses(tmp_path):
    take = _take(tmp_path / "take.wav", [(1.0, True), (0.6, False), (1.6, True), (0.6, False), (0.8, True)])
    pieces = rv.split_take(take, ["Câu một ngắn.", "Câu thứ hai dài hơn một chút.", "Câu ba."], tmp_path)
    lengths = [rv.duration(p) for p in pieces]
    assert len(pieces) == 3
    assert lengths[0] == pytest.approx(1.3, abs=0.2)
    assert lengths[1] == pytest.approx(2.2, abs=0.25)
    assert sum(lengths) == pytest.approx(rv.duration(take), abs=0.05)


def test_recording_name_tracks_text():
    assert rv.recording_name(3, "vi", 0, "Xin chào  các bạn") == rv.recording_name(3, "vi", 0, "Xin chào các bạn")
    assert rv.recording_name(3, "vi", 0, "Xin chào") != rv.recording_name(3, "vi", 0, "Tạm biệt")


@needs_ffmpeg
def test_upload_recording_then_video_uses_it(as_admin, as_user, monkeypatch, tmp_path):
    import app.api as api
    from tests.test_studio import _drafted_episode

    monkeypatch.setattr("config.settings.DATA_DIR", tmp_path)
    monkeypatch.setattr("app.storage.commit_volume", lambda: None)
    import app.studio as st
    monkeypatch.setattr(st, "start_task", lambda flask_app, kind, target_id: st.run_task(flask_app, kind, target_id))
    monkeypatch.setattr("core.studio.plan_series", lambda topic, audience, count: {
        "title": "S", "description": "", "style_guide": "", "glossary": [],
        "episodes": [{"title": "T", "goal": "", "key_points": []}]})
    series_id, ep_id = _drafted_episode(as_admin)
    wav = _take(tmp_path / "t.wav", [(1.0, True)]).read_bytes()

    def upload(index, client=as_admin):
        return client.post(f"/api/studio/episodes/{ep_id}/recordings",
                           data={"lang": "vi", "index": str(index), "file": (io.BytesIO(wav), "doan.webm")},
                           content_type="multipart/form-data")

    assert upload(0).status_code == 200
    assert upload(5).status_code == 400
    assert upload(0, as_user).status_code == 403
    ep = as_admin.get(f"/api/studio/series/{series_id}").get_json()["episodes"][0]
    assert ep["paragraphs_vi"] == ["Câu một. Câu hai.", "Câu ba."]
    assert ep["recordings_vi"][0] and ep["recordings_vi"][1] is None
    assert as_admin.get(f"/quan-tri/xuong-kich-ban/thu-am/{ep_id}/{ep['recordings_vi'][0]}").status_code == 200

    captured: dict = {}
    monkeypatch.setattr(api, "start_script_job", lambda flask_app, job_id, config: captured.update(config=config))
    form = {"script": "Câu một. Câu hai.\nCâu ba.", "target_language": "vi", "tts_engine": "edge-tts",
            "tts_voice": "female", "script_images": "explainer", "recorded_episode": str(ep_id)}
    res = as_admin.post("/api/script", data=form)
    assert res.status_code == 400 and "đoạn 2" in res.get_json()["error"]
    assert upload(1).status_code == 200
    assert as_admin.post("/api/script", data=form).status_code == 202
    assert sorted(captured["config"].recorded_paragraphs) == ["0", "1"]


@needs_ffmpeg
def test_pipeline_reads_script_with_recorded_voice(tmp_path):
    pytest.importorskip("pydub")  # đường ghép âm thanh của pipeline (không có trong requirements-dev)
    from core.pipeline import DubbingConfig, DubbingPipeline
    from core.script_reader import split_script
    from core.transcriber import Segment

    text = "Câu một ngắn. Câu thứ hai dài hơn một chút.\nĐoạn hai."
    takes = {"0": str(_take(tmp_path / "p0.wav", [(1.0, True), (0.6, False), (1.6, True)])),
             "1": str(_take(tmp_path / "p1.wav", [(0.9, True)]))}
    segments = [Segment(start=p, end=p, text=s, translated=s) for p, s in split_script(text)]
    config = DubbingConfig(tts_engine="edge-tts", output_dir=tmp_path, target_language="vi",
                           subtitle_mode="target", recorded_paragraphs=takes)
    result = DubbingPipeline(config).synthesize_script(segments)
    assert result.success, result.error
    starts = [t[0] for t in result.play_timings]
    assert len(starts) == 3 and starts[0] == 0.0 and starts[1] > 1.0 and starts[2] > starts[1] + 1.5
