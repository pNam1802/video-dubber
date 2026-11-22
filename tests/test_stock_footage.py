"""Chế độ "Video stock" của Đọc kịch bản: chia cảnh, chọn clip Pixabay, cache."""
from __future__ import annotations

import json

import pytest

from core import stock_footage as sf

TIMINGS = [(0.0, 4.0), (4.0, 9.0), (9.0, 15.0), (15.0, 20.0)]


def _hit(id_, duration=10, url=True):
    return {"id": id_, "duration": duration,
            "videos": {"small": {"url": f"https://cdn/{id_}.mp4" if url else "", "width": 1280, "height": 720}}}


def test_parse_shots_orders_clamps_and_forces_first_start():
    raw = json.dumps({"shots": [
        {"start": 3, "queries": ["city night", "street"]},
        {"start": 2, "queries": ["woman writing notebook"]},
        {"start": 99, "queries": ["mountain summit"]},
        {"start": 1, "queries": []},  # không có từ khoá -> bỏ
        "rác",
    ]})
    assert sf.parse_shots(raw, 4) == [(0, ["woman writing notebook"]), (2, ["city night", "street"]),
                                      (3, ["mountain summit"])]


def _boom(_):
    raise RuntimeError("429 RESOURCE_EXHAUSTED")


def test_plan_uses_deepseek_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(sf, "ask_gemini_json", _boom)
    monkeypatch.setattr(sf, "ask_deepseek_json", lambda req: json.dumps(
        {"shots": [{"start": 1, "queries": ["man running sunrise"]}, {"start": 3, "queries": ["city night"]}]}))
    queries, slides = sf.plan(["a", "b", "c", "d"], TIMINGS)
    assert queries == [["man running sunrise"], ["city night"]]
    assert [(a, b) for _, a, b in slides] == [(0.0, 9.0), (9.0, 20.0)]


def test_plan_skips_deepseek_when_gemini_works(monkeypatch):
    monkeypatch.setattr(sf, "ask_gemini_json", lambda req: json.dumps({"shots": [{"start": 1, "queries": ["sky"]}]}))
    monkeypatch.setattr(sf, "ask_deepseek_json", lambda req: pytest.fail("không được gọi DeepSeek"))
    assert sf.plan(["a", "b", "c", "d"], TIMINGS)[0] == [["sky"]]


def test_plan_falls_back_to_generic_shots_when_both_fail(monkeypatch):
    monkeypatch.setattr(sf, "ask_gemini_json", _boom)
    monkeypatch.setattr(sf, "ask_deepseek_json", lambda req: "{}")  # trả về nhưng không có cảnh
    queries, slides = sf.plan(["a", "b", "c", "d"], TIMINGS)
    assert queries and all(q[0] in sf.GENERIC_QUERIES for q in queries)
    assert slides[0][1] == 0.0 and slides[-1][2] == 20.0


def test_choose_clip_prefers_unused_and_long_enough():
    hits = [_hit(1, 3), _hit(2, 4), _hit(3, 12)]
    assert sf.choose_clip(hits, 6.0, used={3})["id"] == 1  # không clip nào đủ dài -> giữ thứ hạng
    assert sf.choose_clip(hits, 6.0, used=set())["id"] == 3
    assert sf.choose_clip(hits, 1.0, used={1, 2, 3}) is None


class _FakePixabay:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.results.get(query, [])


def test_find_clips_avoids_repeats_and_fills_gaps():
    pixabay = _FakePixabay({"a": [_hit(1), _hit(2)], "b": [_hit(1)]})
    slides = [(0, 0.0, 5.0), (1, 5.0, 10.0)]
    chosen = sf.find_clips(pixabay, [["a"], ["b"]], slides)
    # cảnh 2 không lấy lại clip 1 -> rơi xuống từ khoá chung chung, không có gì -> dùng lại clip gần nhất
    assert [hit["id"] for hit in chosen] == [1, 1]
    assert pixabay.queries[:2] == ["a", "b"]


def test_find_clips_raises_when_nothing_found():
    with pytest.raises(sf.StockFootageError):
        sf.find_clips(_FakePixabay({}), [["x"]], [(0, 0.0, 5.0)])


def test_frame_counts_do_not_drift():
    slides = [(0, 0.0, 1.01), (1, 1.01, 2.02), (2, 2.02, 3.03)]
    assert sum(sf.frame_counts(slides)) == round(3.03 * sf.FPS)


class _Response:
    def __init__(self, status=200, payload=None, body=b""):
        self.status_code, self.payload, self.body = status, payload, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload

    def iter_content(self, _size):
        yield self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Session:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_search_is_cached_and_download_reuses_file(tmp_path):
    session = _Session([_Response(payload={"hits": [_hit(7), _hit(8, url=False)]}), _Response(body=b"mp4")])
    pixabay = sf.Pixabay("key", tmp_path, session=session)
    assert [h["id"] for h in pixabay.search("City Night")] == [7]  # bỏ clip không có link tải
    assert [h["id"] for h in pixabay.search("city night")] == [7]  # cache, không gọi lại
    path = pixabay.download(_hit(7))
    assert path.read_bytes() == b"mp4" and pixabay.download(_hit(7)) == path
    assert session.calls == 2
    assert not list(tmp_path.rglob("*.part"))


def test_rate_limit_and_missing_key_give_friendly_errors(tmp_path):
    with pytest.raises(sf.StockFootageError):
        sf.Pixabay("", tmp_path)
    pixabay = sf.Pixabay("key", tmp_path, session=_Session([_Response(status=429)]))
    with pytest.raises(sf.StockFootageError, match="hết lượt"):
        pixabay.search("x")


def test_api_stock_needs_key_then_passes_mode_to_job(app, as_user, monkeypatch):
    import app.api as api

    captured: dict = {}
    monkeypatch.setattr(api, "start_script_job", lambda flask_app, job_id, config: captured.update(config=config))
    data = {"script": "Xin chào các bạn. Đây là kịch bản.", "target_language": "vi",
            "tts_engine": "edge-tts", "tts_voice": "female", "script_images": "stock"}

    monkeypatch.setattr(api, "PIXABAY_API_KEY", "")
    res = as_user.post("/api/script", data=data)
    assert res.status_code == 400 and "PIXABAY_API_KEY" in res.get_json()["error"]

    monkeypatch.setattr(api, "PIXABAY_API_KEY", "key")
    res = as_user.post("/api/script", data=data)
    assert res.status_code == 202, res.get_json()
    assert captured["config"].script_images == "stock"


def test_religious_or_violent_clips_are_filtered(tmp_path):
    quran = {**_hit(1), "tags": "quran, islam, woman reading"}
    book = {**_hit(2), "tags": "woman, book, reading"}
    gun = {**_hit(3), "tags": "man, hand gun"}
    session = _Session([_Response(payload={"hits": [quran, book, gun]})])
    assert [h["id"] for h in sf.Pixabay("key", tmp_path, session=session).search("woman reading")] == [2]
