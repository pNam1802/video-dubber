"""Đăng YouTube: kết nối kênh (app/youtube.py) và lời gọi API (core/youtube.py), không gọi mạng thật."""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from app.models import User
from core import youtube as yt


def test_seal_roundtrip_and_wrong_key():
    sealed = yt.seal("1//refresh", "key-a")
    assert "refresh" not in sealed
    assert yt.unseal(sealed, "key-a") == "1//refresh"
    with pytest.raises(yt.YouTubeError):
        yt.unseal(sealed, "key-b")


def test_auth_url_asks_offline_upload_and_captions():
    q = parse_qs(urlparse(yt.auth_url("cid", "https://x/cb", "st")).query)
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"] and q["state"] == ["st"]
    assert set(q["scope"][0].split()) == set(yt.SCOPES)


def test_connect_is_admin_only(as_user):
    assert as_user.get("/quan-tri/youtube/ket-noi").status_code == 403
    assert as_user.get("/api/youtube").status_code == 403


def test_connect_redirects_to_google_with_state(as_admin):
    res = as_admin.get("/quan-tri/youtube/ket-noi?next=/quan-tri/xuong-kich-ban/1")
    assert res.status_code == 302 and res.location.startswith(yt.AUTH_URL)
    q = parse_qs(urlparse(res.location).query)
    assert q["redirect_uri"][0].endswith("/quan-tri/youtube/callback")
    with as_admin.session_transaction() as s:
        assert s["youtube_oauth_state"] == q["state"][0]


def test_callback_rejects_wrong_state(as_admin, app, monkeypatch):
    monkeypatch.setattr(yt, "exchange_code", lambda *a: pytest.fail("không được đổi mã khi state sai"))
    as_admin.get("/quan-tri/youtube/ket-noi")
    res = as_admin.get("/quan-tri/youtube/callback?state=sai&code=abc")
    assert res.status_code == 302
    with app.app_context():
        assert User.query.filter_by(username="boss").one().youtube_token is None


def test_callback_stores_sealed_token_and_channel(as_admin, app, monkeypatch):
    monkeypatch.setattr(yt, "exchange_code", lambda code, *a: {"access_token": "at", "refresh_token": "rt-" + code})
    monkeypatch.setattr(yt, "channel_title", lambda token: "Kênh Nam")
    as_admin.get("/quan-tri/youtube/ket-noi?next=/quan-tri/xuong-kich-ban/1")
    with as_admin.session_transaction() as s:
        state = s["youtube_oauth_state"]
    res = as_admin.get(f"/quan-tri/youtube/callback?state={state}&code=abc")
    assert res.location.endswith("/quan-tri/xuong-kich-ban/1")
    with app.app_context():
        user = User.query.filter_by(username="boss").one()
        assert user.youtube_channel == "Kênh Nam"
        assert "rt-abc" not in user.youtube_token
        assert yt.unseal(user.youtube_token, app.config["SECRET_KEY"]) == "rt-abc"
    assert as_admin.get("/api/youtube").get_json() == {"connected": True, "channel": "Kênh Nam"}

    monkeypatch.setattr(yt, "revoke", lambda token: None)
    assert as_admin.delete("/api/youtube").get_json()["connected"] is False
    with app.app_context():
        assert User.query.filter_by(username="boss").one().youtube_token is None


def test_next_param_cannot_redirect_offsite(as_admin, monkeypatch):
    as_admin.get("/quan-tri/youtube/ket-noi?next=//evil.example")
    with as_admin.session_transaction() as s:
        state = s["youtube_oauth_state"]
    res = as_admin.get(f"/quan-tri/youtube/callback?state={state}&error=access_denied")
    assert "evil" not in res.location


class _Res:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._data, self.headers, self.text = status, data or {}, headers or {}, ""

    def json(self):
        return self._data


def test_video_body_cleans_fields():
    body = yt.video_body({"title": "<b>Agent</b> " + "x" * 200, "description": "a > b", "tags": ['"llm"', "x" * 70] * 20},
                         "vi", "2099-01-01T00:00:00Z")
    assert "<" not in body["snippet"]["title"] and len(body["snippet"]["title"]) == 100
    assert "›" in body["snippet"]["description"]
    assert body["snippet"]["tags"][0] == "llm" and sum(len(t) + 3 for t in body["snippet"]["tags"]) <= 480
    assert body["snippet"]["categoryId"] == "27" and body["status"]["publishAt"] == "2099-01-01T00:00:00Z"
    assert "publishAt" not in yt.video_body({"title": "t"}, "vi")["status"]


def test_upload_video_resumable(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"12345")
    seen = {}

    def post(url, params, headers, json, timeout):
        seen["post"] = (params, headers["X-Upload-Content-Length"], json)
        return _Res(200, {}, {"Location": "https://upload/session"})

    def put(url, data, headers, timeout):
        seen["put"] = (url, data.read(), headers["Content-Length"])
        return _Res(200, {"id": "VID"})

    monkeypatch.setattr(yt.requests, "post", post)
    monkeypatch.setattr(yt.requests, "put", put)
    assert yt.upload_video("tok", video, {"snippet": {}}) == "VID"
    assert seen["post"][0] == {"uploadType": "resumable", "part": "snippet,status"} and seen["post"][1] == "5"
    assert seen["put"] == ("https://upload/session", b"12345", "5")


def test_upload_errors_are_readable(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"1")
    monkeypatch.setattr(yt.requests, "post", lambda *a, **k: _Res(403, {"error": {"message": "quotaExceeded"}}))
    with pytest.raises(yt.YouTubeError, match="quotaExceeded"):
        yt.upload_video("tok", video, {})


def test_upload_caption_multipart(monkeypatch):
    seen = {}

    def post(url, params, headers, data, timeout):
        seen.update(url=url, params=params, ctype=headers["Content-Type"], data=data)
        return _Res(200, {"id": "CAP"})

    monkeypatch.setattr(yt.requests, "post", post)
    assert yt.upload_caption("tok", "VID", "vi", "Tiếng Việt", b"1\n00:00:00,000 --> 00:00:01,000\nchao\n") == "CAP"
    boundary = seen["ctype"].split("boundary=")[1]
    assert seen["data"].startswith(f"--{boundary}\r\n".encode()) and seen["data"].endswith(f"--{boundary}--\r\n".encode())
    assert '"videoId": "VID"'.encode() in seen["data"] and "Tiếng Việt".encode() in seen["data"]


def test_playlist_helpers(monkeypatch):
    seen = {}
    monkeypatch.setattr(yt.requests, "get", lambda url, params, headers, timeout: _Res(200, {"items": [
        {"id": "PL123", "snippet": {"title": "Bản Chất LLM Agent"}}]}))

    def post(url, params, headers, json, timeout):
        seen.update(url=url, json=json)
        return _Res(200, {"id": "item"})

    monkeypatch.setattr(yt.requests, "post", post)
    assert yt.list_playlists("tok") == [{"id": "PL123", "title": "Bản Chất LLM Agent"}]
    yt.add_to_playlist("tok", "PL123", "VID")
    assert seen["url"].endswith("/playlistItems")
    assert seen["json"]["snippet"] == {"playlistId": "PL123", "resourceId": {"kind": "youtube#video", "videoId": "VID"}}


def test_playlists_route_needs_connection(as_admin, app, monkeypatch):
    assert as_admin.get("/api/youtube/playlists").status_code == 400
    with app.app_context():
        from app.extensions import db

        User.query.filter_by(username="boss").one().youtube_token = yt.seal("rt", app.config["SECRET_KEY"])
        db.session.commit()
    monkeypatch.setattr(yt, "access_token", lambda rt, cid, sec: "at")
    monkeypatch.setattr(yt, "list_playlists", lambda token: [{"id": "PL1234567890", "title": "A"}])
    assert as_admin.get("/api/youtube/playlists").get_json() == [{"id": "PL1234567890", "title": "A"}]


def test_set_thumbnail_posts_png(monkeypatch):
    seen = {}

    def post(url, params, headers, data, timeout):
        seen.update(url=url, params=params, ctype=headers["Content-Type"], data=data)
        return _Res(200, {"items": []})

    monkeypatch.setattr(yt.requests, "post", post)
    yt.set_thumbnail("tok", "VID", b"\x89PNG")
    assert seen["url"].endswith("/thumbnails/set") and seen["params"] == {"videoId": "VID", "uploadType": "media"}
    assert seen["ctype"] == "image/png" and seen["data"] == b"\x89PNG"
