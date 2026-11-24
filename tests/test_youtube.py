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
