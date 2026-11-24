"""
core/youtube.py
Đăng video lên kênh YouTube của admin qua YouTube Data API v3, gọi HTTP thẳng bằng requests
(không cần google-api-python-client).

Kết nối kênh: OAuth 2.0 "offline" một lần (app/youtube.py), app giữ refresh token đã mã hoá
bằng khoá suy ra từ FLASK_SECRET_KEY. Mỗi lần đăng thì đổi refresh token lấy access token mới.

Lưu ý của YouTube: project Google Cloud chưa qua audit thì video tải lên qua API bị khoá ở chế
độ riêng tư. Vì vậy mặc định đăng "private" (có thể kèm lịch publishAt), người dùng tự duyệt
trên YouTube Studio.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Optional

import requests

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
API = "https://www.googleapis.com/youtube/v3"
UPLOAD_API = "https://www.googleapis.com/upload/youtube/v3"
#: youtube.upload để tải video; youtube.force-ssl để gắn phụ đề (captions.insert) và đọc tên kênh.
SCOPES = ("https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.force-ssl")
TIMEOUT = 30


class YouTubeError(RuntimeError):
    pass


# ── Mã hoá refresh token khi lưu DB ──────────────────────────
def _fernet(secret_key: str):
    from cryptography.fernet import Fernet

    digest = hashlib.sha256(b"youtube-token:" + secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def seal(token: str, secret_key: str) -> str:
    return _fernet(secret_key).encrypt(token.encode("utf-8")).decode("ascii")


def unseal(sealed: str, secret_key: str) -> str:
    from cryptography.fernet import InvalidToken

    try:
        return _fernet(secret_key).decrypt(sealed.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise YouTubeError("Token YouTube đã lưu không đọc được, hãy kết nối lại kênh.") from exc


# ── OAuth ────────────────────────────────────────────────────
def auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    from urllib.parse import urlencode

    return AUTH_URL + "?" + urlencode({
        "client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent",
        "include_granted_scopes": "true", "state": state,
    })


def _check(res: requests.Response, what: str) -> dict:
    try:
        data = res.json()
    except ValueError:
        data = {}
    if res.status_code >= 400:
        err = data.get("error")
        detail = (err.get("message") if isinstance(err, dict) else data.get("error_description") or err) or res.text[:300]
        raise YouTubeError(f"{what}: {detail}")
    return data


def exchange_code(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """Mã code của Google -> {"access_token", "refresh_token", ...}."""
    data = _check(requests.post(TOKEN_URL, data={
        "code": code, "client_id": client_id, "client_secret": client_secret,
        "redirect_uri": redirect_uri, "grant_type": "authorization_code"}, timeout=TIMEOUT), "Không đổi được mã Google")
    if not data.get("refresh_token"):
        raise YouTubeError("Google không trả refresh token. Gỡ quyền của app ở myaccount.google.com rồi kết nối lại.")
    missing = [s for s in SCOPES if s not in str(data.get("scope", "")).split()]
    if missing:
        raise YouTubeError("Chưa cấp đủ quyền YouTube. Hãy kết nối lại và đánh dấu mọi ô quyền.")
    return data


def access_token(refresh_token: str, client_id: str, client_secret: str) -> str:
    data = _check(requests.post(TOKEN_URL, data={
        "refresh_token": refresh_token, "client_id": client_id, "client_secret": client_secret,
        "grant_type": "refresh_token"}, timeout=TIMEOUT), "Không làm mới được quyền YouTube")
    return data["access_token"]


def revoke(token: str) -> None:
    try:
        requests.post(REVOKE_URL, data={"token": token}, timeout=TIMEOUT)
    except requests.RequestException:
        pass  # gỡ ở phía app vẫn được, Google tự hết hạn token không dùng


def channel_title(token: str) -> Optional[str]:
    data = _check(requests.get(f"{API}/channels", params={"part": "snippet", "mine": "true"},
                               headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT), "Không đọc được kênh")
    items = data.get("items") or []
    if not items:
        raise YouTubeError("Tài khoản Google này chưa có kênh YouTube.")
    return (items[0].get("snippet") or {}).get("title")


# ── Tải video + phụ đề ───────────────────────────────────────
#: Thể loại "Education" của YouTube.
CATEGORY_EDUCATION = "27"
UPLOAD_TIMEOUT = 1800


def _plain(text: str, limit: int) -> str:
    """YouTube từ chối tiêu đề/mô tả có dấu < >; cắt theo giới hạn của trường."""
    return str(text or "").replace("<", "‹").replace(">", "›").strip()[:limit]


def video_body(meta: dict, language: str, publish_at: Optional[str] = None) -> dict:
    """Metadata gửi kèm video. Có publish_at thì video riêng tư cho tới giờ đó rồi tự công khai."""
    tags, used = [], 0
    for tag in meta.get("tags") or []:
        tag = _plain(tag, 60).replace('"', "")
        if tag and used + len(tag) + 3 <= 480:  # YouTube giới hạn ~500 ký tự cho cả danh sách tag
            tags.append(tag)
            used += len(tag) + 3
    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "embeddable": True}
    if publish_at:
        status["publishAt"] = publish_at
    return {"snippet": {"title": _plain(meta.get("title"), 100) or "Video", "description": _plain(meta.get("description"), 4900),
                        "tags": tags, "categoryId": CATEGORY_EDUCATION, "defaultLanguage": language,
                        "defaultAudioLanguage": language},
            "status": status}


def upload_video(token: str, path, body: dict) -> str:
    """Tải lên kiểu resumable (một lần PUT cả file). Trả về id video."""
    from pathlib import Path

    path = Path(path)
    size = path.stat().st_size
    auth = {"Authorization": f"Bearer {token}"}
    start = requests.post(f"{UPLOAD_API}/videos", params={"uploadType": "resumable", "part": "snippet,status"},
                          headers={**auth, "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                          json=body, timeout=TIMEOUT)
    _check(start, "YouTube không nhận yêu cầu tải video")
    location = start.headers.get("Location")
    if not location:
        raise YouTubeError("YouTube không trả địa chỉ tải video lên.")
    with path.open("rb") as fh:
        res = requests.put(location, data=fh, headers={**auth, "Content-Type": "video/mp4", "Content-Length": str(size)},
                           timeout=UPLOAD_TIMEOUT)
    video_id = _check(res, "Tải video lên YouTube lỗi").get("id")
    if not video_id:
        raise YouTubeError("YouTube không trả id video.")
    return video_id


def upload_caption(token: str, video_id: str, language: str, name: str, srt: bytes) -> str:
    """Gắn một file phụ đề .srt vào video (multipart). Trả về id phụ đề."""
    import json
    import secrets

    boundary = "nora" + secrets.token_hex(12)
    meta = json.dumps({"snippet": {"videoId": video_id, "language": language, "name": name, "isDraft": False}},
                      ensure_ascii=False)
    body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
            f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n").encode("utf-8") + srt + \
        f"\r\n--{boundary}--\r\n".encode("ascii")
    res = requests.post(f"{UPLOAD_API}/captions", params={"uploadType": "multipart", "part": "snippet"},
                        headers={"Authorization": f"Bearer {token}",
                                 "Content-Type": f"multipart/related; boundary={boundary}"},
                        data=body, timeout=TIMEOUT * 4)
    return _check(res, f"Không gắn được phụ đề {language}").get("id", "")


# ── Danh sách phát ───────────────────────────────────────────
def list_playlists(token: str) -> list:
    """Danh sách phát của kênh (tối đa 50, đủ cho một kênh cá nhân): [{"id", "title"}]."""
    data = _check(requests.get(f"{API}/playlists", params={"part": "snippet", "mine": "true", "maxResults": 50},
                               headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT),
                  "Không đọc được danh sách phát")
    return [{"id": item["id"], "title": (item.get("snippet") or {}).get("title", "")} for item in data.get("items") or []]


def add_to_playlist(token: str, playlist_id: str, video_id: str) -> None:
    _check(requests.post(f"{API}/playlistItems", params={"part": "snippet"},
                         headers={"Authorization": f"Bearer {token}"},
                         json={"snippet": {"playlistId": playlist_id,
                                           "resourceId": {"kind": "youtube#video", "videoId": video_id}}},
                         timeout=TIMEOUT), "Không thêm được video vào danh sách phát")
