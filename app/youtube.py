"""
app/youtube.py
Kết nối kênh YouTube cho Xưởng kịch bản (chỉ admin): xin quyền OAuth "offline" một lần,
lưu refresh token đã mã hoá vào User. Việc tải video lên nằm ở core/youtube.py.

Dùng chung OAuth client với đăng nhập Google (GOOGLE_CLIENT_ID/SECRET). Trong Google Cloud
cần: bật YouTube Data API v3, thêm redirect URI .../quan-tri/youtube/callback, và thêm hai
scope youtube.upload + youtube.force-ssl vào màn hình đồng ý.
"""
from __future__ import annotations

import secrets

from flask import Blueprint, current_app, flash, jsonify, redirect, request, session, url_for
from flask_login import current_user

from app.extensions import db
from app.permissions import require_admin, require_admin_page
from config.settings import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

bp = Blueprint("youtube", __name__)

STATE_KEY = "youtube_oauth_state"
NEXT_KEY = "youtube_oauth_next"


def _back() -> str:
    target = session.pop(NEXT_KEY, "") or ""
    # Chỉ quay về đường dẫn nội bộ, không cho chuyển hướng ra ngoài.
    return target if target.startswith("/") and not target.startswith("//") else url_for("studio.page")


@bp.get("/quan-tri/youtube/ket-noi")
@require_admin_page
def connect():
    from core.youtube import auth_url

    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        flash("Chưa cấu hình Google OAuth (GOOGLE_CLIENT_ID/SECRET).", "danger")
        return redirect(url_for("studio.page"))
    session[STATE_KEY] = secrets.token_urlsafe(24)
    session[NEXT_KEY] = request.args.get("next", "")
    return redirect(auth_url(GOOGLE_CLIENT_ID, url_for("youtube.callback", _external=True), session[STATE_KEY]))


@bp.get("/quan-tri/youtube/callback")
@require_admin_page
def callback():
    from core.youtube import YouTubeError, channel_title, exchange_code, seal

    expected = session.pop(STATE_KEY, None)
    if not expected or not secrets.compare_digest(expected, request.args.get("state", "")):
        flash("Phiên kết nối YouTube không hợp lệ, hãy thử lại.", "danger")
        return redirect(_back())
    if request.args.get("error") or not request.args.get("code"):
        flash("Bạn đã huỷ kết nối YouTube.", "warning")
        return redirect(_back())
    try:
        token = exchange_code(request.args["code"], GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
                              url_for("youtube.callback", _external=True))
        title = channel_title(token["access_token"])
    except YouTubeError as exc:
        flash(str(exc), "danger")
        return redirect(_back())
    current_user.youtube_token = seal(token["refresh_token"], current_app.config["SECRET_KEY"])
    current_user.youtube_channel = (title or "Kênh YouTube")[:200]
    db.session.commit()
    flash(f"Đã kết nối kênh YouTube “{current_user.youtube_channel}”.", "success")
    return redirect(_back())


@bp.get("/api/youtube")
@require_admin
def status():
    return jsonify({"connected": bool(current_user.youtube_token), "channel": current_user.youtube_channel})


@bp.delete("/api/youtube")
@require_admin
def disconnect():
    from core.youtube import YouTubeError, revoke, unseal

    if current_user.youtube_token:
        try:
            revoke(unseal(current_user.youtube_token, current_app.config["SECRET_KEY"]))
        except YouTubeError:
            pass
    current_user.youtube_token = current_user.youtube_channel = None
    db.session.commit()
    return jsonify({"connected": False, "channel": None})
