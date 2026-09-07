"""Đăng nhập bằng mật khẩu, đăng nhập bằng Google, phân quyền cơ bản."""
from __future__ import annotations

import pytest

from app.extensions import db
from app.models import Role, User
from tests.conftest import PASSWORD, make_user


# ── Cấu trúc ứng dụng ────────────────────────────────────────
def test_blueprints_registered(app):
    assert set(app.blueprints) == {"main", "auth", "api", "admin", "oauth"}


def test_session_hardening(app):
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["PERMANENT_SESSION_LIFETIME"].days == 7


def test_healthz(client):
    """Healthz phải báo đúng theo môi trường đang chạy.

    Máy dev hoặc CI có thể không cài ffmpeg — khi đó 503 mới là câu trả lời
    đúng, không phải lỗi. Test kiểm tra logic, không đòi hỏi ffmpeg có mặt.
    """
    response = client.get("/healthz")
    checks = response.get_json()["checks"]
    assert checks["database"] is True

    healthy = all(checks.values())
    assert response.status_code == (200 if healthy else 503)


# ── Đăng ký và đăng nhập ─────────────────────────────────────
def test_register_then_reach_home(app, client):
    assert client.post("/register", data={"username": "moi", "password": PASSWORD}).status_code == 302
    assert client.get("/").status_code == 200


def test_register_saves_optional_email(app, client):
    client.post("/register", data={"username": "moi", "password": PASSWORD, "email": "moi@example.com"})
    with app.app_context():
        user = User.query.filter_by(username="moi").first()
        assert user.email == "moi@example.com"
        assert user.notify_email is True  # bật mặc định


def test_register_without_email_still_works(app, client):
    """Email không bắt buộc — đăng ký vẫn qua, chỉ là không nhận được thông báo."""
    response = client.post("/register", data={"username": "moi", "password": PASSWORD})
    assert response.status_code == 302
    with app.app_context():
        assert User.query.filter_by(username="moi").first().email is None


def test_register_rejects_invalid_email(app, client):
    response = client.post(
        "/register", data={"username": "moi", "password": PASSWORD, "email": "khong-phai-email"},
        follow_redirects=True,
    )
    assert "Email không hợp lệ" in response.get_data(as_text=True)
    with app.app_context():
        assert User.query.filter_by(username="moi").first() is None


def test_duplicate_username_rejected(app, client, user):
    response = client.post("/register", data={"username": "alice", "password": PASSWORD},
                           follow_redirects=True)
    assert "đã tồn tại" in response.get_data(as_text=True)


def test_wrong_password_rejected(app, user, client):
    response = client.post("/login", data={"username": "alice", "password": "sai-roi"},
                           follow_redirects=True)
    assert "Sai thông tin đăng nhập" in response.get_data(as_text=True)


def test_locked_account_cannot_log_in(app, client):
    with app.app_context():
        make_user("bikhoa", is_active=False)
    response = client.post("/login", data={"username": "bikhoa", "password": PASSWORD},
                           follow_redirects=True)
    assert "khoá" in response.get_data(as_text=True)


def test_login_page_is_public_and_has_csrf_field(client):
    body = client.get("/login").get_data(as_text=True)
    assert 'name="csrf_token"' in body
    assert "bootstrap" not in body.lower()


# ── Lỗi trả về đúng định dạng ────────────────────────────────
def test_api_requires_login_returns_json_401(client):
    response = client.get("/api/progress/1")
    assert response.status_code == 401
    assert response.is_json


def test_404_json_under_api_html_elsewhere(as_user):
    assert as_user.get("/api/khong-ton-tai").is_json
    assert not as_user.get("/khong-ton-tai").is_json


def test_oversized_upload_returns_json_413(app, as_user):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    response = as_user.post("/api/upload", data={"blob": "x" * 5000})
    assert response.status_code == 413
    assert response.is_json


def test_csrf_blocks_upload_when_enabled(app, user):
    app.config["WTF_CSRF_ENABLED"] = True
    c = app.test_client()
    response = c.post("/api/upload", data={})
    assert response.status_code == 400
    assert response.is_json


# ── Cài đặt tài khoản (email nhận thông báo) ──────────────────
def test_account_settings_page_renders(as_user):
    response = as_user.get("/cai-dat")
    assert response.status_code == 200
    assert b'name="email"' in response.data


def test_account_settings_updates_email_and_notify(app, as_user, user):
    response = as_user.post("/cai-dat", data={"email": "moi@example.com", "notify_email": "on"})
    assert response.status_code == 302
    with app.app_context():
        u = db.session.get(User, user)
        assert u.email == "moi@example.com"
        assert u.notify_email is True


def test_account_settings_unchecked_box_turns_notify_off(app, as_user, user):
    # Checkbox không tick thì trình duyệt không gửi field đó lên — thiếu
    # "notify_email" trong form nghĩa là người dùng đã bỏ tick, phải tắt.
    as_user.post("/cai-dat", data={"email": "moi@example.com"})
    with app.app_context():
        assert db.session.get(User, user).notify_email is False


def test_account_settings_rejects_invalid_email(app, as_user, user):
    response = as_user.post("/cai-dat", data={"email": "khong-phai-email"}, follow_redirects=True)
    assert "Email không hợp lệ" in response.get_data(as_text=True)
    with app.app_context():
        assert db.session.get(User, user).email is None  # không bị đổi


def test_account_settings_google_user_cannot_edit_email(app, as_user, user):
    """Email của tài khoản Google lấy từ lúc đăng nhập — sửa tay ở đây
    không hợp lý (lần đăng nhập Google kế tiếp sẽ ghi đè lại), nên khoá."""
    with app.app_context():
        u = db.session.get(User, user)
        u.google_sub = "sub-123"
        u.email = "google@example.com"
        db.session.commit()

    as_user.post("/cai-dat", data={"email": "hacked@example.com", "notify_email": "on"})
    with app.app_context():
        assert db.session.get(User, user).email == "google@example.com"


# ── Trang pháp lý công khai ──────────────────────────────────
@pytest.mark.parametrize("path", ["/privacy", "/terms"])
def test_legal_pages_public(client, path):
    assert client.get(path).status_code == 200


# ── Lệnh quản trị ────────────────────────────────────────────
def test_create_admin_command(app):
    result = app.test_cli_runner().invoke(args=["create-admin", "--username", "boss", "--password", PASSWORD])
    assert result.exit_code == 0
    with app.app_context():
        assert User.query.filter_by(username="boss").first().role == Role.ADMIN


def test_set_role_command(app, user):
    result = app.test_cli_runner().invoke(args=["set-role", "alice", "admin"])
    assert result.exit_code == 0
    with app.app_context():
        assert User.query.filter_by(username="alice").first().role == Role.ADMIN


# ── Đăng nhập bằng Google ────────────────────────────────────
class FakeGoogle:
    def __init__(self):
        self.info: dict = {}
        self.fail = False

    def authorize_access_token(self):
        if self.fail:
            raise RuntimeError("state không hợp lệ")
        return {"userinfo": self.info}


@pytest.fixture()
def google(monkeypatch):
    import app.oauth as oauth_mod

    fake = FakeGoogle()
    monkeypatch.setattr(oauth_mod, "oauth", type("O", (), {"google": fake})())
    return fake


def test_google_button_shown_when_configured(client):
    body = client.get("/login").get_data(as_text=True)
    assert "Tiếp tục với Google" in body
    assert "/auth/google/login" in body


def test_google_first_login_creates_account(app, client, google):
    google.info = {"sub": "g-111", "email": "Nam.Nguyen@gmail.com", "email_verified": True,
                   "picture": "https://example.test/a.png"}
    assert client.get("/auth/google/callback").status_code == 302
    with app.app_context():
        created = User.query.filter_by(google_sub="g-111").first()
        assert created is not None
        assert created.username == "nam.nguyen"
        assert created.email == "nam.nguyen@gmail.com"
        assert created.password_hash is None
        assert created.uses_google is True
    assert client.get("/").status_code == 200


def test_google_second_login_reuses_account(app, client, google):
    google.info = {"sub": "g-111", "email": "a@b.com", "email_verified": True}
    client.get("/auth/google/callback")
    app.test_client().get("/auth/google/callback")
    with app.app_context():
        assert User.query.filter_by(google_sub="g-111").count() == 1


def test_google_links_to_existing_account_when_email_verified(app, client, google):
    with app.app_context():
        make_user("cu", email="cu@truong.edu.vn")
    google.info = {"sub": "g-222", "email": "cu@truong.edu.vn", "email_verified": True}
    client.get("/auth/google/callback")
    with app.app_context():
        existing = User.query.filter_by(username="cu").first()
        assert existing.google_sub == "g-222"
        assert existing.password_hash is not None
        assert User.query.filter_by(email="cu@truong.edu.vn").count() == 1


def test_unverified_email_cannot_take_over_account(app, client, google):
    """Nếu bỏ kiểm tra email_verified, ai cũng chiếm được tài khoản bằng email trùng."""
    with app.app_context():
        make_user("nanhan", email="nanhan@truong.edu.vn")
    google.info = {"sub": "ke-gia-mao", "email": "nanhan@truong.edu.vn", "email_verified": False}
    client.get("/auth/google/callback")
    with app.app_context():
        assert User.query.filter_by(username="nanhan").first().google_sub is None
        assert User.query.filter_by(google_sub="ke-gia-mao").count() == 1


def test_google_locked_account_refused(app, client, google):
    with app.app_context():
        make_user("bikhoa", google=True, google_sub="g-333", is_active=False)
    google.info = {"sub": "g-333", "email": "khoa@truong.edu.vn", "email_verified": True}
    client.get("/auth/google/callback")
    assert client.get("/").status_code == 302


def test_google_failure_returns_to_login(client, google):
    google.fail = True
    response = client.get("/auth/google/callback", follow_redirects=True)
    assert "Không đăng nhập được bằng Google" in response.get_data(as_text=True)


def test_google_account_told_to_use_google(app, client, google):
    google.info = {"sub": "g-444", "email": "nam@x.com", "email_verified": True}
    client.get("/auth/google/callback")
    response = app.test_client().post("/login", data={"username": "nam", "password": "doan-bua"},
                                      follow_redirects=True)
    assert "đăng nhập bằng Google" in response.get_data(as_text=True)


def test_username_collision_gets_suffix(app, client, google):
    with app.app_context():
        make_user("nam")
    google.info = {"sub": "g-555", "email": "nam@khac.com", "email_verified": True}
    client.get("/auth/google/callback")
    with app.app_context():
        created = User.query.filter_by(google_sub="g-555").first()
        assert created.username.startswith("nam-")
