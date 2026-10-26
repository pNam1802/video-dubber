"""
app/__init__.py
Application factory: dựng Flask app, nạp cấu hình, bind extension và blueprint.
"""
from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

from flask import Flask, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from app.errors import register_error_handlers, register_unauthorized_handler
from app.extensions import csrf, db, limiter, login_manager, migrate
from config.settings import (
    MAX_UPLOAD_MB,
    SECRET_KEY,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
    SESSION_LIFETIME_DAYS,
    TRUST_PROXY,
    SQLALCHEMY_DATABASE_URI,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def register_asset_helper(app: Flask) -> None:
    """Cho template gọi asset('css/app.css') để lấy URL có gắn mã băm nội dung.

    Cần mã băm vì file được cache một năm: không có nó thì sửa CSS xong người
    dùng cũ vẫn thấy bản cũ cho tới khi tự xoá cache. Băm tính một lần lúc khởi
    động — file tĩnh không đổi giữa chừng, và trên Modal thì mtime của file
    trong image luôn là epoch 0 nên không dùng mtime được.
    """
    digests: dict[str, str] = {}

    def asset(filename: str) -> str:
        if filename not in digests:
            path = Path(app.static_folder or "") / filename
            try:
                digests[filename] = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
            except OSError:
                digests[filename] = "dev"
        return url_for("static", filename=filename, v=digests[filename])

    app.jinja_env.globals["asset"] = asset


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

    app.config.update(
        SECRET_KEY=SECRET_KEY,
        MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
        SESSION_COOKIE_SAMESITE=SESSION_COOKIE_SAMESITE,
        PERMANENT_SESSION_LIFETIME=timedelta(days=SESSION_LIFETIME_DAYS),
        SQLALCHEMY_DATABASE_URI=SQLALCHEMY_DATABASE_URI,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # Neon tu ngu khi khong ai dung va cat het ket noi cu — ket noi chet
        # nam trong pool lam request dau tien sau do loi AdminShutdown.
        # pre_ping thu ket noi truoc khi dung, recycle bo ket noi qua cu.
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True, "pool_recycle": 280},
        # Mac dinh cua Flask la "no-cache": moi lan chuyen trang trinh duyet
        # van phai hoi lai server truoc khi ve duoc gi — tu Viet Nam la them
        # nua giay cho mot vong di ve. Cache mot nam chi an toan vi asset()
        # gan ma bam noi dung vao URL, doi file la doi URL.
        SEND_FILE_MAX_AGE_DEFAULT=timedelta(days=365),
    )
    if config_overrides:
        app.config.update(config_overrides)

    register_asset_helper(app)

    if TRUST_PROXY:
        # Modal dat mot proxy truoc container; khong co buoc nay thi rate limit
        # dem chung tat ca nguoi dung vao IP cua proxy.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    from app import models  # noqa: F401  — để Alembic thấy được metadata

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(models.User, int(user_id))

    from app.admin import bp as admin_bp
    from app.oauth import bp as oauth_bp, init_oauth
    from app.api import bp as api_bp
    from app.auth import bp as auth_bp
    from app.main import bp as main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(oauth_bp)
    init_oauth(app)

    register_error_handlers(app)
    register_unauthorized_handler()

    from app.cli import register_cli

    register_cli(app)

    return app
