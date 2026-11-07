"""
app/auth.py
Đăng ký, đăng nhập, đăng xuất.
"""
from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db, limiter
from app.oauth import is_enabled as google_enabled
from app.models import User
from config.settings import RATELIMIT_LOGIN, RATELIMIT_REGISTER

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
@limiter.limit(RATELIMIT_LOGIN, methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and not user.password_hash:
            flash("Tài khoản này đăng nhập bằng Google.", "info")
            return render_template("login.html", google_enabled=google_enabled())
        if user and check_password_hash(user.password_hash, password):
            if not user.is_active:
                flash("Tài khoản đã bị khoá!", "danger")
                return render_template("login.html", google_enabled=google_enabled())
            login_user(user)
            session.permanent = True
            flash("Đăng nhập thành công!", "success")
            return redirect(url_for("main.index"))
        flash("Sai thông tin đăng nhập!", "danger")

    return render_template("login.html", google_enabled=google_enabled())


@bp.route("/register", methods=["GET", "POST"])
@limiter.limit(RATELIMIT_REGISTER, methods=["POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        # Email chi de nhan thong bao job — khong dung de dang nhap, khong
        # bat buoc, nen validate nhe: co nhap thi phai giong email that.
        email = request.form.get("email", "").strip()
        if email and "@" not in email:
            flash("Email không hợp lệ.", "danger")
            return render_template("register.html", google_enabled=google_enabled())

        if not username or not password:
            flash("Vui lòng điền đủ!", "danger")
            return render_template("register.html", google_enabled=google_enabled())

        if User.query.filter_by(username=username).first():
            flash("Tên đăng nhập đã tồn tại!", "danger")
        else:
            new_user = User(
                username=username, password_hash=generate_password_hash(password), email=email or None,
            )
            db.session.add(new_user)
            db.session.commit()

            login_user(new_user)
            session.permanent = True
            flash("Đăng ký thành công!", "success")
            return redirect(url_for("main.index"))

    return render_template("register.html", google_enabled=google_enabled())


#: Khớp minlength="8" ở form đăng ký — ở đây kiểm cả phía server.
MIN_PASSWORD_LENGTH = 8


@bp.post("/doi-mat-khau")
@login_required
@limiter.limit(RATELIMIT_LOGIN)
def change_password():
    """Đổi mật khẩu từ trang Cài đặt. Bắt nhập mật khẩu hiện tại: phiên đăng
    nhập bị lộ (máy dùng chung quên đăng xuất) không đủ để chiếm tài khoản.
    Rate limit như đăng nhập vì đây cũng là chỗ dò mật khẩu được."""
    back = redirect(url_for("main.account_settings") + "#mat-khau")
    if not current_user.password_hash:
        flash("Tài khoản này đăng nhập bằng Google nên không có mật khẩu để đổi.", "info")
        return back

    current = request.form.get("current_password", "")
    new = request.form.get("new_password", "")
    confirm = request.form.get("confirm_password", "")

    if not check_password_hash(current_user.password_hash, current):
        flash("Mật khẩu hiện tại không đúng.", "danger")
        return back
    if len(new) < MIN_PASSWORD_LENGTH:
        flash(f"Mật khẩu mới cần tối thiểu {MIN_PASSWORD_LENGTH} ký tự.", "danger")
        return back
    if new != confirm:
        flash("Hai mật khẩu mới chưa khớp.", "danger")
        return back
    if new == current:
        flash("Mật khẩu mới phải khác mật khẩu hiện tại.", "danger")
        return back

    current_user.password_hash = generate_password_hash(new)
    db.session.commit()
    flash("Đã đổi mật khẩu.", "success")
    return back


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("Đã đăng xuất!", "info")
    return redirect(url_for("auth.login"))
