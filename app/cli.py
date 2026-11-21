"""
app/cli.py
Lệnh quản trị chạy bằng `flask <tên lệnh>`.
"""
from __future__ import annotations

import click
from flask import Flask
from flask.cli import with_appcontext
from werkzeug.security import generate_password_hash

from app.extensions import db
from app.models import Role, User


def register_cli(app: Flask) -> None:
    app.cli.add_command(create_admin)
    app.cli.add_command(set_role)


@click.command("create-admin")
@click.option("--username", prompt=True, help="Tên đăng nhập của quản trị viên.")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
@with_appcontext
def create_admin(username: str, password: str) -> None:
    """Tạo tài khoản quản trị đầu tiên (hoặc nâng quyền tài khoản đã có)."""
    username = username.strip()
    if not username or not password:
        raise click.ClickException("Tên đăng nhập và mật khẩu không được để trống.")

    user = User.query.filter_by(username=username).first()
    if user:
        user.role = Role.ADMIN
        user.password_hash = generate_password_hash(password)
        db.session.commit()
        click.echo(f"Đã nâng quyền admin cho tài khoản có sẵn: {username}")
        return

    user = User(username=username, password_hash=generate_password_hash(password), role=Role.ADMIN)
    db.session.add(user)
    db.session.commit()
    click.echo(f"Đã tạo tài khoản admin: {username}")


@click.command("set-role")
@click.argument("username")
@click.argument("role", type=click.Choice(list(Role.ALL)))
@with_appcontext
def set_role(username: str, role: str) -> None:
    """Đổi quyền của một tài khoản."""
    user = User.query.filter_by(username=username.strip()).first()
    if user is None:
        raise click.ClickException(f"Không tìm thấy tài khoản: {username}")
    user.role = role
    db.session.commit()
    click.echo(f"{username} -> {role}")
