"""
app/errors.py
Chuẩn hoá lỗi: /api/* luôn trả JSON, phần còn lại giữ trang HTML.
"""
from __future__ import annotations

from flask import (
    Flask,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user
from flask_wtf.csrf import CSRFError
from werkzeug.exceptions import HTTPException

from app.extensions import db
from config.settings import MAX_UPLOAD_MB


#: Tieu de va cau giai thich cho tung ma loi. Werkzeug co san mo ta nhung bang
#: tieng Anh va viet cho lap trinh vien, khong hop de dua thang cho nguoi dung.
_ERROR_PAGES: dict[int, tuple[str, str]] = {
    400: ("Yêu cầu không hợp lệ", "Dữ liệu gửi lên không đúng định dạng. Thử tải lại trang rồi làm lại."),
    401: ("Cần đăng nhập", "Bạn phải đăng nhập để xem trang này."),
    403: ("Không có quyền truy cập", "Tài khoản của bạn không được phép mở trang này."),
    404: ("Không tìm thấy trang", "Đường dẫn không tồn tại, hoặc nội dung đã bị xoá."),
    413: ("File quá lớn", "Video vượt quá dung lượng cho phép."),
    429: ("Thao tác quá nhanh", "Bạn đã gửi quá nhiều yêu cầu. Chờ một lát rồi thử lại."),
    500: ("Lỗi hệ thống", "Có gì đó hỏng ở phía chúng tôi. Sự cố đã được ghi lại."),
}


def render_error_page(code: int, detail: str | None = None):
    title, hint = _ERROR_PAGES.get(code, ("Đã xảy ra lỗi", "Vui lòng thử lại sau."))
    return render_template("error.html", code=code, title=title, hint=hint, detail=detail), code


def wants_json() -> bool:
    if request.path.startswith("/api/"):
        return True
    accept = request.accept_mimetypes
    return accept.accept_json and not accept.accept_html


def register_unauthorized_handler() -> None:
    """Flask-Login mặc định redirect sang trang login — với /api/* thì phải là 401 JSON."""
    from app.extensions import login_manager

    @login_manager.unauthorized_handler
    def unauthorized():
        if wants_json():
            return jsonify({"error": "Cần đăng nhập.", "code": 401}), 401
        flash(login_manager.login_message, login_manager.login_message_category)
        return redirect(url_for("auth.login"))


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(CSRFError)
    def handle_csrf_error(exc: CSRFError):
        if wants_json():
            return jsonify({"error": "Phiên làm việc đã hết hạn, vui lòng tải lại trang.", "code": 400}), 400
        flash("Phiên làm việc đã hết hạn, vui lòng thử lại.", "warning")
        target = "main.index" if current_user.is_authenticated else "auth.login"
        return redirect(url_for(target))

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        if exc.code == 413:
            exc.description = f"File vượt quá giới hạn {MAX_UPLOAD_MB} MB."
        if wants_json():
            return jsonify({"error": exc.description, "code": exc.code}), exc.code
        # Mo ta mac dinh cua Werkzeug la tieng Anh; chi hien khi noi goi abort()
        # da thay bang cau tieng Viet cua rieng minh.
        detail = exc.description if exc.description != type(exc).description else None
        return render_error_page(exc.code or 500, detail)

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        if isinstance(exc, HTTPException):
            return exc
        current_app.logger.exception("Unhandled error on %s", request.path)
        # Loi DB de lai session o trang thai hong: khong rollback thi chinh
        # trang bao loi (load_user khi render) cung no PendingRollbackError.
        try:
            db.session.rollback()
        except Exception:
            pass
        if wants_json():
            return jsonify({"error": "Lỗi hệ thống, vui lòng thử lại.", "code": 500}), 500
        # Khi chay test hay debug thi de exception noi len de con thay traceback.
        if current_app.testing or current_app.debug:
            raise exc
        return render_error_page(500)
