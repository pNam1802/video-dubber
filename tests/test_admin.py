"""Phân quyền quản trị, hạn mức, rate limit, thống kê."""
from __future__ import annotations

import io

import pytest

import app.quota as quota_mod
from app.admin import humanise_error
from app.extensions import db
from app.models import JobStatus, Role, User
from app.quota import check_quota, estimate_cost
from tests.conftest import PASSWORD, make_job, make_user


# ── Phân quyền ───────────────────────────────────────────────
@pytest.mark.parametrize("path", ["/api/admin/users", "/api/admin/stats", "/api/admin/config"])
def test_admin_endpoints_need_admin(as_admin, as_user, client, path):
    assert as_admin.get(path).status_code == 200
    assert as_user.get(path).status_code == 403      # đăng nhập nhưng không đủ quyền
    assert client.get(path).status_code == 401       # chưa đăng nhập


def test_user_sees_own_usage(as_user):
    payload = as_user.get("/api/usage").get_json()
    assert "jobs" in payload and "gpu_seconds" in payload


def test_admin_can_change_role_and_lock(app, as_admin, user):
    assert as_admin.patch(f"/api/admin/users/{user}", json={"role": "admin"}).get_json()["role"] == "admin"
    response = as_admin.patch(f"/api/admin/users/{user}", json={"role": "user", "is_active": False})
    assert response.get_json()["is_active"] is False


def test_admin_cannot_demote_or_lock_self(as_admin, admin):
    assert as_admin.patch(f"/api/admin/users/{admin}", json={"role": "user"}).status_code == 400
    assert as_admin.patch(f"/api/admin/users/{admin}", json={"is_active": False}).status_code == 400


def test_invalid_role_rejected(as_admin, user):
    assert as_admin.patch(f"/api/admin/users/{user}", json={"role": "superuser"}).status_code == 400


def test_normal_user_cannot_change_roles(as_user, admin):
    assert as_user.patch(f"/api/admin/users/{admin}", json={"role": "user"}).status_code == 403


# ── Hạn mức ──────────────────────────────────────────────────
@pytest.fixture()
def small_quota(monkeypatch):
    monkeypatch.setattr(quota_mod, "QUOTA_JOBS_PER_DAY", 3)
    monkeypatch.setattr(quota_mod, "QUOTA_GPU_SECONDS_PER_DAY", 100)
    monkeypatch.setattr(quota_mod, "QUOTA_STORAGE_MB", 1)


def test_daily_job_limit(app, user, small_quota):
    with app.app_context():
        person = db.session.get(User, user)
        for _ in range(3):
            make_job(user, elapsed_sec=10)
        allowed, reason = check_quota(person)
    assert allowed is False
    assert "lượt xử lý" in reason


def test_gpu_seconds_limit(app, user, small_quota):
    with app.app_context():
        person = db.session.get(User, user)
        make_job(user, elapsed_sec=200)
        allowed, reason = check_quota(person)
    assert allowed is False
    assert "giây GPU" in reason


def test_storage_limit_counts_incoming_file(app, user, small_quota):
    with app.app_context():
        person = db.session.get(User, user)
        make_job(user, file_size=900 * 1024)
        assert check_quota(person, incoming_bytes=500 * 1024)[0] is False
        assert check_quota(person, incoming_bytes=10 * 1024)[0] is True


def test_admin_is_exempt_from_quota(app, admin, small_quota):
    with app.app_context():
        boss = db.session.get(User, admin)
        for _ in range(10):
            make_job(admin, elapsed_sec=500)
        assert check_quota(boss)[0] is True


def test_per_user_override_wins(app, user, small_quota):
    with app.app_context():
        person = db.session.get(User, user)
        for _ in range(3):
            make_job(user, elapsed_sec=1)
        person.quota_jobs_per_day = 100
        db.session.commit()
        assert check_quota(person)[0] is True


def test_upload_over_quota_returns_429(app, as_user, user, small_quota):
    """Chặn TRƯỚC khi ghi file, để người vượt hạn mức không kịp làm đầy đĩa."""
    with app.app_context():
        make_job(user, file_size=900 * 1024)
    response = as_user.post(
        "/api/upload",
        data={"video": (io.BytesIO(b"x" * 200_000), "test.mp4"), "gemini_api_key": "test-key"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 429
    assert response.is_json


# ── Ước tính chi phí ─────────────────────────────────────────
def test_estimate_cost_rounds_to_six_places():
    assert estimate_cost(13.1) == round(13.1 * 0.000164, 6)
    assert estimate_cost(None) == 0


# ── Thống kê ─────────────────────────────────────────────────
@pytest.fixture()
def stats_data(app, user):
    with app.app_context():
        make_job(user, translator_engine="marian", translator_actual="marian",
                 elapsed_sec=20, translate_sec=2.1, tts_sec=3.2, estimated_cost_usd=0.003)
        make_job(user, status=JobStatus.FAILED, translator_engine="gemini",
                 error="404 NOT_FOUND", elapsed_sec=5)
        make_job(user, translator_engine="gemini", translator_actual="marian",
                 elapsed_sec=30, translate_sec=78.4, estimated_cost_usd=0.005)


def test_stats_totals_and_breakdowns(as_admin, stats_data):
    data = as_admin.get("/api/admin/stats").get_json()
    assert data["totals"]["jobs"] == 3
    assert JobStatus.DONE in data["by_status"]
    assert "marian" in data["engine_translate_seconds"]
    assert data["avg_seconds_per_step"]["translate"] > 0
    assert data["fallback_count"] == 1
    assert data["last_error"]["error"].startswith("404")
    assert "Modal" in data["note"]      # nhắc rằng đây là ước tính, không phải hoá đơn


def test_stats_includes_real_translate_cost(app, as_admin, user):
    """translate_cost_usd (tiền Gemini/OpenAI THẬT) phải tách riêng khỏi
    estimated_cost_usd (GPU) — cả ở tổng và ở từng ngày, không được cộng
    gộp hay bỏ sót vì đây đúng là chỗ đo lường bị thiếu dẫn tới sự cố
    RESOURCE_EXHAUSTED không ai biết trước."""
    with app.app_context():
        make_job(
            user, translator_engine="gemini", estimated_cost_usd=0.001,
            translate_cost_usd=0.0123, translate_prompt_tokens=1000, translate_completion_tokens=400,
        )
        make_job(user, translator_engine="gemini", estimated_cost_usd=0.002, translate_cost_usd=0.0050)

    data = as_admin.get("/api/admin/stats").get_json()
    assert data["totals"]["translate_cost_usd"] == 0.0173
    assert sum(d["translate_cost_usd"] for d in data["daily"]) == pytest.approx(0.0173)
    # estimated_cost_usd (GPU) không bị lẫn với translate_cost_usd (API).
    assert data["totals"]["estimated_cost_usd"] == pytest.approx(0.003)


def test_config_never_leaks_key_values(as_admin):
    engines = as_admin.get("/api/admin/config").get_json()["engines"]
    assert engines["gemini"]["configured"] in (True, False)
    assert set(engines["gemini"]) == {"configured", "length", "model"}


# ── Rate limit ───────────────────────────────────────────────
def test_login_rate_limited():
    """Flask-Limiter đọc RATELIMIT_ENABLED lúc init_app, nên phải dựng app riêng —
    fixture chung tắt rate limit để các test khác đăng nhập thoải mái."""
    from app import create_app

    application = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False, "RATELIMIT_ENABLED": True})
    with application.app_context():
        db.create_all()

    c = application.test_client()
    codes = [c.post("/login", data={"username": "x", "password": "y"}).status_code for _ in range(12)]
    assert 429 in codes


def test_proxy_fix_splits_limit_per_real_ip(monkeypatch):
    """Sau proxy Modal, không có ProxyFix thì mọi người dùng chung một ô đếm."""
    import config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "TRUST_PROXY", True)
    import importlib

    import app as app_pkg

    importlib.reload(app_pkg)
    application = app_pkg.create_app({"TESTING": True, "WTF_CSRF_ENABLED": False})
    with application.app_context():
        db.create_all()

    c = application.test_client()
    per_ip = [
        c.post("/login", data={"username": "x", "password": "y"},
               headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
        for i in range(10)
    ]
    assert 429 not in per_ip

    importlib.reload(app_pkg)


# ── Trang quản trị (HTML) ────────────────────────────────────
ADMIN_PAGES = ["/quan-tri/thong-ke", "/quan-tri/nguoi-dung", "/quan-tri/he-thong"]


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_admin_pages_open_for_admin(as_admin, path):
    assert as_admin.get(path).status_code == 200


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_admin_pages_forbidden_for_normal_user(as_user, path):
    """Trang HTML trả 403, không phải JSON như các endpoint /api."""
    response = as_user.get(path)
    assert response.status_code == 403
    assert not response.is_json


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_admin_pages_redirect_anonymous(client, path):
    assert client.get(path).status_code == 302


def test_sidebar_hides_admin_menu_from_normal_user(as_user, as_admin):
    assert "/quan-tri/thong-ke" not in as_user.get("/").get_data(as_text=True)
    assert "/quan-tri/thong-ke" in as_admin.get("/").get_data(as_text=True)


def test_stats_page_loads_chart_js(as_admin):
    """Ba biểu đồ vẽ bằng Chart.js trên canvas, không phải HTML dựng sẵn."""
    html = as_admin.get("/quan-tri/thong-ke").get_data(as_text=True)
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js/" in html
    for chart_id in ["dailyChart", "engineChart", "stepChart"]:
        assert f'<canvas id="{chart_id}">' in html


# ── Rút gọn thông báo lỗi ────────────────────────────────────
def test_humanise_error_pulls_message_out_of_provider_payload():
    """Gemini trả về nguyên cả dict; chỉ câu trong 'message' là đáng đọc."""
    raw = (
        "404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
        "models/gemini-2.5-flash is no longer available to new users.', "
        "'status': 'NOT_FOUND'}}"
    )
    assert humanise_error(raw) == (
        "404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to new users."
    )


def test_humanise_error_handles_json_style_quotes_and_noise():
    assert humanise_error('RateLimitError: {"error": {"message": "Quota exceeded"}}') == (
        "RateLimitError: Quota exceeded"
    )
    # Khong khop duoc thi giu nguyen van, chi go bot khoang trang.
    assert humanise_error("ffmpeg  crashed\n  code 1") == "ffmpeg crashed code 1"
    assert humanise_error(None) == ""


def test_humanise_error_truncates():
    assert len(humanise_error("x" * 500)) == 300


# ── Sức khoẻ engine ──────────────────────────────────────────
def test_engine_health_separates_unfinished_jobs(app, as_admin, user):
    """Job đang chạy không được tính vào mẫu của tỷ lệ lỗi."""
    with app.app_context():
        make_job(user, translator_engine="gemini", status=JobStatus.DONE)
        make_job(user, translator_engine="gemini", status=JobStatus.FAILED, error="boom")
        make_job(user, translator_engine="gemini", status=JobStatus.PROCESSING)

    row = as_admin.get("/api/admin/stats").get_json()["engine_health"]["gemini"]
    assert row == {"total": 3, "done": 1, "failed": 1, "unfinished": 1}


def test_last_error_reports_when_it_happened(app, as_admin, user):
    with app.app_context():
        make_job(user, translator_engine="gemini", status=JobStatus.FAILED,
                 error="{'error': {'message': 'model gone'}}")

    last = as_admin.get("/api/admin/stats").get_json()["last_error"]
    assert last["error"] == "model gone"
    assert last["at"]


def test_donut_uses_distinct_hues_not_one_ramp(as_admin):
    """Bản cũ tô hai công cụ bằng hai bậc của cùng dải lime nên nhìn ra một màu.

    Bộ màu hiện tại đã qua validator: tách CVD ΔE 11.1, mắt thường ΔE 26.5.
    Đổi màu thì phải chạy lại validator, đừng chọn bằng mắt.
    """
    html = as_admin.get("/quan-tri/thong-ke").get_data(as_text=True)
    assert '["#6fa30f", "#3987e5", "#d55181"]' in html
    assert '"#bdfd5d", "#9bd93c"' not in html
