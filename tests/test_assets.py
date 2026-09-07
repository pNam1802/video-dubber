"""Canh hai thứ im lặng hỏng: icon thiếu tên, và CSS quên build lại.

Cả hai đều không làm test nào khác đỏ — trang vẫn trả 200, chỉ là người dùng
nhìn thấy chữ "account_circle" thay cho icon, hoặc một trang không có định dạng.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
BUILT_CSS = ROOT / "static" / "css" / "app.css"

#: <span class="material-symbols-outlined ...">ten_icon</span>
ICON_IN_TEMPLATE = re.compile(r"material-symbols-outlined[^<>]*>\s*([a-z_0-9]+)")


def icons_requested() -> set[str]:
    head = (TEMPLATES / "_head.html").read_text(encoding="utf-8")
    match = re.search(r"icon_names=([a-z_0-9,]+)", head)
    assert match, "_head.html phải lọc icon bằng icon_names, không tải cả bộ font"
    return set(match.group(1).split(","))


def test_icon_font_is_subset_not_whole_family():
    head = (TEMPLATES / "_head.html").read_text(encoding="utf-8")
    # Bo font day du nang 1,13 MB; ban loc 32 icon nang khoang 5,8 KB.
    assert "wght,FILL@100..700,0..1" not in head
    assert "icon_names=" in head


# ── CSS đã build ─────────────────────────────────────────────
def test_built_css_exists_and_is_committed():
    """Image Modal không có Node, nó chỉ chép file đã build vào."""
    assert BUILT_CSS.exists(), "Thiếu static/css/app.css — chạy `npm run build:css`"
    assert BUILT_CSS.stat().st_size > 5_000


def test_pages_link_built_css_not_the_play_cdn(client):
    """Kiểm tra HTML đã render, không phải mã nguồn: chú thích Jinja bị bỏ đi
    lúc render, nên soi file gốc sẽ khớp nhầm vào chính lời giải thích."""
    html = client.get("/login").get_data(as_text=True)
    assert "cdn.tailwindcss.com" not in html, "Bản CDN chỉ để thử nghiệm, không dùng cho production"
    assert "/static/css/app.css" in html


def test_built_css_covers_classes_the_templates_use():
    """Tailwind chỉ giữ class nó quét thấy — class dựng bằng JS dễ bị loại nhầm."""
    css = BUILT_CSS.read_text(encoding="utf-8")
    for cls in ["bg-surface-container", "text-on-surface-variant", "font-headline-xl",
                "bg-error", "bg-tertiary", "text-primary", "-translate-x-full"]:
        assert cls in css, f"Class {cls} không có trong CSS đã build"


@pytest.mark.skipif(not (ROOT / "node_modules").exists(), reason="chưa chạy npm install")
def test_built_css_is_up_to_date():
    """Sửa template rồi quên build lại thì trang mất định dạng ở production."""
    before = BUILT_CSS.read_bytes()
    subprocess.run(
        ["npm", "run", "build:css"], cwd=ROOT, check=True,
        capture_output=True, shell=sys.platform == "win32",
    )
    assert BUILT_CSS.read_bytes() == before, (
        "static/css/app.css cũ hơn template — chạy `npm run build:css` rồi commit lại"
    )


# ── Cache ────────────────────────────────────────────────────
def test_css_url_carries_a_content_hash(client):
    """Cache một năm chỉ an toàn khi URL đổi theo nội dung file."""
    html = client.get("/login").get_data(as_text=True)
    match = re.search(r"/static/css/app\.css\?v=([0-9a-f]{10})", html)
    assert match, "URL của app.css phải kèm ?v=<mã băm>"


def test_static_files_are_cached_long(app, client):
    response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert response.cache_control.max_age and response.cache_control.max_age > 86_400


# ── Icon truyền qua biến Jinja ─────────────────────────────────
# Đọc mã nguồn thô bằng regex bỏ sót icon đến từ macro (index.html stat_row),
# dict trạng thái (job.html STATUS), hay vòng lặp (layout.html nav_items,
# auth_layout.html) — ở những chỗ đó span chỉ chứa `{{ icon }}`, không phải
# tên icon literal. Bài học thực tế: 13 icon từng lọt lưới kiểu này và vỡ ra
# production (hourglass_top, payments, bolt, schedule, sync, check_circle,
# cancel, warning, mic, translate, balance, diamond, add_circle) — trình
# duyệt không thấy ligature nên in ra chữ HOA của tên biến, y hệt ảnh chụp
# màn hình người dùng báo lại.
#
# Cách chắc ăn duy nhất: RENDER thật các trang qua nhiều trạng thái khác nhau
# rồi soi đúng HTML đã render — lúc đó mọi {{ icon }} đều đã thành chữ literal.
# Dùng kết quả render này cho CẢ HAI chiều (thiếu lẫn thừa) — so icon dùng
# thật với icon_names, không so với bản đọc mã nguồn tĩnh vốn luôn thiếu.
JOB_STATUSES = ["queued", "processing", "done", "failed", "cancelled", "interrupted"]


def rendered_icon_names(html: str) -> set[str]:
    return set(ICON_IN_TEMPLATE.findall(html))


def all_icons_actually_rendered(app, client, as_user, as_admin, user) -> set[str]:
    from tests.conftest import make_job

    found: set[str] = set()

    for path in ["/login", "/register", "/privacy", "/terms"]:
        found |= rendered_icon_names(client.get(path).get_data(as_text=True))

    for path in ["/", "/lich-su", "/cai-dat"]:
        found |= rendered_icon_names(as_user.get(path).get_data(as_text=True))

    with app.app_context():
        job_ids = [make_job(user, status=status).id for status in JOB_STATUSES]
    for job_id in job_ids:
        found |= rendered_icon_names(as_user.get(f"/job/{job_id}").get_data(as_text=True))

    for path in ["/quan-tri/thong-ke", "/quan-tri/nguoi-dung", "/quan-tri/he-thong"]:
        found |= rendered_icon_names(as_admin.get(path).get_data(as_text=True))

    # error.html có hai nhánh khác nhau (đã đăng nhập / khách vãng lai) —
    # phải gọi cả hai, thiếu nhánh nào thì icon riêng của nhánh đó lọt lưới.
    found |= rendered_icon_names(as_user.get("/khong-ton-tai").get_data(as_text=True))
    found |= rendered_icon_names(client.get("/khong-ton-tai").get_data(as_text=True))
    return found


def test_every_rendered_icon_is_requested(app, client, as_user, as_admin, user):
    """Thiếu tên trong icon_names thì icon hiện ra thành chữ HOA, trang vẫn trả 200."""
    found = all_icons_actually_rendered(app, client, as_user, as_admin, user)
    missing = found - icons_requested()
    assert not missing, (
        "Trang đã render dùng icon nhưng _head.html chưa lọc — sẽ hiện ra thành "
        "chữ HOA trên production: " + ", ".join(sorted(missing))
    )


def test_no_unused_icons_requested(app, client, as_user, as_admin, user):
    """Icon thừa thì font phình ra mà không trang nào thật sự render tới."""
    found = all_icons_actually_rendered(app, client, as_user, as_admin, user)
    unused = icons_requested() - found
    assert not unused, "icon_names khai thừa, không trang nào dùng tới: " + ", ".join(sorted(unused))


# ── Icon tab trình duyệt ─────────────────────────────────────
def test_favicon_links_present_and_cache_busted(client):
    html = client.get("/login").get_data(as_text=True)
    assert re.search(r'rel="icon" href="/static/img/favicon\.svg\?v=[0-9a-f]{10}"', html)
    assert re.search(r'rel="icon" href="/static/img/favicon-32\.png\?v=[0-9a-f]{10}"', html)
    assert re.search(r'rel="icon" href="/static/img/favicon\.ico\?v=[0-9a-f]{10}"', html)
    assert re.search(r'rel="apple-touch-icon" href="/static/img/apple-touch-icon\.png\?v=[0-9a-f]{10}"', html)


def test_favicon_files_exist():
    img_dir = ROOT / "static" / "img"
    for name in ["favicon.svg", "favicon-16.png", "favicon-32.png", "favicon-48.png",
                 "favicon.ico", "apple-touch-icon.png"]:
        assert (img_dir / name).exists(), f"Thiếu static/img/{name}"
