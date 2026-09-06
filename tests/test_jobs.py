"""Vòng đời job: tạo, theo dõi, trang chi tiết, lịch sử, xoá."""
from __future__ import annotations

import io

import pytest

from app.extensions import db
from app.jobs import cancel_job, mark_interrupted_jobs
from app.models import Job, JobStatus
from config.settings import OUTPUT_DIR
from tests.conftest import make_job


# ── Tải lên ──────────────────────────────────────────────────
def test_upload_without_file_rejected(as_user):
    response = as_user.post("/api/upload", data={})
    assert response.status_code == 400
    assert response.is_json


def _upload_and_capture_config(client, monkeypatch, **form_extra):
    """start_job() thật sẽ chạy pipeline cần torch/ffmpeg thật — giả nó để
    bắt đúng DubbingConfig được dựng, không cần chạy pipeline thật."""
    import app.api as api_mod

    captured: dict = {}

    def fake_start_job(flask_app, job_id, path, config):
        captured["config"] = config

    monkeypatch.setattr(api_mod, "start_job", fake_start_job)

    form = {"video": (io.BytesIO(b"fake"), "video.mp4")}
    form.update(form_extra)
    response = client.post("/api/upload", data=form, content_type="multipart/form-data")
    return response, captured.get("config")


def test_upload_passes_chosen_source_language_to_config(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch, source_language="ja")
    assert response.status_code == 202
    assert config.source_language == "ja"


def test_upload_defaults_source_language_to_auto(app, as_user, monkeypatch):
    """Không gửi source_language (form cũ, hoặc JS lỗi) thì phải tự nhận
    dạng, không được vô tình ép về một ngôn ngữ cụ thể."""
    response, config = _upload_and_capture_config(as_user, monkeypatch)
    assert response.status_code == 202
    assert config.source_language == "auto"


def test_upload_invalid_source_language_falls_back_to_auto(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch, source_language="klingon")
    assert response.status_code == 202
    assert config.source_language == "auto"


# ── Ngôn ngữ đích ──────────────────────────────────────────────
def test_upload_passes_chosen_target_language_to_config(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(
        as_user, monkeypatch, target_language="ja", translator_engine="gemini",
        gemini_api_key="test-key",
    )
    assert response.status_code == 202
    assert config.target_language == "ja"


def test_upload_defaults_target_language_to_vietnamese(app, as_user, monkeypatch):
    response, config = _upload_and_capture_config(as_user, monkeypatch)
    assert response.status_code == 202
    assert config.target_language == "vi"


def test_upload_rejects_marian_with_non_vietnamese_target(app, as_user, monkeypatch):
    """MarianMT là model fine-tune riêng EN→VI — chặn ở đây trước khi tốn
    một lượt xử lý cho job chắc chắn sai ngôn ngữ."""
    response, config = _upload_and_capture_config(
        as_user, monkeypatch, target_language="ja", translator_engine="marian",
    )
    assert response.status_code == 400
    assert "MarianMT" in response.get_json()["error"]
    assert config is None  # start_job() không được gọi tới


def test_upload_marian_with_vietnamese_target_still_allowed(app, as_user, monkeypatch):
    """Không được chặn nhầm trường hợp hợp lệ — mặc định vẫn là marian + vi."""
    response, config = _upload_and_capture_config(
        as_user, monkeypatch, target_language="vi", translator_engine="marian",
    )
    assert response.status_code == 202
    assert config.translator_engine == "marian"
    assert config.target_language == "vi"


def test_job_records_chosen_target_language_immediately(app, as_user, user, monkeypatch):
    """Khác source_language (chỉ biết sau khi Whisper chạy xong),
    target_language là lựa chọn CỦA NGƯỜI DÙNG — biết ngay lúc tạo job,
    không cần đợi job chạy xong."""
    monkeypatch.setattr("app.api.start_job", lambda *a, **k: None)
    as_user.post(
        "/api/upload",
        data={
            "video": (io.BytesIO(b"fake"), "video.mp4"),
            "target_language": "ko", "translator_engine": "gemini", "gemini_api_key": "test-key",
        },
        content_type="multipart/form-data",
    )
    with app.app_context():
        job = Job.query.filter_by(user_id=user).order_by(Job.id.desc()).first()
        assert job.target_language == "ko"


# Hành vi khoá MarianMT khi đổi ngôn ngữ đích chạy trên trình duyệt — canh
# cấu trúc mã gửi cho trình duyệt, cùng cách với các tính năng JS khác trong
# bộ test này (pytest không giả lập được thao tác đổi <select> thật).
def test_create_page_ships_marian_gating_for_non_vi_targets(as_user):
    body = as_user.get("/").get_data(as_text=True)
    assert 'id="target_language"' in body
    assert 'option value="ja"' in body
    assert "marianOption.disabled = nonVi" in body
    assert 'engineField.value = "gemini"' in body


def test_upload_wrong_format_rejected(as_user):
    response = as_user.post(
        "/api/upload",
        data={"video": (io.BytesIO(b"fake"), "tailieu.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Định dạng" in response.get_json()["error"]


# Kiểm tra trước khi tải lên chạy trên trình duyệt — pytest không có trình
# duyệt để giả lập kéo-thả hay đọc file.size thật, nên canh HTML đã render
# chứa đúng danh sách định dạng và đúng giới hạn dung lượng của SERVER (không
# phải một con số chép tay có thể lệch dần theo thời gian).
def test_index_page_ships_client_side_upload_validation(as_user):
    from config.settings import MAX_UPLOAD_MB

    body = as_user.get("/").get_data(as_text=True)
    assert 'id="fileValidationError"' in body
    assert '".mp4", ".mov", ".avi", ".mkv", ".webm"' in body
    assert f"MAX_UPLOAD_BYTES = {MAX_UPLOAD_MB} * 1024 * 1024" in body


def test_create_page_keeps_every_field_the_api_reads(as_user):
    """Đổi tên một trường trong template là upload gãy im lặng."""
    body = as_user.get("/").get_data(as_text=True)
    for field in ["video", "translator_engine", "whisper_model", "source_language", "target_language",
                  "compute_device", "tts_engine", "tts_voice", "subtitle_mode", "original_volume",
                  "gemini_api_key", "gemini_model", "openai_api_key", "openai_model",
                  "csrf_token"]:
        assert f'name="{field}"' in body, field


# Ba lựa chọn giữa (base/small/medium) và lý do chọn Gemini/OpenAI thay
# MarianMT trước đây không giải thích gì — người mới dùng phải đoán. Nút info
# bật/tắt bằng click (không phải hover) vì phần lớn người dùng vào bằng điện
# thoại, hover không hoạt động trên cảm ứng.
def test_advanced_fields_have_explanatory_tooltips(as_user):
    body = as_user.get("/").get_data(as_text=True)
    for tip_id, trigger_attr in [("whisperTip", 'data-tip="whisperTip"'),
                                  ("engineTip", 'data-tip="engineTip"')]:
        assert trigger_attr in body
        assert f'id="{tip_id}"' in body
    assert 'class="info-toggle' in body
    assert "document.querySelectorAll(\".info-toggle\")" in body


# ── Theo dõi tiến trình ──────────────────────────────────────
def test_progress_payload_shape(app, user):
    with app.app_context():
        job = make_job(user, translator_engine="gemini", translator_actual="marian",
                       video_name="a.mp4", video_url="/media/output/a.mp4",
                       elapsed_sec=22.6, transcribe_sec=6.4, tts_sec=3.2)
        payload = job.to_progress_dict()
    assert payload["timings"]["transcribe"] == 6.4
    assert payload["fallback_used"] is True
    assert payload["result"]["video_url"].endswith("a.mp4")


def test_progress_hidden_from_other_users(app, as_other, user):
    with app.app_context():
        job_id = make_job(user).id
    assert as_other.get(f"/api/progress/{job_id}").status_code == 404


def test_queue_position_counts_jobs_ahead(app, as_user, user):
    with app.app_context():
        ids = [make_job(user, status=JobStatus.QUEUED, progress=1).id for _ in range(3)]
    assert as_user.get(f"/api/progress/{ids[0]}").get_json()["queue_position"] == 0
    assert as_user.get(f"/api/progress/{ids[2]}").get_json()["queue_position"] == 2


def test_finished_job_has_no_queue_position(app, as_user, user):
    with app.app_context():
        job_id = make_job(user, video_url="/x", video_name="x").id
    assert "queue_position" not in as_user.get(f"/api/progress/{job_id}").get_json()


def test_restart_marks_running_jobs_interrupted(app, user):
    with app.app_context():
        make_job(user, status=JobStatus.PROCESSING, progress=40)
        make_job(user, status=JobStatus.QUEUED, progress=1)

    assert mark_interrupted_jobs(app) == 2
    with app.app_context():
        assert Job.query.filter(Job.status.in_(JobStatus.ACTIVE)).count() == 0
        assert Job.query.filter_by(status=JobStatus.INTERRUPTED).count() == 2


# ── Trang chi tiết ───────────────────────────────────────────
def test_job_page_visible_to_owner_only(app, as_user, as_other, client, user):
    with app.app_context():
        job_id = make_job(user).id
    assert as_user.get(f"/job/{job_id}").status_code == 200
    assert as_other.get(f"/job/{job_id}").status_code == 404
    assert client.get(f"/job/{job_id}").status_code == 302
    assert as_user.get("/job/99999").status_code == 404


def test_job_page_shows_result_and_timings(app, as_user, user):
    with app.app_context():
        job_id = make_job(
            user, source_filename="bai-giang.mp4", translator_engine="gemini",
            translator_actual="marian", segment_count=12, elapsed_sec=22.6,
            estimated_cost_usd=0.0037, video_name="a.mp4", video_url="/media/output/a.mp4",
            srt_name="a.srt", srt_url="/media/output/a.srt",
            extract_sec=0.5, transcribe_sec=6.4, translate_sec=8.9, tts_sec=3.5, compose_sec=4.4,
        ).id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert "bai-giang.mp4" in body
    assert "/media/output/a.mp4" in body
    assert "chuyển sang MarianMT" in body     # cảnh báo fallback
    assert "Thời gian từng bước" in body
    assert "$0.0037" in body


def test_job_page_shows_failure_reason(app, as_user, user):
    with app.app_context():
        job_id = make_job(
            user, status=JobStatus.FAILED, progress=0,
            message="Xử lý thất bại: 404 NOT_FOUND",
            error="404 NOT_FOUND. This model is no longer available.",
        ).id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert "404 NOT_FOUND" in body
    assert "Chi tiết kỹ thuật" in body


def test_job_page_running_state_has_five_stages(app, as_user, user):
    with app.app_context():
        job_id = make_job(user, status=JobStatus.PROCESSING, progress=45, message="Đang dịch...").id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert body.count("data-stage=") == 5
    assert 'id="cancelBtn"' in body


# Lỗi thật (mất mạng vài giây khi đi cầu thang máy, đổi wifi sang 4G...) không
# tự chạy được trong pytest — không có trình duyệt để giả lập fetch() gãy giữa
# chừng. Test dưới đây canh CẤU TRÚC mã đã gửi cho trình duyệt: đúng cơ chế
# phục hồi có mặt, và mẫu cũ (một lần lỗi là dừng polling vĩnh viễn) đã biến
# mất — không canh được hành vi runtime thì ít nhất canh không tái diễn lỗi cũ.
def test_job_page_polling_recovers_instead_of_dying_on_first_error(app, as_user, user):
    with app.app_context():
        job_id = make_job(user, status=JobStatus.PROCESSING, progress=45).id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)

    # Cơ chế phục hồi phải có mặt.
    assert 'id="reconnectBanner"' in body
    assert "AbortController" in body
    assert "MAX_CONSECUTIVE_FAILURES" in body
    assert 'window.addEventListener("online"' in body

    # 401/404 là lỗi không tự khỏi bằng cách thử lại — phải dừng hẳn, không lặp mãi.
    assert "res.status === 401" in body
    assert "res.status === 404" in body

    # Mẫu cũ: bắt được lỗi là clearInterval ngay, không phân biệt tạm thời
    # hay vĩnh viễn. Không được còn dòng nào như vậy.
    assert "clearInterval(poller)" not in body


# ── Lời thoại tương tác ────────────────────────────────────────
# Cũng như polling: hành vi thật (bấm dòng thì video nhảy, dòng đang phát tự
# sáng) chạy trên trình duyệt nên pytest không giả lập được. Test canh cấu
# trúc mã đã gửi cho trình duyệt, và canh đúng cờ IS_DONE theo từng trạng thái.
def test_done_job_page_ships_transcript_panel(app, as_user, user):
    with app.app_context():
        job_id = make_job(user, status=JobStatus.DONE, progress=100).id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert 'id="transcriptPanel"' in body
    assert 'id="transcriptSearch"' in body
    assert "const IS_DONE = true;" in body
    assert f'"/api/jobs/" + JOB_ID + "/segments"' in body
    # Câu dịch thất bại (seg.target trùng hệt seg.source, xem
    # test_llm_translation.py) phải hiện một dòng kèm dấu "chưa dịch", không
    # phải hai dòng giống hệt nhau — bug thật từng gặp trên production.
    assert "seg.target.trim() === seg.source.trim()" in body
    assert "chưa dịch" in body


@pytest.mark.parametrize("status", [JobStatus.PROCESSING, JobStatus.FAILED, JobStatus.QUEUED])
def test_non_done_job_page_skips_transcript_fetch(app, as_user, user, status):
    """Chỉ job đã xong mới có video để nhảy tới — job đang chạy hay job hỏng
    thì chưa/không có gì để hiện, không nên gọi API vô ích."""
    with app.app_context():
        job_id = make_job(user, status=status, progress=10).id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert "const IS_DONE = false;" in body


def test_finished_job_page_has_no_polling_script(app, as_user, user):
    """Job đã xong thì không cần lấy tiến trình nữa — đỡ tốn request vô ích."""
    with app.app_context():
        job_id = make_job(user, status=JobStatus.DONE, progress=100).id
    body = as_user.get(f"/job/{job_id}").get_data(as_text=True)
    assert 'id="reconnectBanner"' in body  # markup vẫn render, chỉ là JS không chạy tới
    assert "if (FINISHED) return;" in body


# ── Huỷ ──────────────────────────────────────────────────────
def test_cancel_running_job(app, as_user, user):
    with app.app_context():
        job_id = make_job(user, status=JobStatus.PROCESSING, progress=45).id
    assert as_user.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    with app.app_context():
        assert db.session.get(Job, job_id).status == JobStatus.CANCELLED


def test_cancel_finished_job_refused(app, as_user, user):
    with app.app_context():
        job_id = make_job(user).id
    assert as_user.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_cancel_other_users_job_refused(app, as_other, user):
    with app.app_context():
        job_id = make_job(user, status=JobStatus.PROCESSING).id
    assert as_other.post(f"/api/jobs/{job_id}/cancel").status_code == 404


def test_thread_runner_reports_it_could_not_stop(app, user):
    """Runner thread không dừng được giữa chừng — API phải nói thật."""
    with app.app_context():
        job = make_job(user, status=JobStatus.PROCESSING)
        assert cancel_job(app, job) is False
        assert db.session.get(Job, job.id).status == JobStatus.CANCELLED


# ── Lịch sử ──────────────────────────────────────────────────
@pytest.fixture()
def many_jobs(app, user):
    with app.app_context():
        for i in range(25):
            make_job(
                user,
                status=JobStatus.DONE if i % 3 else JobStatus.FAILED,
                source_filename=f"bai-giang-{i:02d}.mp4",
                translator_engine="gemini" if i % 5 == 0 else "marian",
                elapsed_sec=20 + i,
            )
        make_job(user, status=JobStatus.PROCESSING, source_filename="dangchay.mp4")


# ── Đa ngôn ngữ ────────────────────────────────────────────────
def test_job_defaults_to_vietnamese_target_with_no_source_yet(app, user):
    """Job tạo trước khi có UI chọn ngôn ngữ vẫn phải xử lý được: đích mặc
    định tiếng Việt như trước giờ, nguồn để trống (chưa nhận dạng/chưa chọn)."""
    with app.app_context():
        job = make_job(user)
        assert job.target_language == "vi"
        assert job.source_language is None


def test_run_job_saves_detected_source_language(app, user, monkeypatch, tmp_path):
    """Whisper để "auto" thì phải LƯU LẠI ngôn ngữ thật đã nhận dạng được —
    không thì lần sau xem lại job chẳng biết video đó tiếng gì."""
    import app.jobs as jobs_mod
    from core.pipeline import DubbingConfig, DubbingResult

    class FakePipeline:
        def __init__(self, config):
            pass

        def run(self, video_path, progress_cb=None):
            return DubbingResult(success=True, source_language_detected="ja")

    monkeypatch.setattr(jobs_mod, "DubbingPipeline", FakePipeline)

    with app.app_context():
        job_id = make_job(user, status=JobStatus.PROCESSING).id
        jobs_mod.run_job(app, job_id, tmp_path / "khong-ton-tai.mp4", DubbingConfig())
        assert db.session.get(Job, job_id).source_language == "ja"


def test_history_page_renders(as_user):
    body = as_user.get("/lich-su").get_data(as_text=True)
    assert 'id="search"' in body
    assert 'id="engineFilter"' in body
    assert 'id="emptyState"' in body


# Bản cũ chỉ tải một lần lúc mở trang — job "Đang xử lý" nằm im trong bảng
# cho tới khi tự tay F5. Cũng như polling ở job.html, hành vi tự làm mới chạy
# hoàn toàn trên trình duyệt nên pytest không giả lập được; canh cấu trúc mã
# đã gửi thay cho hành vi runtime.
def test_history_page_ships_auto_refresh_for_running_jobs(as_user):
    body = as_user.get("/lich-su").get_data(as_text=True)
    assert "scheduleRefresh" in body
    assert 'job.status === "queued" || job.status === "processing"' in body
    # Chỉ hẹn giờ lại khi còn job chưa xong — không thấy dòng chặn này thì
    # trang sẽ hỏi server vô ích mãi kể cả khi mọi job đã xong từ lâu.
    assert "if (items && !hasActiveJob(items)) return;" in body
    # Tự làm mới không được nháy khung xương lên — silent bỏ qua bước đó.
    assert "if (!silent) skeleton.classList.remove" in body
    assert 'addEventListener("visibilitychange"' in body


def test_history_lists_own_jobs_paginated(as_user, many_jobs):
    data = as_user.get("/api/jobs").get_json()
    assert data["total"] == 26
    assert len(data["items"]) == 20
    assert data["pages"] == 2
    assert data["items"][0]["source_filename"] == "dangchay.mp4"   # mới nhất trước
    assert len(as_user.get("/api/jobs?page=2").get_json()["items"]) == 6


def test_history_isolated_between_users(as_other, many_jobs):
    assert as_other.get("/api/jobs").get_json()["total"] == 0


@pytest.mark.parametrize(
    "query,expected",
    [("status=failed", 9), ("engine=gemini", 5), ("q=giang-07", 1), ("q=khong-co-dau", 0)],
)
def test_history_filters(as_user, many_jobs, query, expected):
    assert as_user.get(f"/api/jobs?{query}").get_json()["total"] == expected


def test_history_combined_filters(as_user, many_jobs):
    items = as_user.get("/api/jobs?status=done&engine=marian").get_json()["items"]
    assert all(j["status"] == "done" and j["engine"] == "marian" for j in items)


# ── Xoá ──────────────────────────────────────────────────────
def test_delete_removes_row_and_files(app, as_user, user):
    """Chỉ xoá bản ghi thì video vẫn nằm trên đĩa và vẫn tính vào hạn mức."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video = OUTPUT_DIR / "xoa-thu.mp4"
    srt = OUTPUT_DIR / "xoa-thu.srt"
    video.write_bytes(b"fake video")
    srt.write_text("1\n", encoding="utf-8")

    with app.app_context():
        job_id = make_job(user, video_name="xoa-thu.mp4", srt_name="xoa-thu.srt",
                          video_url="/media/output/xoa-thu.mp4").id

    response = as_user.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert response.get_json()["files_removed"] == 2
    assert not video.exists()
    assert not srt.exists()
    with app.app_context():
        assert db.session.get(Job, job_id) is None


def test_cannot_delete_running_job(app, as_user, user):
    with app.app_context():
        job_id = make_job(user, status=JobStatus.PROCESSING).id
    assert as_user.delete(f"/api/jobs/{job_id}").status_code == 409


def test_cannot_delete_other_users_job(app, as_other, user):
    with app.app_context():
        job_id = make_job(user).id
    assert as_other.delete(f"/api/jobs/{job_id}").status_code == 404
    with app.app_context():
        assert db.session.get(Job, job_id) is not None


# ── Phục vụ file ─────────────────────────────────────────────
def test_media_is_owner_only(app, as_other, user):
    with app.app_context():
        make_job(user, video_name="rieng.mp4", video_url="/media/output/rieng.mp4")
    assert as_other.get("/media/output/rieng.mp4").status_code == 403
