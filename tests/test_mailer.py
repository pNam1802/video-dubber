"""Thông báo email: build đúng nội dung theo mốc trạng thái, không gửi khi
thiếu cấu hình/opt-out, và job_url() dùng đúng APP_BASE_URL khi có."""
from __future__ import annotations

import pytest

from app.mailer import job_url, notify_job_status, send_email


class FakeUser:
    def __init__(self, email="a@b.com", notify_email=True):
        self.email = email
        self.notify_email = notify_email


class FakeJob:
    def __init__(self, user, **kwargs):
        self.user = user
        self.id = kwargs.get("id", 1)
        self.source_filename = kwargs.get("source_filename", "video.mp4")
        self.error = kwargs.get("error")
        self.message = kwargs.get("message")


def test_send_email_skips_silently_without_smtp_config(monkeypatch):
    import app.mailer as mailer_mod

    monkeypatch.setattr(mailer_mod, "MAIL_ENABLED", False)
    assert send_email("a@b.com", "subject", "body") is False


def test_send_email_skips_without_recipient(monkeypatch):
    import app.mailer as mailer_mod

    monkeypatch.setattr(mailer_mod, "MAIL_ENABLED", True)
    assert send_email("", "subject", "body") is False


def test_send_email_failure_is_caught_not_raised(monkeypatch):
    """Gmail sập, sai mật khẩu app password... không được làm hỏng job."""
    import app.mailer as mailer_mod

    monkeypatch.setattr(mailer_mod, "MAIL_ENABLED", True)

    class BoomSMTP:
        def __init__(self, *a, **k):
            raise ConnectionError("SMTP không kết nối được")

    monkeypatch.setattr(mailer_mod.smtplib, "SMTP", BoomSMTP)
    assert send_email("a@b.com", "subject", "body") is False


def test_notify_job_status_skips_when_no_email(monkeypatch):
    import app.mailer as mailer_mod

    calls = []
    monkeypatch.setattr(mailer_mod, "send_email", lambda *a, **k: calls.append(a))
    notify_job_status(FakeJob(FakeUser(email=None)), "done")
    assert calls == []


def test_notify_job_status_skips_when_opted_out(monkeypatch):
    import app.mailer as mailer_mod

    calls = []
    monkeypatch.setattr(mailer_mod, "send_email", lambda *a, **k: calls.append(a))
    notify_job_status(FakeJob(FakeUser(notify_email=False)), "done")
    assert calls == []


def test_notify_job_status_skips_when_user_missing(monkeypatch):
    import app.mailer as mailer_mod

    calls = []
    monkeypatch.setattr(mailer_mod, "send_email", lambda *a, **k: calls.append(a))
    notify_job_status(FakeJob(None), "done")
    assert calls == []


@pytest.mark.parametrize("event,expected_snippet", [
    ("awaiting_review", "cần bạn duyệt"),
    ("done", "đã lồng tiếng xong"),
    ("failed", "thất bại"),
])
def test_notify_job_status_sends_right_content(monkeypatch, event, expected_snippet):
    import app.mailer as mailer_mod

    captured: dict = {}
    monkeypatch.setattr(
        mailer_mod, "send_email",
        lambda to, subject, body: captured.update(to=to, subject=subject, body=body),
    )
    job = FakeJob(FakeUser(email="a@b.com"), error="404 lỗi thật")
    notify_job_status(job, event)
    assert captured["to"] == "a@b.com"
    assert expected_snippet in (captured["subject"] + captured["body"]).lower()


def test_notify_job_status_unknown_event_sends_nothing(monkeypatch):
    import app.mailer as mailer_mod

    calls = []
    monkeypatch.setattr(mailer_mod, "send_email", lambda *a, **k: calls.append(a))
    notify_job_status(FakeJob(FakeUser()), "some_future_status_nobody_wired_yet")
    assert calls == []


def test_job_url_uses_base_url_when_set(monkeypatch):
    import app.mailer as mailer_mod

    monkeypatch.setattr(mailer_mod, "APP_BASE_URL", "https://example.com")
    assert job_url(42) == "https://example.com/job/42"


def test_job_url_falls_back_to_relative_path(monkeypatch):
    import app.mailer as mailer_mod

    monkeypatch.setattr(mailer_mod, "APP_BASE_URL", "")
    assert job_url(42) == "/job/42"
