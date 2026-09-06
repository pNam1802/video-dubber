"""Giai đoạn 1: giữ lại transcript thay vì để nó chỉ nằm trong file .srt."""
from __future__ import annotations

from dataclasses import dataclass

from app.extensions import db
from app.jobs import save_segments
from app.models import Job, TranscriptSegment
from tests.conftest import make_job


@dataclass
class FakeSegment:
    """Giống core.transcriber.Segment, đủ dùng cho save_segments()."""

    start: float
    end: float
    text: str = ""
    translated: str = ""


SAMPLE = [
    FakeSegment(0.0, 2.5, "Gradient descent is an optimisation algorithm.",
                "Gradient descent là một thuật toán tối ưu."),
    FakeSegment(2.5, 5.0, "  It follows the slope.  ", "  Nó đi theo độ dốc.  "),
    FakeSegment(5.0, 7.25, "That is all.", "Vậy thôi."),
]


# ── Ghi vào DB ───────────────────────────────────────────────
def test_segments_are_saved_with_both_languages(app, user):
    with app.app_context():
        job = make_job(user)
        assert save_segments(app, job.id, SAMPLE) == 3

        rows = TranscriptSegment.query.filter_by(job_id=job.id).order_by(TranscriptSegment.idx).all()
        assert [r.idx for r in rows] == [0, 1, 2]
        assert rows[0].text_source.startswith("Gradient descent")
        assert rows[0].text_target.startswith("Gradient descent là")
        assert rows[2].end_sec == 7.25
        assert rows[0].edited is False


def test_whitespace_is_trimmed(app, user):
    with app.app_context():
        job = make_job(user)
        save_segments(app, job.id, SAMPLE)
        row = TranscriptSegment.query.filter_by(job_id=job.id, idx=1).one()
        assert row.text_source == "It follows the slope."
        assert row.text_target == "Nó đi theo độ dốc."


def test_saving_twice_replaces_instead_of_duplicating(app, user):
    """Chạy lại job không được để lại transcript cũ lẫn với bản mới."""
    with app.app_context():
        job = make_job(user)
        save_segments(app, job.id, SAMPLE)
        save_segments(app, job.id, SAMPLE[:2])

        rows = TranscriptSegment.query.filter_by(job_id=job.id).all()
        assert len(rows) == 2


def test_empty_transcript_writes_nothing(app, user):
    with app.app_context():
        job = make_job(user)
        assert save_segments(app, job.id, []) == 0
        assert TranscriptSegment.query.filter_by(job_id=job.id).count() == 0


def test_deleting_job_deletes_its_transcript(app, user):
    """Không có cascade thì mỗi job bị xoá để lại hàng trăm dòng rác."""
    with app.app_context():
        job = make_job(user)
        save_segments(app, job.id, SAMPLE)
        job_id = job.id

        db.session.delete(db.session.get(Job, job_id))
        db.session.commit()

        assert TranscriptSegment.query.filter_by(job_id=job_id).count() == 0


# ── Endpoint ─────────────────────────────────────────────────
def test_endpoint_returns_segments_in_order(app, as_user, user):
    with app.app_context():
        job = make_job(user)
        save_segments(app, job.id, SAMPLE)
        job_id = job.id

    payload = as_user.get(f"/api/jobs/{job_id}/segments").get_json()
    assert payload["count"] == 3
    assert payload["available"] is True
    assert [s["idx"] for s in payload["segments"]] == [0, 1, 2]
    assert payload["segments"][0]["start"] == 0.0
    assert set(payload["segments"][0]) == {"idx", "start", "end", "source", "target", "edited"}


def test_job_without_transcript_returns_empty_not_404(app, as_user, user):
    """Job chạy trước khi có bảng này vẫn tồn tại, chỉ là không còn transcript."""
    with app.app_context():
        job_id = make_job(user).id

    response = as_user.get(f"/api/jobs/{job_id}/segments")
    assert response.status_code == 200
    assert response.get_json() == {"job_id": job_id, "count": 0, "available": False, "segments": []}


def test_cannot_read_someone_elses_transcript(app, as_other, user):
    with app.app_context():
        job = make_job(user)
        save_segments(app, job.id, SAMPLE)
        job_id = job.id

    assert as_other.get(f"/api/jobs/{job_id}/segments").status_code == 404


def test_transcript_requires_login(app, client, user):
    with app.app_context():
        job_id = make_job(user).id
    assert client.get(f"/api/jobs/{job_id}/segments").status_code == 401
