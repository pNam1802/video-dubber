"""Phục vụ file trên volume Modal khi reload() bị chặn bởi file đang mở."""
from __future__ import annotations

import app.storage as storage


class _FakeVolume:
    def __init__(self, files: dict[str, bytes], reload_error: Exception | None = None):
        self.files = files
        self.reload_error = reload_error
        self.reads: list[str] = []

    def reload(self):
        if self.reload_error:
            raise self.reload_error

    def read_file_into_fileobj(self, path, fileobj):
        self.reads.append(path)
        fileobj.write(self.files[path])
        return len(self.files[path])


def _setup(monkeypatch, tmp_path, volume):
    data = tmp_path / "data"
    (data / "outputs").mkdir(parents=True)
    monkeypatch.setattr(storage, "JOB_RUNNER", "modal")
    monkeypatch.setattr(storage, "DATA_DIR", data)
    monkeypatch.setattr(storage, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(storage, "_volume", lambda: volume)
    return data


def test_existing_file_is_served_in_place(monkeypatch, tmp_path):
    volume = _FakeVolume({})
    data = _setup(monkeypatch, tmp_path, volume)
    target = data / "outputs" / "a.mp4"
    target.write_bytes(b"x")
    assert storage.locate_file(target) == target
    assert volume.reads == []


def test_blocked_reload_falls_back_to_api_download(monkeypatch, tmp_path):
    volume = _FakeVolume({"outputs/a.mp4": b"video"},
                         reload_error=RuntimeError("there are open files preventing the operation"))
    data = _setup(monkeypatch, tmp_path, volume)

    found = storage.locate_file(data / "outputs" / "a.mp4")
    assert found.read_bytes() == b"video"
    # Lần sau dùng lại bản đã tải, không gọi API nữa.
    assert storage.locate_file(data / "outputs" / "a.mp4") == found
    assert volume.reads == ["outputs/a.mp4"]


def test_missing_everywhere_returns_original_path(monkeypatch, tmp_path):
    volume = _FakeVolume({}, reload_error=RuntimeError("blocked"))
    data = _setup(monkeypatch, tmp_path, volume)
    target = data / "outputs" / "gone.mp4"
    assert storage.locate_file(target) == target
    assert not (tmp_path / "cache" / "outputs" / "gone.mp4.part").exists()
