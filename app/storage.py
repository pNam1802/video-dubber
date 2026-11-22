"""
app/storage.py
Đồng bộ file giữa container web và container GPU khi chạy trên Modal.

Modal Volume không tự đồng bộ hai chiều: bên ghi phải commit(), bên đọc phải
reload() thì mới thấy. Chạy local (JOB_RUNNER=thread) thì hai bên dùng chung
đĩa nên các hàm này không làm gì cả.
"""
from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

from config.settings import DATA_DIR, JOB_RUNNER, MODAL_DATA_VOLUME

logger = logging.getLogger(__name__)

#: Bản sao trên đĩa container web của file chưa thấy được qua mount volume.
_CACHE_DIR = Path(tempfile.gettempdir()) / "volume-cache"
_download_lock = threading.Lock()


def _volume():
    import modal

    return modal.Volume.from_name(MODAL_DATA_VOLUME)


def commit_volume() -> None:
    """Ghi thay đổi trên volume xuống ngay, không đợi commit nền.

    Modal tự commit vài giây một lần, nhưng có hai lúc không chờ được:
    job được spawn ngay sau khi lưu video (container GPU có thể khởi động
    trước và không thấy file), và khi xoá file thì người dùng cần thấy kết
    quả ngay.
    """
    if JOB_RUNNER != "modal":
        return
    try:
        _volume().commit()
    except Exception:
        logger.exception("Không commit được volume")


#: Tên cũ, giữ lại cho chỗ gọi sau khi lưu upload.
commit_uploads = commit_volume


def locate_file(path: Path) -> Path:
    """Trả về chỗ đọc được ``path`` (file trên volume, ví dụ video kết quả).

    File do container GPU ghi chỉ hiện ra trên mount của container web sau
    khi ``reload()``. Nhưng reload bị Modal từ chối ("there are open files
    preventing the operation") khi container web đang mở bất kỳ file nào
    trên volume — mà nó luôn mở, vì đang stream video khác cho người dùng.
    Khi đó file mới không bao giờ hiện ra và trình phát nhận 404 cho tới khi
    container khởi động lại.

    Nên nếu reload không giúp được, tải thẳng file qua API của volume (luôn
    thấy dữ liệu đã commit) về đĩa tạm của container rồi phục vụ từ đó.
    Trả lại ``path`` nguyên vẹn nếu không tìm thấy ở đâu — bên gọi tự 404.
    """
    if path.exists() or JOB_RUNNER != "modal":
        return path

    try:
        _volume().reload()
    except Exception as exc:
        logger.warning("Không reload được volume (%s), tải thẳng file qua API", exc)
    if path.exists():
        return path

    try:
        relative = path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        return path
    cached = _CACHE_DIR / relative
    if cached.exists():
        return cached

    with _download_lock:
        if cached.exists():  # request song song (Range) vừa tải xong
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        partial = cached.with_name(cached.name + ".part")
        try:
            with partial.open("wb") as fileobj:
                _volume().read_file_into_fileobj(relative.as_posix(), fileobj)
            partial.replace(cached)
        except Exception:
            partial.unlink(missing_ok=True)
            logger.exception("Không tải được %s từ volume", relative)
            return path
    return cached
