"""
app/studio.py
"Xưởng kịch bản" (chỉ admin): agent lên lộ trình series và viết kịch bản từng tập,
trí nhớ series nằm trong bảng Series/Episode. Logic LLM ở core/studio.py.

Mỗi bước LLM chạy nền (Composer trên Modal, thread khi chạy local) vì có thể lâu hơn
150 giây — giới hạn request web của Modal — và container web không cài google-genai.
Trang chỉ việc hỏi lại trạng thái (Series.task / Episode.task).
"""
from __future__ import annotations

import threading

from flask import Blueprint, Flask, current_app, jsonify, render_template, request
from flask_login import current_user

from app.extensions import db
from app.models import Episode, EpisodeStatus, Job, Series
from app.permissions import require_admin, require_admin_page
from config.settings import JOB_RUNNER, MODAL_APP_NAME

bp = Blueprint("studio", __name__)

EPISODE_STEPS = ("write", "translate", "wrap", "board_vi", "board_en", "publish_vi", "publish_en")
#: Bước chạy trên Series (còn lại chạy trên Episode).
SERIES_STEPS = ("plan", "ideas")
TASK_LABELS = {"plan": "Đang lên lộ trình series...", "ideas": "Đang đọc tin và tìm ý tưởng mới...", "write": "Đang nghiên cứu và viết kịch bản...",
               "translate": "Đang dịch sang tiếng Anh...", "wrap": "Đang chốt tập và cập nhật trí nhớ...",
               "board": "Đang phân cảnh video giải thích...", "preview": "Đang vẽ thử một cảnh...",
               "publish": "Đang dịch phụ đề và soạn gói YouTube...", "youtube": "Đang tải video lên YouTube..."}
LANGUAGES = ("vi", "en")
#: Cảnh xem thử dài tối đa bấy nhiêu giây — đủ thấy hình, không tốn CPU như cả video.
PREVIEW_MAX_SEC = 15.0


# ── Chạy nền ─────────────────────────────────────────────────
def start_task(flask_app: Flask, kind: str, target_id: int) -> None:
    if JOB_RUNNER == "modal":
        import modal

        composer = modal.Cls.from_name(MODAL_APP_NAME, "Composer")()
        composer.run_studio.spawn(kind, target_id)
        return
    threading.Thread(target=run_task, args=(flask_app, kind, target_id), daemon=True,
                     name=f"studio-{kind}-{target_id}").start()


def run_task(flask_app: Flask, kind: str, target_id: int) -> None:
    """Chạy một bước của agent và ghi kết quả. Lỗi thì ghi vào .error, không ném ra.

    kind: "plan" (Series), "write" | "translate" | "wrap" | "board_<lang>" | "preview:<lang>:<cảnh>"
    (Episode) — tham số của bước nằm sau dấu ":" vì Composer.run_studio chỉ nhận (kind, id)."""
    with flask_app.app_context():
        target = db.session.get(Series if kind.split(":")[0] in SERIES_STEPS else Episode, target_id)
        if target is None:
            return
        # Tối đa 3 phần: tham số cuối (giờ đăng YouTube) có thể chứa dấu ":".
        name, *args = kind.split(":", 2)
        try:
            _STEPS[name](target, *args)
            target.error = None
        except Exception as exc:
            db.session.rollback()
            target = db.session.get(Series if kind.split(":")[0] in SERIES_STEPS else Episode, target_id)
            target.error = str(exc)[:2000]
            flask_app.logger.exception("Xưởng kịch bản: bước %s của #%s lỗi", kind, target_id)
        finally:
            target.task = None
            db.session.commit()


def run_daily_ideas(flask_app: Flask, limit: int = 3) -> list[int]:
    """Lượt tìm ý tưởng buổi sáng (Modal cron, xem modal_app.daily_ideas): các series của admin đã có
    ít nhất một tập làm video, mới nhất trước, đang rảnh. Trả về id series đã chạy."""
    from app.models import Role, User

    with flask_app.app_context():
        rows = (Series.query.join(User, Series.user_id == User.id)
                .filter(User.role == Role.ADMIN, Series.task.is_(None))
                .order_by(Series.created_at.desc()).all())
        active = [s for s in rows if any(e.job_vi_id or e.job_en_id for e in s.episodes)][:limit]
        for series in active:
            series.task, series.error = "ideas", None
        db.session.commit()
        ids = [s.id for s in active]
    for series_id in ids:
        run_task(flask_app, "ideas", series_id)
    return ids


def _context(episode: Episode) -> str:
    from core.studio import memory_context

    series = episode.series
    return memory_context(series.as_dict(), [e.as_dict() for e in series.episodes], episode.number)


def _step_plan(series: Series) -> None:
    from core.studio import plan_series

    plan = plan_series(series.topic, series.audience, len(series.episodes) or 10)
    series.title, series.description = plan["title"], plan["description"]
    series.style_guide, series.glossary = plan["style_guide"], plan["glossary"]
    for ep in list(series.episodes):
        db.session.delete(ep)
    db.session.flush()
    for number, ep in enumerate(plan["episodes"], start=1):
        series.episodes.append(Episode(number=number, title=ep["title"], goal=ep["goal"],
                                       key_points=ep["key_points"]))


def _step_ideas(series: Series) -> None:
    """Đọc tin trên web (ưu tiên nguồn tiếng Trung) và lưu vài ý tưởng tập mới cho series."""
    from app.models import Idea
    from core.ideas import find_ideas
    from core.studio import memory_context

    context = memory_context(series.as_dict(), [e.as_dict() for e in series.episodes])
    recent = Idea.query.filter_by(series_id=series.id).order_by(Idea.created_at.desc()).limit(60).all()
    for idea in find_ideas(context, series.topic, [i.title for i in recent]):
        db.session.add(Idea(series_id=series.id, **idea))


def _step_write(episode: Episode) -> None:
    from core.studio import merge_glossary, research, review_script, write_script

    context = _context(episode)
    data = episode.as_dict()
    notes, sources = research(context, data)
    draft = write_script(context, data, notes)
    reviewed = review_script(context, data, notes, draft["script"])
    episode.notes, episode.sources, episode.outline = notes, sources, draft["outline"]
    episode.script_vi, episode.quiz, episode.review = reviewed["script"], draft["quiz"], reviewed["issues"]
    episode.script_en = ""  # bản Anh cũ không còn khớp
    clear_storyboard(episode, "vi")
    clear_storyboard(episode, "en")
    episode.status = EpisodeStatus.DRAFTED
    episode.series.glossary = merge_glossary(episode.series.glossary or [], draft["new_terms"])


def _sentences(episode: Episode, lang: str) -> list[str]:
    from core.script_reader import split_script

    return [s for _, s in split_script(getattr(episode, f"script_{lang}"))]


def clear_storyboard(episode: Episode, lang: str) -> None:
    """Kịch bản đổi thì số thứ tự câu đổi: bảng phân cảnh và cảnh xem thử cũ không còn khớp."""
    setattr(episode, f"storyboard_{lang}", [])
    episode.previews = {k: v for k, v in (episode.previews or {}).items() if not k.startswith(lang + ":")}


# ── Giọng tự thu ─────────────────────────────────────────────
# Mỗi đoạn (dòng không trống của kịch bản) một bản thu, lưu trên volume: tên file mang mã nội
# dung đoạn (core/recorded_voice.recording_name) nên sửa chữ trong đoạn thì bản thu cũ tự hết khớp
# — không cần cột mới trong database.
# ── Nhạc nền của series ──────────────────────────────────────
# Mỗi series một bản nhạc (người dùng tự tải lên, ví dụ từ YouTube Audio Library), lưu trên volume
# DATA_DIR/music/<series>.m4a — không cần cột mới. Trộn vào Video giải thích: nhỏ, tự hạ khi có giọng.
MUSIC_MAX_SEC = 600


def music_file(series_id: int):
    from config.settings import DATA_DIR

    return DATA_DIR / "music" / f"{series_id}.m4a"


def music_path_for(user, series_id: int) -> str:
    series = db.session.get(Series, series_id)
    if series is None or series.user_id != user.id:
        raise LookupError("Không tìm thấy series.")
    path = music_file(series_id)
    if not path.is_file():
        raise ValueError("Series này chưa có nhạc nền.")
    return str(path)


def recordings_dir(episode_id: int):
    from config.settings import DATA_DIR

    return DATA_DIR / "recordings" / str(episode_id)


def paragraphs(text: str) -> list[str]:
    """Các đoạn theo đúng cách core/script_reader.split_script đánh số."""
    return [" ".join(line.split()) for line in (text or "").splitlines() if line.strip()]


def recording_status(episode: Episode, lang: str) -> list[str | None]:
    """Tên bản thu khớp chữ của từng đoạn, None = chưa thu (hoặc đã sửa chữ sau khi thu)."""
    from core.recorded_voice import recording_name

    folder = recordings_dir(episode.id)
    out = []
    for i, text in enumerate(paragraphs(getattr(episode, f"script_{lang}"))):
        name = recording_name(episode.id, lang, i, text)
        out.append(name if (folder / name).is_file() else None)
    return out


def recorded_paragraphs_for(user, episode_id: int, lang: str, text: str) -> dict[str, str]:
    """{"<đoạn>": đường dẫn bản thu} cho kịch bản `text`; thiếu đoạn nào thì ném ValueError."""
    from core.recorded_voice import recording_name

    episode = db.session.get(Episode, episode_id)
    if episode is None or episode.series.user_id != user.id:
        raise LookupError("Không tìm thấy tập này.")
    folder = recordings_dir(episode_id)
    found, missing = {}, []
    for i, para in enumerate(paragraphs(text)):
        path = folder / recording_name(episode_id, lang, i, para)
        if path.is_file():
            found[str(i)] = str(path)
        else:
            missing.append(i + 1)
    if missing:
        shown = ", ".join(map(str, missing[:8])) + ("..." if len(missing) > 8 else "")
        raise ValueError(f"Còn {len(missing)} đoạn chưa thu giọng (đoạn {shown}).")
    return found


def _step_board(episode: Episode, lang: str) -> None:
    from core.explainer import board_to_list, estimate_timings, write_board

    sentences = _sentences(episode, lang)
    if not sentences:
        raise RuntimeError("Chưa có kịch bản để phân cảnh.")
    clear_storyboard(episode, lang)
    board = write_board(sentences, estimate_timings(sentences, lang))
    setattr(episode, f"storyboard_{lang}", board_to_list(board))


def _step_preview(episode: Episode, lang: str, index: str) -> None:
    """Vẽ thử một cảnh của bảng phân cảnh, thời lượng theo ước lượng lời đọc (tối đa 15 giây)."""
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    from app.storage import commit_volume
    from config.settings import OUTPUT_DIR
    from core.explainer import FPS, estimate_timings, render_scene, scene_beats

    board, k = getattr(episode, f"storyboard_{lang}") or [], int(index)
    if not 0 <= k < len(board):
        raise RuntimeError("Không có cảnh này trong bảng phân cảnh.")
    timings = estimate_timings(_sentences(episode, lang), lang)
    start = timings[board[k]["start"]][0]
    end = timings[board[k + 1]["start"]][0] if k + 1 < len(board) else timings[-1][1]
    end = min(end, start + PREVIEW_MAX_SEC)
    scene = {key: v for key, v in board[k].items() if key != "start"}
    scene["beats"] = scene_beats(timings, start, end)

    name = f"studio_{episode.id}_{lang}_{k}_{uuid.uuid4().hex[:8]}.mp4"
    work = Path(tempfile.mkdtemp(prefix="studio-preview-"))
    try:
        render_scene(scene, max(25, round((end - start) * FPS)), work / "scene_000.mp4", work)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "scene_000.mp4"), OUTPUT_DIR / name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    previews = dict(episode.previews or {})
    old = previews.get(f"{lang}:{k}")
    if old:
        (OUTPUT_DIR / old).unlink(missing_ok=True)
    previews[f"{lang}:{k}"] = name
    episode.previews = previews
    commit_volume()


def _step_publish(episode: Episode, lang: str) -> None:
    """Video đã xong -> phụ đề ngôn ngữ còn lại (đúng mốc từng câu) + tiêu đề, mô tả, tags, chapters."""
    from app.models import JobStatus
    from app.storage import commit_volume
    from config.settings import OUTPUT_DIR
    from core.studio import srt_text, translate_lines, youtube_package

    job = db.session.get(Job, getattr(episode, f"job_{lang}_id") or 0)
    if job is None or job.status != JobStatus.DONE or not job.video_name:
        raise RuntimeError("Video của bản này chưa làm xong.")
    rows = list(job.segments)
    lines = [row.text_target or row.text_source for row in rows]
    timings = [(row.play_start_sec if row.play_start_sec is not None else row.start_sec,
                row.play_end_sec if row.play_end_sec is not None else row.end_sec) for row in rows]
    if not lines:
        raise RuntimeError("Video này không có lời thoại để làm phụ đề.")
    other = "en" if lang == "vi" else "vi"
    context = _context(episode)
    translated = translate_lines(context, lines, other)
    stem = job.video_name.rsplit(".", 1)[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = {lang: f"{stem}.{lang}.srt", other: f"{stem}.{other}.srt"}
    (OUTPUT_DIR / names[lang]).write_text(srt_text(lines, timings), encoding="utf-8")
    (OUTPUT_DIR / names[other]).write_text(srt_text(translated, timings), encoding="utf-8")
    package = youtube_package(context, episode.as_dict(), lang, lines, timings)
    episode.publish = {**(episode.publish or {}), lang: {**package, "job_id": job.id, "srt": names}}
    commit_volume()


def _step_youtube(episode: Episode, lang: str, publish_at: str = "") -> None:
    """Tải video + phụ đề hai thứ tiếng lên kênh đã kết nối. Video để riêng tư (kèm lịch công khai
    nếu có publish_at); id video lưu ngay sau khi tải để lỗi phụ đề không làm mất dấu video."""
    from app.models import JobStatus, User
    from app.storage import locate_file
    import shutil
    import tempfile
    from pathlib import Path

    from config.settings import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, OUTPUT_DIR
    from core import youtube as yt
    from core.explainer import board_module, has_thumbnail, render_thumbnail

    pack = (episode.publish or {}).get(lang)
    if not pack:
        raise RuntimeError("Chưa soạn gói YouTube cho bản này.")
    if (pack.get("youtube") or {}).get("id"):
        raise RuntimeError("Bản này đã đăng lên YouTube rồi.")
    job = db.session.get(Job, pack.get("job_id") or 0)
    if job is None or job.status != JobStatus.DONE or not job.video_name:
        raise RuntimeError("Video của bản này chưa làm xong.")
    user = db.session.get(User, episode.series.user_id)
    if user is None or not user.youtube_token:
        raise RuntimeError("Chưa kết nối kênh YouTube.")

    token = yt.access_token(yt.unseal(user.youtube_token, current_app.config["SECRET_KEY"]),
                            GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)
    video_id = yt.upload_video(token, locate_file(OUTPUT_DIR / job.video_name), yt.video_body(pack, lang, publish_at or None))
    result = {"id": video_id, "url": f"https://youtu.be/{video_id}",
              "studio_url": f"https://studio.youtube.com/video/{video_id}/edit",
              "publish_at": publish_at or None, "captions": [], "caption_errors": [], "playlist": None}
    _save_youtube(episode, lang, result)
    db.session.commit()
    for code, name in (pack.get("srt") or {}).items():
        try:
            srt = locate_file(OUTPUT_DIR / name).read_bytes()
            yt.upload_caption(token, video_id, code, "Tiếng Việt" if code == "vi" else "English", srt)
            result["captions"].append(code)
        except Exception as exc:
            result["caption_errors"].append(f"{code}: {exc}"[:300])
    module = board_module(getattr(episode, f"storyboard_{lang}"))
    if module and has_thumbnail(module):
        work = Path(tempfile.mkdtemp(prefix="studio-thumb-"))
        try:
            render_thumbnail(module, lang, work / "thumb.png", work)
            yt.set_thumbnail(token, video_id, (work / "thumb.png").read_bytes())
            result["thumbnail"] = True
        except Exception as exc:
            result["thumbnail_error"] = str(exc)[:300]
        finally:
            shutil.rmtree(work, ignore_errors=True)
    playlist = episode.series.youtube_playlist
    if playlist:
        try:
            yt.add_to_playlist(token, playlist, video_id)
            result["playlist"] = playlist
        except Exception as exc:
            result["playlist_error"] = str(exc)[:300]
    _save_youtube(episode, lang, result)


def _save_youtube(episode: Episode, lang: str, result: dict) -> None:
    publish = dict(episode.publish or {})
    publish[lang] = {**publish.get(lang, {}), "youtube": dict(result)}
    episode.publish = publish  # gán lại cả dict để SQLAlchemy biết cột JSON đã đổi


def _step_translate(episode: Episode) -> None:
    from core.studio import translate

    if not episode.script_vi.strip():
        raise RuntimeError("Chưa có kịch bản tiếng Việt để dịch.")
    episode.script_en = translate(_context(episode), episode.script_vi)
    clear_storyboard(episode, "en")
    if episode.status != EpisodeStatus.DONE:
        episode.status = EpisodeStatus.TRANSLATED


def _step_wrap(episode: Episode) -> None:
    from core.studio import merge_glossary, wrap_up

    if not episode.script_vi.strip():
        raise RuntimeError("Chưa có kịch bản để chốt.")
    result = wrap_up(episode.script_vi)
    episode.summary = result["summary"]
    episode.status = EpisodeStatus.DONE
    episode.series.glossary = merge_glossary(episode.series.glossary or [], result["new_terms"])


_STEPS = {"plan": _step_plan, "ideas": _step_ideas, "write": _step_write, "translate": _step_translate, "wrap": _step_wrap,
          "board_vi": lambda ep: _step_board(ep, "vi"), "board_en": lambda ep: _step_board(ep, "en"),
          "preview": _step_preview, "youtube": _step_youtube,
          "publish_vi": lambda ep: _step_publish(ep, "vi"), "publish_en": lambda ep: _step_publish(ep, "en")}


# ── Trang ────────────────────────────────────────────────────
@bp.get("/quan-tri/xuong-kich-ban")
@require_admin_page
def page():
    return render_template("studio.html", series_id=None)


@bp.get("/quan-tri/xuong-kich-ban/<int:series_id>")
@require_admin_page
def series_page(series_id: int):
    from app.main import _voice_preview_urls
    from config.settings import VOICE_SAMPLE_LANGUAGES

    _own_series(series_id)
    # Nghe thử giọng giống trang Đọc kịch bản: cùng file mẫu (core/voice_previews.py),
    # "Giọng của bạn" phát lại chính file giọng mẫu đã tải lên.
    return render_template(
        "studio.html", series_id=series_id, voice_preview_urls=_voice_preview_urls(),
        voice_samples={lang: current_user.has_voice_sample_in(lang) for lang in VOICE_SAMPLE_LANGUAGES},
    )


# ── API ──────────────────────────────────────────────────────
def _own_series(series_id: int) -> Series:
    series = Series.query.filter_by(id=series_id, user_id=current_user.id).first()
    if series is None:
        from flask import abort

        abort(404)
    return series


def _own_episode(episode_id: int) -> Episode:
    episode = db.session.get(Episode, episode_id)
    if episode is None or episode.series.user_id != current_user.id:
        from flask import abort

        abort(404)
    return episode


def _busy(target) -> tuple | None:
    if target.task:
        return jsonify({"error": "Agent đang chạy một bước khác, hãy đợi xong.", "code": 409}), 409
    return None


@bp.get("/api/studio/series")
@require_admin
def list_series():
    rows = Series.query.filter_by(user_id=current_user.id).order_by(Series.created_at.desc()).all()
    return jsonify([{"id": s.id, "title": s.title or s.topic, "topic": s.topic, "task": s.task,
                     "episodes": len(s.episodes),
                     "done": sum(e.status == EpisodeStatus.DONE for e in s.episodes)} for s in rows])


@bp.post("/api/studio/series")
@require_admin
def create_series():
    body = request.get_json(silent=True) or {}
    topic = " ".join(str(body.get("topic", "")).split())[:1000]
    if not topic:
        return jsonify({"error": "Hãy nhập chủ đề series."}), 400
    try:
        count = max(1, min(30, int(body.get("episodes", 12))))
    except (TypeError, ValueError):
        count = 12
    series = Series(user_id=current_user.id, topic=topic, title=topic[:200],
                    audience=" ".join(str(body.get("audience", "")).split())[:500], task="plan")
    # Tập giữ chỗ: _step_plan đọc số tập muốn có từ đây rồi thay bằng lộ trình thật.
    series.episodes = [Episode(number=i + 1, title="") for i in range(count)]
    db.session.add(series)
    db.session.commit()
    start_task(current_app._get_current_object(), "plan", series.id)
    return jsonify(series.as_dict()), 202


@bp.get("/api/studio/series/<int:series_id>")
@require_admin
def get_series(series_id: int):
    data = _own_series(series_id).as_dict()
    data["has_music"] = music_file(series_id).is_file()
    data["task_label"] = TASK_LABELS.get(data["task"] or "")
    for ep in data["episodes"]:
        ep["task_label"] = TASK_LABELS.get((ep["task"] or "").split(":")[0].split("_")[0])
        episode = db.session.get(Episode, ep["id"])
        for lang in LANGUAGES:
            job = db.session.get(Job, ep[f"job_{lang}_id"]) if ep[f"job_{lang}_id"] else None
            ep[f"job_{lang}_status"] = job.status if job else None
            ep[f"sentences_{lang}"] = _sentences(episode, lang)
            ep[f"paragraphs_{lang}"] = paragraphs(getattr(episode, f"script_{lang}"))
            ep[f"recordings_{lang}"] = recording_status(episode, lang)
    return jsonify(data)


@bp.patch("/api/studio/series/<int:series_id>")
@require_admin
def update_series(series_id: int):
    series = _own_series(series_id)
    body = request.get_json(silent=True) or {}
    if "style_guide" in body:
        series.style_guide = str(body["style_guide"])[:4000]
    if "youtube_playlist" in body:
        import re

        playlist = str(body["youtube_playlist"] or "").strip()
        if playlist and not re.fullmatch(r"[A-Za-z0-9_-]{10,64}", playlist):
            return jsonify({"error": "Mã danh sách phát không hợp lệ."}), 400
        series.youtube_playlist = playlist or None
    if "glossary" in body:
        from core.studio import clean_glossary

        series.glossary = clean_glossary(body["glossary"])
    db.session.commit()
    return jsonify(series.as_dict())


@bp.post("/api/studio/series/<int:series_id>/music")
@require_admin
def upload_music(series_id: int):
    import tempfile
    import uuid
    from pathlib import Path

    from app.storage import commit_volume
    from config.settings import FFMPEG_BIN
    from utils.proc import run_command

    _own_series(series_id)
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Chưa chọn file nhạc."}), 400
    raw = Path(tempfile.gettempdir()) / f"music_{uuid.uuid4().hex}{Path(upload.filename).suffix}"
    upload.save(raw)
    target = music_file(series_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_command([FFMPEG_BIN, "-v", "error", "-y", "-i", str(raw), "-vn", "-t", str(MUSIC_MAX_SEC), "-ac", "2",
                     "-ar", "44100", "-c:a", "aac", "-b:a", "128k", str(target)],
                    timeout=180, error_message="Không đọc được file nhạc.")
    except RuntimeError:
        return jsonify({"error": "Không đọc được file nhạc — hãy dùng mp3, m4a, wav hoặc ogg."}), 400
    finally:
        raw.unlink(missing_ok=True)
    commit_volume()
    return jsonify({"has_music": True})


@bp.delete("/api/studio/series/<int:series_id>/music")
@require_admin
def delete_music(series_id: int):
    from app.storage import commit_volume

    _own_series(series_id)
    music_file(series_id).unlink(missing_ok=True)
    commit_volume()
    return jsonify({"has_music": False})


@bp.get("/quan-tri/xuong-kich-ban/nhac/<int:series_id>.m4a")
@require_admin_page
def serve_music(series_id: int):
    from flask import abort, send_from_directory

    _own_series(series_id)
    path = music_file(series_id)
    if not path.is_file():
        abort(404)
    return send_from_directory(path.parent, path.name, max_age=0)


@bp.delete("/api/studio/series/<int:series_id>")
@require_admin
def delete_series(series_id: int):
    series = _own_series(series_id)
    db.session.delete(series)
    db.session.commit()
    return jsonify({"deleted": series_id})


@bp.patch("/api/studio/episodes/<int:episode_id>")
@require_admin
def update_episode(episode_id: int):
    """Sửa tay: tên, mục tiêu, kịch bản; gắn job video đã tạo từ kịch bản."""
    from core.studio import clean_script

    episode = _own_episode(episode_id)
    body = request.get_json(silent=True) or {}
    for key, limit in (("title", 200), ("goal", 1000)):
        if key in body:
            setattr(episode, key, " ".join(str(body[key]).split())[:limit])
    for lang in LANGUAGES:
        key = f"script_{lang}"
        if key in body and clean_script(body[key]) != getattr(episode, key):
            setattr(episode, key, clean_script(body[key]))
            clear_storyboard(episode, lang)
    for lang in LANGUAGES:
        key = f"storyboard_{lang}"
        if key in body:
            from core.explainer import board_from_list, board_to_list

            board = board_from_list(body[key], len(_sentences(episode, lang)))
            if body[key] and not board:
                return jsonify({"error": "Bảng phân cảnh không có cảnh hợp lệ nào."}), 400
            setattr(episode, key, board_to_list(board))
    for key in ("job_vi_id", "job_en_id"):
        if key in body:
            job = Job.query.filter_by(id=body[key], user_id=current_user.id).first() if body[key] else None
            setattr(episode, key, job.id if job else None)
    db.session.commit()
    return jsonify(episode.as_dict())


@bp.post("/api/studio/episodes/<int:episode_id>/preview")
@require_admin
def preview_scene(episode_id: int):
    episode = _own_episode(episode_id)
    body = request.get_json(silent=True) or {}
    lang = body.get("lang")
    try:
        index = int(body.get("index"))
    except (TypeError, ValueError):
        index = -1
    if lang not in LANGUAGES or not 0 <= index < len(getattr(episode, f"storyboard_{lang}") or []):
        return jsonify({"error": "Không có cảnh này trong bảng phân cảnh."}), 400
    busy = _busy(episode) or _busy(episode.series)
    if busy:
        return busy
    episode.task, episode.error = f"preview:{lang}:{index}", None
    db.session.commit()
    start_task(current_app._get_current_object(), episode.task, episode.id)
    return jsonify(episode.as_dict()), 202


@bp.post("/api/studio/episodes/<int:episode_id>/youtube")
@require_admin
def upload_youtube(episode_id: int):
    """Đăng bản `lang` lên YouTube (riêng tư). publish_at (ISO 8601 có múi giờ) = lịch tự công khai."""
    from datetime import datetime, timedelta, timezone

    episode = _own_episode(episode_id)
    body = request.get_json(silent=True) or {}
    lang = body.get("lang")
    if lang not in LANGUAGES:
        return jsonify({"error": "Ngôn ngữ không hợp lệ."}), 400
    pack = (episode.publish or {}).get(lang)
    if not pack:
        return jsonify({"error": "Hãy soạn phụ đề + gói YouTube trước."}), 400
    if (pack.get("youtube") or {}).get("id"):
        return jsonify({"error": "Bản này đã đăng lên YouTube rồi."}), 409
    if not current_user.youtube_token:
        return jsonify({"error": "Chưa kết nối kênh YouTube."}), 400
    publish_at = ""
    raw = str(body.get("publish_at") or "").strip()
    if raw:
        try:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "Giờ đăng không hợp lệ."}), 400
        if when.tzinfo is None:
            return jsonify({"error": "Giờ đăng cần có múi giờ."}), 400
        when = when.astimezone(timezone.utc)
        if when < datetime.now(timezone.utc) + timedelta(minutes=15):
            return jsonify({"error": "Giờ đăng phải sau hiện tại ít nhất 15 phút."}), 400
        publish_at = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    busy = _busy(episode) or _busy(episode.series)
    if busy:
        return busy
    episode.task, episode.error = f"youtube:{lang}:{publish_at}", None
    db.session.commit()
    start_task(current_app._get_current_object(), episode.task, episode.id)
    return jsonify(episode.as_dict()), 202


# ── Ý tưởng mỗi ngày ─────────────────────────────────────────
@bp.get("/api/studio/series/<int:series_id>/ideas")
@require_admin
def list_ideas(series_id: int):
    from app.models import Idea

    _own_series(series_id)
    rows = Idea.query.filter_by(series_id=series_id).order_by(Idea.created_at.desc(), Idea.id).limit(40).all()
    return jsonify([i.as_dict() for i in rows])


@bp.post("/api/studio/series/<int:series_id>/ideas")
@require_admin
def find_ideas_now(series_id: int):
    series = _own_series(series_id)
    busy = _busy(series)
    if busy:
        return busy
    series.task, series.error = "ideas", None
    db.session.commit()
    start_task(current_app._get_current_object(), "ideas", series.id)
    return jsonify({"task": series.task}), 202


def _own_idea(idea_id: int):
    from app.models import Idea

    idea = db.session.get(Idea, idea_id)
    if idea is None:
        from flask import abort

        abort(404)
    _own_series(idea.series_id)
    return idea


@bp.patch("/api/studio/ideas/<int:idea_id>")
@require_admin
def update_idea(idea_id: int):
    from app.models import IdeaStatus

    idea = _own_idea(idea_id)
    status = (request.get_json(silent=True) or {}).get("status")
    if status not in (IdeaStatus.NEW, IdeaStatus.DISMISSED):
        return jsonify({"error": "Trạng thái không hợp lệ."}), 400
    if idea.status == IdeaStatus.USED:
        return jsonify({"error": "Ý tưởng này đã thành một tập."}), 409
    idea.status = status
    db.session.commit()
    return jsonify(idea.as_dict())


@bp.post("/api/studio/ideas/<int:idea_id>/episode")
@require_admin
def idea_to_episode(idea_id: int):
    """Thêm ý tưởng vào cuối lộ trình series thành một tập mới (chưa viết kịch bản)."""
    from app.models import IdeaStatus

    idea = _own_idea(idea_id)
    if idea.status == IdeaStatus.USED:
        return jsonify({"error": "Ý tưởng này đã thành một tập."}), 409
    series = _own_series(idea.series_id)
    number = max([e.number for e in series.episodes] or [0]) + 1
    refs = [f"Tham khảo: {s.get('url')}" for s in (idea.sources or [])[:2] if s.get("url")]
    episode = Episode(number=number, title=idea.title, goal=(idea.summary or "")[:1000],
                      key_points=(idea.key_points or [])[:5] + refs)
    series.episodes.append(episode)
    db.session.flush()
    idea.status, idea.episode_id = IdeaStatus.USED, episode.id
    db.session.commit()
    return jsonify({"idea": idea.as_dict(), "episode": episode.as_dict()}), 201


@bp.get("/quan-tri/xuong-kich-ban/xem-thu/<path:filename>")
@require_admin_page
def serve_preview(filename: str):
    """File cảnh xem thử — chỉ chủ của tập đó xem được."""
    import re

    from flask import abort, send_from_directory

    from app.storage import locate_file
    from config.settings import OUTPUT_DIR

    match = re.fullmatch(r"studio_(\d+)_(vi|en)_\d+_[0-9a-f]{8}\.mp4", filename)
    if match is None:
        abort(404)
    episode = _own_episode(int(match.group(1)))
    if filename not in (episode.previews or {}).values():
        abort(404)
    path = locate_file(OUTPUT_DIR / filename)
    return send_from_directory(path.parent, path.name, as_attachment=False)


@bp.post("/api/studio/episodes/<int:episode_id>/recordings")
@require_admin
def upload_recording(episode_id: int):
    """Lưu bản tự thu của một đoạn (thay bản cũ của đoạn đó)."""
    import tempfile
    import uuid
    from pathlib import Path

    from app.storage import commit_volume
    from core.recorded_voice import normalize_take, recording_name

    episode = _own_episode(episode_id)
    lang = request.form.get("lang")
    try:
        index = int(request.form.get("index", -1))
    except ValueError:
        index = -1
    paras = paragraphs(getattr(episode, f"script_{lang}", "") if lang in LANGUAGES else "")
    if not 0 <= index < len(paras):
        return jsonify({"error": "Không có đoạn này trong kịch bản (kịch bản vừa đổi? Hãy lưu rồi tải lại trang)."}), 400
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "Chưa có bản thu."}), 400
    folder = recordings_dir(episode.id)
    folder.mkdir(parents=True, exist_ok=True)
    raw = Path(tempfile.gettempdir()) / f"take_{uuid.uuid4().hex}{Path(upload.filename or '').suffix or '.webm'}"
    upload.save(raw)
    target = folder / recording_name(episode.id, lang, index, paras[index])
    try:
        seconds = normalize_take(raw, target)
    except RuntimeError:
        return jsonify({"error": "Không đọc được bản thu — thử thu lại."}), 400
    finally:
        raw.unlink(missing_ok=True)
    if seconds < 0.5:
        target.unlink(missing_ok=True)
        return jsonify({"error": "Bản thu gần như im lặng — kiểm tra lại micro."}), 400
    # Bỏ các bản thu cũ của đoạn này (chữ cũ hoặc lần thu trước).
    for old in folder.glob(f"rec_{episode.id}_{lang}_{index}_*.wav"):
        if old != target:
            old.unlink(missing_ok=True)
    commit_volume()
    return jsonify({"name": target.name, "seconds": round(seconds, 1)})


@bp.get("/quan-tri/xuong-kich-ban/phu-de/<int:episode_id>/<filename>")
@require_admin_page
def serve_subtitle(episode_id: int, filename: str):
    """Tải phụ đề đã làm ở bước đăng YouTube — chỉ chủ tập, chỉ file của tập đó."""
    from flask import abort, send_from_directory

    from app.storage import locate_file
    from config.settings import OUTPUT_DIR

    episode = _own_episode(episode_id)
    known = {name for pack in (episode.publish or {}).values() for name in (pack.get("srt") or {}).values()}
    if filename not in known:
        abort(404)
    path = locate_file(OUTPUT_DIR / filename)
    return send_from_directory(path.parent, path.name, as_attachment=True, download_name=filename)


@bp.get("/quan-tri/xuong-kich-ban/thu-am/<int:episode_id>/<filename>")
@require_admin_page
def serve_recording(episode_id: int, filename: str):
    import re

    from flask import abort, send_from_directory

    _own_episode(episode_id)
    if not re.fullmatch(rf"rec_{episode_id}_(vi|en)_\d+_[0-9a-f]{{8}}\.wav", filename):
        abort(404)
    folder = recordings_dir(episode_id)
    if not (folder / filename).is_file():
        abort(404)
    return send_from_directory(folder, filename, as_attachment=False, max_age=0)


@bp.post("/api/studio/episodes/<int:episode_id>/<step>")
@require_admin
def run_episode_step(episode_id: int, step: str):
    if step not in EPISODE_STEPS:
        return jsonify({"error": "Bước không hợp lệ."}), 400
    episode = _own_episode(episode_id)
    busy = _busy(episode) or _busy(episode.series)
    if busy:
        return busy
    episode.task, episode.error = step, None
    db.session.commit()
    start_task(current_app._get_current_object(), step, episode.id)
    return jsonify(episode.as_dict()), 202
