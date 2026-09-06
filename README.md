# AI Video Dubbing (EN to VI)

End-to-end pipeline to dub English videos into Vietnamese, with subtitle export and both UI/CLI workflows.

## Project Overview

This project automates the full dubbing process:
1. Extract audio from video with ffmpeg
2. Transcribe speech to text with Whisper
3. Translate to Vietnamese (or Japanese/Chinese/Korean) with Gemini or OpenAI
4. Synthesize Vietnamese speech with edge-tts, gTTS, or VietTTS
5. Compose dubbed audio back into the original video
6. Export video and SRT subtitles

Supported usage modes:
- Flask web application (login, upload, progress tracking, history)
- CLI for scripted or batch workflows

## Tech Stack

- Python 3.10+
- ffmpeg, ffprobe
- openai-whisper
- transformers + torch
- edge-tts, gTTS
- yt-dlp (optional for download workflows)

## Project Layout

	app/                Flask application package
	  __init__.py       create_app() factory
	  models.py         User, Job (schema + trạng thái job)
	  auth.py           đăng nhập / đăng ký / đăng xuất
	  main.py           trang chính, phục vụ file, /healthz
	  api.py            /api/upload, /api/progress
	  jobs.py           chạy pipeline nền, ghi tiến trình vào DB
	  cli.py            flask create-admin, flask set-role
	  errors.py         chuẩn hoá lỗi JSON/HTML
	core/               pipeline lồng tiếng (không phụ thuộc Flask)
	migrations/         Alembic
	wsgi.py             điểm vào WSGI

## Installation

Prerequisites:
- Python 3.10 or newer
- ffmpeg and ffprobe available in PATH

Setup:

	pip install -r requirements.txt
	copy .env.example .env
	flask --app wsgi db upgrade

If using OpenAI translation, set OPENAI_API_KEY in .env.

Create the first admin account:

	flask --app wsgi create-admin

The database defaults to SQLite at `data/database.db`. Set `DATABASE_URL` to a
Postgres connection string for production (required on Modal — see the roadmap below).

## Run The App

Development (Flask, auto-reload off):

	python wsgi.py

Production (waitress, Windows-friendly):

	set APP_ENV=production
	set FLASK_SECRET_KEY=<random hex>
	waitress-serve --listen=0.0.0.0:8000 wsgi:app

`APP_ENV=production` requires `FLASK_SECRET_KEY`; the app refuses to start without it.
Generate one with:

	python -c "import secrets; print(secrets.token_hex(32))"

Health check: `GET /healthz` returns 200 when the database, ffmpeg and ffprobe are all
reachable, 503 otherwise — point your load balancer at it.

Run from CLI (defaults to Gemini):

	python cli.py path/to/video.mp4 --gemini-api-key AIza...

OpenAI translation example:

	python cli.py path/to/video.mp4 --translator openai --openai-api-key sk-...

Gemini translation example:

	python cli.py path/to/video.mp4 --translator gemini --gemini-api-key AIza...

Mặc định dùng `gemini-3.6-flash`. Model 2.5 đã bị Google ngừng cấp cho tài khoản mới;
liệt kê model khả dụng cho key của bạn bằng `client.models.list()` nếu gặp lỗi 404.

VietTTS example:

	python cli.py path/to/video.mp4 --tts-engine viettts

## Output Artifacts

- Dubbed video: data/outputs/*_dubbed.mp4
- Subtitle file: data/outputs/*_dubbed.srt

## Demo Videos


### Demo 1: Source Video

https://github.com/user-attachments/assets/458dfa22-e69a-4aa6-b86b-cb6ddb3e952f



### Demo 2: Dubbed Video (MarianMT)

https://github.com/user-attachments/assets/3c90c685-9d6f-4107-a983-cd66ad8e9376


## Recommended Repo Hygiene For Demo Videos

- If a video is larger than 100 MB, use Git LFS or attach it in GitHub Releases.
- Keep filenames stable after publishing links in README.
- For filenames with spaces, use percent-encoded links (for example, space becomes %20).

## Notes

- First run may be slower because Whisper models are downloaded.
- Translation requires a Gemini or OpenAI API key; MarianMT (an in-house EN→VI fine-tune) has since
  been locked out of user-facing selection — it mistranslated silently for non-English sources and
  couldn't reach Japanese/Chinese/Korean at all. The code is kept for reference but no job can pick it.
- VietTTS setup may vary by package variant and environment.

---

# Lộ trình lên Production (Modal GPU)

> Cập nhật 29.08.2026 — **Phase 0 đã hoàn thành**.
> Phần này viết bằng tiếng Việt cho tiện theo dõi; phần còn lại của README giữ nguyên tiếng Anh.

## Kiến trúc đích

```
   Trình duyệt
        │  upload / poll
        ▼
┌──────────────────────┐   spawn()   ┌──────────────────────────┐
│  Web · CPU           │────────────▶│  Pipeline · GPU          │
│  @modal.wsgi_app()   │             │  @app.cls(gpu="T4")      │
│  bọc Flask hiện có   │◀────────────│  @modal.enter() load     │
│  min_containers=1    │  call_id    │  Whisper nạp 1 lần       │
└──────────┬───────────┘             └────────────┬─────────────┘
           │                                      │ ghi tiến trình
           ▼                                      ▼
   ┌───────────────────┐              ┌────────────────────────┐
   │ Postgres (Neon)   │◀─────────────│ R2 / Modal Volume      │
   │ user · job · quota│              │ video, srt, model cache│
   └───────────────────┘              └────────────────────────┘
```

Ba ràng buộc của Modal quyết định toàn bộ thiết kế:

| Ràng buộc | Hệ quả |
|---|---|
| Container ephemeral, chạy song song | Không dùng được SQLite và `data/outputs` trên đĩa → Postgres hosted + object storage |
| Web endpoint bị cắt ở **150 giây** | Không dùng SSE. Pattern chuẩn: `spawn()` trả call id → client poll |
| Tính tiền theo giây container (kể cả cold start) | `max_containers`, quota theo user, và tối ưu cold start là bắt buộc |

## Triển khai Modal

Toàn bộ khai báo nằm trong [`modal_app.py`](modal_app.py): container web (CPU) bọc Flask
bằng `@modal.wsgi_app()`, container GPU là `@app.cls(gpu="T4")` với `@modal.enter()` nạp
model một lần. Hai bên dùng chung một Volume cho file và chung Postgres cho tiến trình.

Các bước, chạy một lần:

	pip install modal
	modal setup
	modal volume create dubber-data
	modal volume create dubber-models
	modal secret create video-dubber \
	    APP_ENV=production \
	    FLASK_SECRET_KEY=<random hex> \
	    DATABASE_URL=postgresql://... \
	    JOB_RUNNER=modal \
	    DATA_DIR=/data \
	    GEMINI_API_KEY=...

Mỗi lần deploy:

	modal run modal_app.py::migrate     # flask db upgrade trên Postgres
	modal deploy modal_app.py

Xem log: `modal app logs video-dubber`.

### Cách web và GPU nói chuyện với nhau

| | Cơ chế |
|---|---|
| Gửi việc | Web gọi `Dubber().run.spawn(job_id, path, config)`, lưu `call.object_id` vào `Job.modal_call_id` |
| Truyền file | Cả hai mount chung Volume `dubber-data` tại `/data`; web `commit()`, GPU `reload()` |
| Báo tiến trình | GPU container ghi thẳng vào Postgres, web chỉ đọc DB — không cần Redis |
| Huỷ job | `modal.FunctionCall.from_id(call_id).cancel()` |
| Theo dõi | Client poll `GET /api/progress/<id>`; request HTTP không bao giờ chạm giới hạn 150 giây |

Chạy local vẫn như cũ: `JOB_RUNNER` mặc định là `thread`, job chạy trong thread nền
của tiến trình web, không đụng gì tới Modal.

## Các giai đoạn

### Phase 0 — Chốt an toàn tối thiểu ✅ (đã xong)

- [x] Bắt buộc `FLASK_SECRET_KEY` khi `APP_ENV=production`, fail fast lúc khởi động
- [x] `SESSION_COOKIE_SECURE / HTTPONLY / SAMESITE` + hạn session 7 ngày
- [x] `CSRFProtect` cho toàn bộ form (login, register, logout, upload)
- [x] Error handler trả JSON cho `/api/*`, xử lý 413 quá dung lượng
- [x] `GET /healthz` kiểm tra database + ffmpeg + ffprobe
- [x] Pin toàn bộ `requirements.txt`, thêm `Flask-WTF` và `waitress`

### Phase 1 — Nền dữ liệu ✅ (đã xong)

- [x] Tách `app.py` thành package `app/` với `create_app()` factory và 3 blueprint
- [x] Flask-Migrate + migration baseline (`migrations/versions/e05f1aec0495_*.py`)
- [x] `User`: thêm `role`, `is_active`, `created_at`
- [x] `Job`: thêm `status, progress, message, error, source_filename, file_size, translator_engine, tts_engine, whisper_model, duration_sec, elapsed_sec, segment_count, started_at, finished_at`
- [x] Bỏ dict `JOBS` trong RAM — trạng thái đọc/ghi thẳng từ DB; job dở dang được đánh dấu `interrupted` khi khởi động lại
- [x] CLI `flask create-admin` và `flask set-role`
- [x] Hỗ trợ Postgres qua `DATABASE_URL` (tự đổi scheme `postgres://` → `postgresql+psycopg://`)
- [x] Thêm cột `modal_call_id` vào `Job`

### Phase 2 — Đưa pipeline lên Modal GPU (đang làm)

- [x] `modal_app.py`: image (apt `ffmpeg` + requirements), `modal.Secret`, 2 `modal.Volume`
- [x] Bọc Flask bằng `@modal.wsgi_app()` trên container CPU
- [x] Pipeline thành `@app.cls(gpu="T4")` + `@modal.enter()` nạp model một lần
- [x] Model cache trỏ vào Volume qua `HF_HOME` / `XDG_CACHE_HOME`
- [x] `spawn()` → lưu `modal_call_id`; GPU ghi tiến trình vào Postgres; huỷ bằng `FunctionCall.cancel()`
- [x] `DATA_DIR` cấu hình được để trỏ vào Volume mount
- [x] **Độ trễ:** TTS chạy song song (`ThreadPoolExecutor`, `TTS_CONCURRENCY=8`) — 68,5s → 6,1s
- [x] **Độ trễ:** `faster-whisper` làm backend mặc định, openai-whisper giữ làm dự phòng — nhanh hơn 1,6–1,95×
- [x] `timeout=` cho mọi lệnh ffmpeg/ffprobe qua `utils/proc.py` (`FFMPEG_TIMEOUT`, `FFPROBE_TIMEOUT`)
- [x] Retry có backoff cho batch dịch OpenAI/Gemini (`LLM_MAX_RETRIES`), chỉ thử lại lỗi tạm thời
- [x] **Tự lùi sang engine dịch còn lại** (Gemini ⇄ OpenAI) khi một bên hỏng — một lần Gemini đổi model
      không còn làm hỏng cả job. (MarianMT từng là điểm lùi trước khi hỗ trợ đa ngôn ngữ; nay đã bị khoá
      khỏi lựa chọn của người dùng vì chỉ dịch được EN→VI.)
- [x] **Đo thời gian từng bước** (`extract/transcribe/translate/tts/compose_sec`) — thay việc bấm giờ thủ công
- [x] **Vị trí hàng đợi** trong `/api/progress` khi job đang chờ container GPU
- [ ] Đẩy output lên R2 + presigned URL (hiện đang dùng chung Volume, đủ dùng nhưng mọi lượt xem đều qua container web)
- [x] **Đã deploy và kiểm chứng thật** — https://pnam1802--video-dubber-web.modal.run
- [x] Đồng bộ Volume hai chiều: web `commit()` sau khi lưu upload, `reload()` khi phục vụ file kết quả (`app/storage.py`)

Kết quả chạy thật trên production (video 86 giây, Whisper base + MarianMT + edge-tts):

| | GTX 1650 (local) | **T4 (Modal, container ấm)** |
|---|---|---|
| Thời gian pipeline | 63,2s | **13,1s** |
| Tổng kể cả upload | — | 30,0s |

### Phase 3 — API v1 (2 ngày)

- [ ] Blueprint `/api/v1` trả JSON thuần, template Jinja vẫn chạy song song
- [ ] `POST /auth/login|logout|register`, `GET /me`
- [ ] `POST /jobs` (202 + call id), `GET /jobs` phân trang + lọc, `GET /jobs/<id>`, `DELETE`, `POST /jobs/<id>/cancel`
- [ ] Polling có backoff thay cho SSE
- [ ] `GET /config` trả engine/model khả dụng (bỏ hard-code trùng ở `app.py` và `templates/index.html`)
- [ ] `GET /stats` cho dashboard
- [ ] Chuẩn hoá error shape `{error, code, detail}`

### Phase 4 — Phân quyền & hạn mức ✅ (đã xong)

- [x] `require_role()` trong `app/permissions.py`, áp lên mọi endpoint quản trị
- [x] Admin API: `GET /api/admin/users`, `PATCH /api/admin/users/<id>` (đổi role, khoá, đặt hạn mức riêng), `GET /api/admin/stats`
- [x] `GET /api/usage` cho user xem mức tiêu thụ của chính mình
- [x] Quota theo user: số job/ngày, giây GPU/ngày, dung lượng lưu trữ — admin không bị giới hạn
- [x] Kiểm tra hạn mức **trước** khi ghi file, dùng `Content-Length`
- [x] Flask-Limiter cho login / register / upload
- [x] `ProxyFix` để rate limit đếm theo IP thật, không gộp tất cả vào IP của proxy Modal
- [x] `Job.estimated_cost_usd` + `User.quota_jobs_per_day`, migration `07861866bba4`

Hạn mức mặc định (chỉnh bằng biến môi trường):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `QUOTA_JOBS_PER_DAY` | 10 | Số job trong 24 giờ |
| `QUOTA_GPU_SECONDS_PER_DAY` | 1800 | Tổng giây GPU trong 24 giờ |
| `QUOTA_STORAGE_MB` | 2048 | Dung lượng video kết quả đang giữ |
| `GPU_COST_PER_SECOND` | 0.000164 | Đơn giá T4 để ước tính chi phí |

> **`estimated_cost_usd` là cận dưới, không phải hoá đơn.** Nó chỉ tính thời gian
> pipeline chạy. Modal còn tính thời gian container nằm không sau job, cold start,
> CPU, RAM và container web — thực tế cao hơn nhiều lần. Dùng để so sánh giữa user
> và giữa các video; muốn số thật thì chạy `modal environment billing report`.

### Phase 5 — Frontend React (2–3 tuần)

- [ ] Vite + React + TypeScript + TanStack Query, build tĩnh cho Flask serve (cùng origin, không CORS)
- [ ] Luồng auth + route được bảo vệ
- [ ] Upload kéo thả có progress thật (XHR)
- [ ] Trang job: polling, xem trước video, tải bằng presigned URL, huỷ job
- [ ] Lịch sử: lọc, phân trang, xoá
- [ ] Dashboard: job theo ngày, tỉ lệ thành công, thời gian xử lý, engine hay dùng, GPU-giây đã tiêu
- [ ] Trang admin, rồi xoá `templates/` và `static/` cũ

### Phase 6 — Deploy & vận hành (2 ngày)

- [ ] Hai Modal app dev/prod, mỗi bên một bộ Secret
- [ ] `flask db upgrade` chạy như một Modal function trước mỗi lần deploy
- [ ] Custom domain, `min_containers=1` cho web nếu cold start làm phiền
- [ ] `max_containers` + trần chi tiêu + cảnh báo
- [ ] Logging có cấu trúc + request id thay `print()`, gắn Sentry
- [ ] Vòng đời file trên R2, backup Postgres, gỡ 2 file `.mp4` khỏi git

### Phase 7 — Test & CI ✅ (đã xong)

- [x] **89 test pytest** trong `tests/`, chạy khoảng 30 giây
- [x] API test với Flask test client, mỗi test một DB SQLite sạch
- [x] Unit test phần lõi không cần Flask: fallback dịch, timeout lệnh ngoài, retry API
- [x] GitHub Actions: chạy migration trên DB trống rồi chạy test, mỗi lần push
- [x] `requirements-dev.txt` đủ để chạy test mà **không cần torch/whisper** — CI xong trong khoảng một phút
- [ ] Smoke test `modal run` với video 5 giây (nightly, tốn GPU nên không chạy mỗi PR)

Chạy test:

	pytest

Bộ test bao gồm cả những trường hợp bảo mật dễ bỏ sót: người dùng khác không xem,
không huỷ, không xoá được job của mình; email Google chưa xác minh không chiếm được
tài khoản; hạn mức chặn trước khi ghi file xuống đĩa.

## Tối ưu độ trễ

Xếp theo mức cải thiện trên mỗi công sử dụng. Hai mục đầu **không phụ thuộc Modal** — làm được ngay:

| Nút thắt | Cách xử lý | Ghi chú |
|---|---|---|
| ~~Whisper chậm~~ ✅ **đã làm** | `core/transcriber.py` giờ có 2 backend, mặc định `WHISPER_BACKEND=auto` ưu tiên faster-whisper (CUDA `float16`, CPU `int8`), tự lùi về openai-whisper nếu không khởi tạo được | Đo thật **1,6–1,95×** ở bước nhận dạng (không đạt 3–4× như dự án công bố, có thể do GPU 4 GB) |
| ~~TTS gọi tuần tự~~ ✅ **đã làm** | `synthesize_all()` giờ chạy song song bằng `ThreadPoolExecutor`, mặc định 8 luồng (`TTS_CONCURRENCY`) | Đo thật: **68,5s → 6,1s**. Tổng thời gian cả pipeline 114,2s → 49,4s |
| Cold start container GPU | `@modal.enter()` load model + Volume cache; `scaledown_window` dài hơn để container còn ấm giữa các job | Thời gian load model cũng bị tính tiền |
| Cold start web | `min_containers=1` cho container web (CPU rẻ) | Bỏ được độ trễ ở lần truy cập đầu |
| Import + khởi tạo nặng | Bật memory snapshot của Modal cho phần import/nạp weights vào RAM | Kiểm tra tài liệu xem phiên bản hiện tại đã snapshot được state trên GPU chưa |
| Dịch theo batch tuần tự | Chạy song song vài batch dịch (Gemini/OpenAI đều chịu được) | Marian chạy local thì giới hạn ở GPU, không song song được nhiều |

### Số đo thực tế (29.08.2026)

Video 86 giây, Whisper `base`, MarianMT, edge-tts, máy dev: **GTX 1650 laptop, 4 GB VRAM**.

**TTS song song** — đo hai lần liên tiếp trên cùng máy:

| Bước | Tuần tự | Song song 8 luồng |
|---|---|---|
| Whisper | 26,3s | 26,3s |
| Dịch (Marian) | 16,1s | 14,1s |
| TTS (11 segment) | **68,5s** | **6,1s** |
| Ghép ffmpeg | 2,0s | 2,0s |
| **Tổng** | **114,2s** | **49,4s** |

**faster-whisper vs openai-whisper** — chỉ riêng bước nhận dạng, ba lần đo:

| Điều kiện | openai-whisper | faster-whisper | Tỉ lệ |
|---|---|---|---|
| Đo riêng, model vừa tải | 14,2s | 7,3s | 1,95× |
| Đo riêng, model đã cache | 11,1s | 6,4s | 1,73× |
| Trong pipeline, máy đang nóng | 65,9s | 40,5s | 1,63× |

> **Cảnh báo về số liệu:** máy dev là GPU laptop 4 GB và bị throttle nhiệt rõ rệt sau
> vài lần chạy liên tiếp — cùng một cấu hình cho ra 49,4s lúc máy nguội và 125,5s sau
> nhiều lần chạy. Vì vậy **chỉ nên tin các so sánh đo liền kề nhau**, đừng so số tổng
> giữa các lần chạy cách xa nhau. Con số đáng tin cho việc chọn GPU sẽ đến từ Phase 2,
> khi mỗi job chạy trong một container Modal riêng.
>
> Mức cải thiện của TTS song song cũng phụ thuộc độ dài segment: với vài segment rất
> ngắn (~1 giây) thì song song không nhanh hơn, thậm chí chậm hơn chút vì chi phí luồng.

faster-whisper cũng cắt segment khác đi một chút (12 segment thay vì 11 trên cùng video),
nội dung nhận dạng tương đương.

Chọn GPU: Whisper `base`/`small` chạy thoải mái trên **T4** hoặc **L4**. Chỉ lên A10G/A100 khi thật sự
cần `large` — đắt hơn nhiều mà với model nhỏ thì không nhanh hơn tương ứng.

## Chi phí & tài khoản Modal

- Tài khoản sinh viên/học thuật dùng tốt cho dự án này. Modal có chương trình cấp credit cho
  academics (xem `modal.com/academics`), ngoài ra tài khoản mới có credit miễn phí ban đầu.
- Workload này rất hợp mô hình serverless: job chạy vài phút rồi scale về 0, không có server nằm không.
- Bắt buộc đặt `max_containers` và trần chi tiêu **trước** khi mở cho người ngoài dùng — một job lỗi
  lặp lại hoặc một người upload hàng loạt là đủ đốt sạch credit.
- Nếu định dùng cho mục đích ngoài học tập, đọc kỹ điều khoản của chương trình credit học thuật.
