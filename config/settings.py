from __future__ import annotations

import os
import importlib
from pathlib import Path

try:
	dotenv = importlib.import_module("dotenv")
	dotenv.load_dotenv()
except Exception:
	# dotenv is optional; environment variables can still be provided by shell.
	pass


def _env_bool(name: str, default: bool) -> bool:
	"""Read a boolean flag from env: 1/true/yes/on -> True."""
	raw = os.getenv(name)
	if raw is None or not raw.strip():
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
	try:
		return int(os.getenv(name, "").strip() or default)
	except ValueError:
		return default


APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"

# Secret key: bắt buộc phải có khi chạy production, không im lặng dùng key mặc định.
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "").strip()
if not SECRET_KEY:
	if IS_PRODUCTION:
		raise RuntimeError(
			"FLASK_SECRET_KEY chua duoc thiet lap. Bat buoc phai co khi APP_ENV=production. "
			"Tao key bang: python -c \"import secrets; print(secrets.token_hex(32))\""
		)
	SECRET_KEY = "dev-only-insecure-key"
	print("[config] CANH BAO: dang dung SECRET_KEY mac dinh cho development.")

# Cookie phiên: mặc định bật Secure khi production (đứng sau HTTPS/nginx).
SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", IS_PRODUCTION)
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
SESSION_LIFETIME_DAYS = _env_int("SESSION_LIFETIME_DAYS", 7)
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 2048)


ROOT_DIR = Path(__file__).resolve().parent.parent
# Tren Modal, DATA_DIR tro toi mot Volume duoc mount (vi du /data) vi dia container
# la ephemeral — file ghi vao day moi song sot qua cac lan chay.
DATA_DIR = Path(os.getenv("DATA_DIR", "").strip() or ROOT_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"

for directory in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
	directory.mkdir(parents=True, exist_ok=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ── Email (thong bao khi job can duyet / xong / loi) ────────
# Mac dinh Gmail SMTP + app password — khong can dang ky dich vu ngoai.
# Thieu SMTP_USER/SMTP_PASSWORD thi MAIL_ENABLED=False, app.mailer.send_email()
# tu bo qua trong im lang, khong lam hong job.
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = _env_int("SMTP_PORT", 587)
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "").strip() or SMTP_USER
MAIL_ENABLED = bool(SMTP_USER and SMTP_PASSWORD)
# Dung de dung link that trong email (vd https://pnam1802--video-dubber-web.modal.run).
# Rong thi email van gui duoc, chi thieu link bam thang toi job.
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip().rstrip("/")

# Database: mac dinh SQLite cho dev, production cam DATABASE_URL (Postgres).
_raw_db_url = os.getenv("DATABASE_URL", "").strip()
if _raw_db_url.startswith("postgres://"):
	# Neon/Heroku tra ve scheme cu; SQLAlchemy 2 can driver ro rang.
	_raw_db_url = _raw_db_url.replace("postgres://", "postgresql+psycopg://", 1)
elif _raw_db_url.startswith("postgresql://"):
	_raw_db_url = _raw_db_url.replace("postgresql://", "postgresql+psycopg://", 1)

SQLALCHEMY_DATABASE_URI = _raw_db_url or f"sqlite:///{DATA_DIR / 'database.db'}"

MARIAN_MODEL_EN_VI = os.getenv("MARIAN_MODEL_EN_VI", "pNam1802/marian-finetuned-en-vi-datn")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", "")).strip()

AI_PRESERVE_TERMS = [
	"AI",
	"ML",
	"Deep Learning",
	"Machine Learning",
	"Neural Network",
	"Transformer",
	"LLM",
	"GPU",
	"CPU",
	"Prompt",
	"Fine-tuning",
	"RAG",
	"Embeddings",
	"Token",
	"Inference",
	"Dataset",
	"OpenAI",
	"Google",
	"Microsoft",
	"PyTorch",
	"TensorFlow",
]

# Runner cho job: "thread" (chay trong tien trinh web) | "modal" (spawn GPU function)
JOB_RUNNER = os.getenv("JOB_RUNNER", "thread").strip().lower()
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "video-dubber").strip()
MODAL_DATA_VOLUME = os.getenv("MODAL_DATA_VOLUME", "dubber-data").strip()

# Dang nhap bang Google. De trong thi chi con dang nhap bang mat khau.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

# Backend Whisper: "auto" (uu tien faster-whisper) | "faster" | "openai"
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "auto").strip().lower()

# So segment TTS tong hop song song. edge-tts/gTTS chu yeu la cho mang,
# chay tuan tu khien buoc nay chiem phan lon thoi gian pipeline.
TTS_CONCURRENCY = _env_int("TTS_CONCURRENCY", 8)

# Sau proxy (Modal, nginx), request.remote_addr la IP cua proxy — moi nguoi dung
# se chung mot o rate limit. Bat ProxyFix de doc X-Forwarded-For.
# CHI bat khi that su dung sau proxy, khong thi header nay gia mao duoc.
TRUST_PROXY = _env_bool("TRUST_PROXY", IS_PRODUCTION)

# Han muc cho moi user thuong (admin khong bi gioi han).
QUOTA_JOBS_PER_DAY = _env_int("QUOTA_JOBS_PER_DAY", 10)
QUOTA_GPU_SECONDS_PER_DAY = _env_int("QUOTA_GPU_SECONDS_PER_DAY", 1800)
QUOTA_STORAGE_MB = _env_int("QUOTA_STORAGE_MB", 2048)

# Don gia GPU de uoc tinh chi phi moi job. Mac dinh theo T4 tren Modal.
GPU_COST_PER_SECOND = float(os.getenv("GPU_COST_PER_SECOND", "0.000164"))

# Gia LLM THAT (USD / 1 trieu token) — dung de tinh chi phi DICH thuc su,
# khac han GPU_COST_PER_SECOND (tien Modal, khong lien quan gi den API
# Gemini/OpenAI). Chot tu trang gia chinh thuc luc 08/09/2026, bac Standard:
#   Gemini: https://ai.google.dev/gemini-api/docs/pricing
#   OpenAI: https://developers.openai.com/api/docs/pricing
# Gia doi theo thoi gian va theo tung model — sua o day khi nha cung cap
# cong bo gia moi (vd gemini-3.6-flash tang gap doi tu 01/01/2027 theo
# trang gia hien tai). Model khong co trong bang nay thi KHONG tinh duoc
# chi phi that (ham tra ve None) — thay vi doan bay mot con so co the sai
# va lam nguoi dung tin nham.
LLM_PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
	"gemini-3.6-flash": {"input": 0.75, "output": 3.75},
	"gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Rate limit. Bo nho trong tien trinh: moi container web dem rieng, nen day la
# lop chan tho. Muon dem chung toan he thong thi can Redis (Phase 6).
RATELIMIT_LOGIN = os.getenv("RATELIMIT_LOGIN", "10 per minute")
RATELIMIT_REGISTER = os.getenv("RATELIMIT_REGISTER", "5 per hour")
RATELIMIT_UPLOAD = os.getenv("RATELIMIT_UPLOAD", "20 per hour")

# Timeout cho lenh ngoai. Ghep video la viec nang nen de rong tay;
# ffprobe chi doc metadata nen rat nhanh.
FFMPEG_TIMEOUT = _env_int("FFMPEG_TIMEOUT", 1800)
FFPROBE_TIMEOUT = _env_int("FFPROBE_TIMEOUT", 60)

# So lan thu lai khi API dich tra loi tam thoi (429, 5xx, timeout).
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 4)

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

