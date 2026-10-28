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

# Gemini TTS: giong AI co cam xuc, lam theo chi dan giong dieu (xem
# core/tts_engine.py::_synthesize_gemini). Dung chung GEMINI_API_KEY.
# Flash-Lite: nghe thu 26/10/2026 gan nhu khong khac ban Flash, re hon ~30%
# ($6 so voi $9 / 1 trieu token am thanh) va tra ket qua nhanh hon (~5.5s so voi ~9s).
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_TTS_STYLE = os.getenv(
	"GEMINI_TTS_STYLE",
	"Giọng người kể chuyện review phim, truyện trên TikTok bằng tiếng Việt chuẩn: cuốn hút, "
	"nhấn nhá tự nhiên theo nội dung, tốc độ vừa phải, không đọc như đọc tin tức.",
)
# Chi dan giong dieu khi ngon ngu dich KHONG phai tieng Viet (GEMINI_TTS_STYLE
# o tren ep "tieng Viet chuan" — dung cho ban tieng Anh thi Gemini doc lo lo).
GEMINI_TTS_STYLES: dict[str, str] = {
	"vi": GEMINI_TTS_STYLE,
	"en": "An engaging TikTok movie/story recap narrator speaking natural American English: "
	"expressive, varied intonation that follows the content, moderate pace, not like a news anchor.",
}

# Danh muc giong theo tung engine: khoa luu trong Job.tts_voice -> ten giong
# that cua nha cung cap. "female"/"male" giu nguyen nghia cu cho job da co.
# Giong *Multilingual* cua edge-tts doc duoc tieng Viet (thu that 26/10/2026).
TTS_VOICES: dict[str, dict[str, str]] = {
	"edge-tts": {
		"female": "vi-VN-HoaiMyNeural",
		"male": "vi-VN-NamMinhNeural",
		"vivienne": "fr-FR-VivienneMultilingualNeural",
		"seraphina": "de-DE-SeraphinaMultilingualNeural",
		"florian": "de-DE-FlorianMultilingualNeural",
	},
	"gemini": {
		"female": "Leda",
		"male": "Charon",
	},
	"gtts": {
		"female": "vi",
	},
	# VieNeu-TTS (ma nguon mo, Apache 2.0 — github.com/pnnbao97/VieNeu-TTS):
	# chay tren GPU rieng (modal_app.py::VoiceCloner), ~25 giay cho 10 phut
	# giong doc (do that 26/10/2026 tren T4). Ten bat dau bang "@" la giong
	# CLONE tu file mau, con lai la giong co san cua VieNeu.
	"vieneu": {
		"female": "@nora",          # giong mac dinh cua Nora, xem NORA_VOICE_REF
		"ngochuyen": "Ngọc Huyền",
		"quynhanh": "Quỳnh Anh",
		"myduyen": "Mỹ Duyên",
		"kimthanh": "Kim Thanh",
		"male": "Thiện Minh",
		"ductri": "Đức Trí",
		"mine": "@user",            # giong nguoi dung tu tai len (VOICE_SAMPLE_DIR)
	},
	# Chatterbox-Turbo (Resemble AI, MIT — github.com/resemble-ai/chatterbox):
	# clone giong TIENG ANH tu file mau, chay tren GPU rieng
	# (modal_app.py::EnglishCloner). "@builtin" la giong co san cua model.
	"chatterbox": {
		"female": "@builtin",
		"mine": "@user",
	},
}

# Ngon ngu dich KHAC tieng Viet: giong vi-VN (HoaiMy/NamMinh) va gTTS "vi"
# doc chu nuoc ngoai theo kieu tieng Viet, nen doi sang giong ban xu cung
# khoa. Giong *Multilingual* va giong Gemini tu doc dung theo van ban, giu
# nguyen.
TTS_VOICES_BY_LANGUAGE: dict[str, dict[str, dict[str, str]]] = {
	"en": {
		"edge-tts": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
		"gtts": {"female": "en"},
	},
}
# Engine chi doc duoc mot so ngon ngu: VieNeu chi tieng Viet, Chatterbox-Turbo
# chi tieng Anh. Engine khong co ten o day doc duoc moi ngon ngu dich.
TTS_ENGINE_LANGUAGES: dict[str, set[str]] = {
	"vieneu": {"vi"},
	"chatterbox": {"en"},
}


def tts_engine_supports(engine: str, language: str) -> bool:
	return language in TTS_ENGINE_LANGUAGES.get(engine, {language})

# File mau cua giong mac dinh "Nora": VieNeu clone tu mot doan giong do chinh
# VieNeu tao ra (khong phai giong nguoi that).
NORA_VOICE_REF = ROOT_DIR / "core" / "voices" / "nora.wav"
# Giong mau nguoi dung tai len de clone: moi nguoi mot file
# voices/user_<id>.wav tren volume du lieu.
VOICE_SAMPLE_DIR = DATA_DIR / "voices"
# Moi nguoi mot giong mau cho MOI ngon ngu: clone giu ca ngu dieu/khau am cua
# mau, nen mau tieng Viet doc ban tieng Anh nghe lo lo (va nguoc lai).
VOICE_SAMPLE_LANGUAGES = {"vi": "Tiếng Việt", "en": "English"}
# Giong "mine" cua engine nao clone tu mau ngon ngu nao.
CLONE_SAMPLE_LANGUAGE = {"vieneu": "vi", "chatterbox": "en"}
VOICE_SAMPLE_MIN_SEC = 3.0
VOICE_SAMPLE_MAX_SEC = 15.0   # VieNeu chi can 3-8 giay; dai hon thi cat bot
# Chatterbox-Turbo bat buoc mau > 5 giay: mau ngan hon thi core/chatterbox_client.py
# lap lai cho du, nen van dung chung VOICE_SAMPLE_MIN_SEC voi VieNeu.
VOICE_SAMPLE_MAX_MB = 10

# Tua nhanh hinh o khoang lang cho khop giong doc (core/video_retime.py):
# bat mac dinh, toi da bay nhieu lan.
MAX_VIDEO_SPEED = 2.0

# DeepSeek dung API tuong thich OpenAI — xem core/translator/deepseek_translator.py.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

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
# Giai doan 2 (giong doc + ghep video) chay tren container CPU (Composer trong
# modal_app.py): 1 loi vat ly $0.0000131/s + 2 GiB x $0.00000222/s, gia Modal
# chot 26/10/2026 (https://modal.com/pricing). Luc ffmpeg dung them loi thi
# Modal tinh theo muc dung that — so nay la can duoi.
CPU_COST_PER_SECOND = float(os.getenv("CPU_COST_PER_SECOND", "0.0000175"))

# Gia LLM THAT (USD / 1 trieu token) — dung de tinh chi phi DICH thuc su,
# khac han GPU_COST_PER_SECOND (tien Modal, khong lien quan gi den API
# Gemini/OpenAI). Chot tu trang gia chinh thuc luc 08/09/2026, bac Standard:
#   Gemini: https://ai.google.dev/gemini-api/docs/pricing
#   OpenAI: https://developers.openai.com/api/docs/pricing
#   DeepSeek (chot 26/10/2026): https://api-docs.deepseek.com/quick_start/pricing/
#     DeepSeek co gia gio cao diem / thap diem (thap diem = 1/2) — bang nay
#     ghi gia CAO DIEM, cache miss: chi phi that chi co the bang hoac thap hon.
# Gia doi theo thoi gian va theo tung model — sua o day khi nha cung cap
# cong bo gia moi (vd gemini-3.6-flash tang gap doi tu 01/01/2027 theo
# trang gia hien tai). Model khong co trong bang nay thi KHONG tinh duoc
# chi phi that (ham tra ve None) — thay vi doan bay mot con so co the sai
# va lam nguoi dung tin nham.
LLM_PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
	"gemini-3.6-flash": {"input": 0.75, "output": 3.75},
	"gpt-4o-mini": {"input": 0.15, "output": 0.60},
	"deepseek-flash": {"input": 0.30, "output": 1.20},
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

