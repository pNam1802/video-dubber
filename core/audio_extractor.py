from __future__ import annotations

from pathlib import Path

from config.settings import (
	FFMPEG_BIN,
	FFMPEG_TIMEOUT,
	FFPROBE_BIN,
	FFPROBE_TIMEOUT,
	TEMP_DIR,
)
from utils.proc import run_command


class AudioExtractor:
	"""Extract and inspect audio streams using ffmpeg/ffprobe."""

	def extract(self, video_path: str | Path, output_path: str | Path | None = None) -> Path:
		video_path = Path(video_path)
		if not video_path.exists():
			raise FileNotFoundError(f"Video not found: {video_path}")

		if output_path is None:
			output_path = TEMP_DIR / f"{video_path.stem}.wav"
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		cmd = [
			FFMPEG_BIN,
			"-hide_banner",
			"-v",
			"error",
			"-y",
			"-i",
			str(video_path),
			"-vn",
			"-ac",
			"1",
			"-ar",
			"16000",
			"-c:a",
			"pcm_s16le",
			str(output_path),
		]
		run_command(
			cmd,
			timeout=FFMPEG_TIMEOUT,
			error_message="Cannot extract audio. Check ffmpeg installation.",
		)
		return output_path

	def get_duration(self, media_path: str | Path) -> float:
		media_path = Path(media_path)
		cmd = [
			FFPROBE_BIN,
			"-v",
			"error",
			"-show_entries",
			"format=duration",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			str(media_path),
		]
		completed = run_command(
			cmd,
			timeout=FFPROBE_TIMEOUT,
			error_message="Cannot read media duration. Check ffprobe installation.",
		)
		return float(completed.stdout.strip() or 0)


