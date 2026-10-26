from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from config.settings import FFMPEG_BIN, FFMPEG_TIMEOUT, FFPROBE_BIN, FFPROBE_TIMEOUT, OUTPUT_DIR, TEMP_DIR
from utils.proc import run_command
from core.transcriber import Segment

#: Codec duy nhat chac chan phat duoc tren moi trinh duyet/dien thoai. Video
#: nguon co the la bat ky thu gi (webm=VP9, mov tu iPhone thuong la HEVC,
#: file tai tu YouTube gan day thuong la AV1...) — "-c:v copy" chi chep
#: nguyen codec nguon sang, KHONG kiem tra co phat duoc hay khong.
_BROWSER_SAFE_VIDEO_CODECS = {"h264"}


class VideoComposer:
	"""Compose final dubbed video by delaying and mixing per-segment TTS audio."""

	def _source_video_codec(self, video_path: Path) -> str:
		"""Đọc codec video của file nguồn. Lỗi hay không đọc được thì trả về
		chuỗi rỗng — coi như KHÔNG an toàn, để compose() mã hoá lại cho chắc,
		thà chậm hơn một chút còn hơn ra video không ai xem được."""
		cmd = [
			FFPROBE_BIN, "-v", "error",
			"-select_streams", "v:0",
			"-show_entries", "stream=codec_name",
			"-of", "default=noprint_wrappers=1:nokey=1",
			str(video_path),
		]
		try:
			completed = run_command(
				cmd, timeout=FFPROBE_TIMEOUT,
				error_message="Không đọc được codec video nguồn.",
			)
		except RuntimeError:
			return ""
		return completed.stdout.strip().lower()

	def compose(
		self,
		video_path: str | Path,
		segments: List[Segment],
		tts_paths: List[str | Path],
		original_volume: float = 0.1,
		tts_volume: float = 1.6,
		video_duration: Optional[float] = None,
		start_times: Optional[List[float]] = None,
	) -> Path:
		video_path = Path(video_path)
		if not video_path.exists():
			raise FileNotFoundError(f"Video not found: {video_path}")
		if not tts_paths:
			raise ValueError("No TTS files provided to compose output video.")

		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		out_path = OUTPUT_DIR / f"{video_path.stem}_dubbed.mp4"

		cmd = [FFMPEG_BIN, "-y", "-i", str(video_path)]
		normalized_tts = [Path(p) for p in tts_paths]
		for p in normalized_tts:
			cmd += ["-i", str(p)]

		filter_parts = [f"[0:a]volume={original_volume}[orig]"]
		mix_inputs = ["[orig]"]

		limit = min(len(segments), len(normalized_tts))
		for idx in range(limit):
			# start_times: thời điểm đã xếp bởi TTSEngine.place_on_timeline()
			# (nghe liền mạch); không có thì đặt đúng thời điểm câu gốc.
			start = start_times[idx] if start_times is not None else segments[idx].start
			delay_ms = max(0, int(start * 1000))
			label = f"t{idx}"
			filter_parts.append(
				f"[{idx + 1}:a]volume={tts_volume},adelay={delay_ms}|{delay_ms}[{label}]"
			)
			mix_inputs.append(f"[{label}]")

		filter_parts.append(
			"".join(mix_inputs)
			+ f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=2:normalize=0[mix_raw]"
		)
		# Limit peaks after boosting TTS to avoid clipping while keeping loudness stable.
		filter_parts.append("[mix_raw]alimiter=limit=0.95[mix]")
		filter_complex = ";".join(filter_parts)

		# copy nhanh hon nhieu (khong giai ma/ma hoa lai video), nhung chi an
		# toan khi nguon da la H.264 san — codec duy nhat moi trinh duyet va
		# dien thoai deu phat duoc. Nguon khac (webm=VP9, mov iPhone=HEVC,
		# video moi tai ve thuong la AV1...) thi phai ma hoa lai, khong thi
		# ra video "hoan tat" nhung bam khong xem duoc — da gap that tren
		# production voi mot video nguon AV1.
		source_codec = self._source_video_codec(video_path)
		if source_codec in _BROWSER_SAFE_VIDEO_CODECS:
			video_codec_args = ["-c:v", "copy"]
		else:
			video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]

		cmd += [
			"-filter_complex",
			filter_complex,
			"-map",
			"0:v",
			"-map",
			"[mix]",
			*video_codec_args,
			"-c:a",
			"aac",
			"-shortest",
			str(out_path),
		]

		run_command(
			cmd,
			timeout=FFMPEG_TIMEOUT,
			error_message="Video compose failed. Check ffmpeg installation.",
		)
		return out_path

