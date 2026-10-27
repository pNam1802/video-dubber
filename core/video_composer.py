from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from config.settings import FFMPEG_BIN, FFMPEG_TIMEOUT, FFPROBE_BIN, FFPROBE_TIMEOUT, OUTPUT_DIR, TEMP_DIR
from utils.proc import run_command
from core.transcriber import Segment
from core.video_retime import Window, is_identity

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

	def _source_fps(self, video_path: Path) -> str:
		"""Tốc độ khung hình nguồn (vd "30000/1001"), để video tua nhanh giữ
		nguyên fps thay vì dồn gấp đôi số khung hình."""
		cmd = [
			FFPROBE_BIN, "-v", "error",
			"-select_streams", "v:0",
			"-show_entries", "stream=r_frame_rate",
			"-of", "default=noprint_wrappers=1:nokey=1",
			str(video_path),
		]
		try:
			rate = run_command(cmd, timeout=FFPROBE_TIMEOUT, error_message="Không đọc được fps.").stdout.strip()
		except RuntimeError:
			return "30"
		return rate if rate and not rate.startswith("0") else "30"

	@staticmethod
	def _retime_filters(windows: Sequence[Window], fps: str, original_volume: float) -> list[str]:
		"""Hình: MỘT setpts theo hàm từng khúc (giải mã một lần, không cắt
		thành nhiều luồng). Âm thanh gốc: cắt từng khúc + atempo (giữ cao độ)
		rồi nối lại — atempo không đổi tốc độ theo thời gian được."""
		terms = []
		for i, w in enumerate(windows):
			gate = f"gte(T,{w.src_start:.4f})"
			if i + 1 < len(windows):
				gate += f"*lt(T,{w.src_end:.4f})"
			terms.append(f"{gate}*({w.out_start:.4f}+(T-{w.src_start:.4f})/{w.speed:.4f})")
		# Bọc nháy đơn: dấu phẩy trong gte(T,...) nếu không sẽ bị hiểu là
		# ranh giới giữa hai bộ lọc.
		video = f"[0:v]setpts='({'+'.join(terms)})/TB',fps={fps}[vout]"

		parts = [video, f"[0:a]asplit={len(windows)}" + "".join(f"[as{i}]" for i in range(len(windows)))]
		for i, w in enumerate(windows):
			tempo = f",atempo={w.speed:.4f}" if abs(w.speed - 1.0) > 1e-6 else ""
			parts.append(
				f"[as{i}]atrim=start={w.src_start:.4f}:end={w.src_end:.4f},asetpts=PTS-STARTPTS{tempo}[ar{i}]"
			)
		parts.append(
			"".join(f"[ar{i}]" for i in range(len(windows)))
			+ f"concat=n={len(windows)}:v=0:a=1,volume={original_volume}[orig]"
		)
		return parts

	def compose(
		self,
		video_path: str | Path,
		segments: List[Segment],
		tts_paths: List[str | Path],
		original_volume: float = 0.1,
		tts_volume: float = 1.6,
		video_duration: Optional[float] = None,
		start_times: Optional[List[float]] = None,
		retime: Optional[Sequence[Window]] = None,
	) -> Path:
		"""retime: cửa sổ tua nhanh hình (core/video_retime.py). Có thì
		start_times phải là thời điểm trên video MỚI (đã tua)."""
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

		retimed = bool(retime) and not is_identity(retime)
		if retimed:
			filter_parts = self._retime_filters(retime, self._source_fps(video_path), original_volume)
		else:
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
		if source_codec in _BROWSER_SAFE_VIDEO_CODECS and not retimed:
			video_codec_args = ["-c:v", "copy"]
		else:
			video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]

		cmd += [
			"-filter_complex",
			filter_complex,
			"-map",
			"[vout]" if retimed else "0:v",
			"-map",
			"[mix]",
			*video_codec_args,
			"-c:a",
			"aac",
			# moov ở đầu file: trình duyệt phát được ngay khi đang tải.
			"-movflags",
			"+faststart",
			"-shortest",
			str(out_path),
		]

		run_command(
			cmd,
			timeout=FFMPEG_TIMEOUT,
			error_message="Video compose failed. Check ffmpeg installation.",
		)
		return out_path

