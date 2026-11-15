from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from config.settings import (
	FFMPEG_BIN, FFMPEG_TIMEOUT, FFPROBE_BIN, FFPROBE_TIMEOUT, OUTPUT_DIR, OUTPUT_LOUDNORM, TEMP_DIR,
)
from utils.proc import run_command
from core.transcriber import Segment
from core.video_retime import Window, is_identity

#: Codec duy nhat chac chan phat duoc tren moi trinh duyet/dien thoai. Video
#: nguon co the la bat ky thu gi (webm=VP9, mov tu iPhone thuong la HEVC,
#: file tai tu YouTube gan day thuong la AV1...) — "-c:v copy" chi chep
#: nguyen codec nguon sang, KHONG kiem tra co phat duoc hay khong.
_BROWSER_SAFE_VIDEO_CODECS = {"h264"}

#: Giọng đọc xong SAU khi hết hình (giọng dài hơn lời gốc, câu cuối bị đẩy
#: trễ) thì giữ khung hình cuối đứng yên cho đọc hết câu, cộng thêm nhịp này.
#: Trước đây -shortest + amix duration=first cắt cụt câu cuối giữa chừng
#: (đoạn manhwa 34 giây: câu cuối vào trễ 1,53 giây so với hình).
END_HOLD_SEC = 0.6


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
	def _pts_expr(windows: Sequence[Window]) -> str:
		"""Hàm từng khúc T gốc -> T mới, dựng thành CÂY if() cân bằng chia đôi
		theo mốc src_start (các cửa sổ liền nhau, phủ kín từ 0).

		KHÔNG dùng tổng phẳng gte()*lt()*(...) + ... : ffmpeg 5.1 (bản
		Debian 12 trên Modal) báo "Cannot allocate memory" khi parse biểu
		thức đó từ khoảng 100 cửa sổ trở lên — job #80 (video 9,6 phút, 103
		câu) hỏng ở đây. Dạng cây chạy được tới vài trăm cửa sổ, và mỗi
		khung hình chỉ tính log2(N) phép so sánh thay vì N."""
		if len(windows) == 1:
			w = windows[0]
			return f"({w.out_start:.4f}+(T-{w.src_start:.4f})/{w.speed:.4f})"
		mid = len(windows) // 2
		left = VideoComposer._pts_expr(windows[:mid])
		right = VideoComposer._pts_expr(windows[mid:])
		return f"if(lt(T,{windows[mid].src_start:.4f}),{left},{right})"

	@staticmethod
	def _retime_filters(windows: Sequence[Window], fps: str, original_volume: float) -> list[str]:
		"""Hình: MỘT setpts theo hàm từng khúc (giải mã một lần, không cắt
		thành nhiều luồng). Âm thanh gốc: cắt từng khúc + atempo (giữ cao độ)
		rồi nối lại — atempo không đổi tốc độ theo thời gian được."""
		# Bọc nháy đơn: dấu phẩy trong if(...) nếu không sẽ bị hiểu là ranh
		# giới giữa hai bộ lọc.
		video = f"[0:v]setpts='{VideoComposer._pts_expr(windows)}/TB',fps={fps}[vout]"

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
		content_end: Optional[float] = None,
	) -> Path:
		"""retime: cửa sổ tua nhanh hình (core/video_retime.py). Có thì
		start_times phải là thời điểm trên video MỚI (đã tua).
		content_end: lúc giọng đọc cuối cùng kết thúc (trên video mới) —
		muộn hơn cuối hình thì kéo dài hình bằng khung cuối, xem END_HOLD_SEC."""
		video_path = Path(video_path)
		if not video_path.exists():
			raise FileNotFoundError(f"Video not found: {video_path}")
		if not tts_paths:
			raise ValueError("No TTS files provided to compose output video.")

		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		out_path = OUTPUT_DIR / f"{video_path.stem}_dubbed.mp4"

		cmd = [FFMPEG_BIN, "-hide_banner", "-v", "error", "-y", "-i", str(video_path)]
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

		video_end = retime[-1].out_end if retimed else video_duration
		hold = 0.0
		if content_end and video_end:
			hold = max(0.0, content_end + END_HOLD_SEC - video_end)
		extend = hold > 0.05

		filter_parts.append(
			"".join(mix_inputs)
			+ f"amix=inputs={len(mix_inputs)}:duration={'longest' if extend else 'first'}"
			+ ":dropout_transition=2:normalize=0[mix_raw]"
		)
		# Đưa độ to cả bản mix về chuẩn -14 LUFS (xem OUTPUT_LOUDNORM): mỗi
		# engine TTS ra một mức khác nhau, riêng tts_volume không đủ bù.
		pad = f"apad=whole_dur={video_end + hold:.3f}," if extend else ""
		filter_parts.append(f"[mix_raw]{pad}{OUTPUT_LOUDNORM}[mix]")
		video_map = "[vout]" if retimed else "0:v"
		if extend:
			source = "[vout]" if retimed else "[0:v]"
			filter_parts.append(f"{source}tpad=stop_mode=clone:stop_duration={hold:.3f}[vext]")
			video_map = "[vext]"
		filter_complex = ";".join(filter_parts)

		# copy nhanh hon nhieu (khong giai ma/ma hoa lai video), nhung chi an
		# toan khi nguon da la H.264 san — codec duy nhat moi trinh duyet va
		# dien thoai deu phat duoc. Nguon khac (webm=VP9, mov iPhone=HEVC,
		# video moi tai ve thuong la AV1...) thi phai ma hoa lai, khong thi
		# ra video "hoan tat" nhung bam khong xem duoc — da gap that tren
		# production voi mot video nguon AV1.
		source_codec = self._source_video_codec(video_path)
		if source_codec in _BROWSER_SAFE_VIDEO_CODECS and not retimed and not extend:
			video_codec_args = ["-c:v", "copy"]
		else:
			video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]

		cmd += [
			"-filter_complex",
			filter_complex,
			"-map",
			video_map,
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
			error_message="Ghép video thất bại (ffmpeg):",
		)
		return out_path

