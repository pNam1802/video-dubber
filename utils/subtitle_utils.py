from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.transcriber import Segment


def _fmt_srt_time(seconds: float) -> str:
	ms_total = int(max(0.0, seconds) * 1000)
	hours = ms_total // 3_600_000
	minutes = (ms_total % 3_600_000) // 60_000
	secs = (ms_total % 60_000) // 1000
	ms = ms_total % 1000
	return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(
	segments: Iterable[Segment],
	mode: str = "bilingual",
	output_path: str | Path | None = None,
) -> str:
	lines: list[str] = []
	seg_list = list(segments)

	for idx, seg in enumerate(seg_list, start=1):
		lines.append(str(idx))
		lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")

		# "source" khong nhat thiet la tieng Anh, "target" khong nhat thiet
		# la tieng Viet — ten trung lap vi source_language/target_language
		# gio la tuy chon (Nhat/Trung/Han...), khong con co dinh nhu truoc.
		source_text = (seg.text or "").strip()
		target_text = (seg.translated or "").strip()

		if mode == "source":
			lines.append(source_text)
		elif mode == "target":
			lines.append(target_text or source_text)
		else:
			if source_text:
				lines.append(source_text)
			if target_text:
				lines.append(target_text)

		lines.append("")

	content = "\n".join(lines).strip() + "\n"
	if output_path is not None:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(content, encoding="utf-8")
	return content

