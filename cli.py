from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.settings import GEMINI_API_KEY
from core.pipeline import DubbingConfig, DubbingPipeline


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AI Video Dubbing CLI (EN -> VI)")
	parser.add_argument("video", help="Path to input video file")
	parser.add_argument("--translator", choices=["openai", "gemini"], default="gemini")
	parser.add_argument("--openai-api-key", default="")
	parser.add_argument("--openai-model", default="gpt-4o-mini")
	parser.add_argument("--gemini-api-key", default="")
	parser.add_argument("--gemini-model", default="gemini-2.5-flash")
	parser.add_argument("--whisper-model", default="base")
	parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
	parser.add_argument("--no-sentence-resegment", action="store_true")
	parser.add_argument("--silence-threshold", type=float, default=0.45)
	parser.add_argument("--tts-engine", choices=["edge-tts", "gtts", "viettts"], default="edge-tts")
	parser.add_argument("--tts-voice", choices=["female", "male"], default="female")
	parser.add_argument("--original-volume", type=float, default=0.1)
	parser.add_argument("--subtitle-mode", choices=["bilingual", "vi", "en", "none"], default="bilingual")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	video_path = Path(args.video)
	if not video_path.exists():
		print(f"[ERROR] Video not found: {video_path}")
		return 1

	cfg = DubbingConfig(
		translator_engine=args.translator,
		openai_api_key=args.openai_api_key,
		openai_model=args.openai_model,
		gemini_api_key=args.gemini_api_key or GEMINI_API_KEY,
		gemini_model=args.gemini_model,
		whisper_model=args.whisper_model,
		compute_device=args.device,
		sentence_resegment=not args.no_sentence_resegment,
		silence_threshold=args.silence_threshold,
		tts_engine=args.tts_engine,
		tts_voice=args.tts_voice,
		original_volume=args.original_volume,
		subtitle_mode=args.subtitle_mode,
	)

	pipeline = DubbingPipeline(cfg)
	result = pipeline.run(video_path)
	if not result.success:
		print(f"[ERROR] {result.error}")
		return 2

	print("[OK] Dubbing completed")
	print(f"Video: {result.output_video}")
	if result.srt_path:
		print(f"Subtitle: {result.srt_path}")
	print(f"Segments: {len(result.segments)}")
	print(f"Elapsed: {result.elapsed_seconds:.2f}s")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

