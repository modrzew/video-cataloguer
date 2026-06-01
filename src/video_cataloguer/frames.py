from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

MAX_FRAMES = 30
FRAME_SIZE = (320, 240)  # Scale down for LLM vision input


def extract_frames(
    video_path: Path,
    max_frames: int = MAX_FRAMES,
) -> list[tuple[float, Image.Image]]:
    """Extract uniformly spaced frames from *video_path*.

    Returns a list of (timestamp_seconds, PIL.Image) tuples.
    """
    duration = _get_duration(video_path)
    if duration <= 0:
        logger.warning("Could not determine duration for %s, skipping frames", video_path.name)
        return []

    # Cap the number of frames based on duration
    num_frames = min(max_frames, max(1, int(duration / 10)))  # ~1 frame per 10 seconds, max 30
    timestamps = _uniform_timestamps(duration, num_frames)

    frames: list[tuple[float, Image.Image]] = []
    for ts in timestamps:
        try:
            img = _extract_frame_at(video_path, ts)
            frames.append((ts, img))
        except Exception as e:
            logger.warning("Failed to extract frame at %.1fs in %s: %s", ts, video_path.name, e)

    logger.info("Extracted %d frames from %s", len(frames), video_path.name)
    return frames


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


def _uniform_timestamps(duration: float, count: int) -> list[float]:
    """Return *count* uniformly spaced timestamps across *duration* seconds."""
    if count == 1:
        return [duration / 2]
    interval = duration / (count + 1)
    return [round(interval * (i + 1), 2) for i in range(count)]


def _extract_frame_at(video_path: Path, timestamp: float) -> Image.Image:
    """Extract a single frame at *timestamp* seconds."""
    # Lazy-import ffmpeg-python — it's a heavy optional dependency that
    # also triggers PyTorch imports when used alongside Whisper.
    import ffmpeg  # type: ignore[import-untyped]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        (
            ffmpeg.input(str(video_path), ss=timestamp)
            .output(str(tmp.name), vframes=1, f="mjpeg")
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        img = Image.open(tmp.name)
        img.thumbnail(FRAME_SIZE, Image.Resampling.LANCZOS)
        return img
