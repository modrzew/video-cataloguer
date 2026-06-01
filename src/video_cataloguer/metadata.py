from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from video_cataloguer.models import VideoMetadata

logger = logging.getLogger(__name__)


def extract_metadata(video_path: Path) -> VideoMetadata:
    """Run ffprobe on *video_path* and return structured metadata."""
    probe = _run_ffprobe(video_path)
    stream = _find_video_stream(probe)
    format_data = probe.get("format", {})

    duration = float(format_data.get("duration", 0))
    file_size = int(format_data.get("size", 0))

    # Try to get recording date from tags
    tags = format_data.get("tags", {})
    recording_date = tags.get("creation_time") or tags.get("date") or None

    # GPS from ISO-6709 location tag (used by iOS, Android, etc.)
    gps_lat, gps_lon = _parse_iso6709(tags.get("location"))

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    fps_raw = stream.get("r_frame_rate", "0/1")
    fps = _parse_fps(fps_raw)

    video_codec = stream.get("codec_name", "unknown")

    # Find audio stream
    audio_codec = "none"
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            audio_codec = s.get("codec_name", "unknown")
            break

    return VideoMetadata(
        file_path=str(video_path),
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=video_codec,
        audio_codec=audio_codec,
        file_size_bytes=file_size,
        recording_date=recording_date,
        gps_latitude=gps_lat,
        gps_longitude=gps_lon,
    )


def _run_ffprobe(video_path: Path) -> dict:
    """Run ffprobe and return parsed JSON."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _find_video_stream(probe: dict) -> dict:
    """Return the first video stream from ffprobe output."""
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError(f"No video stream found in {probe.get('format', {}).get('filename', '?')}")


def _parse_fps(fps_str: str) -> float:
    """Parse a fraction string like '30000/1001' into a float."""
    try:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    except ValueError, ZeroDivisionError:
        return 0.0


def _parse_iso6709(value: str | None) -> tuple[float | None, float | None]:
    """Parse an ISO-6709 coordinate string like '+37.77-122.41/' into (lat, lon).

    Returns (None, None) if the string cannot be parsed.

    ISO-6709 compact form:
        [+/-]latitude[+/-]longitude[/altitude]
        e.g. +37.7749-122.4194/, +37.7749-122.4194+010.123/, or +51.5074-001.2561/51h
    """
    if not value:
        return None, None

    nums = re.findall(r"[+-]\d+(?:\.\d+)?", value)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None
