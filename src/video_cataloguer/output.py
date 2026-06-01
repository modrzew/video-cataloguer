from __future__ import annotations

import logging
from pathlib import Path

from video_cataloguer.models import VideoCatalogEntry

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(minutes):02d}:{int(secs):02d}"


def _fmt_size(bytes_: int) -> str:
    """Format bytes as human-readable size."""
    size = float(bytes_)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} EB"


def _compose_markdown(entry: VideoCatalogEntry) -> str:
    """Compose the full Markdown catalog document from structured data."""
    meta = entry.metadata
    lines: list[str] = [
        f"# {Path(entry.video_path).name}",
        "",
        "## Metadata",
        f"- Duration: {_fmt_duration(meta.duration_seconds)}",
        f"- Resolution: {meta.width}x{meta.height}",
        f"- FPS: {meta.fps}",
        f"- Video codec: {meta.video_codec}",
        f"- Audio codec: {meta.audio_codec}",
        f"- File size: {_fmt_size(meta.file_size_bytes)}",
    ]

    if meta.recording_date:
        lines.append(f"- Recording date: {meta.recording_date}")

    if meta.gps_latitude is not None and meta.gps_longitude is not None:
        lines.append(f"- Location: {meta.gps_latitude}, {meta.gps_longitude}")

    lines += [
        "",
        "## Summary",
        entry.summary,
        "",
        "## Transcript",
        entry.transcript.to_timestamped_text(),
    ]

    return "\n".join(lines)


def save_catalog_entry(entry: VideoCatalogEntry, output_dir: str | None = None) -> Path:
    """Save the catalog entry as a Markdown file.

    If *output_dir* is given, writes there. Otherwise writes alongside the source video.
    """
    video_path = Path(entry.video_path)
    stem = video_path.stem

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"{stem}.md"
    else:
        dest = video_path.with_suffix(".md")

    markdown = _compose_markdown(entry)
    dest.write_text(markdown, encoding="utf-8")
    logger.info("Saved catalog: %s", dest)
    return dest
