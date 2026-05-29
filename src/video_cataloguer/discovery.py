from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


def discover_videos(folder: str) -> list[Path]:
    """Recursively find all video files in *folder*."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    videos: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    logger.info("Discovered %d video(s) in %s", len(videos), root)
    return videos
