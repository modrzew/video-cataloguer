from __future__ import annotations

import logging
from pathlib import Path

from video_cataloguer.models import VideoCatalogEntry

logger = logging.getLogger(__name__)


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

    dest.write_text(entry.summary, encoding="utf-8")
    logger.info("Saved catalog: %s", dest)
    return dest
