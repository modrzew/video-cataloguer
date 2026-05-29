from __future__ import annotations

import asyncio
import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_cataloguer.discovery import discover_videos
from video_cataloguer.frames import extract_frames
from video_cataloguer.llm import configure, describe_frames, generate_summary
from video_cataloguer.metadata import extract_metadata
from video_cataloguer.models import VideoCatalogEntry
from video_cataloguer.output import save_catalog_entry
from video_cataloguer.transcription import transcribe

logger = logging.getLogger(__name__)


def process_single_video(
    video_path: Path,
    whisper_model: str = "tiny",
    llm_provider: str = "ollama",
    llm_base_url: str | None = None,
    vision_model: str | None = None,
    summary_model: str | None = None,
) -> VideoCatalogEntry:
    """Process one video: metadata → transcript → frames → descriptions → summary.

    Runs synchronously so it can be called from a thread executor.
    """
    # Configure the LLM provider in this worker process
    configure(provider=llm_provider, base_url=llm_base_url)

    name = video_path.name
    logger.info("[%s] Extracting metadata", name)
    metadata = extract_metadata(video_path)

    logger.info("[%s] Transcribing audio (model: %s)", name, whisper_model)
    transcript = transcribe(video_path, model_name=whisper_model)

    logger.info("[%s] Extracting frames", name)
    frames = extract_frames(video_path)

    if frames:
        logger.info("[%s] Describing frames", name)
        frame_descriptions = describe_frames(frames, model=vision_model)
    else:
        frame_descriptions = []

    # Build metadata dict for the LLM prompt
    metadata_dict = dataclasses.asdict(metadata)

    logger.info("[%s] Generating summary", name)
    summary = generate_summary(
        video_path=str(video_path),
        metadata=metadata_dict,
        transcript=transcript,
        frame_descriptions=frame_descriptions,
        model=summary_model,
    )

    return VideoCatalogEntry(
        video_path=str(video_path),
        metadata=metadata,
        transcript=transcript,
        frame_descriptions=frame_descriptions,
        summary=summary,
    )


async def process_videos(
    folder: str,
    output_dir: str | None = None,
    max_concurrency: int = 2,
    whisper_model: str = "tiny",
    llm_provider: str = "ollama",
    llm_base_url: str | None = None,
    vision_model: str | None = None,
    summary_model: str | None = None,
    progress_callback=None,
) -> list[VideoCatalogEntry]:
    """Process all videos in *folder* with bounded concurrency.

    Each video is processed in a separate thread. Results are saved as
    Markdown files.
    """
    videos = discover_videos(folder)
    total = len(videos)

    if total == 0:
        logger.warning("No videos found in %s", folder)
        return []

    if progress_callback:
        progress_callback(total, "discovering")

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[VideoCatalogEntry] = []
    completed = 0

    async def _process(video: Path) -> VideoCatalogEntry:
        nonlocal completed
        async with semaphore:
            try:
                if progress_callback:
                    progress_callback(total, "processing", completed, video.name)
                entry = await loop.run_in_executor(
                    executor,
                    process_single_video,
                    video,
                    whisper_model,
                    llm_provider,
                    llm_base_url,
                    vision_model,
                    summary_model,
                )
                completed += 1
                if progress_callback:
                    progress_callback(total, "done", completed, video.name)
                results.append(entry)
                return entry
            except Exception as e:
                logger.error("Failed to process %s: %s", video.name, e, exc_info=True)
                completed += 1
                if progress_callback:
                    progress_callback(total, "error", completed, video.name, str(e))
                raise

    tasks = [_process(v) for v in videos]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Save all results
    for entry in results:
        try:
            save_catalog_entry(entry, output_dir)
        except Exception as e:
            logger.error("Failed to save catalog for %s: %s", entry.video_path, e)

    executor.shutdown(wait=False)
    logger.info("Processed %d / %d videos", len(results), total)
    return results
