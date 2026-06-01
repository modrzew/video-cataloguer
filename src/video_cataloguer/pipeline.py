from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_cataloguer.discovery import discover_videos
from video_cataloguer.frames import extract_frames, get_frame_count
from video_cataloguer.llm import configure, describe_frames, generate_summary
from video_cataloguer.metadata import extract_metadata
from video_cataloguer.models import VideoCatalogEntry
from video_cataloguer.output import save_catalog_entry
from video_cataloguer.transcription import transcribe

logger = logging.getLogger(__name__)


def process_single_video(
    video_path: Path,
    whisper_model: str = "large",
    vision_model: str | None = None,
    summary_model: str | None = None,
    total_steps: int = 0,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> VideoCatalogEntry:
    """Process one video: metadata → transcript → frames → descriptions → summary.

    Runs synchronously so it can be called from a thread executor.

    The LLM provider must be configured (via ``llm.configure()``) before
    calling this function — it runs in a worker thread that shares the
    main process memory.

    *total_steps* is the expected step count (computed by the caller).
    *progress_callback* is called as ``(phase, current_step, total_steps)``.
    Phases: "transcribing", "describing_frames", "generating_summary".
    """

    name = video_path.name
    step = 0

    def _report(phase: str) -> None:
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(phase, step, total_steps)

    logger.info("[%s] Extracting metadata", name)
    metadata = extract_metadata(video_path)

    logger.info("[%s] Transcribing audio (model: %s)", name, whisper_model)
    transcript = transcribe(video_path, model_name=whisper_model)
    _report("transcribing")

    logger.info("[%s] Extracting frames", name)
    frames = extract_frames(video_path)

    if frames:
        logger.info("[%s] Describing frames (%d frames)", name, len(frames))

        def _frame_progress(current: int, total: int) -> None:
            logger.info("[%s] Describing frame %d/%d", name, current, total)
            _report("describing_frames")

        frame_descriptions = describe_frames(
            frames, model=vision_model, progress_callback=_frame_progress
        )
    else:
        frame_descriptions = []

    logger.info("[%s] Generating summary", name)
    summary = generate_summary(
        frame_descriptions=frame_descriptions,
        transcript=transcript,
        model=summary_model,
    )
    _report("generating_summary")

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
    whisper_model: str = "large",
    llm_provider: str = "ollama",
    llm_base_url: str | None = None,
    vision_model: str | None = None,
    summary_model: str | None = None,
    progress_callback: Callable[..., None] | None = None,
    videos: list[Path] | None = None,
) -> list[VideoCatalogEntry]:
    """Process all videos in *folder* with bounded concurrency.

    Each video is processed in a separate thread. Results are saved as
    Markdown files incrementally — each video is saved as soon as it
    finishes so a crash doesn't lose completed work.

    If *videos* is provided, skip discovery and use the supplied list.
    """
    if videos is None:
        videos = discover_videos(folder)
    total = len(videos)

    if total == 0:
        logger.warning("No videos found in %s", folder)
        return []

    # Pre-compute per-video step counts so the TUI knows totals upfront
    video_steps: dict[Path, int] = {}
    for video in videos:
        num_frames = get_frame_count(video)
        video_steps[video] = 1 + num_frames + 1  # transcribe + frames + summary

    if progress_callback:
        progress_callback(total, "discovering", steps_map=video_steps)

    # Configure LLM once in the main thread; worker threads share process memory.
    configure(provider=llm_provider, base_url=llm_base_url)

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[VideoCatalogEntry] = []
    completed = 0

    async def _process(video: Path) -> VideoCatalogEntry:
        nonlocal completed
        async with semaphore:
            try:
                steps = video_steps[video]
                if progress_callback:
                    progress_callback(
                        total,
                        "processing",
                        completed=completed,
                        name=video.name,
                        phase="transcribing",
                        current_step=0,
                        total_steps=steps,
                    )

                def _video_progress(phase: str, current: int, _total: int) -> None:
                    if progress_callback:
                        progress_callback(
                            total,
                            "step",
                            completed=completed,
                            name=video.name,
                            phase=phase,
                            current_step=current,
                            total_steps=steps,
                        )

                entry = await loop.run_in_executor(
                    executor,
                    process_single_video,
                    video,
                    whisper_model,
                    vision_model,
                    summary_model,
                    steps,
                    _video_progress,
                )
                completed += 1
                if progress_callback:
                    progress_callback(total, "done", completed, video.name)
                results.append(entry)
                # Save incrementally so a crash doesn't lose completed work
                try:
                    save_catalog_entry(entry, output_dir)
                except Exception as e:
                    logger.error("Failed to save catalog for %s: %s", video.name, e)
                return entry
            except Exception as e:
                logger.error("Failed to process %s: %s", video.name, e, exc_info=True)
                completed += 1
                if progress_callback:
                    progress_callback(total, "error", completed, video.name, str(e))
                raise

    tasks = [_process(v) for v in videos]
    await asyncio.gather(*tasks, return_exceptions=True)

    executor.shutdown(wait=False)
    logger.info("Processed %d / %d videos", len(results), total)
    return results
