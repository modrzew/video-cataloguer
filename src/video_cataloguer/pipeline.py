from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from video_cataloguer.discovery import discover_videos
from video_cataloguer.frames import extract_frames
from video_cataloguer.llm import configure, describe_frames, generate_summary
from video_cataloguer.metadata import extract_metadata
from video_cataloguer.models import FrameDescription, Transcript, VideoCatalogEntry, VideoMetadata
from video_cataloguer.output import save_catalog_entry
from video_cataloguer.transcription import transcribe

logger = logging.getLogger(__name__)


class StageState(Enum):
    """Per-stage state for a single video."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class _StageTracker:
    """Tracks the status of each independent step per video.

    Each entry has state: ``"pending" | "running" | "done"``.
    """

    def __init__(self) -> None:
        self.metadata = StageState.PENDING
        self.audio = StageState.PENDING
        self.frames = StageState.PENDING
        self.summary = StageState.PENDING

    @property
    def all_done(self) -> bool:
        return (
            self.metadata == StageState.DONE
            and self.audio == StageState.DONE
            and self.frames == StageState.DONE
        )


def process_single_video(
    video_path: Path,
    whisper_model: str = "large",
    vision_model: str | None = None,
    summary_model: str | None = None,
    progress_callback: Callable[[str, str, str], None] | None = None,
) -> VideoCatalogEntry:
    """Process one video: metadata, audio transcription, frame extraction/description.

    Runs synchronously so it can be called from a thread executor.

    The LLM provider must be configured (via ``llm.configure()``) before
    calling this function — it runs in a worker thread that shares the
    main process memory.

    *progress_callback* is called as ``(video_name, stage, event)`` where
    *stage* is one of ``"metadata"``, ``"audio"``, ``"frames"``,
    ``"summary"``, and *event* is ``"start"`` or ``"done"``.
    """

    name = video_path.name

    def _report(stage: str, event: str) -> None:
        if progress_callback:
            progress_callback(name, stage, event)

    logger.info("[%s] Extracting metadata", name)
    _report("metadata", "start")
    metadata = extract_metadata(video_path)
    _report("metadata", "done")

    return VideoCatalogEntry(
        video_path=str(video_path),
        metadata=metadata,
        transcript=Transcript(),
        frame_descriptions=[],
        summary="",
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
    """Process all videos in *folder* with bounded per-type concurrency.

    Each processing step (metadata, audio transcription, frame description,
    summary generation) is submitted as an independent task to a shared
    executor. Per-type semaphores limit: ``max_concurrency`` concurrent
    Whisper calls AND ``max_concurrency`` concurrent LLM calls.

    Dependencies: metadata, audio transcription, and frame extraction/description
    run concurrently across all videos. Summary for each video waits until
    that video's metadata + audio + frames are all complete.

    Results are saved as Markdown files incrementally — each video is saved
    as soon as it finishes so a crash doesn't lose completed work.

    If *videos* is provided, skip discovery and use the supplied list.
    """
    if videos is None:
        videos = discover_videos(folder)
    total = len(videos)

    if total == 0:
        logger.warning("No videos found in %s", folder)
        return []

    if progress_callback:
        progress_callback("discovering")

    # Configure LLM once in the main thread; worker threads share process memory.
    configure(provider=llm_provider, base_url=llm_base_url)

    executor = ThreadPoolExecutor(max_workers=max_concurrency * 2)

    # Per-type semaphores: one for audio (Whisper), one for LLM (frames + summary)
    audio_semaphore = asyncio.Semaphore(max_concurrency)
    llm_semaphore = asyncio.Semaphore(max_concurrency)

    results: list[VideoCatalogEntry] = []
    completed = 0

    # Shared tracker and result storage per video
    trackers: dict[Path, _StageTracker] = {v: _StageTracker() for v in videos}
    meta_results: dict[Path, VideoMetadata] = {}
    audio_results: dict[Path, Transcript] = {}
    frames_results: dict[Path, list[FrameDescription]] = {}

    async def _run_audio(video: Path) -> Transcript:
        name = video.name
        await audio_semaphore.acquire()
        try:
            # Notify start
            if progress_callback:
                progress_callback("stage_start", name=name, stage="audio")

            loop = asyncio.get_running_loop()

            def _do_work() -> Transcript:
                return transcribe(video, model_name=whisper_model)

            transcript: Transcript = await loop.run_in_executor(executor, _do_work)
            audio_results[video] = transcript
            trackers[video].audio = StageState.DONE
            if progress_callback:
                progress_callback("stage_done", name=name, stage="audio")
            return transcript
        finally:
            audio_semaphore.release()

    async def _run_frames(video: Path) -> list[FrameDescription]:
        name = video.name
        await llm_semaphore.acquire()
        try:
            if progress_callback:
                progress_callback("stage_start", name=name, stage="frames")
            loop = asyncio.get_running_loop()

            def _do_work() -> list[FrameDescription]:
                frames = extract_frames(video)

                def _frame_progress(current: int, total: int) -> None:
                    if progress_callback:
                        progress_callback("stage_done", name=name, stage="frames")

                if frames:
                    return describe_frames(
                        frames, model=vision_model, progress_callback=_frame_progress
                    )
                return []

            result = await loop.run_in_executor(executor, _do_work)
            frames_results[video] = result
            trackers[video].frames = StageState.DONE
            if progress_callback:
                progress_callback("stage_done", name=name, stage="frames")
            return result
        finally:
            llm_semaphore.release()

    async def _run_summary(video: Path) -> str:
        name = video.name
        await llm_semaphore.acquire()
        try:
            if progress_callback:
                progress_callback("stage_start", name=name, stage="summary")
            loop = asyncio.get_running_loop()

            def _do_work() -> str:
                return generate_summary(
                    frame_descriptions=frames_results[video],
                    transcript=audio_results[video],
                    model=summary_model,
                )

            result: str = await loop.run_in_executor(executor, _do_work)
            trackers[video].summary = StageState.DONE
            if progress_callback:
                progress_callback("stage_done", name=name, stage="summary")
            return result
        finally:
            llm_semaphore.release()

    async def _run_metadata(video: Path) -> VideoMetadata:
        name = video.name
        if progress_callback:
            progress_callback("stage_start", name=name, stage="metadata")

        meta = extract_metadata(video)
        meta_results[video] = meta
        trackers[video].metadata = StageState.DONE
        if progress_callback:
            progress_callback("stage_done", name=name, stage="metadata")
        return meta

    async def _schedule_summary_when_ready(video: Path) -> None:
        """Wait until metadata+audio+frames are all done, then dispatch summary."""
        while True:
            t = trackers[video]
            if (
                t.metadata == StageState.DONE
                and t.audio == StageState.DONE
                and t.frames == StageState.DONE
            ):
                await _run_summary(video)
                return
            await asyncio.sleep(0.1)

    async def _process_video(video: Path) -> None:
        nonlocal completed
        try:
            # Dispatch independent stages immediately
            meta_task = asyncio.create_task(_run_metadata(video))
            audio_task = asyncio.create_task(_run_audio(video))
            frames_task = asyncio.create_task(_run_frames(video))

            # Wait for the three independent stages to finish
            await asyncio.gather(meta_task, audio_task, frames_task)

            # Schedule summary (waits internally for upstream deps)
            await _schedule_summary_when_ready(video)

            completed += 1
            if progress_callback:
                progress_callback("video_done", completed=completed, name=video.name)
            entry = VideoCatalogEntry(
                video_path=str(video),
                metadata=meta_results[video],
                transcript=audio_results[video],
                frame_descriptions=frames_results[video],
                summary=trackers[video].summary.value
                if trackers[video].summary == StageState.DONE
                else "",
            )
            results.append(entry)
            try:
                save_catalog_entry(entry, output_dir)
            except Exception as e:
                logger.error("Failed to save catalog for %s: %s", video.name, e)

        except Exception as e:
            logger.error("Failed to process %s: %s", video.name, e, exc_info=True)
            completed += 1
            if progress_callback:
                progress_callback("video_error", completed=completed, name=video.name, error=str(e))

    tasks = [_process_video(v) for v in videos]
    await asyncio.gather(*tasks, return_exceptions=True)

    executor.shutdown(wait=False)
    logger.info("Processed %d / %d videos", len(results), total)
    return results
