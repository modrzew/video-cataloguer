from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Label, Log, ProgressBar, Rule

from video_cataloguer.discovery import discover_videos
from video_cataloguer.frames import get_frame_count
from video_cataloguer.pipeline import process_videos

logger = logging.getLogger(__name__)


@dataclass
class VideoProgress:
    """Tracks progress for a single video."""

    name: str
    status: str = "pending"  # pending | processing | done | error
    phase: str = ""
    current_step: int = 0
    total_steps: int = 0
    detail: str = ""


class StatusBanner(Label):
    """Top banner showing overall status."""

    DEFAULT_CSS = """
    StatusBanner {
        width: 100%;
        content-align: center middle;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
"""


class VideoRow(Container):
    """Single row showing a video's filename, progress bar, and status."""

    DEFAULT_CSS = """
    VideoRow {
        layout: horizontal;
        height: auto;
        width: 100%;
    }

    VideoRow .vr-name {
        width: 1fr;
        padding: 0 1;
        content-align: left middle;
    }

    VideoRow .vr-bar {
        width: 30;
        padding: 0 1;
        content-align: center middle;
    }

    VideoRow .vr-status {
        width: 1fr;
        padding: 0 1;
        content-align: left middle;
    }
"""

    def __init__(self, name: str, total_steps: int) -> None:
        super().__init__()
        self._video_name = name
        self.total_steps = total_steps
        self._name_label: Label
        self._bar: ProgressBar
        self._status_label: Label

    def compose(self) -> ComposeResult:
        self._name_label = Label(f"  {self._video_name}")
        self._name_label.classes = "vr-name"
        yield self._name_label

        self._bar = ProgressBar(total=self.total_steps, show_percentage=False, show_eta=False)
        self._bar.classes = "vr-bar"
        yield self._bar

        self._status_label = Label("")
        self._status_label.classes = "vr-status"
        yield self._status_label

    def update(self, current: int, total: int, status: str) -> None:
        """Update the progress bar and status text."""
        self._bar.update(progress=current, total=total)
        self._status_label.update(f"  {status}")


class VideoList(Container):
    """Container holding one VideoRow per video. Scrollbar appears only when needed."""

    DEFAULT_CSS = """
    VideoList {
        height: 1fr;
        border: solid $accent;
        padding: 0;
        overflow-y: auto;
    }
"""

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.rows: dict[str, VideoRow] = {}

    def add_row(self, name: str, total_steps: int) -> None:
        row = VideoRow(name, total_steps)
        self.rows[name] = row
        self.mount(row)

    def update_row(self, name: str, current: int, total: int, status: str) -> None:
        row = self.rows.get(name)
        if row:
            row.update(current, total, status)


class CataloguerApp(App):
    """TUI for the video cataloguer."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #video-list {
        height: 1fr;
        border: solid $accent;
        padding: 0;
    }

    #details-log {
        height: 1fr;
        text-wrap: wrap;
        max-width: 100%;
        overflow-y: auto;
    }
"""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "details", "Toggle Details"),
    ]

    # Disable built-in command palette and theme switching
    command_palette_enabled = False
    theme_switcher_enabled = False

    def __init__(
        self,
        folder: str,
        output_dir: str | None = None,
        max_concurrency: int = 2,
        whisper_model: str = "tiny",
        llm_provider: str = "ollama",
        llm_base_url: str | None = None,
        vision_model: str | None = None,
        summary_model: str | None = None,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.output_dir = output_dir
        self.max_concurrency = max_concurrency
        self.whisper_model = whisper_model
        self.llm_provider = llm_provider
        self.llm_base_url = llm_base_url
        self.vision_model = vision_model
        self.summary_model = summary_model

        self.videos: list[VideoProgress] = []
        self.show_details = True
        self.video_steps: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBanner("Discovering videos...", id="status-banner")
        yield VideoList(id="video-list")
        yield Rule()
        yield Log(id="details-log")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_logging()
        self._start_processing()

    def _setup_logging(self) -> None:
        """Route log output to the details panel."""
        handler = _TextualLogHandler(self)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def _start_processing(self) -> None:
        """Kick off async processing."""
        asyncio.create_task(self._run_pipeline())

    async def _run_pipeline(self) -> None:
        banner = self.query_one("#status-banner", StatusBanner)
        video_list = self.query_one("#video-list", VideoList)

        try:
            # Discover
            discovered = discover_videos(self.folder)
            self.videos = [VideoProgress(name=v.name) for v in discovered]

            # Pre-compute step counts so the TUI knows totals upfront
            for i, video in enumerate(discovered):
                num_frames = get_frame_count(video)
                steps = 1 + num_frames + 1  # transcribe + frames + summary
                self.video_steps[self.videos[i].name] = steps

            # Create rows in the UI
            for vp in self.videos:
                video_list.add_row(vp.name, self.video_steps[vp.name])

            banner.update(f"Processing {len(self.videos)} video(s)...")

            def progress_callback(
                total: int,
                state: str,
                completed: int = 0,
                name: str = "",
                error: str = "",
                phase: str = "",
                current_step: int = 0,
                total_steps: int = 0,
                steps_map: dict | None = None,
            ) -> None:
                if state == "discovering" and steps_map is not None:
                    # Steps map already set above; no-op
                    pass
                elif state == "processing":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "processing"
                            vp.phase = "transcribing"
                            vp.current_step = 0
                            vp.total_steps = self.video_steps.get(name, 0)
                    self._refresh_video_list()
                elif state == "step":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.phase = phase
                            vp.current_step = current_step
                            vp.total_steps = total_steps
                    self._refresh_video_list()
                elif state == "done":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "done"
                            vp.current_step = vp.total_steps
                    self.call_later(banner.update, f"Completed: {completed}/{total}")
                    self._refresh_video_list()
                elif state == "error":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "error"
                            vp.detail = error
                    self._refresh_video_list()

            await process_videos(
                folder=self.folder,
                output_dir=self.output_dir,
                max_concurrency=self.max_concurrency,
                whisper_model=self.whisper_model,
                llm_provider=self.llm_provider,
                llm_base_url=self.llm_base_url,
                vision_model=self.vision_model,
                summary_model=self.summary_model,
                progress_callback=progress_callback,
            )

            banner.update(f"Done! Processed {len(self.videos)} video(s).")

        except Exception as e:
            banner.update(f"Error: {e}")
            logger.error("Pipeline failed: %s", e, exc_info=True)

    def _refresh_video_list(self) -> None:
        """Schedule a refresh of all VideoRow widgets with current progress.

        Defers the actual widget update so the Textual event loop can render
        between progress callbacks — otherwise rapid synchronous updates get
        batched and only the final state is painted.
        """
        if threading.current_thread() is threading.main_thread():
            # Called from the event-loop thread: defer to the next tick so
            # the render cycle runs before we mutate widget state again.
            self.call_later(self._refresh_video_list_now)
        else:
            # Called from a worker thread: marshal back to the main thread.
            self.call_from_thread(self._refresh_video_list_now)

    def _refresh_video_list_now(self) -> None:
        """Apply the current progress state to all VideoRow widgets."""
        video_list = self.query_one("#video-list", VideoList)

        for vp in self.videos:
            if vp.status == "done":
                status_label = "Done"
            elif vp.status == "error":
                status_label = f"Error: {vp.detail}"
            elif vp.status == "processing":
                status_label = _phase_label(vp.phase)
            else:
                status_label = "Pending"

            video_list.update_row(vp.name, vp.current_step, vp.total_steps, status_label)

    def action_details(self) -> None:
        self.show_details = not self.show_details
        log = self.query_one("#details-log", Log)
        log.display = self.show_details

    def action_command_palette(self) -> None:
        pass

    def action_change_theme(self) -> None:
        pass


class _TextualLogHandler(logging.Handler):
    """Routes Python log records to a Textual Log widget.

    Log writes must happen on the Textual main thread, so we delegate
    through `call_from_thread` when called from worker threads.
    """

    def __init__(self, app: CataloguerApp) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        # call_from_thread raises when called from the event-loop thread,
        # so write directly on the main thread and delegate from workers.
        if threading.current_thread() is threading.main_thread():
            self._write_log(msg)
        else:
            with contextlib.suppress(Exception):
                self.app.call_from_thread(self._write_log, msg)

    def _write_log(self, msg: str) -> None:
        log = self.app.query_one("#details-log", Log)
        log.write(msg + "\n")


def _phase_label(phase: str) -> str:
    """Return a human-readable status label for a processing phase."""
    labels = {
        "transcribing": "Transcribing audio",
        "describing_frames": "Describing video frames",
        "generating_summary": "Generating summary",
    }
    return labels.get(phase, phase)
