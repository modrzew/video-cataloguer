from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Label, Log, Rule

from video_cataloguer.discovery import discover_videos
from video_cataloguer.pipeline import process_videos

logger = logging.getLogger(__name__)


# Icon codepoints for each processing stage
STAGE_ICONS: dict[str, str] = {
    "metadata": "\ueb6d",
    "audio": "\uf001",
    "frames": "\uf03e",
    "summary": "\ueb26",
}

SPINNER_CHARS = "⠋⠙⠸⢰⣰⣤⣆⡇"


@dataclass(slots=True)
class StageStatus:
    """Tracks the state of a single processing stage."""

    state: str = "pending"  # pending | running | done


@dataclass
class VideoProgress:
    """Tracks progress for a single video."""

    name: str
    status: str = "pending"  # pending | processing | done | error
    detail: str = ""
    stages: dict[str, StageStatus] = field(
        default_factory=lambda: {
            "metadata": StageStatus("pending"),
            "audio": StageStatus("pending"),
            "frames": StageStatus("pending"),
            "summary": StageStatus("pending"),
        }
    )


class StatusBanner(Label):
    """Top banner showing overall status."""

    pass  # CSS defined in app-level CSS block


class StageIcon(Label):
    """A single icon that shows pending/running/done state for a processing stage."""

    pass  # CSS defined in app-level CSS block

    def __init__(self, stage_name: str) -> None:
        super().__init__()
        self.stage_name = stage_name
        self._state: str = "pending"
        self._spinner_index: int = 0
        self._timer: Any = None
        self.add_class("pending")
        self.update(STAGE_ICONS[stage_name])

    def _update_display(self) -> None:
        icon = STAGE_ICONS.get(self.stage_name, "?")
        if self._state == "running":
            self.update(SPINNER_CHARS[self._spinner_index])
        else:
            self.update(icon)

    def set_state(self, state: str) -> None:
        """Change the stage state (pending | running | done)."""
        if state == self._state:
            return
        self._state = state
        self.remove_class("pending", "running", "done")
        self.add_class(state)
        if state == "running":
            self._start_spinner()
        else:
            self._stop_spinner()
            self._spinner_index = 0
        self._update_display()

    def _start_spinner(self) -> None:
        self._stop_spinner()
        self._timer = self.set_interval(0.1, self._spin)

    def _stop_spinner(self) -> None:
        if self._timer is not None:
            with contextlib.suppress(AttributeError):
                self._timer.remove()
            self._timer = None

    def _spin(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_CHARS)
        self._update_display()


class VideoRow(Container):
    """Single row showing a video's filename and per-stage icons."""

    pass  # CSS defined in app-level CSS block

    def __init__(self, name: str) -> None:
        super().__init__()
        self._video_name = name
        self._name_label: Label
        self._icons: dict[str, StageIcon] = {}

    def compose(self) -> ComposeResult:
        self._name_label = Label(f"  {self._video_name}")
        self._name_label.styles.width = 40
        yield self._name_label

        for stage in ("metadata", "audio", "frames", "summary"):
            icon = StageIcon(stage)
            self._icons[stage] = icon
            yield icon

    def set_stage_state(self, stage: str, state: str) -> None:
        icon = self._icons.get(stage)
        if icon:
            icon.set_state(state)


class VideoList(Container):
    """Container holding one VideoRow per video. Scrollbar appears only when needed."""

    pass  # CSS defined in app-level CSS block

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.rows: dict[str, VideoRow] = {}

    def add_row(self, name: str) -> None:
        row = VideoRow(name)
        self.rows[name] = row
        self.mount(row)

    def set_stage_state(self, name: str, stage: str, state: str) -> None:
        row = self.rows.get(name)
        if row:
            row.set_stage_state(stage, state)


class CataloguerApp(App):
    """TUI for the video cataloguer."""

    CSS = """
    Screen {
        layout: vertical;
    }

    StatusBanner {
        width: 100%;
        content-align: center middle;
        background: $boost;
        color: $text;
        padding: 0 1;
    }

    StageIcon {
        width: 4;
        padding: 0 1;
        content-align: center middle;
        color: gray;
    }

    StageIcon.running {
        color: green;
    }

    StageIcon.done {
        color: green;
    }

    VideoRow {
        layout: horizontal;
        height: auto;
        width: 100%;
    }

    VideoList {
        height: 1fr;
        border: solid $accent;
        padding: 0;
        overflow-y: auto;
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

    def compose(self) -> ComposeResult:
        yield StatusBanner("Discovering videos...", id="status-banner")
        yield VideoList(id="video-list")
        yield Rule()
        yield Log(id="details-log")
        yield Footer(show_command_palette=False)

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

            # Create rows in the UI
            for vp in self.videos:
                video_list.add_row(vp.name)

            banner.update(f"Processing {len(self.videos)} video(s)...")

            def progress_callback(
                *args,
                **kwargs,
            ) -> None:
                state = kwargs.get("state", args[0] if args else "")
                completed = kwargs.get("completed", 0)
                name = kwargs.get("name", "")
                error = kwargs.get("error", "")
                stage = kwargs.get("stage", "")
                if state == "discovering":
                    pass  # no-op, list already populated
                elif state == "stage_start":
                    for vp in self.videos:
                        if vp.name == name and stage in vp.stages:
                            vp.stages[stage].state = "running"
                            vp.status = "processing"
                    video_list.set_stage_state(name, stage, "running")
                elif state == "stage_done":
                    for vp in self.videos:
                        if vp.name == name and stage in vp.stages:
                            vp.stages[stage].state = "done"
                    video_list.set_stage_state(name, stage, "done")
                elif state == "video_done":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "done"
                    self.call_later(banner.update, f"Completed: {completed} video(s)")
                elif state == "video_error":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "error"
                            vp.detail = error
                    video_list.set_stage_state(name, "metadata", "error")
                    video_list.set_stage_state(name, "audio", "error")
                    video_list.set_stage_state(name, "frames", "error")
                    video_list.set_stage_state(name, "summary", "error")

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
