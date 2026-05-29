from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Label, Log, Rule

from video_cataloguer.discovery import discover_videos
from video_cataloguer.pipeline import process_videos

logger = logging.getLogger(__name__)


@dataclass
class VideoProgress:
    """Tracks progress for a single video."""

    name: str
    status: str = "pending"  # pending | processing | done | error
    detail: str = ""


class VideoList(Log):
    """Log widget that shows per-video progress."""

    DEFAULT_CSS = """
    VideoList {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }
"""


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


class CataloguerApp(App):
    """TUI for the video cataloguer."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-gutter: 1;
    }

    .main-container {
        height: 1fr;
        layout: vertical;
    }
"""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "details", "Toggle Details"),
    ]

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

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusBanner("Discovering videos...", id="status-banner")
        with Container(id="main-container"):
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
            video_list.write(f"[bold]{len(self.videos)} video(s) found[/]\n")
            banner.update(f"Processing {len(self.videos)} video(s)...")

            def progress_callback(
                total: int, state: str, completed: int = 0, name: str = "", error: str = ""
            ) -> None:
                if state == "processing":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "processing"
                            vp.detail = "Extracting metadata & transcribing..."
                    self._refresh_video_list()
                elif state == "done":
                    for vp in self.videos:
                        if vp.name == name:
                            vp.status = "done"
                    banner.update(f"Completed: {completed}/{total}")
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
            video_list.write("\n[bold green]All done![/]")

        except Exception as e:
            banner.update(f"Error: {e}")
            video_list.write(f"\n[red]Processing failed: {e}[/]")
            logger.error("Pipeline failed: %s", e, exc_info=True)

    def _refresh_video_list(self) -> None:
        """Update the video list widget with current progress."""
        video_list = self.query_one("#video-list", VideoList)
        video_list.clear()

        for vp in self.videos:
            if vp.status == "done":
                status_str = f"[green]✓[/] {vp.name}"
            elif vp.status == "processing":
                status_str = f"[yellow]⟳[/] {vp.name} — {vp.detail}"
            elif vp.status == "error":
                status_str = f"[red]✗[/] {vp.name} — {vp.detail}"
            else:
                status_str = f"[dim]○[/] {vp.name}"
            video_list.write(status_str)

    def action_details(self) -> None:
        self.show_details = not self.show_details
        log = self.query_one("#details-log", Log)
        log.display = self.show_details


class _TextualLogHandler(logging.Handler):
    """Routes Python log records to a Textual Log widget."""

    def __init__(self, app: CataloguerApp) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            log = self.app.query_one("#details-log", Log)
            log.write(msg)
        except Exception:
            pass  # Widget may not exist yet
