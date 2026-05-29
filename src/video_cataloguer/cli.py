from __future__ import annotations

import click

from video_cataloguer.tui import CataloguerApp


@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=True, file_okay=False),
    default=None,
    help="Output directory for Markdown files. Defaults to alongside source videos.",
)
@click.option(
    "-c",
    "--concurrency",
    type=int,
    default=2,
    help="Max number of videos processed in parallel (default: 2).",
)
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default="tiny",
    help="Whisper model size for transcription (default: tiny).",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["ollama", "lmstudio"]),
    default="ollama",
    help="LLM provider: ollama or lmstudio (default: ollama).",
)
@click.option(
    "--llm-base-url",
    type=str,
    default=None,
    help="Custom LLM base URL (overrides provider default).",
)
@click.option(
    "--vision-model",
    type=str,
    default=None,
    help="Vision model for frame descriptions. Defaults to provider default.",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Text model for summary generation. Defaults to provider default.",
)
@click.option(
    "--headless",
    is_flag=True,
    help="Run without the TUI — just process and exit.",
)
def main(
    folder: str,
    output: str | None,
    concurrency: int,
    whisper_model: str,
    llm_provider: str,
    llm_base_url: str | None,
    vision_model: str | None,
    summary_model: str | None,
    headless: bool,
) -> None:
    """Catalogue videos in FOLDER.

    Extracts metadata, transcribes audio, describes frames, and generates
    Markdown summaries using a local LLM (Ollama or LM Studio).

    Examples:

        video-cataloguer ./my-videos

        video-cataloguer ./my-videos --llm-provider lmstudio

        video-cataloguer ./my-videos -o ./catalogs --whisper-model base

        video-cataloguer ./my-videos --headless
    """
    if headless:
        import asyncio
        import logging

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        async def _run() -> None:
            from video_cataloguer.pipeline import process_videos

            await process_videos(
                folder=folder,
                output_dir=output,
                max_concurrency=concurrency,
                whisper_model=whisper_model,
                llm_provider=llm_provider,
                llm_base_url=llm_base_url,
                vision_model=vision_model,
                summary_model=summary_model,
            )

        asyncio.run(_run())
    else:
        app = CataloguerApp(
            folder=folder,
            output_dir=output,
            max_concurrency=concurrency,
            whisper_model=whisper_model,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            vision_model=vision_model,
            summary_model=summary_model,
        )
        app.run()
