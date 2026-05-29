# Commands

- **Run:** `uv run video-cataloguer`
- **Lint & fix:** `uv run ruff check --fix`
- **Format:** `uv run ruff format`
- **Type check:** `uv run ty check`
- **Pre-commit (all checks):** `uv run pre-commit run --all-files`

# Architecture

- **Package:** `src/video_cataloguer` (src layout, hatch builds from `src/video_cataloguer`)
- **Entry point:** `video_cataloguer.cli:main` (click CLI)
- **TUI:** Built with textual (>=6.0.0). All UI lives in the package; run `uv run video-cataloguer` to start the app.
- **Video processing:** `ffmpeg-python` for frame extraction, `openai-whisper` for transcription, `Pillow` for image handling.
- **Data model:** `pydantic` models for video metadata, transcripts, and frame descriptions.

# Gotchas

- Requires Python **3.14+**. The `.python-version` and `pyproject.toml` enforce this.
- `ty` is the type checker (not mypy or pyright). Run it with `uv run ty check`.
- `ruff` handles both linting and formatting (no separate black/isort).
- No test suite exists yet.
