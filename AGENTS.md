# Commands

- **Run:** `uv run video-cataloguer`
- **Lint & fix:** `uv run ruff check --fix`
- **Format:** `uv run ruff format`
- **Type check:** `uv run ty check`
- **Pre-commit (all checks):** `uv run pre-commit run --all-files`

# After every code change

Run `uv run ty check` and `uv run ruff check`. These will also be run as pre-commit hooks.

# Architecture

- **Package:** `src/video_cataloguer` (src layout, hatch builds from `src/video_cataloguer`)
- **Entry point:** `video_cataloguer.cli:main` (click CLI)
- **TUI:** Built with textual (>=6.0.0). All UI lives in the package; run `uv run video-cataloguer` to start the app.
- **Video processing:** `ffmpeg-python` for frame extraction, `openai-whisper` for transcription, `Pillow` for image handling.
- **Data model:** `pydantic` models for video metadata, transcripts, and frame descriptions.

# Gotchas

- Whisper must run in its own subprocess: in-process, PyTorch's fork() inherits the TUI's file descriptors and crashes ("bad value in fds_to_keep"). Don't inline it.
- Treat model/LLM output as untrusted text: request a parseable structure and handle mismatches explicitly.
