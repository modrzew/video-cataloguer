from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm as tqdm_class  # type: ignore[import-untyped]

from video_cataloguer.models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


def transcribe(video_path: Path, model_name: str = "large") -> Transcript:
    """Extract audio from *video_path* and transcribe it with Whisper.

    Runs Whisper in a subprocess so that PyTorch's internal fork() never
    inherits the TUI's extra file descriptors (the "bad value in fds_to_keep"
    crash).
    """
    # Extract audio first (ffmpeg, no fork issue).  delete=False so the
    # subprocess can access the file after the context manager closes it.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name
    try:
        _extract_audio(video_path, Path(audio_path))
        result = _run_transcribe_subprocess(audio_path, model_name)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    segments = [
        TranscriptSegment(start=s["start"], end=s["end"], text=s["text"].strip())
        for s in result.get("segments", [])
    ]

    logger.info("Transcribed %d segments from %s", len(segments), video_path.name)
    return Transcript(segments=segments)


def _run_transcribe_subprocess(audio_path: str, model_name: str) -> dict:
    """Run Whisper transcription in a clean subprocess with no inherited FDs.

    Uses a temp file for the result so that PyTorch/Whisper stdout noise
    cannot corrupt the JSON output.
    """
    # Create a temp file for the subprocess to write its JSON result into.
    # delete=False so it persists after we close it.
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="whisper_result_", delete=False
    ) as result_file:
        result_path = result_file.name
    # result_file is closed here; subprocess will open it for writing.

    script = (
        f"from video_cataloguer.transcription import _transcribe_worker\n"
        f"import json\n"
        f"result = _transcribe_worker({audio_path!r}, {model_name!r})\n"
        f"with open({result_path!r}, 'w') as f:\n"
        f"    json.dump(result, f)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Whisper subprocess failed (exit {proc.returncode}): {proc.stderr}")
        with open(result_path) as f:
            return json.loads(f.read())
    finally:
        Path(result_path).unlink(missing_ok=True)


def _transcribe_worker(audio_path: str, model_name: str) -> dict:
    """Worker run inside the subprocess — loads model and transcribes."""
    import whisper  # type: ignore[import-untyped]

    model = whisper.load_model(model_name)
    # Disable tqdm to avoid any progress output leaking to the TUI.
    if hasattr(tqdm_class, "disable"):
        tqdm_class.disable = True
    try:
        return model.transcribe(audio_path, verbose=False)
    finally:
        if hasattr(tqdm_class, "disable"):
            tqdm_class.disable = False


def _extract_audio(video_path: Path, output_path: Path) -> None:
    """Extract audio track from video to a WAV file."""
    import ffmpeg  # type: ignore[import-untyped]

    (
        ffmpeg.input(str(video_path))
        .output(str(output_path), acodec="pcm_s16le", ac=1, ar="16k", format="wav")
        .overwrite_output()
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )
