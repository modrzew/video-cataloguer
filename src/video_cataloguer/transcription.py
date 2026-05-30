from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import whisper  # type: ignore[import-untyped]
from tqdm import tqdm as tqdm_class  # type: ignore[import-untyped]

from video_cataloguer.models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, whisper.Whisper] = {}


def transcribe(video_path: Path, model_name: str = "large") -> Transcript:
    """Extract audio from *video_path* and transcribe it with Whisper."""
    model = _load_model(model_name)

    # Whisper can transcribe video files directly, but extracting audio first
    # is more reliable and avoids issues with certain codecs.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as audio_file:
        _extract_audio(video_path, Path(audio_file.name))
        # Disable tqdm to avoid multiprocessing lock issues when fork() is
        # called from a process with extra open file descriptors (e.g. TUI).
        # Use hasattr/guard — some environments (e.g. textual) may resolve
        # `tqdm` to a different class lacking the `disable` attribute.
        if hasattr(tqdm_class, "disable"):
            disabled = tqdm_class.disable
            tqdm_class.disable = True
        try:
            result = model.transcribe(audio_file.name, verbose=False)
        finally:
            if hasattr(tqdm_class, "disable"):
                tqdm_class.disable = disabled

    segments = [
        TranscriptSegment(start=s["start"], end=s["end"], text=s["text"].strip())
        for s in result.get("segments", [])
    ]

    logger.info("Transcribed %d segments from %s", len(segments), video_path.name)
    return Transcript(segments=segments)


def _load_model(model_name: str) -> whisper.Whisper:
    """Load (or cache) a Whisper model."""
    if model_name not in _MODEL_CACHE:
        logger.info("Loading Whisper model: %s", model_name)
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def _extract_audio(video_path: Path, output_path: Path) -> None:
    """Extract audio track from video to a WAV file."""
    import ffmpeg  # type: ignore[import-untyped]

    (
        ffmpeg.input(str(video_path))
        .output(str(output_path), acodec="pcm_s16le", ac=1, ar="16k", format="wav")
        .overwrite_output()
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )
