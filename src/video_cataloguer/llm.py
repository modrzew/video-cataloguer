from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from video_cataloguer.models import FrameDescription, Transcript, _format_time

logger = logging.getLogger(__name__)

# Provider defaults
PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "vision_model": "llama3.2-vision",
        "summary_model": "llama3.3",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234",
        "vision_model": "qwen3.6-35b-a3b-mtp",
        "summary_model": "qwen3.6-35b-a3b-mtp",
    },
}

# Current provider config (set by CLI)
_provider: str = "ollama"
_base_url: str = PROVIDER_DEFAULTS["ollama"]["base_url"]


def configure(provider: str = "ollama", base_url: str | None = None) -> None:
    """Set the LLM provider. Must be called before any LLM requests."""
    global _provider, _base_url
    _provider = provider
    if base_url:
        _base_url = base_url
    elif provider in PROVIDER_DEFAULTS:
        _base_url = PROVIDER_DEFAULTS[provider]["base_url"]
    logger.info("LLM provider: %s (%s)", _provider, _base_url)


def describe_frames(
    frames: list[tuple[float, Any]],
    model: str | None = None,
) -> list[FrameDescription]:
    """Send extracted frames to a vision model and get descriptions."""
    if model is None:
        model = PROVIDER_DEFAULTS[_provider]["vision_model"]

    descriptions: list[FrameDescription] = []
    batch_size = 3

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        try:
            result = _vision_request(model, batch)
            for (ts, _img), desc in zip(batch, result, strict=True):
                descriptions.append(FrameDescription(timestamp=ts, description=desc))
        except Exception as e:
            logger.error("Frame description failed at %.1fs: %s", batch[0][0], e)
            for ts, _img in batch:
                descriptions.append(
                    FrameDescription(timestamp=ts, description="Failed to describe frame.")
                )

    logger.info("Described %d frames", len(descriptions))
    return descriptions


def generate_summary(
    video_path: str,
    metadata: dict,
    transcript: Transcript,
    frame_descriptions: list[FrameDescription],
    model: str | None = None,
) -> str:
    """Generate a Markdown summary of the video using a local LLM."""
    if model is None:
        model = PROVIDER_DEFAULTS[_provider]["summary_model"]

    prompt = _build_summary_prompt(video_path, metadata, transcript, frame_descriptions)

    try:
        return _text_request(model, prompt)
    except Exception as e:
        logger.error("Summary generation failed for %s: %s", video_path, e)
        return _fallback_summary(video_path, metadata, transcript, frame_descriptions)


# --- Shared frame encoding ---


def _encode_frames(frames: list[tuple[float, Any]]) -> tuple[list[str], list[float]]:
    """Encode frames to base64 JPEG and return (b64_strings, timestamps)."""
    images_b64: list[str] = []
    timestamps: list[float] = []

    for ts, img in frames:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        timestamps.append(ts)

    return images_b64, timestamps


def _build_vision_prompt(frames: list[tuple[float, Any]]) -> str:
    """Build the vision prompt shared by all providers."""
    timestamps = [ts for ts, _ in frames]
    return (
        "You are describing video frames. For each image provided, give a concise "
        "one-sentence description of what is visible. Return exactly one description "
        "per image, in order. Output a JSON array of strings, one per frame.\n\n"
        f"There are {len(frames)} frames from timestamps: "
        f"{', '.join(f'{t:.1f}s' for t in timestamps)}."
    )


# --- Ollama provider ---


def _ollama_vision(model: str, frames: list[tuple[float, Any]]) -> list[str]:
    """Send frames to Ollama /api/chat for vision description."""
    images_b64, _timestamps = _encode_frames(frames)
    prompt = _build_vision_prompt(frames)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{_base_url}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")

    return _split_descriptions(content, len(frames))


def _ollama_text(model: str, prompt: str) -> str:
    """Send a text-only prompt to Ollama /api/chat."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{_base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")


# --- LM Studio (OpenAI-compatible) provider ---


def _lmstudio_vision(model: str, frames: list[tuple[float, Any]]) -> list[str]:
    """Send frames to LM Studio's OpenAI-compatible /v1/chat/completions."""
    images_b64, _timestamps = _encode_frames(frames)
    prompt = _build_vision_prompt(frames)

    # OpenAI format: images embedded in content as image_url objects
    content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_b64 in images_b64:
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "stream": False,
        "temperature": 0.1,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{_base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    return _split_descriptions(content, len(frames))


def _lmstudio_text(model: str, prompt: str) -> str:
    """Send a text-only prompt to LM Studio's /v1/chat/completions."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }

    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{_base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# --- Dispatch ---


def _vision_request(model: str, frames: list[tuple[float, Any]]) -> list[str]:
    """Dispatch vision request to the configured provider."""
    if _provider == "lmstudio":
        return _lmstudio_vision(model, frames)
    return _ollama_vision(model, frames)


def _text_request(model: str, prompt: str) -> str:
    """Dispatch text request to the configured provider."""
    if _provider == "lmstudio":
        return _lmstudio_text(model, prompt)
    return _ollama_text(model, prompt)


# --- Shared helpers ---


def _split_descriptions(content: str, count: int) -> list[str]:
    """Split LLM response into per-frame descriptions.

    Tries JSON array first, then falls back to numbered lines (``1.`` / ``1)``),
    then plain non-empty lines. Pads with a placeholder if the model returned
    fewer descriptions than requested.
    """
    # Strategy 1: JSON array
    stripped = content.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                descs = [str(d).strip() for d in parsed if str(d).strip()]
                if descs:
                    while len(descs) < count:
                        descs.append("A frame from the video.")
                    return descs[:count]
        except json.JSONDecodeError, ValueError:
            pass

    # Strategy 2: Numbered lines ("1. ...", "1) ...", "1:..." etc.)
    numbered: list[str] = []
    for line in content.splitlines():
        m = re.match(r"^\d+[\.\)\:]\s*(.+)", line.strip())
        if m:
            numbered.append(m.group(1).strip())
    if numbered:
        while len(numbered) < count:
            numbered.append("A frame from the video.")
        return numbered[:count]

    # Strategy 3: Plain non-empty lines (last resort)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    while len(lines) < count:
        lines.append("A frame from the video.")
    return lines[:count]


def _build_summary_prompt(
    video_path: str,
    metadata: dict,
    transcript: Transcript,
    frame_descriptions: list[FrameDescription],
) -> str:
    """Build the prompt for the summary generation LLM call."""
    frame_text = "\n".join(
        f"- [{_format_time(f.timestamp)}] {f.description}" for f in frame_descriptions
    )

    return f"""You are a video cataloguer. Create a Markdown document describing
the following video.

## Video Information
- File: {Path(video_path).name}
- Duration: {_fmt_duration(metadata.get("duration_seconds", 0))}
- Resolution: {metadata.get("width", "?")}x{metadata.get("height", "?")}
- FPS: {metadata.get("fps", "?")}
- Video codec: {metadata.get("video_codec", "?")}
- Audio codec: {metadata.get("audio_codec", "?")}
- File size: {_fmt_size(metadata.get("file_size_bytes", 0))}
{
        f"- Recording date: {metadata.get('recording_date', 'N/A')}"
        if metadata.get("recording_date")
        else ""
    }
{
        f"- Location: {metadata.get('gps_latitude', '?')}, {metadata.get('gps_longitude', '?')}"
        if metadata.get("gps_latitude")
        else ""
    }

## Key Frame Descriptions
{frame_text}

## Full Transcript
{transcript.to_timestamped_text()}

---

Please produce a Markdown document with the following structure:

1. A title line with the video filename
2. A **Metadata** section with the extracted metadata
3. A **Summary** section describing what the video is about, who/what is shown,
   and the scenery
4. A **Transcript** section with the full timestamped transcript

Be concise but informative in the Summary section. Use the frame descriptions
to ground your description of what is visually present."""


def _fallback_summary(
    video_path: str,
    metadata: dict,
    transcript: Transcript,
    frame_descriptions: list[FrameDescription],
) -> str:
    """Return a basic Markdown summary when the LLM is unavailable."""
    lines = [
        f"# {Path(video_path).name}",
        "",
        "## Metadata",
        f"- Duration: {_fmt_duration(metadata.get('duration_seconds', 0))}",
        f"- Resolution: {metadata.get('width', '?')}x{metadata.get('height', '?')}",
        f"- File size: {_fmt_size(metadata.get('file_size_bytes', 0))}",
        "",
        "## Summary",
        "LLM summary unavailable. The LLM service was not reachable.",
        "",
        "## Transcript",
        transcript.to_timestamped_text(),
    ]
    return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
    return f"{int(minutes):02d}:{int(secs):02d}"


def _fmt_size(bytes_: int) -> str:
    """Format bytes as human-readable size."""
    size = float(bytes_)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} EB"
