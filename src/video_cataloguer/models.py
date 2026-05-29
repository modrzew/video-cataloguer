from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file via ffprobe."""

    file_path: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    file_size_bytes: int
    recording_date: str | None = None
    gps_latitude: float | None = None
    gps_longitude: float | None = None


@dataclass
class TranscriptSegment:
    """A single segment of a Whisper transcript."""

    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """Full transcript of a video's audio track."""

    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_text(self) -> str:
        """Return the full transcript as a single string."""
        return " ".join(seg.text for seg in self.segments).strip()

    def to_timestamped_text(self) -> str:
        """Return the transcript with timestamps."""
        lines: list[str] = []
        for seg in self.segments:
            start = _format_time(seg.start)
            end = _format_time(seg.end)
            lines.append(f"[{start} → {end}] {seg.text}")
        return "\n".join(lines)


@dataclass
class FrameDescription:
    """Description of a single extracted frame."""

    timestamp: float
    description: str


@dataclass
class VideoCatalogEntry:
    """Complete catalog entry for one video."""

    video_path: str
    metadata: VideoMetadata
    transcript: Transcript
    frame_descriptions: list[FrameDescription] = field(default_factory=list)
    summary: str = ""


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ms."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
