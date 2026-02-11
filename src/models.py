"""Pydantic models for the multimodal pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Transcription Models ─────────────────────────────────────────────────────


class TranscriptionSegment(BaseModel):
    """A single timestamped segment from the transcription."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class TranscriptionResult(BaseModel):
    """Complete transcription output."""

    text: str = Field(..., description="Full transcribed text")
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = Field(default="en")
    duration_seconds: float = Field(..., ge=0.0)
    backend: str = Field(default="whisper", description="whisper | gcp")
    model_name: str = Field(default="base")


# ── Analysis Models ──────────────────────────────────────────────────────────


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Sentiment(BaseModel):
    label: SentimentLabel
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class NamedEntity(BaseModel):
    text: str
    type: str = Field(..., description="E.g. PERSON, ORG, DATE, LOCATION, DELIVERABLE")
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class ActionItem(BaseModel):
    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[str] = Field(None, description="high | medium | low")


class AnalysisResult(BaseModel):
    """Structured LLM analysis of transcribed text."""

    sentiment: Sentiment
    entities: list[NamedEntity] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    summary: str = Field(..., description="Concise summary of the content")
    action_items: list[ActionItem] = Field(default_factory=list)
    key_quotes: list[str] = Field(default_factory=list)
    word_count: int = Field(default=0)


# ── Pipeline Output ──────────────────────────────────────────────────────────


class PipelineOutput(BaseModel):
    """Combined output of the full multimodal pipeline."""

    source_file: str
    transcription: TranscriptionResult
    analysis: AnalysisResult
    pipeline_version: str = Field(default="1.0.0")
    processing_time_seconds: Optional[float] = None

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"📁 Source: {self.source_file}",
            f"⏱️  Duration: {self.transcription.duration_seconds:.1f}s",
            f"🗣️  Language: {self.transcription.language}",
            f"💬 Sentiment: {self.analysis.sentiment.label.value} ({self.analysis.sentiment.score:.0%})",
            f"📝 Summary: {self.analysis.summary}",
            f"🏷️  Topics: {', '.join(self.analysis.topics)}",
            f"👤 Entities: {len(self.analysis.entities)}",
            f"✅ Action Items: {len(self.analysis.action_items)}",
        ]
        if self.processing_time_seconds:
            lines.append(f"⚡ Processed in {self.processing_time_seconds:.2f}s")
        return "\n".join(lines)
