"""Unit tests for the multimodal pipeline components."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    ActionItem,
    AnalysisResult,
    NamedEntity,
    PipelineOutput,
    Sentiment,
    SentimentLabel,
    TranscriptionResult,
    TranscriptionSegment,
)


# ── Model Tests ──────────────────────────────────────────────────────────────


class TestModels:
    def test_transcription_segment(self):
        seg = TranscriptionSegment(start=0.0, end=3.5, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 3.5
        assert seg.text == "Hello world"

    def test_transcription_result(self):
        result = TranscriptionResult(
            text="Hello world",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="Hello world")],
            language="en",
            duration_seconds=1.0,
        )
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.backend == "whisper"

    def test_analysis_result(self):
        result = AnalysisResult(
            sentiment=Sentiment(label=SentimentLabel.POSITIVE, score=0.85),
            entities=[NamedEntity(text="Disney", type="ORG")],
            topics=["entertainment", "animation"],
            summary="A discussion about animation.",
            action_items=[ActionItem(task="Review storyboards", priority="high")],
            key_quotes=["This is great work"],
            word_count=50,
        )
        assert result.sentiment.label == SentimentLabel.POSITIVE
        assert result.sentiment.score == 0.85
        assert len(result.entities) == 1
        assert result.entities[0].type == "ORG"
        assert len(result.action_items) == 1

    def test_pipeline_output_summary(self):
        output = PipelineOutput(
            source_file="test.wav",
            transcription=TranscriptionResult(
                text="Test", duration_seconds=5.0, language="en"
            ),
            analysis=AnalysisResult(
                sentiment=Sentiment(label=SentimentLabel.NEUTRAL, score=0.5),
                summary="A test.",
                topics=["testing"],
                word_count=1,
            ),
            processing_time_seconds=1.23,
        )
        summary = output.to_summary()
        assert "test.wav" in summary
        assert "5.0s" in summary
        assert "neutral" in summary

    def test_pipeline_output_serialization(self):
        output = PipelineOutput(
            source_file="test.wav",
            transcription=TranscriptionResult(
                text="Test", duration_seconds=5.0, language="en"
            ),
            analysis=AnalysisResult(
                sentiment=Sentiment(label=SentimentLabel.NEUTRAL, score=0.5),
                summary="A test.",
                word_count=1,
            ),
        )
        data = output.model_dump()
        assert data["source_file"] == "test.wav"
        assert data["transcription"]["text"] == "Test"

        # Round-trip
        restored = PipelineOutput.model_validate(data)
        assert restored.source_file == output.source_file


# ── Transcriber Tests ────────────────────────────────────────────────────────


class TestTranscriber:
    def test_validate_unsupported_format(self):
        from src.transcriber import Transcriber

        t = Transcriber()
        with pytest.raises(ValueError, match="Unsupported format"):
            from pathlib import Path
            # Create a temporary file with bad extension
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
                f.write(b"fake")
                bad_path = Path(f.name)
            try:
                t.transcribe(bad_path)
            finally:
                bad_path.unlink()

    def test_validate_missing_file(self):
        from src.transcriber import Transcriber

        t = Transcriber()
        with pytest.raises(FileNotFoundError):
            t.transcribe("/nonexistent/audio.wav")


# ── Analyzer Tests ───────────────────────────────────────────────────────────


class TestAnalyzer:
    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_analyze_parses_response(self, mock_model_cls, mock_configure):
        from src.analyzer import Analyzer

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sentiment": {"label": "positive", "score": 0.9},
            "entities": [{"text": "Disney", "type": "ORG"}],
            "topics": ["animation"],
            "summary": "Great meeting.",
            "action_items": [{"task": "Ship it", "priority": "high"}],
            "key_quotes": ["Ship it!"],
        })

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model

        analyzer = Analyzer(api_key="test-key")
        result = analyzer.analyze("Some transcribed text here.")

        assert result.sentiment.label == SentimentLabel.POSITIVE
        assert result.sentiment.score == 0.9
        assert result.entities[0].text == "Disney"
        assert len(result.action_items) == 1

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_analyze_empty_text_raises(self, mock_model_cls, mock_configure):
        from src.analyzer import Analyzer

        analyzer = Analyzer(api_key="test-key")
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze("")


# ── Pipeline Config Tests ────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_loads_default_config(self):
        from src.pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.whisper_model == "base"
        assert config.gemini_model == "gemini-1.5-pro"
        assert config.max_retries == 3
