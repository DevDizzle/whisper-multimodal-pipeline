"""Audio transcription via OpenAI Whisper or Google Cloud Speech-to-Text."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydub import AudioSegment

from .models import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "flac", "ogg", "webm"}


class Transcriber:
    """Unified transcription interface supporting Whisper and GCP backends."""

    def __init__(
        self,
        backend: Literal["whisper", "gcp"] = "whisper",
        model_name: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.language = language
        self._device = device
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe an audio file and return structured results."""
        audio_path = Path(audio_path)
        self._validate_file(audio_path)

        logger.info("Transcribing %s with %s (%s)", audio_path.name, self.backend, self.model_name)

        if self.backend == "whisper":
            return self._transcribe_whisper(audio_path)
        elif self.backend == "gcp":
            return self._transcribe_gcp(audio_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ── Whisper Backend ───────────────────────────────────────────────────

    def _load_whisper_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            import whisper

            device = self._device or ("cuda" if self._check_cuda() else "cpu")
            logger.info("Loading Whisper model '%s' on %s", self.model_name, device)
            self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def _transcribe_whisper(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper."""
        model = self._load_whisper_model()

        options = {}
        if self.language:
            options["language"] = self.language

        result = model.transcribe(str(audio_path), **options)

        segments = [
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("avg_logprob"),
            )
            for seg in result.get("segments", [])
        ]

        duration = self._get_audio_duration(audio_path)

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", self.language or "en"),
            duration_seconds=duration,
            backend="whisper",
            model_name=self.model_name,
        )

    # ── Google Cloud Speech-to-Text Backend ───────────────────────────────

    def _transcribe_gcp(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using Google Cloud Speech-to-Text V2."""
        from google.cloud import speech_v1 as speech

        client = speech.SpeechClient()

        # Convert to WAV mono 16kHz for optimal GCP compatibility
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_channels(1).set_frame_rate(16000)
        wav_bytes = audio.export(format="wav").read()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.language or "en-US",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        audio_content = speech.RecognitionAudio(content=wav_bytes)
        response = client.recognize(config=config, audio=audio_content)

        full_text_parts: list[str] = []
        segments: list[TranscriptionSegment] = []

        for result in response.results:
            alt = result.alternatives[0]
            full_text_parts.append(alt.transcript)

            if alt.words:
                seg_start = alt.words[0].start_time.total_seconds()
                seg_end = alt.words[-1].end_time.total_seconds()
            else:
                seg_start = 0.0
                seg_end = 0.0

            segments.append(
                TranscriptionSegment(
                    start=seg_start,
                    end=seg_end,
                    text=alt.transcript.strip(),
                    confidence=alt.confidence,
                )
            )

        duration = self._get_audio_duration(audio_path)

        return TranscriptionResult(
            text=" ".join(full_text_parts).strip(),
            segments=segments,
            language=self.language or "en",
            duration_seconds=duration,
            backend="gcp",
            model_name="gcp-speech-v1",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _validate_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        suffix = path.suffix.lstrip(".").lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '.{suffix}'. Supported: {SUPPORTED_FORMATS}")

    @staticmethod
    def _get_audio_duration(path: Path) -> float:
        """Get audio duration in seconds using pydub."""
        audio = AudioSegment.from_file(str(path))
        return len(audio) / 1000.0

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
