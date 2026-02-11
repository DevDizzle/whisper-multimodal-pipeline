"""Multimodal pipeline orchestration: Audio → Transcription → Analysis → Output."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from .analyzer import Analyzer
from .models import PipelineOutput
from .transcriber import Transcriber

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"


class PipelineConfig:
    """Load and access pipeline configuration."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

    @property
    def whisper_model(self) -> str:
        return self._cfg.get("transcription", {}).get("whisper_model", "base")

    @property
    def transcription_backend(self) -> str:
        return self._cfg.get("transcription", {}).get("backend", "whisper")

    @property
    def language(self) -> Optional[str]:
        return self._cfg.get("transcription", {}).get("language")

    @property
    def gemini_model(self) -> str:
        return self._cfg.get("analysis", {}).get("gemini_model", "gemini-1.5-pro")

    @property
    def temperature(self) -> float:
        return self._cfg.get("analysis", {}).get("temperature", 0.2)

    @property
    def max_retries(self) -> int:
        return self._cfg.get("pipeline", {}).get("max_retries", 3)

    @property
    def retry_delay(self) -> float:
        return self._cfg.get("pipeline", {}).get("retry_delay_seconds", 2.0)

    @property
    def output_dir(self) -> Path:
        return Path(self._cfg.get("pipeline", {}).get("output_dir", "outputs"))


class Pipeline:
    """Orchestrates the full audio → intelligence pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        transcriber: Optional[Transcriber] = None,
        analyzer: Optional[Analyzer] = None,
    ) -> None:
        self.config = config or PipelineConfig()

        self.transcriber = transcriber or Transcriber(
            backend=self.config.transcription_backend,
            model_name=self.config.whisper_model,
            language=self.config.language,
        )

        self.analyzer = analyzer or Analyzer(
            model_name=self.config.gemini_model,
            temperature=self.config.temperature,
        )

    def run(self, audio_path: str | Path) -> PipelineOutput:
        """Run the full pipeline synchronously with retry logic."""
        audio_path = Path(audio_path)
        start = time.time()

        # Step 1: Transcribe
        logger.info("Step 1/2 — Transcribing: %s", audio_path.name)
        transcription = self._with_retry(
            lambda: self.transcriber.transcribe(audio_path),
            step_name="transcription",
        )

        # Step 2: Analyze
        logger.info("Step 2/2 — Analyzing transcription (%d words)", len(transcription.text.split()))
        analysis = self._with_retry(
            lambda: self.analyzer.analyze(transcription.text),
            step_name="analysis",
        )

        elapsed = time.time() - start

        output = PipelineOutput(
            source_file=str(audio_path),
            transcription=transcription,
            analysis=analysis,
            processing_time_seconds=round(elapsed, 2),
        )

        logger.info("Pipeline complete in %.2fs", elapsed)
        return output

    async def run_async(self, audio_path: str | Path) -> PipelineOutput:
        """Run the pipeline with async analysis step."""
        audio_path = Path(audio_path)
        start = time.time()

        # Transcription (sync — Whisper doesn't have async API)
        transcription = await asyncio.to_thread(
            self.transcriber.transcribe, audio_path
        )

        # Analysis (async)
        analysis = await self._with_retry_async(
            lambda: self.analyzer.analyze_async(transcription.text),
            step_name="analysis",
        )

        elapsed = time.time() - start

        return PipelineOutput(
            source_file=str(audio_path),
            transcription=transcription,
            analysis=analysis,
            processing_time_seconds=round(elapsed, 2),
        )

    async def run_batch_async(self, audio_paths: list[str | Path]) -> list[PipelineOutput]:
        """Process multiple audio files concurrently."""
        tasks = [self.run_async(p) for p in audio_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def save_output(self, output: PipelineOutput, output_dir: Optional[Path] = None) -> Path:
        """Save pipeline output as JSON."""
        out_dir = output_dir or self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(output.source_file).stem
        out_path = out_dir / f"{stem}_output.json"

        with open(out_path, "w") as f:
            json.dump(output.model_dump(), f, indent=2, default=str)

        logger.info("Saved output to %s", out_path)
        return out_path

    # ── Retry Logic ───────────────────────────────────────────────────────

    def _with_retry(self, fn, step_name: str = "step"):
        """Execute with exponential backoff retry."""
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    step_name, attempt, self.config.max_retries, e, delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"{step_name} failed after {self.config.max_retries} retries: {last_error}")

    async def _with_retry_async(self, fn, step_name: str = "step"):
        """Async retry with exponential backoff."""
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await fn()
            except Exception as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    step_name, attempt, self.config.max_retries, e, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"{step_name} failed after {self.config.max_retries} retries: {last_error}")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """CLI entry point for the pipeline."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Whisper Multimodal Pipeline")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    config = PipelineConfig(args.config) if args.config else PipelineConfig()
    pipeline = Pipeline(config=config)

    result = pipeline.run(args.audio)

    out_dir = Path(args.output_dir) if args.output_dir else None
    saved = pipeline.save_output(result, output_dir=out_dir)

    print("\n" + result.to_summary())
    print(f"\n💾 Full output: {saved}")


if __name__ == "__main__":
    main()
