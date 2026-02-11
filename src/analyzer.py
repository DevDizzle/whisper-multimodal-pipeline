"""LLM-powered text analysis using Google Gemini."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import google.generativeai as genai

from .models import (
    ActionItem,
    AnalysisResult,
    NamedEntity,
    Sentiment,
    SentimentLabel,
)

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_PROMPT = """\
You are an expert content analyst. Analyze the following transcribed audio and return a JSON object with these fields:

1. **sentiment**: {{"label": "positive"|"negative"|"neutral"|"mixed", "score": 0.0-1.0}}
2. **entities**: list of {{"text": "...", "type": "PERSON|ORG|DATE|LOCATION|DELIVERABLE|PRODUCT|EVENT"}}
3. **topics**: list of topic strings (3-7 topics)
4. **summary**: 1-3 sentence summary
5. **action_items**: list of {{"task": "...", "assignee": "..." or null, "deadline": "..." or null, "priority": "high"|"medium"|"low"}}
6. **key_quotes**: list of notable direct quotes (max 5)

Return ONLY valid JSON. No markdown fences, no explanation.

---
TRANSCRIPT:
{text}
---
"""


class Analyzer:
    """Analyze transcribed text using Google Gemini for structured intelligence extraction."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.prompt_template = prompt_template or DEFAULT_ANALYSIS_PROMPT
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set. Pass api_key or set the environment variable.")
        genai.configure(api_key=key)

        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
            ),
        )

    def analyze(self, text: str) -> AnalysisResult:
        """Analyze transcribed text and return structured results."""
        if not text.strip():
            raise ValueError("Cannot analyze empty text.")

        logger.info("Analyzing %d characters with %s", len(text), self.model_name)

        prompt = self.prompt_template.format(text=text)
        response = self._model.generate_content(prompt)

        raw = self._parse_json_response(response.text)
        return self._build_result(raw, text)

    async def analyze_async(self, text: str) -> AnalysisResult:
        """Async version of analyze."""
        if not text.strip():
            raise ValueError("Cannot analyze empty text.")

        logger.info("Analyzing (async) %d characters with %s", len(text), self.model_name)

        prompt = self.prompt_template.format(text=text)
        response = await self._model.generate_content_async(prompt)

        raw = self._parse_json_response(response.text)
        return self._build_result(raw, text)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Raw response: %s", text[:500])
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    @staticmethod
    def _build_result(raw: dict, original_text: str) -> AnalysisResult:
        """Build a validated AnalysisResult from raw LLM JSON."""
        sentiment_data = raw.get("sentiment", {"label": "neutral", "score": 0.5})
        sentiment = Sentiment(
            label=SentimentLabel(sentiment_data.get("label", "neutral")),
            score=float(sentiment_data.get("score", 0.5)),
        )

        entities = [
            NamedEntity(text=e.get("text", ""), type=e.get("type", "UNKNOWN"))
            for e in raw.get("entities", [])
        ]

        action_items = [
            ActionItem(
                task=a.get("task", ""),
                assignee=a.get("assignee"),
                deadline=a.get("deadline"),
                priority=a.get("priority"),
            )
            for a in raw.get("action_items", [])
        ]

        return AnalysisResult(
            sentiment=sentiment,
            entities=entities,
            topics=raw.get("topics", []),
            summary=raw.get("summary", ""),
            action_items=action_items,
            key_quotes=raw.get("key_quotes", []),
            word_count=len(original_text.split()),
        )
