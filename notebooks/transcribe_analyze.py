# %% [markdown]
# # 🎙️ Whisper Multimodal Pipeline — Full Walkthrough
#
# **Audio → Transcription → LLM Analysis → Structured Output**
#
# This notebook demonstrates the complete cross-modal pipeline:
# 1. Load an audio file
# 2. Transcribe with OpenAI Whisper (or Google Cloud Speech-to-Text)
# 3. Analyze transcription with Google Gemini
# 4. Output structured JSON with sentiment, entities, topics, and action items

# %% [markdown]
# ## Setup

# %%
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from src.transcriber import Transcriber
from src.analyzer import Analyzer
from src.pipeline import Pipeline, PipelineConfig
from src.models import PipelineOutput

# %% [markdown]
# ## Configuration
#
# Set your Gemini API key and choose an audio file.

# %%
# Set API key (or export GEMINI_API_KEY in your environment)
# os.environ["GEMINI_API_KEY"] = "your-key-here"

AUDIO_FILE = project_root / "sample_audio" / "sample.wav"
CONFIG_PATH = project_root / "configs" / "pipeline_config.yaml"

print(f"Audio file: {AUDIO_FILE}")
print(f"File exists: {AUDIO_FILE.exists()}")

# %% [markdown]
# ## Option A: OpenAI Whisper Transcription (Local)
#
# Runs entirely on your machine. No API key needed for transcription.

# %%
whisper_transcriber = Transcriber(
    backend="whisper",
    model_name="base",  # tiny | base | small | medium | large
)

# Uncomment when you have a sample audio file:
# transcription = whisper_transcriber.transcribe(AUDIO_FILE)
# print(f"Transcribed: {transcription.text[:200]}...")
# print(f"Segments: {len(transcription.segments)}")
# print(f"Duration: {transcription.duration_seconds:.1f}s")

# %% [markdown]
# ## Option B: Google Cloud Speech-to-Text (Cloud)
#
# Requires GCP credentials. Better for production workloads.

# %%
# gcp_transcriber = Transcriber(
#     backend="gcp",
#     language="en-US",
# )
# transcription_gcp = gcp_transcriber.transcribe(AUDIO_FILE)
# print(f"GCP Transcription: {transcription_gcp.text[:200]}...")

# %% [markdown]
# ## Gemini Analysis
#
# Analyze the transcription for sentiment, entities, topics, summary, and action items.

# %%
analyzer = Analyzer(
    model_name="gemini-1.5-pro",
    temperature=0.2,
)

# Demo with sample text (replace with actual transcription)
sample_text = """
Good morning everyone. Let's start with the update on the character animation pipeline.
Sarah, can you walk us through where we are with the new Moana sequences?
Sure. We've completed the rigging for three of the five main characters. The facial
expressions are looking really good — we got positive feedback from John in the review
yesterday. The remaining two characters should be done by Thursday. One concern: the
water simulation rendering is taking longer than expected. We might need to allocate
additional GPU resources from the render farm. I'd suggest we talk to the infrastructure
team today. Overall, I'm feeling good about hitting our March 15th milestone, but the
rendering bottleneck is the biggest risk right now.
"""

analysis = analyzer.analyze(sample_text)

print("=== Analysis Results ===\n")
print(f"Sentiment: {analysis.sentiment.label.value} ({analysis.sentiment.score:.0%})")
print(f"\nTopics: {', '.join(analysis.topics)}")
print(f"\nSummary: {analysis.summary}")
print(f"\nEntities:")
for ent in analysis.entities:
    print(f"  - {ent.text} ({ent.type})")
print(f"\nAction Items:")
for item in analysis.action_items:
    print(f"  - [{item.priority or '?'}] {item.task}")
    if item.assignee:
        print(f"    Assignee: {item.assignee}")
    if item.deadline:
        print(f"    Deadline: {item.deadline}")

# %% [markdown]
# ## Full Pipeline — End to End
#
# Run the complete pipeline with one call.

# %%
config = PipelineConfig(CONFIG_PATH)
pipeline = Pipeline(config=config)

# Uncomment when you have a sample audio file:
# result = pipeline.run(AUDIO_FILE)
# print(result.to_summary())

# %% [markdown]
# ## Save Structured Output

# %%
# Save to JSON
# output_path = pipeline.save_output(result)
# print(f"Saved to: {output_path}")

# # Preview the JSON
# with open(output_path) as f:
#     data = json.load(f)
# print(json.dumps(data, indent=2)[:2000])

# %% [markdown]
# ## Async Batch Processing
#
# Process multiple audio files concurrently.

# %%
import asyncio

async def batch_demo():
    """Process multiple files concurrently."""
    audio_dir = project_root / "sample_audio"
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    if not audio_files:
        print("No audio files found in sample_audio/. Add some to test batch processing.")
        return

    pipeline = Pipeline(config=PipelineConfig(CONFIG_PATH))
    results = await pipeline.run_batch_async(audio_files)

    for r in results:
        if isinstance(r, Exception):
            print(f"Error: {r}")
        else:
            print(f"\n{'='*60}")
            print(r.to_summary())

# Uncomment to run:
# await batch_demo()  # in Jupyter
# asyncio.run(batch_demo())  # in script

# %% [markdown]
# ## Summary
#
# This pipeline demonstrates **cross-modal AI capability**:
#
# | Modality | Technology | Output |
# |----------|-----------|--------|
# | **Audio** | Whisper / GCP STT | Raw waveform → timestamped text |
# | **Text** | Gemini 1.5 Pro | Unstructured text → structured intelligence |
# | **Structured** | Pydantic | Validated JSON with types and constraints |
#
# **Key engineering decisions:**
# - Dual transcription backends for flexibility
# - Async support for batch processing
# - Retry logic with exponential backoff
# - Pydantic validation for reliable output schemas
# - Configurable via YAML — no code changes needed for tuning
