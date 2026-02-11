# 🎙️ Whisper Multimodal Pipeline

**Cross-modal AI: Audio → Text → Intelligence → Structured Output**

A production-ready pipeline that demonstrates multimodal AI capabilities by combining speech-to-text transcription (OpenAI Whisper) with LLM-powered analysis (Google Gemini) to extract structured intelligence from audio content.

---

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Audio Input │───▶│   Transcriber    │───▶│    Analyzer      │───▶│ Structured Output│
│              │    │                  │    │                  │    │                  │
│ .wav .mp3    │    │ OpenAI Whisper   │    │ Google Gemini    │    │ JSON / Pydantic  │
│ .m4a .flac   │    │ — or —           │    │                  │    │                  │
│              │    │ GCP Speech-to-   │    │ • Sentiment      │    │ • Transcription  │
│              │    │ Text             │    │ • Entities       │    │ • Analysis       │
│              │    │                  │    │ • Topics         │    │ • Action Items   │
│              │    │ Timestamped      │    │ • Summary        │    │ • Confidence     │
│              │    │ Segments         │    │ • Action Items   │    │   Scores         │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

## ✨ Key Features

- **Dual transcription backends** — OpenAI Whisper (local) or Google Cloud Speech-to-Text (cloud)
- **LLM-powered analysis** — Sentiment, named entities, topic classification, summarization, action item extraction via Gemini
- **Structured output** — Pydantic-validated JSON results with confidence scores
- **Async pipeline** — Full async/await support for concurrent processing
- **Retry logic** — Exponential backoff with configurable retry policies
- **Multi-format audio** — WAV, MP3, M4A, FLAC, OGG support via pydub/ffmpeg

## 🎯 Use Cases

| Domain | Application |
|--------|-------------|
| **Media & Entertainment** | Automated content analysis for audio/video assets |
| **Customer Experience** | Call center transcript analysis with sentiment tracking |
| **Accessibility** | Real-time captioning with contextual intelligence |
| **Content Production** | Meeting transcription → action items → task assignment |
| **Research** | Interview analysis with entity extraction and topic modeling |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Transcription | OpenAI Whisper, Google Cloud Speech-to-Text |
| Analysis | Google Gemini 1.5 Pro |
| Data Models | Pydantic v2 |
| Audio Processing | pydub, ffmpeg |
| ML Runtime | PyTorch |
| Async | asyncio, aiofiles |
| Testing | pytest, unittest.mock |
| Config | YAML |

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/whisper-multimodal-pipeline.git
cd whisper-multimodal-pipeline
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Run the pipeline
python -m src.pipeline --audio sample_audio/your_clip.wav
```

See [GUIDE.md](GUIDE.md) for detailed setup instructions.

## 📁 Project Structure

```
whisper-multimodal-pipeline/
├── src/
│   ├── __init__.py
│   ├── transcriber.py      # Whisper/GCP transcription wrapper
│   ├── analyzer.py          # Gemini LLM analysis engine
│   ├── pipeline.py          # Orchestration with async + retry
│   └── models.py            # Pydantic data models
├── notebooks/
│   └── transcribe_analyze.py  # Full pipeline walkthrough (percent script)
├── configs/
│   └── pipeline_config.yaml   # Model sizes, prompts, settings
├── sample_audio/
│   └── .gitkeep
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── .gitignore
├── GUIDE.md
└── README.md
```

## 📊 Example Output

```json
{
  "transcription": {
    "text": "We need to finalize the character designs by Friday...",
    "segments": [
      {"start": 0.0, "end": 3.2, "text": "We need to finalize the character designs by Friday"}
    ],
    "language": "en",
    "duration_seconds": 45.2
  },
  "analysis": {
    "sentiment": {"label": "neutral-urgent", "score": 0.78},
    "entities": [
      {"text": "Friday", "type": "DATE"},
      {"text": "character designs", "type": "DELIVERABLE"}
    ],
    "topics": ["project management", "creative production", "deadlines"],
    "summary": "Team discussion about finalizing character designs with a Friday deadline.",
    "action_items": [
      {"task": "Finalize character designs", "assignee": null, "deadline": "Friday"}
    ]
  }
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built to demonstrate cross-modal AI engineering: transforming raw audio into structured, actionable intelligence.*
