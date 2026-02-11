# 📋 Setup Guide — Whisper Multimodal Pipeline

**Estimated time: 2-3 hours** (including model download and experimentation)

---

## Step 1: Install System Dependencies

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Verify
ffmpeg -version
```

FFmpeg is required by both Whisper and pydub for audio format handling.

---

## Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> **Note:** PyTorch will download ~2GB. Whisper model weights download on first use (base = ~150MB, large = ~3GB).

---

## Step 3: Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key and set it:

```bash
export GEMINI_API_KEY="your-key-here"
```

Or add to a `.env` file (gitignored):

```
GEMINI_API_KEY=your-key-here
```

**Free tier** includes 60 requests/minute for Gemini 1.5 Pro — more than enough for testing.

---

## Step 4: Add Sample Audio

Place audio files in `sample_audio/`:

```bash
# Example: record a quick clip
# Or download a sample:
# - https://www.openslr.org/12/ (LibriSpeech)
# - Any podcast clip, meeting recording, or voice memo

cp ~/Downloads/meeting_recording.wav sample_audio/
```

**Supported formats:** WAV, MP3, M4A, FLAC, OGG

**Recommended for testing:**
- 30-60 second clips work great
- Meeting recordings with multiple topics show off the analysis best
- Try clips with clear action items and named entities

---

## Step 5: Run the Pipeline

### Option A: Notebook (recommended for exploration)

```bash
cd notebooks
# Open in VS Code, Jupyter, or run as script:
python transcribe_analyze.py
```

### Option B: CLI

```bash
python -m src.pipeline --audio sample_audio/your_clip.wav
```

### Option C: Python API

```python
from src.pipeline import Pipeline

pipeline = Pipeline()
result = pipeline.run("sample_audio/your_clip.wav")
print(result.to_summary())
```

Output is saved to `outputs/` as structured JSON.

---

## Step 6: Review & Update README

After running the pipeline:

1. Check `outputs/` for the JSON results
2. Copy an interesting example into the README's "Example Output" section
3. Update with your actual results to show real capability

---

## Optional: Run Tests

```bash
pytest tests/ -v
```

---

## Optional: Try Google Cloud Speech-to-Text

If you want to test the GCP backend:

1. Set up a [GCP project](https://console.cloud.google.com)
2. Enable the Speech-to-Text API
3. Create a service account and download credentials
4. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

5. Update `configs/pipeline_config.yaml`:

```yaml
transcription:
  backend: gcp
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ffmpeg not found` | Install ffmpeg (Step 1) |
| `GEMINI_API_KEY not set` | Export the key (Step 3) |
| `CUDA out of memory` | Use a smaller Whisper model (`tiny` or `base`) in config |
| `ModuleNotFoundError` | Activate your venv and `pip install -r requirements.txt` |
| Slow transcription | Use `tiny` model for testing, `large-v3` for production quality |

---

*Once you've run the pipeline successfully, you have a working cross-modal AI demo ready for your portfolio.*
