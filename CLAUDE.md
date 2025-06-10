# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Personal Guidelines

- Get right to the point. No yapping. Take a forward-thinking, step-by-step approach. No icon, use a clear, smart and formal tone.

- For complex topics/terms, leverage analogies and examples to explain (ELI5).

- For coding or ill-defined questions: First, ask questions to clarify the user's need, without self-assuming anything. Second, carefully derive a detailed plan, including all steps, project layout, role of each file, function of each file, etc. Finally, once approved by the user, execute based on the plan.

## Common Commands

### Development Setup
```bash
# Create virtual environment and install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# For Google Cloud TTS support
pip install -r requirements-google.txt

# Quick setup using provided script
./setup.sh
```

### Running the Pipeline
```bash
# Basic usage - convert PDF to podcast (uses single-speaker TTS)
python src/main.py document.pdf

# With specific TTS backend and voices
python src/main.py paper.pdf --tts eleven --voices Rachel Josh

# Generate with Gemini multi-speaker TTS (recommended)
python src/main.py document.pdf --tts gemini

# Generate with multiple TTS backends for comparison
python src/main.py document.pdf --dual-tts
```

### Multi-Speaker TTS Regeneration
```bash
# Regenerate podcast from existing script using multi-speaker TTS
python utils/regenerate_podcast_multi_speaker.py script_file.txt --model gemini

# List available voices for multi-speaker models
python utils/regenerate_podcast_multi_speaker.py --model gemini --list-voices

# Custom voices and style for multi-speaker
python utils/regenerate_podcast_multi_speaker.py script_file.txt --model gemini \
  --host-voice Zephyr --guest-voice Puck \
  --style-prompt "Generate energetic Vietnamese podcast dialogue"
```

### Single-Speaker TTS Regeneration
```bash
# Regenerate using traditional single-speaker approach with mixing
python utils/regenerate_podcast_single_speaker.py script_file.txt --tts eleven

# With custom volume and speed adjustments
python utils/regenerate_podcast_single_speaker.py script_file.txt --tts eleven \
  --male-volume 1.1 --female-volume 1.0 --male-speed 1.15 --female-speed 1.12
```

### Testing
```bash
# Run test suite
pytest tests/

# Run interactive test demonstration
python tests/test_pipeline.py

# Test with verbose output
pytest tests/ -v
```

## Architecture Overview

This is a TTS (Text-to-Speech) podcast pipeline that converts PDF documents into engaging audio dialogues. The system supports **two distinct TTS approaches**:

### 1. Traditional Single-Speaker TTS (Multi-Backend)
- **Process**: Generate audio for each dialogue line separately, then mix together
- **Backends**: ElevenLabs, OpenAI, Azure, Google Cloud, Coqui
- **Mixing**: Uses `audio_mixer.py` to combine individual audio segments
- **Benefits**: Wide backend support, fine-grained control over each line
- **Usage**: Default mode in `src/main.py`, regeneration via `utils/regenerate_podcast_single_speaker.py`

### 2. True Multi-Speaker TTS (Gemini Only)
- **Process**: Generate entire conversation in single API call with multiple speakers
- **Backend**: Gemini 2.5 Pro TTS with native multi-speaker support
- **No Mixing**: Returns complete podcast audio directly
- **Benefits**: More natural speaker transitions, better conversational flow
- **Usage**: `--tts gemini` in main pipeline, dedicated `src/gemini_pipeline.py`, regeneration via `utils/regenerate_podcast_multi_speaker.py`

### Core Pipeline (8 Steps)

1. **PDF Text Extraction** (`pdf_ingest.py`) - Extract and clean text from PDF files
2. **Language Detection** (`lang_detect.py`) - Automatically detect document language 
3. **AI Summarization** (`summarizer.py`) - Create focused summaries using GPT-4o-mini via OpenRouter
4. **Script Generation** (`script_gen.py`) - Convert summaries to natural 2-person dialogue with emotional expressions
5. **TTS Backend Selection** (`main.py`) - Auto-select optimal voice synthesis service
6. **Voice Assignment** (`tts/` modules) - Map speakers to appropriate voices
7. **Audio Synthesis** - Generate speech using selected approach:
   - **Single-speaker**: Individual line synthesis + mixing (`audio_mixer.py`)
   - **Multi-speaker**: Complete conversation synthesis (`src/gemini_pipeline.py`)
8. **Final Output** - MP3 podcast with metadata

### Key Components

- **Main orchestrator**: `src/main.py` - CLI entry point supporting both TTS approaches
- **Multi-speaker pipeline**: `src/gemini_pipeline.py` - Dedicated Gemini multi-speaker processing
- **TTS backends**: Modular system in `src/tts/` directory
  - Single-speaker: `eleven.py`, `openai.py`, `azure.py`, `google.py`, `coqui.py`
  - Multi-speaker: `gemini.py` with native multi-speaker API support
- **Audio processing**: 
  - Single-speaker: `audio_mixer.py` for segment mixing and normalization
  - Multi-speaker: Direct audio output from Gemini TTS
- **Utilities**: 
  - `utils/regenerate_podcast_single_speaker.py` - Traditional regeneration
  - `utils/regenerate_podcast_multi_speaker.py` - Multi-speaker regeneration

### Configuration

The system uses `.env` file for API keys and settings:
- **Required**: `OPENROUTER_API_KEY` for summarization/script generation
- **TTS providers**: 
  - Multi-speaker: `GEMINI_API_KEY` (recommended)
  - Single-speaker: At least one of ElevenLabs, OpenAI, Azure, Google Cloud keys
- **Default voices**: 
  - Gemini: Zephyr (host), Puck (guest) - Vietnamese-optimized
  - ElevenLabs: Elli/The Hao - Vietnamese voices

### Output Files

- **MP3 podcast**: Final audio at 192kbps (single-speaker mixed or multi-speaker direct)
- **Analysis file**: Text summary and metadata  
- **Script file**: Complete dialogue with speaker annotations
- **Metadata JSON**: Processing details, costs, and file information

### Voice Configuration

**Multi-Speaker TTS (Gemini)**:
- Uses voice names: Zephyr, Puck, Fenrir, Leda, etc.
- Configured via `--host-voice` and `--guest-voice` parameters
- Supports style prompts for conversation tone

**Single-Speaker TTS**:
- **ElevenLabs**: Custom voice IDs (e.g., "MF3mGyEYCl7XYWbV9V6O" for Elli)
- **OpenAI**: Predefined names (e.g., "shimmer", "onyx")  
- **Google**: Language-model combinations (e.g., "vi-VN-Neural2-A")

### Recommendations

- **For highest quality**: Use Gemini multi-speaker TTS (`--tts gemini`)
- **For backend flexibility**: Use traditional single-speaker approach with ElevenLabs
- **For regeneration**: Use appropriate utility script based on desired approach
- **For Vietnamese content**: Gemini or ElevenLabs provide best results