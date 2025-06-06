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
# Basic usage - convert PDF to podcast
python src/main.py document.pdf

# With specific TTS backend and voices
python src/main.py paper.pdf --tts eleven --voices Rachel Josh

# Generate with multiple TTS backends for comparison
python src/main.py document.pdf --dual-tts
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

This is a TTS (Text-to-Speech) podcast pipeline that converts PDF documents into engaging audio dialogues. The system follows an 8-step pipeline:

1. **PDF Text Extraction** (`pdf_ingest.py`) - Extract and clean text from PDF files
2. **Language Detection** (`lang_detect.py`) - Automatically detect document language 
3. **AI Summarization** (`summarizer.py`) - Create focused summaries using GPT-4o-mini via OpenRouter
4. **Script Generation** (`script_gen.py`) - Convert summaries to natural 2-person dialogue with emotional expressions
5. **TTS Backend Selection** (`main.py`) - Auto-select optimal voice synthesis service
6. **Voice Assignment** (`tts/` modules) - Map speakers to appropriate voices
7. **Audio Synthesis** (TTS backends) - Generate speech for each dialogue line
8. **Audio Mixing** (`audio_mixer.py`) - Combine and normalize final podcast MP3

### Key Components

- **Main orchestrator**: `src/main.py` - CLI entry point that coordinates entire pipeline
- **TTS backends**: Modular system supporting ElevenLabs, OpenAI, Azure, Google Cloud, and local Coqui
- **Natural dialogue**: Script generator creates realistic conversations with pauses, laughter, emotional expressions
- **Audio processing**: Professional mixing with configurable bitrates and normalization

### Configuration

The system uses `.env` file for API keys and settings:
- **Required**: `OPENROUTER_API_KEY` for summarization/script generation
- **TTS providers**: At least one of ElevenLabs, OpenAI, Azure, Google Cloud keys
- **Default voices**: Vietnamese voices (Elli/The Hao) for ElevenLabs backend

### Output Files

- **MP3 podcast**: Final mixed audio at 192kbps
- **Analysis file**: Text summary and metadata  
- **Script file**: Complete dialogue with speaker annotations
- **Metadata JSON**: Processing details, costs, and file information

### Error Handling

The pipeline includes comprehensive error handling with structured JSON logging. Common failure points:
- Missing API keys (check `.env` configuration)
- PDF extraction issues (ensure text-based PDFs)
- TTS backend unavailability (auto-fallback to available providers)

### Voice Configuration

Different TTS backends use different voice IDs:
- **ElevenLabs**: Custom voice IDs (e.g., "MF3mGyEYCl7XYWbV9V6O" for Elli)
- **OpenAI**: Predefined names (e.g., "shimmer", "onyx")  
- **Google**: Language-model combinations (e.g., "en-US-Neural2-C")

The system automatically selects appropriate default voices per backend and allows custom voice specification via CLI.