# Utility Scripts

This directory contains utility scripts for advanced TTS podcast pipeline operations.

## 🔧 Available Utilities

### `regenerate_podcast.py`
Regenerate podcast audio from an existing script file with custom speaker adjustments.

**Features:**
- Parse existing script files
- Custom volume adjustments per speaker
- Custom speed adjustments per speaker
- Support for all TTS backends
- Voice customization

**Usage:**
```bash
python utils/regenerate_podcast.py script.txt --male-volume 1.1 --female-volume 1.0
python utils/regenerate_podcast.py script.txt --tts eleven --voices "Rachel" "Josh"
```

### `recover_audio.py`
Recovery script to generate partial audio from successful segments when synthesis fails partway through.

**Features:**
- Resume from partial synthesis
- Process only successful segments
- Useful for quota/rate limit recovery

**Usage:**
```bash
python utils/recover_audio.py
```

### `customization_examples.py`
Interactive examples showing how to customize various aspects of the pipeline.

**Features:**
- Voice customization examples
- Script style modifications
- Character personality adjustments
- Emotion mapping customization

**Usage:**
```bash
python utils/customization_examples.py
```

### `get_voice_info.py`
Get detailed information about available TTS voices.

**Features:**
- List available voices for each TTS backend
- Voice capability information
- Language support details

**Usage:**
```bash
python utils/get_voice_info.py --backend eleven
python utils/get_voice_info.py --backend google --language vi
```

## 📋 Requirements

All utilities require the same dependencies as the main pipeline:
- Python 3.8+
- Required API keys set in `.env`
- Installed dependencies from `requirements.txt`

## 🔗 Integration

These utilities are designed to work alongside the main pipeline (`src/main.py`) and can be used for:
- Post-processing existing podcasts
- Advanced customization workflows
- Debugging and recovery operations
- Voice and backend exploration 