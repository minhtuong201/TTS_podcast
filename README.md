# TTS Podcast Pipeline

Convert PDF documents into engaging podcast dialogues using AI summarization, script generation, and text-to-speech synthesis.

## 🎯 Features

- **PDF Text Extraction**: Extract and process text from PDF documents
- **AI Summarization**: Use GPT-4o-mini to create concise summaries
- **Natural Dialogue Generation**: Create engaging 2-person conversations with natural elements like pauses, laughter, and emotional expressions
- **Multiple TTS Backends**: Support for ElevenLabs, OpenAI TTS, Azure Speech, Google Cloud TTS, and local Coqui-XTTS
- **Audio Mixing**: Automatic audio processing and mixing for professional-sounding podcasts
- **Language Detection**: Automatic language detection with manual override support

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd TTS_podcast

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` with your API keys:

```bash
# Required: OpenRouter API for GPT-4o-mini
OPENROUTER_API_KEY=your_openrouter_api_key_here

# At least one TTS provider:
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google/credentials.json
```

### Basic Usage

```bash
# Convert a PDF to podcast (auto-select TTS backend)
python src/main.py your_document.pdf

# Specify TTS backend and voices
python src/main.py paper.pdf --tts eleven --voices Rachel Josh

# Custom output location and settings
python src/main.py research.pdf --output my_podcast.mp3 --target-words 1200
```

## 📖 Command Line Options

```bash
python src/main.py PDF_FILE [OPTIONS]

Required:
  PDF_FILE                  Path to PDF file to process

Optional:
  --tts {eleven,openai,azure,google,coqui,auto}
                           TTS backend to use (default: auto)
  --voices FEMALE MALE     Voice IDs for female host and male guest
  --output OUTPUT_FILE     Output MP3 file path
  --summary-length {short,medium,long}
                           Summary length (default: short)
  --target-words WORDS     Target word count for dialogue (default: 900)
  --language LANG          Force specific language (ISO code)
  --log-level {DEBUG,INFO,WARNING,ERROR}
                           Logging level (default: INFO)
  --keep-temp              Keep temporary audio files
  --dual-tts               Generate with both OpenAI and ElevenLabs
```

## 🔊 TTS Backend Configuration

### ElevenLabs (Recommended)
- **Quality**: Excellent for podcasts with natural prosody
- **Voices**: 10+ multilingual voices, natural breathing and pauses
- **Cost**: 10k characters free/month, then ~$5-11 per 100k characters
- **Setup**: Get API key from [ElevenLabs](https://elevenlabs.io)

Popular voices:
- Female: `Rachel`, `Domi`, `Bella`, `MF3mGyEYCl7XYWbV9V6O` (Elli - Vietnamese)
- Male: `Josh`, `Antoni`, `Arnold`, `M0rVwr32hdQ5UXpkI3ni` (The Hao - Vietnamese)

### OpenAI TTS
- **Quality**: Very good, fewer voices but excellent prosody
- **Voices**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- **Cost**: ~$0.015 per minute of audio
- **Setup**: Use existing OpenAI API key

### Google Cloud TTS
- **Quality**: Premium Neural2 voices, 40+ languages
- **Voices**: `en-US-Neural2-C` (female), `en-US-Neural2-A` (male)
- **Cost**: 1M characters free/month, then $16 per 1M characters (Neural2)
- **Setup**: Create service account, download JSON credentials

#### Google Cloud Setup
1. Install dependencies: `pip install -r requirements-google.txt`
2. Set credentials: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json`
3. Test: `python src/main.py document.pdf --tts google`

### Azure Speech Services
- **Quality**: Good, 400+ voices in 140+ languages
- **Cost**: 0.5M characters free for 30 days, then ~$16 per 1M characters
- **Setup**: Requires Azure subscription

### Coqui-XTTS (Local/Offline)
- **Quality**: Good, completely offline
- **Cost**: Free (requires GPU for best performance)
- **Setup**: Uncomment `TTS==0.21.1` in requirements.txt

## 🎭 Natural Dialogue Features

The script generator creates realistic conversations with:

- **Emotional expressions**: `[laughs]`, `[chuckles]`, `[surprised]`, `[thoughtful]`
- **Natural pauses**: Automatic pause detection and insertion
- **Conversational elements**: "Well...", "You know...", "That's fascinating!"
- **Question-answer dynamics**: Natural back-and-forth between host and guest
- **Discovery moments**: Reactions of surprise and engagement

Example generated dialogue:
```
HOST: So this research is looking at something pretty fascinating about neural networks. [excited] Can you break down what makes this approach different?

GUEST: [chuckles] Well, you know, traditional neural networks process information in a very linear way... [pause] But what these researchers discovered is actually quite surprising!

HOST: Ooh, tell me more! [curious]

GUEST: [thoughtful] So instead of just passing data through layers sequentially...
```

## 🔧 Advanced Usage

### Dual TTS Generation
Generate podcasts with both OpenAI and ElevenLabs for comparison:

```bash
python src/main.py document.pdf --dual-tts --target-words 150
```

Creates:
- `output/{pdf_name}_openai.mp3`
- `output/{pdf_name}_eleven.mp3`
- Shared script and summary files

### Custom Voice Configuration
```bash
# ElevenLabs with specific voices
python src/main.py paper.pdf --tts eleven --voices "Rachel" "Josh"

# Google Cloud with Neural2 voices  
python src/main.py paper.pdf --tts google --voices "en-US-Neural2-C" "en-US-Neural2-A"
```

### Multi-language Support
```bash
# Force Spanish language
python src/main.py documento.pdf --language es --tts google --voices "es-ES-Neural2-A" "es-ES-Neural2-B"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run tests with verbose output
pytest tests/ -v

# Run interactive test demonstration
python tests/test_pipeline.py
```

## 📊 Output Files

- **Format**: MP3 audio file at 192kbps
- **Duration**: Approximately 8 minutes (configurable)
- **Structure**: Natural conversation between female host and male guest
- **Metadata**: JSON files with processing details and costs

## 🔍 Troubleshooting

### Common Issues

**"No TTS backends available"**
- Check that at least one API key is configured in `.env`
- Verify API keys are valid and have sufficient credits

**"PDF extraction failed"**
- Ensure PDF is text-based (not scanned images)
- Try with `--target-words` set to a lower value

**"Script generation timeout"**
- Check OpenRouter API key and credits
- Reduce `--target-words` for shorter processing

### Enable Debug Logging

```bash
python src/main.py document.pdf --log-level DEBUG
```

## 🏗️ Pipeline Architecture

The system follows a clear 8-step process:

1. **PDF Text Extraction** - Extract and clean text from PDF
2. **Language Detection** - Automatically detect document language
3. **AI Summarization** - Create focused summary using GPT-4o-mini
4. **Script Generation** - Convert summary to natural dialogue
5. **TTS Backend Selection** - Choose optimal voice synthesis service
6. **Voice Assignment** - Map speakers to appropriate voices
7. **Audio Synthesis** - Generate speech for each dialogue line
8. **Audio Mixing** - Combine and normalize final podcast

## 📋 Requirements

- **Python**: 3.8+
- **OpenRouter API**: Required for summarization and script generation
- **TTS Provider**: At least one of ElevenLabs, OpenAI, Azure, Google Cloud, or local Coqui

## 📄 License

MIT License - see scaffold.md for full project specifications. 