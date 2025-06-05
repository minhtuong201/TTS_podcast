# Changelog

## Repository Cleanup and Restructuring

### Files Removed
- ❌ **Redundant test files**: `test_sample.py`, `demo_core.py`, `test_eleven_settings.py`, `test_elevenlabs_api.py`, `test_google_tts.py`, `test_vietnamese_prompts.py`, `troubleshoot_elevenlabs.py`, `test_voices.py`
- ❌ **Redundant documentation**: `DUAL_TTS_README.md`, `GOOGLE_CLOUD_TTS_SETUP.md`
- ❌ **Redundant scripts**: `dual_tts_example.py`
- ❌ **Committed credentials**: `genuine-segment-461316-s6-aa9afa4b2a72.json`

### Files Moved and Organized

#### Created `utils/` directory for utility scripts:
- 📁 `utils/regenerate_podcast.py` - Regenerate podcast from existing scripts
- 📁 `utils/recover_audio.py` - Recovery script for partial synthesis
- 📁 `utils/customization_examples.py` - Customization examples and guides
- 📁 `utils/get_voice_info.py` - Voice information utility
- 📁 `utils/README.md` - Documentation for utility scripts

#### Created `tests/` directory with consolidated test suite:
- 📁 `tests/test_pipeline.py` - Comprehensive test suite with proper pytest structure

### Files Updated

#### Configuration Files:
- ✅ **`.gitignore`**: Enhanced with comprehensive patterns for Python projects, credentials, temporary files, and OS-specific files
- ✅ **`requirements.txt`**: Updated with version ranges, added test dependencies, better organization
- ✅ **`setup.cfg`**: New configuration file with project metadata, test configuration, and code quality settings
- ✅ **`setup.py`**: Simplified to use setuptools with setup.cfg

#### Documentation:
- ✅ **`README.md`**: Consolidated information from multiple documentation files, added comprehensive sections for all TTS backends, improved organization with emojis and clear structure

### Repository Structure (After Cleanup)

```
TTS_podcast/
├── src/                          # Main source code
│   ├── main.py                   # Main CLI entry point
│   ├── pdf_ingest.py            # PDF text extraction
│   ├── lang_detect.py           # Language detection
│   ├── summarizer.py            # AI summarization
│   ├── script_gen.py            # Dialogue script generation
│   ├── audio_mixer.py           # Audio mixing and processing
│   ├── tts/                     # TTS backend implementations
│   │   ├── base.py              # Base TTS interface
│   │   ├── eleven.py            # ElevenLabs TTS
│   │   ├── openai.py            # OpenAI TTS
│   │   ├── google.py            # Google Cloud TTS
│   │   ├── azure.py             # Azure Speech Services
│   │   └── coqui.py             # Local Coqui XTTS
│   └── utils/
│       └── log_cfg.py           # Logging configuration
├── tests/
│   └── test_pipeline.py         # Comprehensive test suite
├── utils/                       # Utility scripts
│   ├── README.md                # Utility documentation
│   ├── regenerate_podcast.py    # Regenerate from existing scripts
│   ├── recover_audio.py         # Recovery utility
│   ├── customization_examples.py # Customization guide
│   └── get_voice_info.py        # Voice information tool
├── output/                      # Generated podcasts (gitignored)
├── pdf_folder/                  # Input PDFs (gitignored)
├── .env                         # Environment variables (gitignored)
├── requirements.txt             # Core dependencies
├── requirements-google.txt      # Google Cloud TTS dependencies
├── setup.cfg                    # Project configuration
├── setup.py                     # Simplified setuptools config
├── setup.sh                     # Installation script
├── env_example.txt             # Environment template
├── README.md                    # Main documentation
├── CHANGELOG.md                 # This file
├── scaffold.md                  # Project specifications
├── system_status.md            # System status information
└── .gitignore                   # Enhanced ignore patterns
```

### Benefits of This Restructuring

1. **🧹 Cleaner Repository**: Removed 8+ redundant test files and 3 redundant documentation files
2. **📁 Better Organization**: Utility scripts organized in dedicated directory with documentation
3. **🧪 Proper Testing**: Consolidated test suite with pytest structure and proper organization
4. **📚 Comprehensive Documentation**: Single README with all essential information
5. **🔒 Better Security**: Removed committed credentials, enhanced .gitignore
6. **⚙️ Modern Configuration**: Added setup.cfg for proper Python project structure
7. **🚀 Easier Maintenance**: Clear separation of concerns and logical file organization

### Breaking Changes
- Test files moved from root to `tests/` directory
- Utility scripts moved from root to `utils/` directory
- Some redundant documentation removed (information consolidated into main README)

### Migration Guide
- **For testing**: Use `pytest tests/` instead of individual test files
- **For utilities**: Scripts moved to `utils/` directory - update any scripts that reference them
- **For documentation**: Refer to main README.md for all setup and usage information 