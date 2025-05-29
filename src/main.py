#!/usr/bin/env python3
"""
TTS Podcast Pipeline - Main CLI Orchestrator

This is the main entry point that coordinates the entire pipeline:
1. Extract PDF text
2. Detect language
3. Summarize content
4. Generate dialogue script
5. Choose TTS backend
6. Synthesize audio
7. Mix final podcast

Usage:
    python src/main.py input.pdf --tts eleven --voices female_1 male_2
"""
import argparse
import logging
from pathlib import Path
import sys
import os
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pdf_ingest import extract
from lang_detect import detect as detect_language
from summarizer import llm_summary, SummaryConfig
from script_gen import ScriptGenerator, ScriptConfig
from tts.base import TTSConfig
from tts.eleven import ElevenLabsTTSBackend
from audio_mixer import AudioMixer, AudioSegmentInfo, MixingConfig
from utils.log_cfg import JSONFormatter, PipelineTimer, log_pipeline_metrics

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup structured logging for the pipeline"""
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)


def get_available_tts_backends() -> List[str]:
    """Get list of available TTS backends"""
    backends = []
    
    # Check ElevenLabs
    if os.getenv('ELEVENLABS_API_KEY'):
        backends.append('eleven')
    
    # Check OpenAI
    if os.getenv('OPENAI_API_KEY'):
        backends.append('openai')
    
    # Check Azure
    if os.getenv('AZURE_SPEECH_KEY') and os.getenv('AZURE_SPEECH_REGION'):
        backends.append('azure')
    
    # Check Coqui (always available if installed)
    try:
        import torch
        from TTS.api import TTS
        backends.append('coqui')
    except ImportError:
        pass
    
    return backends


def create_tts_backend(backend_name: str, language: str, config: TTSConfig):
    """Factory function to create TTS backend instances"""
    
    if backend_name == 'eleven':
        return ElevenLabsTTSBackend(config)
    elif backend_name == 'openai':
        from tts.openai import OpenAITTSBackend
        return OpenAITTSBackend(config)
    elif backend_name == 'azure':
        from tts.azure import AzureTTSBackend
        return AzureTTSBackend(config)
    elif backend_name == 'coqui':
        from tts.coqui import CoquiTTSBackend
        return CoquiTTSBackend(config)
    else:
        raise ValueError(f"Unknown TTS backend: {backend_name}")


def main():
    """Main pipeline orchestrator"""
    
    parser = argparse.ArgumentParser(
        description="TTS Podcast Pipeline - Convert PDF to podcast dialogue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py paper.pdf --tts eleven
  python src/main.py doc.pdf --tts eleven --voices female_1 male_2 --output my_podcast.mp3
  python src/main.py research.pdf --summary-length medium --target-words 1200
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--tts', choices=['eleven', 'openai', 'azure', 'coqui', 'auto'], 
                       default='auto', help='TTS backend to use (default: auto-select)')
    parser.add_argument('--voices', nargs=2, metavar=('FEMALE', 'MALE'),
                       help='Voice IDs for female host and male guest')
    parser.add_argument('--output', '-o', help='Output MP3 file path (default: auto-generated)')
    parser.add_argument('--summary-length', choices=['short', 'medium', 'long'], 
                       default='short', help='Summary length (default: short)')
    parser.add_argument('--target-words', type=int, default=900, 
                       help='Target word count for dialogue (default: 900)')
    parser.add_argument('--max-chars', type=int, 
                       help='Maximum characters to extract from PDF')
    parser.add_argument('--language', help='Force specific language (ISO code)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Optional log file path')
    parser.add_argument('--keep-temp', action='store_true', 
                       help='Keep temporary audio files')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        
        # Setup output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"output/{pdf_path.stem}_podcast.mp3")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting TTS podcast pipeline for: {pdf_path}")
        
        # Step 1: Extract PDF text
        with PipelineTimer("PDF extraction", logger):
            text = extract(pdf_path, max_chars=args.max_chars)
            logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Step 2: Detect language
        if args.language:
            language_code = args.language
            confidence = 1.0
            lang_metadata = {}
            logger.info(f"Using forced language: {language_code}")
        else:
            with PipelineTimer("Language detection", logger):
                language_code, confidence, lang_metadata = detect_language(text)
                logger.info(f"Detected language: {language_code} (confidence: {confidence:.3f})")
        
        # Step 3: Summarize text
        with PipelineTimer("Text summarization", logger):
            summary_config = SummaryConfig(target_length=args.summary_length)
            summary = llm_summary(text, target_len=args.summary_length)
            logger.info(f"Generated summary: {len(summary)} characters")
        
        # Step 4: Generate dialogue script
        with PipelineTimer("Script generation", logger):
            script_config = ScriptConfig(
                target_words=args.target_words,
                language=language_code
            )
            script_generator = ScriptGenerator()
            dialogue_lines = script_generator.generate_dialogue(summary, script_config)
            logger.info(f"Generated script with {len(dialogue_lines)} dialogue lines")
        
        # Step 5: Choose TTS backend
        if args.tts == 'auto':
            available_backends = get_available_tts_backends()
            if not available_backends:
                logger.error("No TTS backends available. Check your API keys.")
                sys.exit(1)
            
            # Prefer ElevenLabs for quality, fall back to others
            tts_backend_name = available_backends[0] if 'eleven' in available_backends else available_backends[0]
        else:
            tts_backend_name = args.tts
        
        logger.info(f"Using TTS backend: {tts_backend_name}")
        
        # Step 6: Setup TTS configuration
        tts_config = TTSConfig(language=language_code)
        if args.voices:
            tts_config.female_voice = args.voices[0]
            tts_config.male_voice = args.voices[1]
        
        tts_backend = create_tts_backend(tts_backend_name, language_code, tts_config)
        
        # Step 7: Synthesize audio for each dialogue line
        audio_segments = []
        total_cost = 0.0
        
        with PipelineTimer("Audio synthesis", logger):
            for i, line in enumerate(dialogue_lines):
                logger.debug(f"Synthesizing line {i+1}/{len(dialogue_lines)}: {line.speaker.value}")
                
                # Determine voice based on speaker
                voice_id = tts_config.female_voice if line.speaker.value == "HOST" else tts_config.male_voice
                
                # Synthesize the line
                result = tts_backend.synthesize(line.text, voice_id)
                
                # Create audio segment info
                segment_info = AudioSegmentInfo(
                    audio_data=result.audio_data,
                    duration_seconds=result.duration_seconds,
                    speaker=line.speaker.value.lower(),
                    text=line.text,
                    pause_before=line.pause_before,
                    pause_after=line.pause_after
                )
                
                audio_segments.append(segment_info)
                total_cost += result.cost_estimate
                
                logger.debug(f"Synthesized {result.duration_seconds:.1f}s audio")
        
        logger.info(f"Synthesized {len(audio_segments)} audio segments, estimated cost: ${total_cost:.3f}")
        
        # Step 8: Mix final audio
        with PipelineTimer("Audio mixing", logger):
            mixer_config = MixingConfig()
            mixer = AudioMixer(mixer_config)
            
            mix_result = mixer.mix_segments(audio_segments, output_path)
            logger.info(f"Final podcast created: {output_path}")
            logger.info(f"Duration: {mix_result['duration_seconds']:.1f}s, Size: {mix_result['file_size_bytes']/1024/1024:.1f}MB")
        
        # Log final pipeline metrics
        final_metrics = {
            'input_file': str(pdf_path),
            'output_file': str(output_path),
            'language': language_code,
            'language_confidence': confidence,
            'tts_backend': tts_backend_name,
            'text_chars': len(text),
            'summary_chars': len(summary),
            'script_lines': len(dialogue_lines),
            'audio_segments': len(audio_segments),
            'final_duration_seconds': mix_result['duration_seconds'],
            'final_file_size_bytes': mix_result['file_size_bytes'],
            'estimated_cost': total_cost
        }
        log_pipeline_metrics("pipeline_complete", final_metrics, logger)
        
        logger.info("TTS Podcast Pipeline completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 