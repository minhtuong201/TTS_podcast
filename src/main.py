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
import json
from datetime import datetime
import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pdf_ingest import extract
from lang_detect import detect as detect_language
from summarizer import llm_summary, SummarizerConfig
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
    logger = logging.getLogger(__name__)
    backends = []
    
    # Check ElevenLabs with connectivity test
    if os.getenv('ELEVENLABS_API_KEY'):
        try:
            # Quick connectivity test
            response = requests.get('https://api.elevenlabs.io/v1/voices', 
                                  headers={'xi-api-key': os.getenv('ELEVENLABS_API_KEY')}, 
                                  timeout=5)
            if response.status_code in [200, 401]:  # 401 means API key works, just unauthorized for this endpoint
                backends.append('eleven')
                logger.info("ElevenLabs API connectivity: ✓")
            else:
                logger.warning(f"ElevenLabs API issue: {response.status_code}")
        except Exception as e:
            logger.warning(f"ElevenLabs connectivity test failed: {e}")
    
    # Check Gemini TTS
    if os.getenv('GEMINI_API_KEY'):
        try:
            import google.generativeai as genai
            backends.append('gemini')
            logger.info("Gemini TTS API key found: ✓")
        except ImportError:
            logger.warning("Gemini TTS: google-generativeai package not installed")
        except Exception as e:
            logger.warning(f"Gemini TTS check failed: {e}")
    
    # Check OpenAI
    if os.getenv('OPENAI_API_KEY'):
        backends.append('openai')
    
    # Check Google Cloud TTS
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        backends.append('google')
    
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
    elif backend_name == 'gemini':
        from tts.gemini import GeminiTTSBackend
        return GeminiTTSBackend(config)
    elif backend_name == 'openai':
        from tts.openai import OpenAITTSBackend
        return OpenAITTSBackend(config)
    elif backend_name == 'google':
        from tts.google import GoogleCloudTTSBackend
        return GoogleCloudTTSBackend(config)
    elif backend_name == 'azure':
        from tts.azure import AzureTTSBackend
        return AzureTTSBackend(config)
    elif backend_name == 'coqui':
        from tts.coqui import CoquiTTSBackend
        return CoquiTTSBackend(config)
    else:
        raise ValueError(f"Unknown TTS backend: {backend_name}")


def process_tts_synthesis(tts_backend_name: str, dialogue_lines, language_code, voices, output_base_path, logger):
    """Process TTS synthesis for a given backend and return results"""
    
    # Set voices based on TTS backend
    if voices:
        female_voice, male_voice = voices
    else:
        # Default voices for different backends
        if tts_backend_name == 'eleven':
            female_voice = "MF3mGyEYCl7XYWbV9V6O"  # Elli - Vietnamese female voice
            male_voice = "M0rVwr32hdQ5UXpkI3ni"    # The Hao - Vietnamese male voice
        elif tts_backend_name == 'gemini':
            # Use Vietnamese-optimized voices from Gemini
            female_voice = "Fenrir"  # Excitable, energetic host
            male_voice = "Leda"      # Youthful, engaged guest
        elif tts_backend_name == 'openai':
            female_voice = "shimmer"  # OpenAI female voice
            male_voice = "onyx"       # OpenAI male voice
        elif tts_backend_name == 'google':
            female_voice = "en-US-Neural2-C"  # Google Cloud female voice (high quality)
            male_voice = "en-US-Neural2-A"    # Google Cloud male voice (high quality)
        else:
            female_voice = "default_female"
            male_voice = "default_male"
    
    logger.info(f"Processing with {tts_backend_name} TTS backend - Female: {female_voice}, Male: {male_voice}")
    
    # Create TTS backend
    dummy_config = TTSConfig(voice_id=female_voice)
    tts_backend = create_tts_backend(tts_backend_name, language_code, dummy_config)
    
    # Handle Gemini TTS special multi-speaker capability
    if tts_backend_name == 'gemini':
        # Use Gemini's native multi-speaker synthesis
        with PipelineTimer(f"{tts_backend_name.title()} multi-speaker synthesis", logger):
            result = tts_backend.synthesize_multi_speaker(
                dialogue_lines, 
                host_voice=female_voice, 
                guest_voice=male_voice
            )
            
            # Create single audio segment for the entire conversation
            segment_info = AudioSegmentInfo(
                audio_data=result.audio_data,
                duration_seconds=result.duration_seconds,
                speaker="multi",  # Indicates multi-speaker audio
                text="Multi-speaker dialogue"
            )
            
            audio_segments = [segment_info]
            total_cost = result.cost_estimate
            
            logger.info(f"Gemini multi-speaker synthesis completed: {result.duration_seconds:.1f}s audio")
    else:
        # Standard per-line synthesis for other backends
        audio_segments = []
        total_cost = 0.0
        
        with PipelineTimer(f"{tts_backend_name.title()} audio synthesis", logger):
            for i, line in enumerate(dialogue_lines):
                logger.debug(f"Synthesizing line {i+1}/{len(dialogue_lines)}: {line.speaker.value} [{tts_backend_name}]")
                
                # Determine voice based on speaker
                voice_id = female_voice if line.speaker.value == "HOST" else male_voice
                
                # Create TTS config for this line with voice-specific settings
                if line.speaker.value == "HOST":
                    # Female voice settings (Elli)
                    tts_config = TTSConfig(
                        voice_id=voice_id,
                        speed=1.12,  # Female speed updated
                        stability=0.5,  # 50% stability for female voice
                        similarity_boost=0.75,  # 75% similarity boost
                        style=0.0  # 0% style
                    )
                else:
                    # Male voice settings (The Hao)
                    tts_config = TTSConfig(
                        voice_id=voice_id,
                        speed=1.15,  # Male speed updated
                        stability=0.25,  # 25% stability for male voice
                        similarity_boost=0.75,  # 75% similarity boost
                        style=0.0  # 0% style
                    )
                
                # Synthesize the line
                result = tts_backend.synthesize(line.text, tts_config)
                
                # Create audio segment info
                segment_info = AudioSegmentInfo(
                    audio_data=result.audio_data,
                    duration_seconds=result.duration_seconds,
                    speaker=line.speaker.value.lower(),
                    text=line.text
                )
                
                audio_segments.append(segment_info)
                total_cost += result.cost_estimate
                
                logger.debug(f"Synthesized {result.duration_seconds:.1f}s audio [{tts_backend_name}]")
    
    logger.info(f"{tts_backend_name.title()} synthesized {len(audio_segments)} audio segments, estimated cost: ${total_cost:.3f}")
    
    # Mix final audio
    output_path = output_base_path.parent / f"{output_base_path.stem}_{tts_backend_name}.mp3"
    
    if tts_backend_name == 'gemini':
        # For Gemini, we already have the complete audio, just save it
        with PipelineTimer(f"{tts_backend_name.title()} audio saving", logger):
            with open(output_path, 'wb') as f:
                f.write(audio_segments[0].audio_data)
            
            # Create mix_result compatible with existing code
            mix_result = {
                'duration_seconds': audio_segments[0].duration_seconds,
                'file_size_bytes': len(audio_segments[0].audio_data)
            }
            
            logger.info(f"{tts_backend_name.title()} podcast created: {output_path}")
            logger.info(f"Duration: {mix_result['duration_seconds']:.1f}s, Size: {mix_result['file_size_bytes']/1024/1024:.1f}MB")
    else:
        # Standard mixing for other backends
        with PipelineTimer(f"{tts_backend_name.title()} audio mixing", logger):
            mixer_config = MixingConfig()
            mixer = AudioMixer(mixer_config)
            
            mix_result = mixer.mix_segments(audio_segments, output_path)
            logger.info(f"{tts_backend_name.title()} podcast created: {output_path}")
            logger.info(f"Duration: {mix_result['duration_seconds']:.1f}s, Size: {mix_result['file_size_bytes']/1024/1024:.1f}MB")
    
    return {
        'backend': tts_backend_name,
        'output_path': output_path,
        'audio_segments': audio_segments,
        'total_cost': total_cost,
        'mix_result': mix_result,
        'voices': {'female': female_voice, 'male': male_voice}
    }


def main():
    """Main pipeline orchestrator"""
    
    parser = argparse.ArgumentParser(
        description="TTS Podcast Pipeline - Convert PDF to podcast dialogue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py paper.pdf
  python src/main.py doc.pdf --voices female_id male_id --output my_podcast.mp3
  python src/main.py research.pdf --summary-length medium --target-words 1200
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--tts', choices=['eleven', 'gemini', 'openai', 'google', 'azure', 'coqui', 'auto'], 
                       default='auto', help='TTS backend to use (default: auto-select with fallback)')
    parser.add_argument('--dual-tts', action='store_true',
                       help='Generate podcasts using both OpenAI and ElevenLabs TTS')
    parser.add_argument('--voices', nargs=2, metavar=('FEMALE', 'MALE'),
                       help='Voice IDs for female host and male guest')
    parser.add_argument('--output', '-o', help='Output MP3 file path (default: auto-generated)')
    parser.add_argument('--summary-length', choices=['short', 'medium', 'long'], 
                       default='short', help='Analysis length (default: short)')
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
        
        # Step 3: Analyze text (previously called summarize)
        with PipelineTimer("Text analysis", logger):
            summary_config = SummarizerConfig(target_length=args.summary_length)
            summary = llm_summary(text, target_len=args.summary_length)
            logger.info(f"Generated analysis: {len(summary)} characters")
        
        # Save analysis to file
        summary_path = output_path.parent / f"{pdf_path.stem}_analysis.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Analysis of {pdf_path.name}\n\n")
            f.write(f"Language: {language_code} (confidence: {confidence:.3f})\n")
            f.write(f"Original text: {len(text)} characters\n")
            f.write(f"Analysis: {len(summary)} characters\n")
            f.write(f"Compression ratio: {len(text)/len(summary):.1f}x\n\n")
            f.write(summary)
        logger.info(f"Analysis saved to: {summary_path}")
        
        # Step 4: Generate dialogue script
        with PipelineTimer("Script generation", logger):
            script_config = ScriptConfig(
                target_words=args.target_words,
                language=language_code
            )
            script_generator = ScriptGenerator()
            dialogue_lines = script_generator.generate_dialogue(summary, script_config)
            logger.info(f"Generated script with {len(dialogue_lines)} dialogue lines")
        
        # Save script to file
        script_path = output_path.parent / f"{pdf_path.stem}_script.txt"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(f"# Podcast Script - {pdf_path.name}\n\n")
            f.write(f"Language: {language_code}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total lines: {len(dialogue_lines)}\n")
            f.write("="*50 + "\n\n")
            
            for i, line in enumerate(dialogue_lines):
                f.write(f"[{i+1:02d}] {line.speaker.value}: {line.text}\n")
                f.write("\n")
        logger.info(f"Script saved to: {script_path}")
        
        # Step 5: Choose TTS backend(s)
        if args.dual_tts:
            # Check if both OpenAI and ElevenLabs are available
            available_backends = get_available_tts_backends()
            if 'openai' not in available_backends:
                logger.error("OpenAI TTS not available. Check OPENAI_API_KEY environment variable.")
                sys.exit(1)
            if 'eleven' not in available_backends:
                logger.error("ElevenLabs TTS not available. Check ELEVENLABS_API_KEY environment variable.")
                sys.exit(1)
            
            tts_backends = ['openai', 'eleven']
            logger.info("Dual TTS mode: Will generate podcasts using both OpenAI and ElevenLabs")
        else:
            if args.tts == 'auto':
                available_backends = get_available_tts_backends()
                if not available_backends:
                    logger.error("No TTS backends available. Check your API keys.")
                    sys.exit(1)
                
                # Prefer ElevenLabs for quality, then Gemini, fall back to OpenAI, then others
                if 'eleven' in available_backends:
                    tts_backend_name = 'eleven'
                elif 'gemini' in available_backends:
                    tts_backend_name = 'gemini'
                    logger.info("ElevenLabs not available, using Gemini TTS as fallback")
                elif 'openai' in available_backends:
                    tts_backend_name = 'openai'
                    logger.info("ElevenLabs and Gemini not available, using OpenAI TTS as fallback")
                else:
                    tts_backend_name = available_backends[0]
                    logger.info(f"Using {tts_backend_name} TTS as fallback")
            else:
                tts_backend_name = args.tts
            
            tts_backends = [tts_backend_name]
            logger.info(f"Using TTS backend: {tts_backend_name}")
        
        # Step 6: Process TTS synthesis for each backend
        synthesis_results = []
        all_output_files = []
        total_cost_all = 0.0
        
        for backend in tts_backends:
            logger.info(f"Processing TTS synthesis with {backend}")
            synthesis_result = process_tts_synthesis(backend, dialogue_lines, language_code, args.voices, output_path, logger)
            synthesis_results.append(synthesis_result)
            all_output_files.append(synthesis_result['output_path'])
            total_cost_all += synthesis_result['total_cost']
        
        # Log final pipeline metrics
        final_metrics = {
            'input_file': str(pdf_path),
            'output_files': [str(file) for file in all_output_files],
            'language': language_code,
            'language_confidence': confidence,
            'tts_backends': tts_backends,
            'text_chars': len(text),
            'analysis_chars': len(summary),
            'analysis_words': len(summary.split()),
            'script_lines': len(dialogue_lines),
            'audio_segments': sum(len(result['audio_segments']) for result in synthesis_results),
            'final_duration_seconds': max(result['mix_result']['duration_seconds'] for result in synthesis_results),
            'final_file_size_bytes': max(result['mix_result']['file_size_bytes'] for result in synthesis_results),
            'estimated_cost': total_cost_all
        }
        log_pipeline_metrics("pipeline_complete", final_metrics, logger)
        
        # Save comprehensive metadata file
        metadata_path = all_output_files[0].parent / f"{all_output_files[0].stem}_metadata.json"
        metadata = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'input_file': str(pdf_path),
                'input_file_size': pdf_path.stat().st_size
            },
            'content_analysis': {
                'language': language_code,
                'language_confidence': confidence,
                'original_text_chars': len(text),
                'original_text_words': len(text.split()),
                'analysis_chars': len(summary),
                'analysis_words': len(summary.split()),
                'compression_ratio': len(text) / len(summary)
            },
            'script_generation': {
                'target_words': args.target_words,
                'actual_words': sum(len(line.text.split()) for line in dialogue_lines),
                'dialogue_lines': len(dialogue_lines),
                'host_lines': len([l for l in dialogue_lines if l.speaker.value == 'HOST']),
                'guest_lines': len([l for l in dialogue_lines if l.speaker.value == 'GUEST']),
            },
            'tts_synthesis': {
                'backends_used': tts_backends,
                'voices_used': [result['voices'] for result in synthesis_results],
                'total_characters': sum(len(line.text) for line in dialogue_lines),
                'estimated_cost': total_cost_all
            },
            'output_files': {
                'podcasts': [str(file) for file in all_output_files],
                'analysis': str(summary_path),
                'script': str(script_path),
                'metadata': str(metadata_path),
                'duration_seconds': max(result['mix_result']['duration_seconds'] for result in synthesis_results),
                'file_size_bytes': max(result['mix_result']['file_size_bytes'] for result in synthesis_results)
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        logger.info("TTS Podcast Pipeline completed successfully!")
        
        # Print analysis of all generated files
        logger.info("📁 Generated files:")
        logger.info(f"  📄 Analysis: {summary_path}")
        logger.info(f"  📝 Script: {script_path}")
        for file in all_output_files:
            logger.info(f"  🎧 Podcast: {file}")
        logger.info(f"  📊 Metadata: {metadata_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 