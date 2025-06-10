#!/usr/bin/env python3
"""
Regenerate Podcast using Multi-Speaker TTS (Gemini Only)

This script reads an existing podcast script and regenerates the audio
using true multi-speaker TTS models that can generate multiple speakers
in a single API call.
"""
import argparse
import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Optional, Dict
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from script_gen import DialogueLine, SpeakerRole
from gemini_pipeline import create_gemini_pipeline
from utils.log_cfg import JSONFormatter, PipelineTimer, log_pipeline_metrics


def setup_logging(log_level: str = "INFO"):
    """Setup structured logging"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)


def parse_script_file(script_path: Path) -> List[DialogueLine]:
    """Parse the podcast script file and return DialogueLine objects"""
    logger = logging.getLogger(__name__)
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    dialogue_lines = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and headers
        if not line or line.startswith('#') or line.startswith('==') or line.startswith('Language:') or line.startswith('Generated:') or line.startswith('Total lines:'):
            i += 1
            continue
        
        # Look for dialogue pattern: [XX] SPEAKER: text
        dialogue_match = re.match(r'^\[(\d+)\]\s+(HOST|GUEST):\s*(.+)$', line)
        if dialogue_match:
            line_num, speaker_str, text = dialogue_match.groups()
            
            # Convert speaker string to enum
            speaker = SpeakerRole.HOST if speaker_str == "HOST" else SpeakerRole.GUEST
            
            # Clean up the text
            text = text.strip()
            
            dialogue_line = DialogueLine(speaker=speaker, text=text)
            dialogue_lines.append(dialogue_line)
            
            logger.debug(f"Parsed line {line_num}: {speaker_str} - {len(text)} chars")
        
        i += 1
    
    logger.info(f"Parsed {len(dialogue_lines)} dialogue lines from script")
    return dialogue_lines


def generate_output_filename(script_path: Path, output_path: Optional[str], model: str) -> Path:
    """Generate output filename with model suffix"""
    if output_path:
        output_file = Path(output_path)
        # If no extension, add .mp3
        if not output_file.suffix:
            output_file = output_file.with_suffix('.mp3')
    else:
        # Auto-generate from script name
        script_stem = script_path.stem
        if script_stem.endswith('_script'):
            script_stem = script_stem[:-7]  # Remove '_script' suffix
        
        output_file = script_path.parent / f"{script_stem}_{model}_multispeaker.mp3"
    
    return output_file


def generate_metadata_filename(audio_path: Path) -> Path:
    """Generate metadata filename from audio path"""
    return audio_path.with_suffix('.json')


def list_available_voices(model: str) -> None:
    """List available voices for the specified model"""
    logger = logging.getLogger(__name__)
    
    if model != "gemini":
        logger.error(f"Multi-speaker voice listing not supported for model: {model}")
        sys.exit(1)
    
    try:
        # Create pipeline to access voice info
        pipeline = create_gemini_pipeline()
        voices = pipeline.get_available_voices()
        
        print(f"\nAvailable Gemini TTS voices ({len(voices)}):")
        print("=" * 50)
        
        for voice in voices:
            print(f"ID: {voice['id']}")
            print(f"  Name: {voice['name']}")
            print(f"  Characteristic: {voice.get('characteristic', 'N/A')}")
            print(f"  Energy: {voice.get('energy', 'N/A')}")
            print(f"  Vietnamese Suitability: {voice.get('vietnamese_suitability', 'N/A')}")
            print(f"  Podcast Role: {voice.get('podcast_role', 'N/A')}")
            if voice.get('description'):
                print(f"  Description: {voice['description']}")
            print()
        
        # Show default voices
        defaults = pipeline.get_default_voices()
        print(f"Default voices:")
        print(f"  Host: {defaults['host']}")
        print(f"  Guest: {defaults['guest']}")
        
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        sys.exit(1)


def main():
    """Main function to regenerate podcast using multi-speaker TTS"""
    parser = argparse.ArgumentParser(
        description="Regenerate podcast from existing script using multi-speaker TTS models"
    )
    
    parser.add_argument('script_path', nargs='?', help='Path to existing podcast script file')
    parser.add_argument('--model', choices=['gemini'], required=True,
                       help='Multi-speaker TTS model to use (currently only gemini supported)')
    parser.add_argument('--output', '-o', help='Output MP3 file path (auto-generated if not specified)')
    parser.add_argument('--host-voice', help='Voice ID for HOST speaker (default: Zephyr)')
    parser.add_argument('--guest-voice', help='Voice ID for GUEST speaker (default: Rasalgethi)')
    parser.add_argument('--style-prompt', 
                        default=(
                        "Both speakers should speak in the Northern Vietnamese dialect, "
                        "sounding very enthusiastic and youthful. Include natural conversational elements like excitement, curiosity, and occasional laughter. "
                        "The host should be more energetic and guiding, while the guest should be responsive and engaged. "
                        "In the script, emotion tags in square brackets '[]' are used to indicate emotions; do NOT read them out loud."
                ),
                        help='Global style instructions for both speakers (default: engaging Vietnamese podcast style)')
    parser.add_argument('--list-voices', action='store_true',
                       help='List available voices for the specified model and exit')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--temperature', type=float, default=1.5, 
                       help='Temperature for TTS generation (0.0-2.0, default: 1.5, higher = more variation)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # List voices if requested
    if args.list_voices:
        list_available_voices(args.model)
        return
    
    # Validate script path (required if not listing voices)
    if not args.script_path:
        logger.error("Script path is required when not using --list-voices")
        sys.exit(1)
    
    script_path = Path(args.script_path)
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        sys.exit(1)
    
    logger.info(f"Regenerating podcast using multi-speaker TTS: {args.model}")
    logger.info(f"Script file: {script_path}")
    
    try:
        with PipelineTimer("Multi-speaker podcast regeneration", logger):
            
            # Parse script file
            dialogue_lines = parse_script_file(script_path)
            if not dialogue_lines:
                logger.error("No dialogue lines found in script file")
                sys.exit(1)
            
            # Generate output paths
            output_file = generate_output_filename(script_path, args.output, args.model)
            metadata_file = generate_metadata_filename(output_file)
            
            logger.info(f"Output file: {output_file}")
            logger.info(f"Metadata file: {metadata_file}")
            
            # Create multi-speaker pipeline
            if args.model == "gemini":
                pipeline = create_gemini_pipeline(
                    host_voice=args.host_voice,
                    guest_voice=args.guest_voice,
                    style_prompt=args.style_prompt,
                    temperature=args.temperature
                )
                
                logger.info(f"Using voices - Host: {pipeline.host_voice}, Guest: {pipeline.guest_voice}")
                if args.style_prompt:
                    logger.info(f"Style prompt: {args.style_prompt[:100]}...")
            else:
                logger.error(f"Unsupported multi-speaker model: {args.model}")
                sys.exit(1)
            
            # Process script using multi-speaker pipeline
            result = pipeline.process_script(dialogue_lines)
            
            # Save audio file
            pipeline.save_audio(result, output_file)
            
            # Save metadata
            pipeline.save_metadata(result, metadata_file)
            
            # Summary
            logger.info(f"Multi-speaker podcast regenerated successfully: {output_file}")
            logger.info(f"Duration: {result.duration_seconds:.1f}s")
            logger.info(f"File size: {len(result.audio_data) / 1024 / 1024:.1f}MB")
            logger.info(f"Estimated cost: ${result.cost_estimate:.3f}")
            logger.info(f"Character count: {result.character_count:,}")
            logger.info(f"Model: {args.model} (multi-speaker)")
            
            # Final metrics
            final_metrics = {
                'script_file': str(script_path),
                'output_file': str(output_file),
                'model': args.model,
                'pipeline_type': 'multi_speaker',
                'dialogue_lines': len(dialogue_lines),
                'character_count': result.character_count,
                'duration_seconds': result.duration_seconds,
                'file_size_bytes': len(result.audio_data),
                'cost_estimate': result.cost_estimate,
                'host_voice': pipeline.host_voice,
                'guest_voice': pipeline.guest_voice,
                'style_prompt_length': len(pipeline.style_prompt) if pipeline.style_prompt else 0,
                'temperature': args.temperature
            }
            log_pipeline_metrics("multi_speaker_podcast_regeneration", final_metrics, logger)
            
            logger.info("Multi-speaker podcast regeneration completed successfully!")
            
    except Exception as e:
        logger.error(f"Multi-speaker podcast regeneration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()