#!/usr/bin/env python3
"""
Regenerate Podcast from Existing Script

This script reads an existing podcast script and regenerates the audio
with custom speaker volume adjustments.
"""
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from script_gen import DialogueLine, SpeakerRole
from tts.base import TTSConfig
from tts.eleven import ElevenLabsTTSBackend
from tts.openai import OpenAITTSBackend
from tts.google import GoogleCloudTTSBackend
from audio_mixer import AudioMixer, AudioSegmentInfo, MixingConfig
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
        
        # Look for dialogue lines that start with [XX] HOST: or [XX] GUEST:
        match = re.match(r'^\[(\d+)\]\s+(HOST|GUEST):\s+(.+)', line)
        if match:
            line_num, speaker, text = match.groups()
            
            dialogue_line = DialogueLine(
                speaker=SpeakerRole(speaker),
                text=text.strip()
            )
            
            dialogue_lines.append(dialogue_line)
            
            i = i + 1
        else:
            i += 1
    
    logger.info(f"Parsed {len(dialogue_lines)} dialogue lines from script")
    return dialogue_lines

def create_tts_backend(backend_name: str, config: TTSConfig):
    """Factory function to create TTS backend instances"""
    
    if backend_name == 'eleven':
        return ElevenLabsTTSBackend(config)
    elif backend_name == 'openai':
        return OpenAITTSBackend(config)
    elif backend_name == 'google':
        return GoogleCloudTTSBackend(config)
    else:
        raise ValueError(f"Unknown TTS backend: {backend_name}")

def get_available_tts_backends() -> List[str]:
    """Get list of available TTS backends"""
    backends = []
    
    if os.getenv('ELEVENLABS_API_KEY'):
        backends.append('eleven')
    if os.getenv('OPENAI_API_KEY'):
        backends.append('openai')
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        backends.append('google')
    
    return backends

def main():
    """Main function to regenerate podcast"""
    parser = argparse.ArgumentParser(
        description="Regenerate podcast from existing script with speaker volume adjustments"
    )
    
    parser.add_argument('script_path', help='Path to existing podcast script file')
    parser.add_argument('--tts', choices=['eleven', 'openai', 'google', 'auto'], 
                       default='auto', help='TTS backend to use (default: auto-select)')
    parser.add_argument('--output', '-o', help='Output MP3 file path')
    parser.add_argument('--male-volume', type=float, default=1.1, 
                       help='Volume multiplier for male speaker (default: 1.1 = 10% louder)')
    parser.add_argument('--female-volume', type=float, default=1.0,
                       help='Volume multiplier for female speaker (default: 1.0)')
    parser.add_argument('--male-speed', type=float, default=1.15,
                       help='Speed multiplier for male speaker (default: 1.15 to match ElevenLabs settings)')
    parser.add_argument('--female-speed', type=float, default=1.12,
                       help='Speed multiplier for female speaker (default: 1.12 to match ElevenLabs settings)')
    parser.add_argument('--voices', nargs=2, metavar=('FEMALE', 'MALE'),
                       help='Voice IDs for female host and male guest')
    parser.add_argument('--language', default='vi', help='Language code (default: vi for Vietnamese)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input
        script_path = Path(args.script_path)
        if not script_path.exists():
            logger.error(f"Script file not found: {script_path}")
            sys.exit(1)
        
        # Setup output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = script_path.parent / f"{script_path.stem}_regenerated.mp3"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Regenerating podcast from script: {script_path}")
        logger.info(f"Male speaker volume: {args.male_volume:.1%} ({args.male_volume}x)")
        logger.info(f"Female speaker volume: {args.female_volume:.1%} ({args.female_volume}x)")
        logger.info(f"Male speaker speed: {args.male_speed:.1%} ({args.male_speed}x)")
        logger.info(f"Female speaker speed: {args.female_speed:.1%} ({args.female_speed}x)")
        
        # Parse script file
        dialogue_lines = parse_script_file(script_path)
        if not dialogue_lines:
            logger.error("No dialogue lines found in script file")
            sys.exit(1)
        
        # Choose TTS backend
        if args.tts == 'auto':
            available_backends = get_available_tts_backends()
            if not available_backends:
                logger.error("No TTS backends available. Check your API keys.")
                sys.exit(1)
            
            # Prefer ElevenLabs for Vietnamese, fall back to others
            if 'eleven' in available_backends:
                tts_backend_name = 'eleven'
            elif 'openai' in available_backends:
                tts_backend_name = 'openai'
            else:
                tts_backend_name = available_backends[0]
        else:
            tts_backend_name = args.tts
        
        logger.info(f"Using TTS backend: {tts_backend_name}")
        
        # Set voices based on TTS backend
        if args.voices:
            female_voice, male_voice = args.voices
        else:
            # Default voices for different backends (Vietnamese)
            # Using ElevenLabs Eleven Turbo v2.5 model which supports Vietnamese and 32 languages
            if tts_backend_name == 'eleven':
                female_voice = "MF3mGyEYCl7XYWbV9V6O"  # Elli (Vietnamese female voice)
                male_voice = "M0rVwr32hdQ5UXpkI3ni"    # Vietnamese male voice (The Hao)
            elif tts_backend_name == 'openai':
                female_voice = "shimmer"  # OpenAI female voice
                male_voice = "onyx"       # OpenAI male voice
            elif tts_backend_name == 'google':
                female_voice = "vi-VN-Neural2-A"  # Vietnamese female voice
                male_voice = "vi-VN-Neural2-D"    # Vietnamese male voice
            else:
                female_voice = "default_female"
                male_voice = "default_male"
        
        logger.info(f"Using voices - Female: {female_voice}, Male: {male_voice}")
        
        # Create TTS backend
        dummy_config = TTSConfig(voice_id=female_voice)
        tts_backend = create_tts_backend(tts_backend_name, dummy_config)
        
        # Synthesize audio for each dialogue line
        audio_segments = []
        total_cost = 0.0
        
        with PipelineTimer("Audio synthesis", logger):
            for i, line in enumerate(dialogue_lines):
                logger.debug(f"Synthesizing line {i+1}/{len(dialogue_lines)}: {line.speaker.value}")
                
                # Determine voice, speaker type, and speed for this line
                if line.speaker.value == "HOST":
                    voice_id = female_voice
                    speaker_type = "female"
                    speaking_speed = args.female_speed
                else:
                    voice_id = male_voice
                    speaker_type = "male"
                    speaking_speed = args.male_speed
                
                # Create TTS config for this line with speed control
                if line.speaker.value == "HOST":
                    # Female voice settings (Elli)
                    tts_config = TTSConfig(
                        voice_id=voice_id,
                        speed=speaking_speed,  # Use female speed (default 1.12)
                        stability=0.5,  # 50% stability for female voice
                        similarity_boost=0.75,  # 75% similarity boost
                        style=0.0  # 0% style
                    )
                else:
                    # Male voice settings (The Hao)
                    tts_config = TTSConfig(
                        voice_id=voice_id,
                        speed=speaking_speed,  # Use male speed (default 1.15)
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
                    speaker=speaker_type,  # Use speaker type for volume adjustment
                    text=line.text
                )
                
                audio_segments.append(segment_info)
                total_cost += result.cost_estimate
                
                logger.debug(f"Synthesized {result.duration_seconds:.1f}s audio")
        
        logger.info(f"Synthesized {len(audio_segments)} audio segments, estimated cost: ${total_cost:.3f}")
        
        # Mix final audio with speaker volume adjustments
        with PipelineTimer("Audio mixing", logger):
            mixer_config = MixingConfig(
                speaker_volumes={
                    "female": args.female_volume,
                    "male": args.male_volume
                }
            )
            mixer = AudioMixer(mixer_config)
            
            mix_result = mixer.mix_segments(audio_segments, output_path)
            
            logger.info(f"Podcast regenerated successfully: {output_path}")
            logger.info(f"Duration: {mix_result['duration_seconds']:.1f}s")
            logger.info(f"File size: {mix_result['file_size_bytes']/1024/1024:.1f}MB")
        
        logger.info("Podcast regeneration completed successfully!")
        
    except Exception as e:
        logger.error(f"Podcast regeneration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 