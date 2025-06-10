#!/usr/bin/env python3
"""
Gemini Multi-Speaker TTS Pipeline

Dedicated pipeline for Gemini TTS multi-speaker podcast generation.
This pipeline bypasses traditional audio mixing since Gemini generates
complete multi-speaker audio in a single API call.
"""
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from script_gen import DialogueLine, SpeakerRole
from tts.base import TTSConfig, SynthesisResult
from tts.gemini import GeminiTTSBackend
from utils.log_cfg import PipelineTimer, log_pipeline_metrics


@dataclass
class GeminiPipelineResult:
    """Result from Gemini multi-speaker pipeline"""
    audio_data: bytes
    duration_seconds: float
    cost_estimate: float
    character_count: int
    metadata: Dict[str, Any]
    output_path: Optional[Path] = None


class GeminiMultiSpeakerPipeline:
    """Pipeline for Gemini multi-speaker TTS processing"""
    
    def __init__(self, host_voice: str = "Zephyr", guest_voice: str = "Rasalgethi", 
                 style_prompt: str = None, temperature: float = 1.5):
        """
        Initialize Gemini multi-speaker pipeline
        
        Args:
            host_voice: Voice ID for HOST speaker (default: Zephyr)
            guest_voice: Voice ID for GUEST speaker (default: Rasalgethi)
            style_prompt: Global style instructions for both speakers
            temperature: Temperature for generation (0.0-2.0, higher = more variation)
        """
        self.logger = logging.getLogger(__name__)
        self.host_voice = host_voice
        self.guest_voice = guest_voice
        self.style_prompt = style_prompt
        self.temperature = temperature
        
        # Initialize TTS backend
        config = TTSConfig(
            voice_id=host_voice,  # Primary voice (required for init)
            sample_rate=44100,
            speed=1.0
        )
        
        try:
            self.tts_backend = GeminiTTSBackend(config)
            self.logger.info(f"Gemini pipeline initialized: host={host_voice}, guest={guest_voice}")
            if style_prompt:
                self.logger.info(f"Style prompt configured: {style_prompt[:100]}...")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini TTS backend: {e}")
            raise
    
    def validate_voices(self) -> None:
        """Validate that specified voices are available in Gemini TTS"""
        try:
            available_voices = self.tts_backend.get_available_voices()
            available_voice_ids = {voice['id'] for voice in available_voices}
            
            # Check host voice
            if self.host_voice not in available_voice_ids:
                raise ValueError(f"Host voice '{self.host_voice}' not available. "
                               f"Available voices: {sorted(available_voice_ids)}")
            
            # Check guest voice
            if self.guest_voice not in available_voice_ids:
                raise ValueError(f"Guest voice '{self.guest_voice}' not available. "
                               f"Available voices: {sorted(available_voice_ids)}")
            
            self.logger.info(f"Voice validation successful: {self.host_voice}, {self.guest_voice}")
            
        except Exception as e:
            self.logger.error(f"Voice validation failed: {e}")
            raise
    
    def process_script(self, dialogue_lines: List[DialogueLine]) -> GeminiPipelineResult:
        """
        Process dialogue lines using Gemini multi-speaker TTS
        
        Args:
            dialogue_lines: List of parsed dialogue lines
            
        Returns:
            GeminiPipelineResult with audio data and metadata
        """
        if not dialogue_lines:
            raise ValueError("No dialogue lines provided")
        
        # Validate voices before processing
        self.validate_voices()
        
        # Calculate total characters
        total_chars = sum(len(line.text) for line in dialogue_lines)
        
        with PipelineTimer(f"Gemini multi-speaker pipeline ({total_chars} chars)", self.logger):
            
            # Synthesize using multi-speaker API
            try:
                synthesis_result = self.tts_backend.synthesize_multi_speaker(
                    dialogue_lines=dialogue_lines,
                    host_voice=self.host_voice,
                    guest_voice=self.guest_voice,
                    style_prompt=self.style_prompt,
                    temperature=self.temperature
                )
                
                # Create pipeline result
                pipeline_result = GeminiPipelineResult(
                    audio_data=synthesis_result.audio_data,
                    duration_seconds=synthesis_result.duration_seconds,
                    cost_estimate=synthesis_result.cost_estimate,
                    character_count=total_chars,
                    metadata={
                        **synthesis_result.metadata,
                        'pipeline_type': 'gemini_multi_speaker',
                        'dialogue_lines_count': len(dialogue_lines),
                        'host_voice': self.host_voice,
                        'guest_voice': self.guest_voice,
                        'style_prompt': self.style_prompt,
                        'processing_timestamp': time.time(),
                        'audio_size_bytes': len(synthesis_result.audio_data)
                    }
                )
                
                # Log pipeline metrics
                metrics = {
                    'dialogue_lines': len(dialogue_lines),
                    'character_count': total_chars,
                    'duration_seconds': synthesis_result.duration_seconds,
                    'cost_estimate': synthesis_result.cost_estimate,
                    'host_voice': self.host_voice,
                    'guest_voice': self.guest_voice,
                    'style_prompt_length': len(self.style_prompt) if self.style_prompt else 0,
                    'audio_size_bytes': len(synthesis_result.audio_data)
                }
                log_pipeline_metrics("gemini_multi_speaker_pipeline", metrics, self.logger)
                
                self.logger.info(f"Pipeline completed: {len(synthesis_result.audio_data)} bytes, "
                               f"{synthesis_result.duration_seconds:.1f}s, ${synthesis_result.cost_estimate:.3f}")
                
                return pipeline_result
                
            except Exception as e:
                self.logger.error(f"Multi-speaker synthesis failed: {e}")
                raise
    
    def save_audio(self, result: GeminiPipelineResult, output_path: Path) -> None:
        """
        Save audio data to file
        
        Args:
            result: Pipeline result with audio data
            output_path: Path to save audio file
        """
        try:
            with open(output_path, 'wb') as f:
                f.write(result.audio_data)
            
            result.output_path = output_path
            self.logger.info(f"Audio saved to: {output_path}")
            self.logger.info(f"File size: {len(result.audio_data) / 1024 / 1024:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio to {output_path}: {e}")
            raise
    
    def save_metadata(self, result: GeminiPipelineResult, metadata_path: Path) -> None:
        """
        Save metadata to JSON file
        
        Args:
            result: Pipeline result with metadata
            metadata_path: Path to save metadata file
        """
        try:
            metadata = {
                'pipeline_info': {
                    'pipeline_type': 'gemini_multi_speaker',
                    'host_voice': self.host_voice,
                    'guest_voice': self.guest_voice,
                    'processing_timestamp': result.metadata.get('processing_timestamp'),
                    'dialogue_lines_count': result.metadata.get('dialogue_lines_count')
                },
                'audio_info': {
                    'duration_seconds': result.duration_seconds,
                    'file_size_bytes': len(result.audio_data),
                    'file_size_mb': len(result.audio_data) / 1024 / 1024,
                    'character_count': result.character_count,
                    'format': 'mp3',
                    'sample_rate': 44100
                },
                'cost_info': {
                    'estimated_cost_usd': result.cost_estimate,
                    'cost_per_character': result.cost_estimate / result.character_count if result.character_count > 0 else 0
                },
                'tts_metadata': result.metadata,
                'output_file': str(result.output_path) if result.output_path else None
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata to {metadata_path}: {e}")
            raise
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices from Gemini TTS
        
        Returns:
            List of voice dictionaries with metadata
        """
        try:
            return self.tts_backend.get_available_voices(language="vi")
        except Exception as e:
            self.logger.error(f"Failed to get available voices: {e}")
            raise
    
    def get_default_voices(self) -> Dict[str, str]:
        """
        Get default voice configuration
        
        Returns:
            Dictionary with host and guest voice IDs
        """
        return {
            "host": self.host_voice,
            "guest": self.guest_voice
        }


def create_gemini_pipeline(host_voice: str = None, guest_voice: str = None, 
                          style_prompt: str = None, temperature: float = 1.5) -> GeminiMultiSpeakerPipeline:
    """
    Factory function to create Gemini multi-speaker pipeline
    
    Args:
        host_voice: Custom host voice ID (optional)
        guest_voice: Custom guest voice ID (optional)
        style_prompt: Global style instructions (optional)
        temperature: Temperature for generation (0.0-2.0, optional)
        
    Returns:
        Initialized GeminiMultiSpeakerPipeline
    """
    # Use defaults if not specified
    host_voice = host_voice or "Zephyr"
    guest_voice = guest_voice or "Rasalgethi"
    
    return GeminiMultiSpeakerPipeline(
        host_voice=host_voice, 
        guest_voice=guest_voice,
        style_prompt=style_prompt,
        temperature=temperature
    )