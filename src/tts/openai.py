"""
OpenAI TTS backend for TTS Podcast Pipeline
"""
import logging
import os
from typing import Optional, Dict, Any
import tempfile
from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseTTSBackend, TTSConfig, SynthesisResult, AudioFormat
from ..utils.log_cfg import PipelineTimer

logger = logging.getLogger(__name__)


class OpenAITTSBackend(BaseTTSBackend):
    """OpenAI TTS backend implementation"""
    
    # Available voices from OpenAI TTS API
    VOICE_MAPPINGS = {
        'alloy': {'gender': 'neutral', 'description': 'Balanced, natural voice'},
        'echo': {'gender': 'male', 'description': 'Deep, resonant male voice'},
        'fable': {'gender': 'neutral', 'description': 'Expressive storytelling voice'},
        'onyx': {'gender': 'male', 'description': 'Strong, authoritative male voice'},
        'nova': {'gender': 'female', 'description': 'Bright, energetic female voice'},
        'shimmer': {'gender': 'female', 'description': 'Warm, friendly female voice'}
    }
    
    # Cost per 1000 characters (as of 2024)
    COST_PER_1K_CHARS = 0.015
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Set default voices if not specified
        if not config.female_voice:
            config.female_voice = 'shimmer'  # Default female voice
        if not config.male_voice:
            config.male_voice = 'onyx'      # Default male voice
            
        self.config = config
        
        logger.info(f"Initialized OpenAI TTS backend with voices: {config.female_voice}, {config.male_voice}")
    
    def get_available_voices(self, language: Optional[str] = None) -> Dict[str, Any]:
        """Get available voices for the language"""
        
        # OpenAI TTS voices work with all supported languages
        return {
            voice_id: {
                'id': voice_id,
                'name': voice_id.title(),
                'gender': info['gender'],
                'description': info['description'],
                'language_support': 'multilingual'
            }
            for voice_id, info in self.VOICE_MAPPINGS.items()
        }
    
    def validate_voice(self, voice_id: str, language: Optional[str] = None) -> bool:
        """Validate if voice exists and supports the language"""
        return voice_id in self.VOICE_MAPPINGS
    
    def estimate_cost(self, text: str, voice_id: str) -> float:
        """Estimate synthesis cost"""
        char_count = len(text)
        return (char_count / 1000) * self.COST_PER_1K_CHARS
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    def _synthesize_with_retry(self, text: str, voice_id: str, output_format: str) -> bytes:
        """Make TTS API call with retry logic"""
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1",  # Use standard quality model
                voice=voice_id,
                input=text,
                response_format=output_format
            )
            
            return response.content
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise ValueError("Invalid OpenAI API key")
        except openai.BadRequestError as e:
            logger.error(f"OpenAI bad request: {e}")
            raise ValueError(f"Invalid request: {e}")
        except Exception as e:
            logger.error(f"OpenAI TTS API error: {e}")
            raise
    
    def synthesize(self, text: str, voice_id: str, **kwargs) -> SynthesisResult:
        """
        Synthesize speech using OpenAI TTS
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            **kwargs: Additional parameters
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not self.validate_voice(voice_id):
            raise ValueError(f"Invalid voice ID: {voice_id}")
        
        # Clean and prepare text
        clean_text = self.clean_text_for_synthesis(text)
        
        # Validate text length (OpenAI has a 4096 character limit)
        if len(clean_text) > 4000:
            logger.warning(f"Text length {len(clean_text)} exceeds recommended limit, truncating")
            clean_text = clean_text[:4000] + "..."
        
        with PipelineTimer(f"OpenAI TTS synthesis ({len(clean_text)} chars)", logger):
            try:
                # Synthesize audio
                audio_format = "mp3"  # OpenAI supports mp3, opus, aac, flac
                audio_data = self._synthesize_with_retry(clean_text, voice_id, audio_format)
                
                if not audio_data:
                    raise ValueError("Received empty audio data from OpenAI")
                
                # Estimate duration (rough estimate: ~150 chars per second for speech)
                estimated_duration = len(clean_text) / 150.0
                
                # Calculate cost
                cost = self.estimate_cost(clean_text, voice_id)
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=AudioFormat.MP3,
                    sample_rate=24000,  # OpenAI TTS default sample rate
                    duration_seconds=estimated_duration,
                    voice_id=voice_id,
                    text=clean_text,
                    cost_estimate=cost,
                    metadata={
                        'backend': 'openai',
                        'model': 'tts-1',
                        'voice_gender': self.VOICE_MAPPINGS[voice_id]['gender'],
                        'characters_processed': len(clean_text)
                    }
                )
                
                logger.info(f"OpenAI synthesis completed: {estimated_duration:.1f}s, ${cost:.4f}")
                return result
                
            except Exception as e:
                logger.error(f"OpenAI TTS synthesis failed: {e}")
                raise
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about this TTS backend"""
        return {
            'name': 'OpenAI TTS',
            'provider': 'OpenAI',
            'models': ['tts-1', 'tts-1-hd'],
            'supported_formats': ['mp3', 'opus', 'aac', 'flac'],
            'supported_languages': [
                'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi', 'tr', 'pl', 'nl'
            ],
            'max_text_length': 4096,
            'sample_rate': 24000,
            'cost_per_1k_chars': self.COST_PER_1K_CHARS,
            'available_voices': list(self.VOICE_MAPPINGS.keys()),
            'voice_details': self.VOICE_MAPPINGS
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the backend is healthy and accessible"""
        try:
            # Try a minimal synthesis to test connectivity
            test_audio = self._synthesize_with_retry("Test", "alloy", "mp3")
            
            return {
                'status': 'healthy',
                'backend': 'openai',
                'api_accessible': True,
                'test_synthesis': len(test_audio) > 0
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'openai',
                'error': str(e),
                'api_accessible': False
            } 