"""
Google Cloud Text-to-Speech backend implementation
Provides high-quality TTS synthesis using Google Cloud TTS API
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from google.cloud import texttospeech
from google.oauth2 import service_account
import google.api_core.exceptions

from .base import BaseTTSBackend, TTSConfig, SynthesisResult, AudioFormat, TTSError, QuotaExceededError
from utils.log_cfg import PipelineTimer, log_pipeline_metrics

logger = logging.getLogger(__name__)


class GoogleCloudTTSBackend(BaseTTSBackend):
    """Google Cloud Text-to-Speech backend implementation"""
    
    # Popular Google Cloud TTS voices
    POPULAR_VOICES = {
        'en-US': {
            'female': ['en-US-Wavenet-C', 'en-US-Wavenet-F', 'en-US-Neural2-C', 'en-US-Neural2-F'],
            'male': ['en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Neural2-A', 'en-US-Neural2-D']
        },
        'en-GB': {
            'female': ['en-GB-Wavenet-A', 'en-GB-Wavenet-C', 'en-GB-Neural2-A', 'en-GB-Neural2-C'],
            'male': ['en-GB-Wavenet-B', 'en-GB-Wavenet-D', 'en-GB-Neural2-B', 'en-GB-Neural2-D']
        },
        'en-AU': {
            'female': ['en-AU-Wavenet-A', 'en-AU-Wavenet-C', 'en-AU-Neural2-A', 'en-AU-Neural2-C'],
            'male': ['en-AU-Wavenet-B', 'en-AU-Wavenet-D', 'en-AU-Neural2-B', 'en-AU-Neural2-D']
        }
    }
    
    def __init__(self, config: TTSConfig):
        # Setup authentication
        self.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not self.credentials_path:
            raise ValueError(
                "Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable to point to your service account JSON file."
            )
        
        # Verify credentials file exists
        if not os.path.exists(self.credentials_path):
            raise ValueError(f"Credentials file not found: {self.credentials_path}")
        
        super().__init__(api_key=None)  # Google uses credentials file, not API key
        
        self.config = config
        
        # Initialize the Google Cloud TTS client
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info("Google Cloud TTS client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Google Cloud TTS client: {e}")
        
        # Cache for voices to reduce API calls
        self._voices_cache = None
        self._cache_timestamp = 0
        self._cache_duration = 3600  # 1 hour
    
    def get_max_text_length(self) -> int:
        """Google Cloud TTS supports up to 5000 characters"""
        return 5000
    
    def supports_ssml(self) -> bool:
        """Google Cloud TTS supports SSML"""
        return True
    
    def synthesize(self, text: str, config: TTSConfig) -> SynthesisResult:
        """
        Synthesize text using Google Cloud TTS API
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        self.validate_text(text)
        
        with PipelineTimer(f"Google Cloud TTS synthesis ({len(text)} chars)", logger):
            
            # Clean and prepare text
            clean_text = self.clean_text(text)
            
            # Detect if text contains SSML
            has_ssml = '<' in clean_text and '>' in clean_text
            
            try:
                logger.info(f"Synthesizing with Google Cloud TTS: voice={config.voice_id}, chars={len(clean_text)}, ssml={has_ssml}")
                
                # Parse voice ID to get language and voice name
                language_code, voice_name = self._parse_voice_id(config.voice_id)
                
                # Set up the synthesis input - use SSML if markup was added
                if has_ssml:
                    # Wrap in SSML speak tags if not already present
                    if not clean_text.strip().startswith('<speak>'):
                        clean_text = f'<speak>{clean_text}</speak>'
                    synthesis_input = texttospeech.SynthesisInput(ssml=clean_text)
                else:
                    synthesis_input = texttospeech.SynthesisInput(text=clean_text)
                
                # Configure voice
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    name=voice_name
                )
                
                # Configure audio output
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=config.speed,
                    pitch=config.pitch,
                    sample_rate_hertz=config.sample_rate
                )
                
                # Perform the text-to-speech request
                response = self.client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
                
                audio_data = response.audio_content
                
                if not audio_data:
                    raise TTSError("Received empty audio data from Google Cloud TTS")
                
                # Estimate duration (rough calculation: ~150 chars per minute) - use original text length
                estimated_duration = len(clean_text) / 150 * 60
                
                # Estimate cost (Google Cloud TTS pricing: ~$4.00 per 1M characters for WaveNet)
                # Neural2 voices are ~$16.00 per 1M characters - use original text length
                if 'Neural2' in voice_name or 'neural2' in voice_name.lower():
                    cost_estimate = len(clean_text) / 1000000 * 16.00
                elif 'Wavenet' in voice_name or 'wavenet' in voice_name.lower():
                    cost_estimate = len(clean_text) / 1000000 * 4.00
                else:
                    cost_estimate = len(clean_text) / 1000000 * 4.00  # Default to WaveNet pricing
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=AudioFormat.MP3,
                    duration_seconds=estimated_duration,
                    sample_rate=config.sample_rate,
                    character_count=len(clean_text),
                    cost_estimate=cost_estimate,
                    metadata={
                        "voice_id": config.voice_id,
                        "language_code": language_code,
                        "voice_name": voice_name,
                        "speaking_rate": config.speed,
                        "pitch": config.pitch,
                        "backend": "google_cloud",
                        "ssml_used": has_ssml
                    }
                )
                
                # Log metrics
                metrics = {
                    'voice_id': config.voice_id,
                    'language_code': language_code,
                    'voice_name': voice_name,
                    'character_count': len(clean_text),
                    'audio_size_bytes': len(audio_data),
                    'estimated_duration': estimated_duration,
                    'cost_estimate': cost_estimate,
                    'speaking_rate': config.speed,
                    'pitch': config.pitch,
                    'ssml_used': has_ssml
                }
                log_pipeline_metrics("google_cloud_synthesis", metrics, logger)
                
                logger.info(f"Google Cloud TTS synthesis completed: {len(audio_data)} bytes, ~{estimated_duration:.1f}s, ${cost_estimate:.4f}")
                
                return result
                
            except google.api_core.exceptions.ResourceExhausted as e:
                logger.error(f"Google Cloud TTS quota exceeded: {e}")
                raise QuotaExceededError("Google Cloud TTS quota exceeded")
            except google.api_core.exceptions.PermissionDenied as e:
                logger.error(f"Google Cloud TTS permission denied: {e}")
                raise TTSError("Google Cloud TTS permission denied - check credentials and API enablement")
            except Exception as e:
                logger.error(f"Google Cloud TTS synthesis failed: {e}")
                raise TTSError(f"Google Cloud TTS synthesis failed: {e}")
    
    def _parse_voice_id(self, voice_id: str) -> tuple[str, str]:
        """
        Parse voice ID to extract language code and voice name
        
        Args:
            voice_id: Full voice ID like 'en-US-Wavenet-C' or just 'en-US-Wavenet-C'
            
        Returns:
            Tuple of (language_code, voice_name)
        """
        # Handle different voice ID formats
        if '-' in voice_id and len(voice_id.split('-')) >= 3:
            # Full voice ID like 'en-US-Wavenet-C'
            parts = voice_id.split('-')
            language_code = f"{parts[0]}-{parts[1]}"
            voice_name = voice_id
        else:
            # Assume it's just a voice name, use default language
            language_code = "en-US"
            voice_name = f"{language_code}-{voice_id}" if not voice_id.startswith('en-') else voice_id
        
        return language_code, voice_name
    
    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available voices from Google Cloud TTS
        
        Args:
            language: Optional language filter (e.g., 'en-US')
            
        Returns:
            List of voice dictionaries
        """
        # Check cache
        current_time = time.time()
        if (self._voices_cache and 
            current_time - self._cache_timestamp < self._cache_duration):
            voices = self._voices_cache
        else:
            # Fetch voices from API
            try:
                response = self.client.list_voices()
                voices = []
                
                for voice in response.voices:
                    for language_code in voice.language_codes:
                        voice_info = {
                            'id': voice.name,
                            'name': voice.name,
                            'language': language_code,
                            'gender': voice.ssml_gender.name.lower(),
                            'type': self._get_voice_type(voice.name),
                            'quality': self._get_voice_quality(voice.name)
                        }
                        voices.append(voice_info)
                
                # Update cache
                self._voices_cache = voices
                self._cache_timestamp = current_time
                
            except Exception as e:
                logger.error(f"Failed to fetch Google Cloud TTS voices: {e}")
                return []
        
        # Filter by language if specified
        if language:
            voices = [v for v in voices if v['language'].startswith(language)]
        
        return voices
    
    def _get_voice_type(self, voice_name: str) -> str:
        """Determine voice type from name"""
        if 'Neural2' in voice_name:
            return 'Neural2'
        elif 'Wavenet' in voice_name:
            return 'WaveNet'
        elif 'Standard' in voice_name:
            return 'Standard'
        else:
            return 'Unknown'
    
    def _get_voice_quality(self, voice_name: str) -> str:
        """Determine voice quality from name"""
        if 'Neural2' in voice_name:
            return 'Premium'
        elif 'Wavenet' in voice_name:
            return 'High'
        else:
            return 'Standard'
    
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Dictionary with voice information
        """
        voices = self.get_available_voices()
        
        for voice in voices:
            if voice['id'] == voice_id or voice['name'] == voice_id:
                return voice
        
        # If not found in cache, return basic info
        language_code, voice_name = self._parse_voice_id(voice_id)
        return {
            'id': voice_id,
            'name': voice_name,
            'language': language_code,
            'gender': 'unknown',
            'type': self._get_voice_type(voice_name),
            'quality': self._get_voice_quality(voice_name)
        }
    
    def estimate_cost(self, text: str, config: TTSConfig) -> float:
        """
        Estimate cost for synthesis in USD
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            Estimated cost in USD
        """
        char_count = len(text)
        voice_name = config.voice_id
        
        # Google Cloud TTS pricing (as of 2024)
        if 'Neural2' in voice_name or 'neural2' in voice_name.lower():
            return char_count / 1000000 * 16.00  # $16 per 1M characters
        elif 'Wavenet' in voice_name or 'wavenet' in voice_name.lower():
            return char_count / 1000000 * 4.00   # $4 per 1M characters
        else:
            return char_count / 1000000 * 4.00   # Default to WaveNet pricing
    
    def get_service_limits(self) -> Dict[str, Any]:
        """
        Get Google Cloud TTS service limits
        
        Returns:
            Dictionary with service limits
        """
        return {
            'max_characters_per_request': 5000,
            'max_requests_per_minute': 1000,  # Generous limit
            'supported_formats': ['mp3', 'wav', 'ogg'],
            'supported_sample_rates': [8000, 16000, 22050, 24000, 32000, 44100, 48000],
            'pricing_per_1m_chars': {
                'Standard': 4.00,
                'WaveNet': 4.00,
                'Neural2': 16.00
            },
            'free_tier_chars_per_month': 1000000,  # 1M chars/month free
            'max_audio_length_seconds': 900,  # 15 minutes
            'api_version': 'v1'
        }
    
    def _detect_gender(self, voice_name: str) -> str: 