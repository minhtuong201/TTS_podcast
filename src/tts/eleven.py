"""
ElevenLabs TTS Backend for high-quality podcast voices
"""
import logging
import os
import time
from typing import Optional, Dict, Any, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

from .base import (
    BaseTTSBackend, TTSConfig, SynthesisResult, AudioFormat,
    TTSError, VoiceNotFoundError, QuotaExceededError, AudioTooLongError
)
from utils.log_cfg import PipelineTimer, log_pipeline_metrics

load_dotenv()
logger = logging.getLogger(__name__)


class ElevenLabsTTSBackend(BaseTTSBackend):
    """ElevenLabs TTS backend implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable.")
        
        super().__init__(api_key=self.api_key)
        
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Cache for voices to reduce API calls
        self._voices_cache = None
        self._cache_timestamp = 0
        self._cache_duration = 3600  # 1 hour
    
    def get_max_text_length(self) -> int:
        """ElevenLabs limit is 5000 characters per request"""
        return 5000
    
    def supports_emotion(self) -> bool:
        """ElevenLabs supports style/emotion control"""
        return True
    
    def supports_ssml(self) -> bool:
        """ElevenLabs supports basic SSML"""
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make API request with retry logic"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(method, url, headers=self.headers, timeout=60, **kwargs)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, will retry...")
                raise requests.exceptions.RequestException("Rate limited")
            
            # Handle quota exceeded
            elif response.status_code == 402:
                raise QuotaExceededError("ElevenLabs quota exceeded")
            
            # Handle authentication errors
            elif response.status_code == 401:
                raise TTSError("Invalid ElevenLabs API key")
            
            # Handle other client errors
            elif 400 <= response.status_code < 500:
                error_msg = response.json().get('detail', {}).get('message', response.text) if response.text else "Client error"
                raise TTSError(f"ElevenLabs API error: {error_msg}")
            
            # Handle server errors (retryable)
            elif response.status_code >= 500:
                logger.warning(f"Server error {response.status_code}, will retry...")
                raise requests.exceptions.RequestException("Server error")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logger.warning("Request timeout, will retry...")
            raise
        except Exception as e:
            if "Rate limited" in str(e) or "Server error" in str(e):
                raise
            logger.error(f"ElevenLabs API request failed: {e}")
            raise TTSError(f"ElevenLabs API request failed: {e}")
    
    def synthesize(self, text: str, config: TTSConfig) -> SynthesisResult:
        """
        Synthesize text using ElevenLabs API
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        self.validate_text(text)
        
        with PipelineTimer(f"ElevenLabs synthesis ({len(text)} chars)", logger):
            
            # Clean and prepare text
            clean_text = self.clean_text(text)
            
            # Add pauses and emotion if supported
            if config.pause_before > 0 or config.pause_after > 0:
                clean_text = self.add_pauses_to_text(clean_text, config.pause_before, config.pause_after)
            
            if config.emotion:
                clean_text = self.add_emotion_to_text(clean_text, config.emotion)
            
            # Prepare request payload
            payload = {
                "text": clean_text,
                "model_id": "eleven_multilingual_v2",  # Best quality model
                "voice_settings": {
                    "stability": config.stability,
                    "similarity_boost": config.similarity_boost,
                    "style": config.style,
                    "use_speaker_boost": True
                }
            }
            
            try:
                logger.info(f"Synthesizing with ElevenLabs: voice={config.voice_id}, chars={len(clean_text)}")
                
                # Make synthesis request
                response = self._make_request(
                    "POST",
                    f"text-to-speech/{config.voice_id}",
                    json=payload
                )
                
                audio_data = response.content
                
                if not audio_data:
                    raise TTSError("Received empty audio data from ElevenLabs")
                
                # Estimate duration (rough calculation: ~150 chars per minute)
                estimated_duration = len(clean_text) / 150 * 60
                
                # Estimate cost (ElevenLabs pricing: ~$0.30 per 1K characters)
                cost_estimate = len(clean_text) / 1000 * 0.30
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=AudioFormat.MP3,  # ElevenLabs returns MP3
                    duration_seconds=estimated_duration,
                    sample_rate=config.sample_rate,
                    character_count=len(clean_text),
                    cost_estimate=cost_estimate,
                    metadata={
                        "voice_id": config.voice_id,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": payload["voice_settings"],
                        "backend": "elevenlabs"
                    }
                )
                
                # Log metrics
                metrics = {
                    'voice_id': config.voice_id,
                    'character_count': len(clean_text),
                    'audio_size_bytes': len(audio_data),
                    'estimated_duration': estimated_duration,
                    'cost_estimate': cost_estimate,
                    'stability': config.stability,
                    'similarity_boost': config.similarity_boost,
                    'style': config.style
                }
                log_pipeline_metrics("elevenlabs_synthesis", metrics, logger)
                
                logger.info(f"ElevenLabs synthesis completed: {len(audio_data)} bytes, ~{estimated_duration:.1f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"ElevenLabs synthesis failed: {e}")
                raise
    
    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available voices from ElevenLabs
        
        Args:
            language: Optional language filter
            
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
                response = self._make_request("GET", "voices")
                data = response.json()
                voices = data.get('voices', [])
                
                # Update cache
                self._voices_cache = voices
                self._cache_timestamp = current_time
                
            except Exception as e:
                logger.error(f"Failed to fetch ElevenLabs voices: {e}")
                return []
        
        # Process and filter voices
        processed_voices = []
        for voice in voices:
            voice_info = {
                'id': voice.get('voice_id'),
                'name': voice.get('name'),
                'gender': self._detect_gender(voice.get('name', '')),
                'language': 'multilingual',  # ElevenLabs voices are multilingual
                'category': voice.get('category', 'generated'),
                'description': voice.get('description', ''),
                'use_case': voice.get('use_case', ''),
                'accent': voice.get('accent', ''),
                'age': voice.get('age', ''),
                'available': True,
                'backend': 'elevenlabs'
            }
            
            # Language filtering (basic since ElevenLabs voices are multilingual)
            if language and language != 'en' and language != 'multilingual':
                # For non-English, we might want to suggest specific voices
                # but for now, include all since they're multilingual
                pass
            
            processed_voices.append(voice_info)
        
        logger.info(f"Retrieved {len(processed_voices)} ElevenLabs voices")
        return processed_voices
    
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Dictionary with voice information
        """
        try:
            response = self._make_request("GET", f"voices/{voice_id}")
            voice_data = response.json()
            
            return {
                'id': voice_data.get('voice_id'),
                'name': voice_data.get('name'),
                'gender': self._detect_gender(voice_data.get('name', '')),
                'language': 'multilingual',
                'category': voice_data.get('category', 'generated'),
                'description': voice_data.get('description', ''),
                'use_case': voice_data.get('use_case', ''),
                'accent': voice_data.get('accent', ''),
                'age': voice_data.get('age', ''),
                'settings': voice_data.get('settings', {}),
                'samples': voice_data.get('samples', []),
                'available': True,
                'backend': 'elevenlabs'
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise VoiceNotFoundError(f"Voice {voice_id} not found")
            raise TTSError(f"Failed to get voice info: {e}")
    
    def estimate_cost(self, text: str, config: TTSConfig) -> float:
        """
        Estimate cost for ElevenLabs synthesis
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            Estimated cost in USD
        """
        clean_text = self.clean_text(text)
        character_count = len(clean_text)
        
        # ElevenLabs pricing (as of 2024):
        # Free tier: 10,000 chars/month
        # Creator: $5/month + $0.30 per 1K chars
        # Pro: $11/month + $0.24 per 1K chars
        # Scale: $99/month + $0.18 per 1K chars
        
        # Using Creator tier pricing as default
        cost_per_1k_chars = 0.30
        return (character_count / 1000) * cost_per_1k_chars
    
    def get_service_limits(self) -> Dict[str, Any]:
        """
        Get ElevenLabs service limits
        
        Returns:
            Dictionary with service limits
        """
        return {
            'max_characters_per_request': 5000,
            'max_requests_per_minute': 20,  # Varies by plan
            'supported_formats': ['mp3'],
            'supported_sample_rates': [22050, 44100],
            'free_tier_chars_per_month': 10000,
            'rate_limit_headers': ['x-ratelimit-remaining-characters', 'x-ratelimit-reset'],
            'max_voice_clones': 3,  # Varies by plan
            'api_version': 'v1'
        }
    
    def add_emotion_to_text(self, text: str, emotion: Optional[str]) -> str:
        """
        Add emotion/style hints for ElevenLabs
        
        Args:
            text: Original text
            emotion: Emotion to apply
            
        Returns:
            Text with emotion hints
        """
        if not emotion:
            return text
        
        # ElevenLabs responds well to contextual emotion hints
        emotion_map = {
            'happy': 'cheerful',
            'excited': 'enthusiastic',
            'sad': 'melancholy',
            'angry': 'stern',
            'surprised': 'amazed',
            'thoughtful': 'contemplative',
            'curious': 'inquisitive',
            'amused': 'lighthearted',
            'interested': 'engaged'
        }
        
        emotion_hint = emotion_map.get(emotion.lower(), emotion)
        
        # Add subtle emotion hint (ElevenLabs picks up on context)
        if emotion_hint in ['cheerful', 'enthusiastic', 'lighthearted']:
            return f"{text}"  # Natural expression works better
        elif emotion_hint in ['contemplative', 'thoughtful']:
            return f"{text}"  # Let natural pauses handle this
        else:
            return text
    
    def _detect_gender(self, voice_name: str) -> str:
        """
        Detect gender from voice name (heuristic)
        
        Args:
            voice_name: Name of the voice
            
        Returns:
            'male', 'female', or 'unknown'
        """
        name_lower = voice_name.lower()
        
        # Common male names
        male_names = {'adam', 'antony', 'arnold', 'brian', 'callum', 'charlie', 'clyde', 
                     'daniel', 'dave', 'ethan', 'fin', 'george', 'gideon', 'giovanni',
                     'harry', 'james', 'jeremy', 'josh', 'liam', 'marcus', 'matthew',
                     'michael', 'paul', 'ryan', 'sam', 'thomas', 'will', 'william'}
        
        # Common female names  
        female_names = {'alice', 'bella', 'charlotte', 'domi', 'dorothy', 'elena', 'elli',
                       'emily', 'emma', 'freya', 'grace', 'isabella', 'jessica', 'lily',
                       'lucy', 'matilda', 'nicole', 'rachel', 'sarah', 'serena', 'sophia'}
        
        for male_name in male_names:
            if male_name in name_lower:
                return 'male'
                
        for female_name in female_names:
            if female_name in name_lower:
                return 'female'
        
        return 'unknown' 