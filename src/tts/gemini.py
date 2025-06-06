"""
Gemini 2.5 Pro TTS Backend for high-quality multi-speaker podcast generation
"""
import logging
import os
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

from .base import (
    BaseTTSBackend, TTSConfig, SynthesisResult, AudioFormat,
    TTSError, VoiceNotFoundError, QuotaExceededError, AudioTooLongError
)
from utils.log_cfg import PipelineTimer, log_pipeline_metrics

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiTTSBackend(BaseTTSBackend):
    """Gemini 2.5 Pro TTS backend implementation with multi-speaker support"""
    
    def __init__(self, config: TTSConfig):
        # Extract API key from environment
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        super().__init__(api_key=self.api_key)
        
        self.config = config
        self.model_name = "gemini-2.5-pro-preview-tts"
        
        # Configure Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Load voice profiles
        self.voices_data = self._load_voice_profiles()
        
        # Cache for voices to reduce processing
        self._voices_cache = None
        self._cache_timestamp = 0
        self._cache_duration = 3600  # 1 hour
        
        logger.info(f"Initialized Gemini TTS backend with model: {self.model_name}")
    
    def _load_voice_profiles(self) -> Dict[str, Any]:
        """Load voice profiles from JSON file"""
        voices_file = Path(__file__).parent / "gemini_voices.json"
        try:
            with open(voices_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Voice profiles file not found: {voices_file}")
            return {"voices": {}, "default_voices": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse voice profiles: {e}")
            return {"voices": {}, "default_voices": {}}
    
    def get_max_text_length(self) -> int:
        """Gemini TTS supports up to 32k tokens (~25k words)"""
        return 25000  # Conservative estimate in characters
    
    def supports_ssml(self) -> bool:
        """Gemini TTS supports natural language style prompts"""
        return True
    
    def get_default_voices(self) -> Dict[str, str]:
        """Get default Vietnamese-optimized voices"""
        defaults = self.voices_data.get("default_voices", {}).get("vietnamese_podcast", {})
        return {
            "host": defaults.get("host", "Fenrir"),
            "guest": defaults.get("guest", "Leda")
        }
    
    def _create_multi_speaker_prompt(self, dialogue_lines, host_voice: str, guest_voice: str) -> str:
        """Create multi-speaker prompt for Gemini TTS"""
        
        # Get style instructions
        style_prompt = self.voices_data.get("style_prompts", {}).get("vietnamese_podcast", "")
        
        # Build the dialogue text
        dialogue_text = ""
        for line in dialogue_lines:
            speaker_name = "Host" if line.speaker.value == "HOST" else "Guest"
            dialogue_text += f"{speaker_name}: {line.text}\n\n"
        
        # Create the full prompt with voice assignments and style
        full_prompt = f"""
{style_prompt}

Voice assignments:
- Host: Use voice '{host_voice}' (energetic, enthusiastic)
- Guest: Use voice '{guest_voice}' (youthful, engaged)

Generate this dialogue as a natural Vietnamese podcast conversation:

{dialogue_text}

Make sure both speakers sound natural, engaged, and maintain the conversational flow with appropriate emotions and pacing.
"""
        
        return full_prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def _generate_audio_with_retry(self, prompt: str, host_voice: str, guest_voice: str) -> bytes:
        """Generate audio with retry logic"""
        
        try:
            # Configure speech generation using the new API syntax
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=host_voice,
                            )
                        )
                    ),
                )
            )
            
            # Extract audio data from response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            return part.inline_data.data
            
            raise TTSError("No audio data found in Gemini response")
            
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                raise QuotaExceededError(f"Gemini TTS quota exceeded: {e}")
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise TTSError(f"Gemini TTS authentication failed: {e}")
            else:
                logger.warning(f"Gemini TTS API error: {e}, will retry...")
                raise
    
    def synthesize_multi_speaker(self, dialogue_lines, host_voice: str = None, guest_voice: str = None) -> SynthesisResult:
        """
        Synthesize multi-speaker dialogue using Gemini TTS
        
        Args:
            dialogue_lines: List of dialogue lines with speaker and text
            host_voice: Voice for host speaker
            guest_voice: Voice for guest speaker
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        
        # Get default voices if not specified
        if not host_voice or not guest_voice:
            defaults = self.get_default_voices()
            host_voice = host_voice or defaults["host"]
            guest_voice = guest_voice or defaults["guest"]
        
        # Calculate total text length
        total_text = " ".join([line.text for line in dialogue_lines])
        self.validate_text(total_text)
        
        with PipelineTimer(f"Gemini multi-speaker synthesis ({len(total_text)} chars)", logger):
            
            # Create multi-speaker prompt
            prompt = self._create_multi_speaker_prompt(dialogue_lines, host_voice, guest_voice)
            
            try:
                logger.info(f"Synthesizing with Gemini TTS: host={host_voice}, guest={guest_voice}, lines={len(dialogue_lines)}")
                
                # Generate audio
                audio_data = self._generate_audio_with_retry(prompt, host_voice, guest_voice)
                
                if not audio_data:
                    raise TTSError("Received empty audio data from Gemini TTS")
                
                # Estimate duration (rough calculation: ~150 chars per minute)
                estimated_duration = len(total_text) / 150 * 60
                
                # Estimate cost (Gemini TTS pricing - estimated based on Google's pricing model)
                # Using a conservative estimate similar to other premium TTS services
                cost_estimate = len(total_text) / 1000 * 0.20  # ~$0.20 per 1K characters
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=AudioFormat.MP3,  # Assuming MP3 format
                    duration_seconds=estimated_duration,
                    sample_rate=self.config.sample_rate,
                    character_count=len(total_text),
                    cost_estimate=cost_estimate,
                    metadata={
                        "host_voice": host_voice,
                        "guest_voice": guest_voice,
                        "model": self.model_name,
                        "dialogue_lines": len(dialogue_lines),
                        "backend": "gemini",
                        "multi_speaker": True
                    }
                )
                
                # Log metrics
                metrics = {
                    'host_voice': host_voice,
                    'guest_voice': guest_voice,
                    'dialogue_lines': len(dialogue_lines),
                    'character_count': len(total_text),
                    'audio_size_bytes': len(audio_data),
                    'estimated_duration': estimated_duration,
                    'cost_estimate': cost_estimate,
                    'model': self.model_name
                }
                log_pipeline_metrics("gemini_multi_speaker_synthesis", metrics, logger)
                
                logger.info(f"Gemini synthesis completed: {len(audio_data)} bytes, ~{estimated_duration:.1f}s, ${cost_estimate:.3f}")
                
                return result
                
            except Exception as e:
                logger.error(f"Gemini TTS synthesis failed: {e}")
                raise
    
    def synthesize(self, text: str, config: TTSConfig) -> SynthesisResult:
        """
        Synthesize single text using Gemini TTS (fallback method)
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        self.validate_text(text)
        
        with PipelineTimer(f"Gemini single synthesis ({len(text)} chars)", logger):
            
            # Clean and prepare text
            clean_text = self.clean_text(text)
            
            # Use default voice from config
            voice_name = config.voice_id or self.get_default_voices()["host"]
            
            try:
                logger.info(f"Synthesizing with Gemini TTS: voice={voice_name}, chars={len(clean_text)}")
                
                # Create simple prompt for single speaker
                prompt = f"Generate this text as natural Vietnamese speech: {clean_text}"
                
                # Generate audio using the same method as multi-speaker
                audio_data = self._generate_audio_with_retry(prompt, voice_name, voice_name)
                
                if not audio_data:
                    raise TTSError("Received empty audio data from Gemini TTS")
                
                # Estimate duration
                estimated_duration = len(clean_text) / 150 * 60
                
                # Estimate cost
                cost_estimate = len(clean_text) / 1000 * 0.20
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=AudioFormat.MP3,
                    duration_seconds=estimated_duration,
                    sample_rate=config.sample_rate,
                    character_count=len(clean_text),
                    cost_estimate=cost_estimate,
                    metadata={
                        "voice_name": voice_name,
                        "model": self.model_name,
                        "backend": "gemini",
                        "multi_speaker": False
                    }
                )
                
                logger.info(f"Gemini synthesis completed: {len(audio_data)} bytes, ~{estimated_duration:.1f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Gemini TTS synthesis failed: {e}")
                raise
    
    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available voices from Gemini TTS
        
        Args:
            language: Optional language filter (Vietnamese preferred)
            
        Returns:
            List of voice dictionaries
        """
        voices = []
        
        for voice_name, voice_info in self.voices_data.get("voices", {}).items():
            voice_dict = {
                'id': voice_name,
                'name': voice_name,
                'characteristic': voice_info.get('characteristic', ''),
                'energy': voice_info.get('energy', 'medium'),
                'speed_preference': voice_info.get('speed_preference', 'medium'),
                'emotion_range': voice_info.get('emotion_range', 'medium'),
                'vietnamese_suitability': voice_info.get('vietnamese_suitability', 'unknown'),
                'podcast_role': voice_info.get('podcast_role', 'general'),
                'description': voice_info.get('description', ''),
                'backend': 'gemini'
            }
            
            # Filter for Vietnamese suitability if language specified
            if language and language.lower() in ['vi', 'vietnamese']:
                if voice_info.get('vietnamese_suitability') in ['excellent', 'good']:
                    voices.append(voice_dict)
            else:
                voices.append(voice_dict)
        
        logger.info(f"Retrieved {len(voices)} Gemini TTS voices")
        return voices
    
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Dictionary with voice information
        """
        voice_data = self.voices_data.get("voices", {}).get(voice_id)
        
        if not voice_data:
            raise VoiceNotFoundError(f"Voice {voice_id} not found")
        
        return {
            'id': voice_id,
            'name': voice_id,
            'characteristic': voice_data.get('characteristic', ''),
            'energy': voice_data.get('energy', 'medium'),
            'speed_preference': voice_data.get('speed_preference', 'medium'),
            'emotion_range': voice_data.get('emotion_range', 'medium'),
            'vietnamese_suitability': voice_data.get('vietnamese_suitability', 'unknown'),
            'podcast_role': voice_data.get('podcast_role', 'general'),
            'description': voice_data.get('description', ''),
            'backend': 'gemini'
        }
    
    def estimate_cost(self, text: str, config: TTSConfig) -> float:
        """
        Estimate cost for Gemini TTS synthesis
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            Estimated cost in USD
        """
        clean_text = self.clean_text(text)
        character_count = len(clean_text)
        
        # Gemini TTS pricing (estimated based on premium TTS services)
        # Using conservative estimate similar to other premium services
        cost_per_1k_chars = 0.20
        return (character_count / 1000) * cost_per_1k_chars
    
    def get_service_limits(self) -> Dict[str, Any]:
        """
        Get Gemini TTS service limits
        
        Returns:
            Dictionary with service limits
        """
        return {
            'max_characters_per_request': 25000,  # Conservative estimate for 32k tokens
            'max_requests_per_minute': 60,  # Typical Google API limit
            'supported_formats': ['mp3'],
            'supported_sample_rates': [22050, 44100],
            'max_speakers': 2,  # Multi-speaker support limit
            'supported_languages': 24,
            'cost_per_1k_chars': 0.20,
            'max_dialogue_lines': 100,  # Practical limit for podcast generation
            'api_version': 'v1beta',
            'model': self.model_name
        }