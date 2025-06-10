"""
Gemini 2.5 Pro TTS Backend for high-quality multi-speaker podcast generation
"""
import logging
import os
import json
import time
import io
import struct
import mimetypes
from typing import Optional, Dict, Any, List, Tuple
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
            "host": defaults.get("host", "Zephyr"),
            "guest": defaults.get("guest", "Rasalgethi")
        }
    
    def _create_multi_speaker_dialogue(self, dialogue_lines) -> str:
        """Create clean dialogue text for multi-speaker API"""
        
        # Build the dialogue text with proper speaker labels for API
        dialogue_text = ""
        for line in dialogue_lines:
            # Use exact speaker labels that match the API configuration
            speaker_label = line.speaker.value  # This should be "HOST" or "GUEST"
            dialogue_text += f"{speaker_label}: {line.text}\n\n"
        
        return dialogue_text.strip()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def _generate_audio_with_retry(self, dialogue_text: str, host_voice: str, guest_voice: str, 
                                  style_prompt: str = None, temperature: float = 1.5) -> Tuple[bytes, str]:
        """Generate audio with retry logic using proper multi-speaker API
        
        Returns:
            Tuple of (audio_data, mime_type)
        """
        
        try:
            # Create content with style prompt + dialogue text
            if style_prompt:
                full_text = f"{style_prompt}\n\n{dialogue_text}"
            else:
                full_text = dialogue_text
                
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=full_text)
                    ]
                )
            ]
            
            # Configure speech generation using proper multi-speaker API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,  # Natural variation for engaging audio
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(
                                    speaker="HOST",
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=host_voice
                                        )
                                    )
                                ),
                                types.SpeakerVoiceConfig(
                                    speaker="GUEST",
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=guest_voice
                                        )
                                    )
                                )
                            ]
                        )
                    ),
                )
            )
            
            # Extract audio data and format from response (following Google AI Studio pattern)
            if (
                response.candidates is None
                or response.candidates[0].content is None
                or response.candidates[0].content.parts is None
            ):
                raise TTSError("No audio data found in Gemini response")
            
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    
                    # Validate audio data
                    if not audio_data or len(audio_data) < 100:  # Basic sanity check
                        logger.warning(f"Received suspiciously small audio data: {len(audio_data) if audio_data else 0} bytes")
                    
                    logger.debug(f"Gemini TTS returned audio: {len(audio_data)} bytes, MIME type: {mime_type}")
                    
                    return audio_data, mime_type
            
            raise TTSError("No audio data found in Gemini response")
            
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                raise QuotaExceededError(f"Gemini TTS quota exceeded: {e}")
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise TTSError(f"Gemini TTS authentication failed: {e}")
            else:
                logger.warning(f"Gemini TTS API error: {e}, will retry...")
                raise
    
    def synthesize_multi_speaker(self, dialogue_lines, host_voice: str = None, guest_voice: str = None,
                                style_prompt: str = None, temperature: float = 1.5) -> SynthesisResult:
        """
        Synthesize multi-speaker dialogue using Gemini TTS
        
        Args:
            dialogue_lines: List of dialogue lines with speaker and text
            host_voice: Voice for host speaker
            guest_voice: Voice for guest speaker
            style_prompt: Global style instructions for both speakers
            temperature: Temperature for generation (0.0-2.0, higher = more variation)
            
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
            
            # Create clean dialogue text for multi-speaker API
            dialogue_text = self._create_multi_speaker_dialogue(dialogue_lines)
            
            try:
                logger.info(f"Synthesizing with Gemini TTS: host={host_voice}, guest={guest_voice}, lines={len(dialogue_lines)}")
                if style_prompt:
                    logger.info(f"Using style prompt: {style_prompt[:100]}...")
                
                # Generate audio using proper multi-speaker API
                audio_data, mime_type = self._generate_audio_with_retry(
                    dialogue_text, host_voice, guest_voice, style_prompt, temperature
                )
                
                if not audio_data:
                    raise TTSError("Received empty audio data from Gemini TTS")
                
                # Determine audio format and handle conversion (Google AI Studio approach)
                file_extension = mimetypes.guess_extension(mime_type)
                if file_extension is None:
                    file_extension = ".wav"
                    # Convert to WAV first using Google AI Studio method
                    audio_data = self._convert_to_wav(audio_data, mime_type)
                    audio_format = AudioFormat.WAV
                    logger.info(f"Converted audio from {mime_type} to WAV format")
                else:
                    audio_format = self._detect_audio_format(mime_type)
                    logger.info(f"Gemini TTS returned {audio_format.value} format (MIME: {mime_type})")
                
                # Convert to MP3 if not already MP3
                if audio_format != AudioFormat.MP3:
                    audio_data = self._convert_audio_to_mp3(audio_data, audio_format, mime_type)
                    audio_format = AudioFormat.MP3
                    logger.info(f"Converted audio to MP3 format")
                
                # Estimate duration (rough calculation: ~150 chars per minute)
                estimated_duration = len(total_text) / 150 * 60
                
                # Estimate cost (Gemini TTS pricing - estimated based on Google's pricing model)
                # Using a conservative estimate similar to other premium TTS services
                cost_estimate = len(total_text) / 1000 * 0.20  # ~$0.20 per 1K characters
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=audio_format,
                    duration_seconds=estimated_duration,
                    sample_rate=self.config.sample_rate,
                    character_count=len(total_text),
                    cost_estimate=cost_estimate,
                    metadata={
                        "host_voice": host_voice,
                        "guest_voice": guest_voice,
                        "style_prompt": style_prompt,
                        "temperature": temperature,
                        "model": self.model_name,
                        "dialogue_lines": len(dialogue_lines),
                        "backend": "gemini",
                        "multi_speaker": True,
                        "original_mime_type": mime_type
                    }
                )
                
                # Log metrics
                metrics = {
                    'host_voice': host_voice,
                    'guest_voice': guest_voice,
                    'style_prompt_length': len(style_prompt) if style_prompt else 0,
                    'temperature': temperature,
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
                
                # For single speaker, use the dialogue format but with one speaker
                single_speaker_text = f"HOST: {clean_text}"
                
                # Generate audio using multi-speaker method with same voice for both
                audio_data, mime_type = self._generate_audio_with_retry(single_speaker_text, voice_name, voice_name)
                
                if not audio_data:
                    raise TTSError("Received empty audio data from Gemini TTS")
                
                # Determine audio format and handle conversion (Google AI Studio approach)
                file_extension = mimetypes.guess_extension(mime_type)
                if file_extension is None:
                    file_extension = ".wav"
                    # Convert to WAV first using Google AI Studio method
                    audio_data = self._convert_to_wav(audio_data, mime_type)
                    audio_format = AudioFormat.WAV
                    logger.info(f"Converted audio from {mime_type} to WAV format")
                else:
                    audio_format = self._detect_audio_format(mime_type)
                    logger.info(f"Gemini TTS returned {audio_format.value} format (MIME: {mime_type})")
                
                # Convert to MP3 if not already MP3
                if audio_format != AudioFormat.MP3:
                    audio_data = self._convert_audio_to_mp3(audio_data, audio_format, mime_type)
                    audio_format = AudioFormat.MP3
                    logger.info(f"Converted audio to MP3 format")
                
                # Estimate duration
                estimated_duration = len(clean_text) / 150 * 60
                
                # Estimate cost
                cost_estimate = len(clean_text) / 1000 * 0.20
                
                result = SynthesisResult(
                    audio_data=audio_data,
                    format=audio_format,
                    duration_seconds=estimated_duration,
                    sample_rate=config.sample_rate,
                    character_count=len(clean_text),
                    cost_estimate=cost_estimate,
                    metadata={
                        "voice_name": voice_name,
                        "model": self.model_name,
                        "backend": "gemini",
                        "multi_speaker": False,
                        "original_mime_type": mime_type
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
            'supported_formats': ['mp3', 'wav', 'flac'],  # Gemini supports multiple formats
            'supported_sample_rates': [22050, 44100],
            'max_speakers': 2,  # Multi-speaker support limit
            'supported_languages': 24,
            'cost_per_1k_chars': 0.20,
            'max_dialogue_lines': 100,  # Practical limit for podcast generation
            'api_version': 'v1beta',
            'model': self.model_name,
            'multi_speaker_support': True,  # Now properly supported
            'streaming_support': False  # Not implemented yet
        }
    
    def _detect_audio_format(self, mime_type: str) -> AudioFormat:
        """Detect audio format from MIME type
        
        Args:
            mime_type: MIME type string from Gemini response
            
        Returns:
            AudioFormat enum value
        """
        if not mime_type:
            logger.warning("No MIME type provided, defaulting to MP3")
            return AudioFormat.MP3
            
        mime_to_format = {
            'audio/mpeg': AudioFormat.MP3,
            'audio/mp3': AudioFormat.MP3,
            'audio/wav': AudioFormat.WAV,
            'audio/wave': AudioFormat.WAV,
            'audio/x-wav': AudioFormat.WAV,
            'audio/flac': AudioFormat.WAV,  # Treat FLAC as WAV for conversion
            'audio/ogg': AudioFormat.OGG,
        }
        
        # Handle PCM format specifically (Gemini TTS returns this)
        if mime_type.startswith('audio/L16') or 'pcm' in mime_type.lower():
            logger.info(f"Detected PCM audio format from MIME type: {mime_type}")
            return AudioFormat.WAV  # Treat PCM as WAV for processing
        
        # Default to MP3 for unknown types
        detected_format = mime_to_format.get(mime_type.lower(), AudioFormat.MP3)
        
        if mime_type.lower() not in mime_to_format:
            logger.warning(f"Unknown MIME type '{mime_type}', defaulting to MP3 format")
        
        logger.debug(f"Detected audio format {detected_format.value} from MIME type: {mime_type}")
        return detected_format
    
    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Convert audio data to WAV format using Google AI Studio approach
        
        Args:
            audio_data: Raw audio data
            mime_type: MIME type from Gemini response
            
        Returns:
            WAV-encoded audio data
        """
        try:
            import struct
            import mimetypes
            
            # Use the same conversion approach as Google AI Studio sample
            parameters = self._parse_audio_mime_type(mime_type)
            bits_per_sample = parameters["bits_per_sample"]
            sample_rate = parameters["rate"]
            num_channels = 1
            data_size = len(audio_data)
            bytes_per_sample = bits_per_sample // 8
            block_align = num_channels * bytes_per_sample
            byte_rate = sample_rate * block_align
            chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

            # http://soundfile.sapp.org/doc/WaveFormat/
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",          # ChunkID
                chunk_size,       # ChunkSize (total file size - 8 bytes)
                b"WAVE",          # Format
                b"fmt ",          # Subchunk1ID
                16,               # Subchunk1Size (16 for PCM)
                1,                # AudioFormat (1 for PCM)
                num_channels,     # NumChannels
                sample_rate,      # SampleRate
                byte_rate,        # ByteRate
                block_align,      # BlockAlign
                bits_per_sample,  # BitsPerSample
                b"data",          # Subchunk2ID
                data_size         # Subchunk2Size (size of audio data)
            )
            
            wav_data = header + audio_data
            logger.info(f"Converted {len(audio_data)} bytes ({mime_type}) to {len(wav_data)} bytes (WAV)")
            
            return wav_data
            
        except Exception as e:
            logger.error(f"Failed to convert audio from {mime_type} to WAV: {e}")
            # Return original data if conversion fails
            return audio_data
    
    def _parse_audio_mime_type(self, mime_type: str) -> dict:
        """Parse bits per sample and rate from an audio MIME type string (Google AI Studio approach)
        
        Args:
            mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000")
            
        Returns:
            Dictionary with "bits_per_sample" and "rate" keys
        """
        bits_per_sample = 16
        rate = 24000

        # Extract rate from parameters
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass  # Keep rate as default
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass  # Keep bits_per_sample as default if conversion fails

        return {"bits_per_sample": bits_per_sample, "rate": rate}
    
    def _save_binary_file(self, file_name: str, data: bytes) -> None:
        """Save binary file (following Google AI Studio pattern)
        
        Args:
            file_name: Name of file to save
            data: Binary data to save
        """
        try:
            with open(file_name, "wb") as f:
                f.write(data)
            logger.info(f"File saved to: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save file {file_name}: {e}")
    
    def _convert_audio_to_mp3(self, audio_data: bytes, source_format: AudioFormat, mime_type: str = None) -> bytes:
        """Convert audio data to MP3 format using Google AI Studio compatible approach
        
        Args:
            audio_data: Raw audio data
            source_format: Source audio format
            mime_type: Original MIME type for additional format info
            
        Returns:
            MP3-encoded audio data
        """
        try:
            from pydub import AudioSegment
            
            logger.debug(f"Converting {len(audio_data)} bytes from {source_format.value} to MP3")
            
            # Handle PCM format specifically using Google AI Studio approach
            if mime_type and (mime_type.startswith('audio/L16') or 'pcm' in mime_type.lower()):
                # First convert to WAV using Google AI Studio method
                wav_data = self._convert_to_wav(audio_data, mime_type)
                
                # Then load WAV into pydub for MP3 conversion
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(wav_data),
                    format="wav"
                )
            else:
                # Load audio from bytes for other formats
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(audio_data),
                    format=source_format.value
                )
            
            # Ensure audio properties are set correctly
            if audio_segment.frame_rate != 44100:
                audio_segment = audio_segment.set_frame_rate(44100)
            if audio_segment.channels != 1:
                audio_segment = audio_segment.set_channels(1)  # Mono for podcasts
            
            # Convert to MP3
            output_buffer = io.BytesIO()
            audio_segment.export(
                output_buffer,
                format="mp3",
                bitrate="192k",
                parameters=["-q:a", "2"]  # High quality encoding
            )
            
            mp3_data = output_buffer.getvalue()
            logger.info(f"Converted {len(audio_data)} bytes ({source_format.value}) to {len(mp3_data)} bytes (MP3)")
            
            return mp3_data
            
        except ImportError:
            logger.error("pydub not available for audio conversion")
            return audio_data
        except Exception as e:
            logger.error(f"Failed to convert audio from {source_format.value} to MP3: {e}")
            # Return original data if conversion fails - the audio mixer might still handle it
            return audio_data