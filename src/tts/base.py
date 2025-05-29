"""
Base TTS interface for all TTS backends in the podcast pipeline
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis"""
    voice_id: str
    speed: float = 1.0          # Speed multiplier (0.5-2.0)
    pitch: float = 0.0          # Pitch adjustment (-20 to +20 semitones)
    stability: float = 0.75     # Voice stability (0.0-1.0, ElevenLabs specific)
    similarity_boost: float = 0.75  # Similarity boost (0.0-1.0, ElevenLabs specific)
    style: float = 0.0          # Style strength (0.0-1.0, ElevenLabs specific)
    emotion: Optional[str] = None  # Emotion hint for synthesis
    pause_before: float = 0.0   # Pause before speaking (seconds)
    pause_after: float = 0.0    # Pause after speaking (seconds)
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 44100    # Sample rate in Hz
    bitrate: str = "192k"       # Audio bitrate


@dataclass
class SynthesisResult:
    """Result of TTS synthesis"""
    audio_data: bytes
    format: AudioFormat
    duration_seconds: float
    sample_rate: int
    character_count: int
    cost_estimate: Optional[float] = None  # Estimated cost in USD
    metadata: Optional[Dict[str, Any]] = None


class TTSError(Exception):
    """Base exception for TTS operations"""
    pass


class VoiceNotFoundError(TTSError):
    """Raised when requested voice is not available"""
    pass


class QuotaExceededError(TTSError):
    """Raised when TTS service quota is exceeded"""
    pass


class AudioTooLongError(TTSError):
    """Raised when audio exceeds service limits"""
    pass


class BaseTTSBackend(ABC):
    """Abstract base class for all TTS backends"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.backend_name = self.__class__.__name__.lower().replace('ttsbackend', '').replace('backend', '')
        
    @abstractmethod
    def synthesize(self, text: str, config: TTSConfig) -> SynthesisResult:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            SynthesisResult with audio data and metadata
            
        Raises:
            TTSError: For various TTS-related errors
        """
        pass
    
    @abstractmethod
    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available voices
        
        Args:
            language: Optional language filter (ISO code)
            
        Returns:
            List of voice dictionaries with id, name, gender, language, etc.
        """
        pass
    
    @abstractmethod
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Dictionary with voice information
            
        Raises:
            VoiceNotFoundError: If voice doesn't exist
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, text: str, config: TTSConfig) -> float:
        """
        Estimate cost for synthesis in USD
        
        Args:
            text: Text to synthesize
            config: TTS configuration
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    @abstractmethod
    def get_service_limits(self) -> Dict[str, Any]:
        """
        Get service limits and quotas
        
        Returns:
            Dictionary with limits (characters, requests, etc.)
        """
        pass
    
    def validate_text(self, text: str) -> bool:
        """
        Validate text for synthesis
        
        Args:
            text: Text to validate
            
        Returns:
            True if text is valid
            
        Raises:
            TTSError: If text is invalid
        """
        if not text or not text.strip():
            raise TTSError("Text is empty")
        
        # Remove common TTS-unfriendly content
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) > self.get_max_text_length():
            raise AudioTooLongError(f"Text too long: {len(cleaned_text)} characters (max: {self.get_max_text_length()})")
        
        if len(cleaned_text.split()) < 1:
            raise TTSError("Text has no meaningful words")
        
        return True
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for better TTS synthesis
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common problematic characters
        text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
        
        # Ensure proper sentence endings
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def get_max_text_length(self) -> int:
        """
        Get maximum text length for this backend
        
        Returns:
            Maximum characters that can be synthesized in one request
        """
        return 5000  # Default conservative limit
    
    def supports_emotion(self) -> bool:
        """
        Check if backend supports emotion/style control
        
        Returns:
            True if emotions are supported
        """
        return False
    
    def supports_ssml(self) -> bool:
        """
        Check if backend supports SSML markup
        
        Returns:
            True if SSML is supported
        """
        return False
    
    def add_pauses_to_text(self, text: str, pause_before: float = 0.0, pause_after: float = 0.0) -> str:
        """
        Add pause markup to text if supported
        
        Args:
            text: Original text
            pause_before: Pause before speaking (seconds)
            pause_after: Pause after speaking (seconds)
            
        Returns:
            Text with pause markup
        """
        if not self.supports_ssml():
            return text
        
        result = text
        
        if pause_before > 0:
            result = f'<break time="{pause_before}s"/>{result}'
        
        if pause_after > 0:
            result = f'{result}<break time="{pause_after}s"/>'
        
        return result
    
    def add_emotion_to_text(self, text: str, emotion: Optional[str]) -> str:
        """
        Add emotion markup to text if supported
        
        Args:
            text: Original text
            emotion: Emotion to apply
            
        Returns:
            Text with emotion markup
        """
        if not emotion or not self.supports_emotion():
            return text
        
        # Default implementation - subclasses can override
        return text
    
    def save_audio(self, audio_data: bytes, output_path: Path, format: AudioFormat = AudioFormat.MP3) -> Path:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio bytes
            output_path: Output file path
            format: Audio format
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure correct extension
        if output_path.suffix.lower() != f'.{format.value}':
            output_path = output_path.with_suffix(f'.{format.value}')
        
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Saved audio to {output_path} ({len(audio_data)} bytes)")
        return output_path
    
    def convert_format(self, audio_data: bytes, from_format: AudioFormat, to_format: AudioFormat) -> bytes:
        """
        Convert audio between formats
        
        Args:
            audio_data: Input audio data
            from_format: Source format
            to_format: Target format
            
        Returns:
            Converted audio data
        """
        if from_format == to_format:
            return audio_data
        
        # Use pydub for conversion
        try:
            from pydub import AudioSegment
            from io import BytesIO
            
            # Load audio
            audio = AudioSegment.from_file(BytesIO(audio_data), format=from_format.value)
            
            # Convert to target format
            output_buffer = BytesIO()
            audio.export(output_buffer, format=to_format.value)
            
            return output_buffer.getvalue()
            
        except ImportError:
            logger.error("pydub not available for format conversion")
            return audio_data
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return audio_data
    
    def __str__(self):
        return f"{self.backend_name.title()} TTS Backend"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(backend_name='{self.backend_name}')>" 