"""
Coqui-XTTS Local TTS Backend
Provides high-quality offline text-to-speech synthesis
"""
import logging
import os
import tempfile
import shutil
from typing import Optional, Dict, List, Union
from pathlib import Path
import io

try:
    import torch
    import torchaudio
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

from .base import TTSBackend, TTSResult, TTSConfig

logger = logging.getLogger(__name__)


class CoquiTTSBackend(TTSBackend):
    """Coqui-XTTS local TTS implementation"""
    
    # Available languages for XTTS v2
    SUPPORTED_LANGUAGES = [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 
        'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko'
    ]
    
    # Voice samples directory (can be expanded with custom voices)
    DEFAULT_VOICES = {
        'female': {
            'en': 'female_en_1.wav',
            'es': 'female_es_1.wav',
            'fr': 'female_fr_1.wav',
            'de': 'female_de_1.wav'
        },
        'male': {
            'en': 'male_en_1.wav',
            'es': 'male_es_1.wav', 
            'fr': 'male_fr_1.wav',
            'de': 'male_de_1.wav'
        }
    }
    
    def __init__(self, config: TTSConfig):
        if not COQUI_AVAILABLE:
            raise ImportError(
                "Coqui TTS not available. Install with: pip install TTS\n"
                "For best performance, also install: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
        
        super().__init__(config)
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize XTTS model
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Loading Coqui XTTS model: {self.model_name}")
        
        try:
            self.tts = TTS(model_name=self.model_name).to(self.device)
            logger.info("Coqui XTTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Coqui XTTS model: {e}")
            raise
        
        # Setup voice samples directory
        self.voices_dir = Path("src/tts/voices")
        self.voices_dir.mkdir(exist_ok=True)
        
        # Setup default voices
        self._setup_voices()
        
        logger.info(f"Initialized Coqui TTS backend for language: {config.language}")
    
    def _setup_voices(self):
        """Setup default voice samples"""
        # For now, we'll use built-in samples or user-provided ones
        # In a full implementation, you'd download or include sample voice files
        
        lang_code = self.config.language
        if lang_code not in self.SUPPORTED_LANGUAGES:
            lang_code = 'en'  # Fallback to English
        
        # Set default voice paths (these would be actual audio files)
        if not self.config.female_voice:
            self.config.female_voice = f"female_{lang_code}"
        
        if not self.config.male_voice:
            self.config.male_voice = f"male_{lang_code}"
        
        logger.info(f"Using voices - Female: {self.config.female_voice}, Male: {self.config.male_voice}")
    
    def _create_sample_voice_file(self, voice_id: str) -> Optional[str]:
        """
        Create or find a sample voice file for voice cloning
        In a real implementation, this would manage a library of voice samples
        """
        voice_file = self.voices_dir / f"{voice_id}.wav"
        
        if voice_file.exists():
            return str(voice_file)
        
        # For demo purposes, create a synthetic sample using TTS
        # In practice, you'd have pre-recorded voice samples
        logger.warning(f"Voice sample not found for {voice_id}, using default synthesis")
        return None
    
    def _extract_emotion_from_text(self, text: str) -> tuple[str, str]:
        """Extract emotion annotations and clean text"""
        import re
        
        # Find emotion markers like [laughs], [pause], [excited]
        emotion_pattern = r'\[([^\]]+)\]'
        emotion_matches = re.findall(emotion_pattern, text)
        cleaned_text = re.sub(emotion_pattern, '', text).strip()
        
        # Map emotions to speech adjustments
        emotion = emotion_matches[0].lower() if emotion_matches else 'neutral'
        
        return cleaned_text, emotion
    
    def _adjust_for_emotion(self, text: str, emotion: str) -> str:
        """Adjust text for emotional expression"""
        # Add SSML-like adjustments for emotional expression
        if emotion in ['laugh', 'chuckle', 'happy']:
            # Add slight pauses and emphasis
            text = text.replace('.', '... ')  # Extend pauses
            text = text.replace('!', '! ')    # Add emphasis
        elif emotion in ['pause', 'thoughtful']:
            # Add longer pauses
            text = text.replace(',', ', ... ')
            text = text.replace('.', ' ... ')
        elif emotion in ['excited', 'surprised']:
            # Add emphasis and speed
            text = text.upper() if len(text) < 50 else text  # Short excited phrases
        
        return text
    
    def synthesize(self, text: str, voice_id: str) -> TTSResult:
        """
        Synthesize text to speech using Coqui XTTS
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier for sample selection
            
        Returns:
            TTSResult with audio data and metadata
        """
        try:
            logger.debug(f"Synthesizing with Coqui voice: {voice_id}")
            
            # Clean text and extract emotions
            cleaned_text, emotion = self._extract_emotion_from_text(text)
            
            # Adjust text for emotional expression
            adjusted_text = self._adjust_for_emotion(cleaned_text, emotion)
            
            # Get voice sample file
            speaker_wav = self._create_sample_voice_file(voice_id)
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            try:
                # Synthesize using XTTS
                if speaker_wav and os.path.exists(speaker_wav):
                    # Voice cloning mode
                    self.tts.tts_to_file(
                        text=adjusted_text,
                        speaker_wav=speaker_wav,
                        language=self.config.language,
                        file_path=output_path
                    )
                else:
                    # Use built-in speaker (if available)
                    available_speakers = getattr(self.tts, 'speakers', None)
                    if available_speakers:
                        # Use first available speaker as fallback
                        speaker = available_speakers[0] if isinstance(available_speakers, list) else None
                        self.tts.tts_to_file(
                            text=adjusted_text,
                            speaker=speaker,
                            language=self.config.language,
                            file_path=output_path
                        )
                    else:
                        # Basic synthesis without speaker
                        self.tts.tts_to_file(
                            text=adjusted_text,
                            language=self.config.language,
                            file_path=output_path
                        )
                
                # Read the generated audio file
                with open(output_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                # Get audio metadata using torchaudio
                try:
                    waveform, sample_rate = torchaudio.load(output_path)
                    duration_seconds = waveform.shape[1] / sample_rate
                except:
                    # Fallback duration estimation
                    duration_seconds = len(audio_data) / (22050 * 2)  # Rough estimate
                
                # Clean up temporary file
                os.unlink(output_path)
                
                # Calculate character count and cost (free for local)
                character_count = len(text)
                cost_estimate = 0.0  # Local synthesis is free
                
                logger.debug(f"Coqui synthesis successful: {duration_seconds:.1f}s, {character_count} chars")
                
                return TTSResult(
                    audio_data=audio_data,
                    format="wav",
                    duration_seconds=duration_seconds,
                    character_count=character_count,
                    cost_estimate=cost_estimate,
                    metadata={
                        'voice_id': voice_id,
                        'backend': 'coqui',
                        'emotion': emotion,
                        'device': self.device,
                        'model': self.model_name,
                        'sample_rate': getattr(self, 'sample_rate', 22050)
                    }
                )
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(output_path):
                    os.unlink(output_path)
                raise e
                
        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            raise
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available voice samples"""
        voices = []
        
        # List available voice sample files
        if self.voices_dir.exists():
            for voice_file in self.voices_dir.glob("*.wav"):
                voice_name = voice_file.stem
                gender = 'female' if 'female' in voice_name.lower() else 'male'
                
                voices.append({
                    'id': voice_name,
                    'name': voice_name.replace('_', ' ').title(),
                    'gender': gender,
                    'file_path': str(voice_file),
                    'type': 'custom'
                })
        
        # Add built-in speakers if available
        if hasattr(self.tts, 'speakers') and self.tts.speakers:
            for i, speaker in enumerate(self.tts.speakers):
                voices.append({
                    'id': f"builtin_{i}",
                    'name': f"Built-in Speaker {i+1}",
                    'gender': 'unknown',
                    'type': 'builtin'
                })
        
        return voices
    
    def get_cost_estimate(self, character_count: int) -> float:
        """Get cost estimate for character count (free for local)"""
        return 0.0
    
    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.SUPPORTED_LANGUAGES
    
    def get_voice_gender(self, voice_id: str) -> str:
        """Get gender of voice"""
        if 'female' in voice_id.lower():
            return 'female'
        elif 'male' in voice_id.lower():
            return 'male'
        return 'unknown'
    
    def add_voice_sample(self, voice_id: str, audio_file_path: str) -> bool:
        """
        Add a custom voice sample for voice cloning
        
        Args:
            voice_id: Identifier for the voice
            audio_file_path: Path to the audio sample file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(audio_file_path)
            if not source_path.exists():
                logger.error(f"Voice sample file not found: {audio_file_path}")
                return False
            
            # Copy to voices directory
            target_path = self.voices_dir / f"{voice_id}.wav"
            
            if source_path.suffix.lower() == '.wav':
                shutil.copy2(source_path, target_path)
            else:
                # Convert to WAV if needed
                try:
                    waveform, sample_rate = torchaudio.load(str(source_path))
                    torchaudio.save(str(target_path), waveform, sample_rate)
                except Exception as e:
                    logger.error(f"Failed to convert audio file: {e}")
                    return False
            
            logger.info(f"Added voice sample: {voice_id} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add voice sample: {e}")
            return False 