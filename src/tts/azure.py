"""
Azure Speech Services TTS Backend
Provides high-quality text-to-speech with extensive language and voice support
"""
import logging
import os
import tempfile
from typing import Optional, Dict, List
from dataclasses import dataclass
import io

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from .base import TTSBackend, TTSResult, TTSConfig

logger = logging.getLogger(__name__)


@dataclass
class AzureVoiceConfig:
    """Azure-specific voice configuration"""
    voice_name: str
    language: str
    gender: str
    style: Optional[str] = None  # cheerful, sad, excited, etc.
    style_degree: float = 1.0    # 0.01 to 2.0


class AzureTTSBackend(TTSBackend):
    """Azure Speech Services TTS implementation"""
    
    # Popular Azure voices by language and gender
    VOICE_MAPPING = {
        'en': {
            'female': [
                'en-US-AriaNeural',      # Natural, conversational
                'en-US-JennyNeural',     # Friendly, warm
                'en-GB-SoniaNeural',     # British accent
                'en-AU-NatashaNeural'    # Australian accent
            ],
            'male': [
                'en-US-GuyNeural',       # Confident, clear
                'en-US-DavisNeural',     # Professional
                'en-GB-RyanNeural',      # British accent
                'en-AU-WilliamNeural'    # Australian accent
            ]
        },
        'es': {
            'female': ['es-ES-ElviraNeural', 'es-MX-DaliaNeural'],
            'male': ['es-ES-AlvaroNeural', 'es-MX-JorgeNeural']
        },
        'fr': {
            'female': ['fr-FR-DeniseNeural', 'fr-CA-SylvieNeural'],
            'male': ['fr-FR-HenriNeural', 'fr-CA-AntoineNeural']
        },
        'de': {
            'female': ['de-DE-KatjaNeural', 'de-AT-IngridNeural'],
            'male': ['de-DE-ConradNeural', 'de-CH-JanNeural']
        },
        'it': {
            'female': ['it-IT-ElsaNeural'],
            'male': ['it-IT-DiegoNeural']
        },
        'pt': {
            'female': ['pt-BR-FranciscaNeural', 'pt-PT-RaquelNeural'],
            'male': ['pt-BR-AntonioNeural', 'pt-PT-DuarteNeural']
        },
        'ja': {
            'female': ['ja-JP-NanamiNeural'],
            'male': ['ja-JP-KeitaNeural']
        },
        'ko': {
            'female': ['ko-KR-SunHiNeural'],
            'male': ['ko-KR-InJoonNeural']
        },
        'zh': {
            'female': ['zh-CN-XiaoxiaoNeural', 'zh-TW-HsiaoChenNeural'],
            'male': ['zh-CN-YunxiNeural', 'zh-TW-YunJheNeural']
        }
    }
    
    # Speaking styles that work with neural voices
    SPEAKING_STYLES = [
        'cheerful', 'chat', 'customerservice', 'newscast', 
        'assistant', 'friendly', 'excited', 'sad', 'angry',
        'fearful', 'disgruntled', 'serious', 'calm'
    ]
    
    def __init__(self, config: TTSConfig):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Speech SDK not available. Install with: pip install azure-cognitiveservices-speech")
        
        super().__init__(config)
        
        # Get Azure credentials
        self.speech_key = os.getenv('AZURE_SPEECH_KEY')
        self.speech_region = os.getenv('AZURE_SPEECH_REGION')
        
        if not self.speech_key or not self.speech_region:
            raise ValueError("Azure Speech credentials not found. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables.")
        
        # Configure Azure Speech SDK
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        
        # Set audio format (48kHz, 16-bit, mono)
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio48Khz16BitMonoPcm
        )
        
        # Auto-select voices if not specified
        self._setup_voices()
        
        # Initialize synthesizer
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None  # We'll handle audio output manually
        )
        
        logger.info(f"Initialized Azure TTS backend for language: {config.language}")
    
    def _setup_voices(self):
        """Setup default voices based on language"""
        lang_voices = self.VOICE_MAPPING.get(self.config.language, self.VOICE_MAPPING['en'])
        
        if not self.config.female_voice:
            self.config.female_voice = lang_voices['female'][0]
        
        if not self.config.male_voice:
            self.config.male_voice = lang_voices['male'][0]
        
        logger.info(f"Using voices - Female: {self.config.female_voice}, Male: {self.config.male_voice}")
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices"""
        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            voices_result = synthesizer.get_voices_async().get()
            
            if voices_result.reason == speechsdk.ResultReason.VoicesListRetrieved:
                voices = []
                for voice in voices_result.voices:
                    if voice.locale.startswith(self.config.language):
                        voices.append({
                            'id': voice.short_name,
                            'name': voice.local_name,
                            'gender': voice.gender.name.lower(),
                            'locale': voice.locale,
                            'style_list': getattr(voice, 'style_list', [])
                        })
                return voices
            else:
                logger.warning("Could not retrieve voice list from Azure")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving Azure voices: {e}")
            return []
    
    def _build_ssml(self, text: str, voice_name: str, style: Optional[str] = None) -> str:
        """Build SSML markup for enhanced speech synthesis"""
        
        # Extract emotion annotations from text
        import re
        
        # Clean text and extract annotations
        cleaned_text = text
        emotions = []
        
        # Find emotion markers like [laughs], [pause], [excited]
        emotion_pattern = r'\[([^\]]+)\]'
        emotion_matches = re.findall(emotion_pattern, text)
        cleaned_text = re.sub(emotion_pattern, '', text).strip()
        
        # Determine speaking style based on emotions
        speaking_style = style
        if not speaking_style and emotion_matches:
            emotion = emotion_matches[0].lower()
            if emotion in ['happy', 'excited', 'laugh', 'chuckle']:
                speaking_style = 'cheerful'
            elif emotion in ['serious', 'thoughtful']:
                speaking_style = 'serious'
            elif emotion in ['friendly', 'warm']:
                speaking_style = 'friendly'
            elif emotion in ['sad', 'disappointed']:
                speaking_style = 'sad'
        
        # Build SSML
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.config.language}">'
        ssml += f'<voice name="{voice_name}">'
        
        # Add speaking style if supported
        if speaking_style and speaking_style in self.SPEAKING_STYLES:
            ssml += f'<mstts:express-as style="{speaking_style}">'
            ssml += cleaned_text
            ssml += '</mstts:express-as>'
        else:
            ssml += cleaned_text
        
        ssml += '</voice></speak>'
        
        return ssml
    
    def synthesize(self, text: str, voice_id: str) -> TTSResult:
        """
        Synthesize text to speech using Azure Speech Services
        
        Args:
            text: Text to synthesize
            voice_id: Azure voice name (e.g., 'en-US-AriaNeural')
            
        Returns:
            TTSResult with audio data and metadata
        """
        try:
            logger.debug(f"Synthesizing with Azure voice: {voice_id}")
            
            # Build SSML for natural speech
            ssml = self._build_ssml(text, voice_id)
            
            # Synthesize speech
            result = self.synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingSpeechCompleted:
                # Convert to our audio format
                audio_data = result.audio_data
                
                # Estimate duration (48kHz, 16-bit, mono = 96000 bytes per second)
                duration_seconds = len(audio_data) / 96000.0
                
                # Estimate cost (approximate)
                character_count = len(text)
                cost_per_char = 0.000016  # ~$16 per 1M characters
                cost_estimate = character_count * cost_per_char
                
                logger.debug(f"Azure synthesis successful: {duration_seconds:.1f}s, {character_count} chars")
                
                return TTSResult(
                    audio_data=audio_data,
                    format="wav",
                    duration_seconds=duration_seconds,
                    character_count=character_count,
                    cost_estimate=cost_estimate,
                    metadata={
                        'voice_id': voice_id,
                        'backend': 'azure',
                        'sample_rate': 48000,
                        'bit_depth': 16,
                        'channels': 1
                    }
                )
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                error_msg = f"Azure TTS canceled: {cancellation_details.reason}"
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_msg += f" - {cancellation_details.error_details}"
                
                logger.error(error_msg)
                raise Exception(error_msg)
            
            else:
                raise Exception(f"Azure TTS failed with reason: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure TTS synthesis failed: {e}")
            raise
    
    def get_cost_estimate(self, character_count: int) -> float:
        """Get cost estimate for character count"""
        # Azure pricing: ~$16 per 1M characters (Neural voices)
        return character_count * 0.000016
    
    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.VOICE_MAPPING
    
    def get_voice_gender(self, voice_id: str) -> str:
        """Get gender of voice"""
        for lang, voices in self.VOICE_MAPPING.items():
            if voice_id in voices['female']:
                return 'female'
            elif voice_id in voices['male']:
                return 'male'
        return 'unknown' 