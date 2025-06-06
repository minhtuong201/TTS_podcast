"""Azure Speech Services TTS Backend"""
import logging
import os
from typing import Optional, Dict, Any, List

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AZURE_AVAILABLE = False

from .base import BaseTTSBackend, TTSConfig, SynthesisResult, AudioFormat

logger = logging.getLogger(__name__)


class AzureTTSBackend(BaseTTSBackend):
    """Simplified Azure Speech Services backend."""

    def __init__(self, config: TTSConfig):
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Speech SDK not available. Install with: pip install azure-cognitiveservices-speech"
            )

        super().__init__(api_key=None)
        self.config = config

        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        if not self.speech_key or not self.speech_region:
            raise ValueError(
                "Azure Speech credentials not found. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars."
            )

        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.speech_region
        )
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio48Khz16BitMonoPcm
        )

        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )

    # ------------------------------------------------------------------
    # BaseTTSBackend API
    # ------------------------------------------------------------------
    def get_max_text_length(self) -> int:
        return 5000

    def supports_ssml(self) -> bool:
        return True

    def synthesize(self, text: str, config: TTSConfig) -> SynthesisResult:
        self.validate_text(text)
        voice_id = config.voice_id
        ssml = f"<speak><voice name='{voice_id}'>{text}</voice></speak>"
        result = self.synthesizer.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingSpeechCompleted:
            raise RuntimeError("Azure synthesis failed")

        audio = result.audio_data
        duration = len(audio) / 96000.0
        cost = self.estimate_cost(text, config)
        return SynthesisResult(
            audio_data=audio,
            format=AudioFormat.WAV,
            duration_seconds=duration,
            sample_rate=48000,
            character_count=len(text),
            cost_estimate=cost,
        )

    def get_available_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            voices_result = self.synthesizer.get_voices_async().get()
            voices = []
            for voice in voices_result.voices:
                if language and not voice.locale.startswith(language):
                    continue
                voices.append(
                    {
                        "id": voice.short_name,
                        "name": voice.local_name,
                        "gender": voice.gender.name.lower(),
                        "locale": voice.locale,
                    }
                )
            return voices
        except Exception as e:  # pragma: no cover - API failure
            logger.error(f"Error retrieving Azure voices: {e}")
            return []

    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        for voice in self.get_available_voices():
            if voice["id"] == voice_id:
                return voice
        raise ValueError(f"Voice not found: {voice_id}")

    def estimate_cost(self, text: str, config: TTSConfig) -> float:
        return len(text) * 0.000016  # ~$16 per 1M characters

    def get_service_limits(self) -> Dict[str, Any]:
        return {
            "max_characters": 5000,
            "sample_rate": 48000,
        }
