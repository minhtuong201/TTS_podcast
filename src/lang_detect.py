"""
Language detection module for TTS Podcast Pipeline
"""
import logging
from typing import Tuple, Optional
import re

from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from utils.log_cfg import PipelineTimer, log_pipeline_metrics

logger = logging.getLogger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0

# Language code mappings for TTS services
LANGUAGE_MAPPINGS = {
    'en': {'name': 'English', 'direction': 'ltr', 'tts_supported': True},
    'es': {'name': 'Spanish', 'direction': 'ltr', 'tts_supported': True},
    'fr': {'name': 'French', 'direction': 'ltr', 'tts_supported': True},
    'de': {'name': 'German', 'direction': 'ltr', 'tts_supported': True},
    'it': {'name': 'Italian', 'direction': 'ltr', 'tts_supported': True},
    'pt': {'name': 'Portuguese', 'direction': 'ltr', 'tts_supported': True},
    'ru': {'name': 'Russian', 'direction': 'ltr', 'tts_supported': True},
    'ja': {'name': 'Japanese', 'direction': 'ltr', 'tts_supported': True},
    'ko': {'name': 'Korean', 'direction': 'ltr', 'tts_supported': True},
    'zh-cn': {'name': 'Chinese (Simplified)', 'direction': 'ltr', 'tts_supported': True},
    'zh-tw': {'name': 'Chinese (Traditional)', 'direction': 'ltr', 'tts_supported': True},
    'ar': {'name': 'Arabic', 'direction': 'rtl', 'tts_supported': True},
    'hi': {'name': 'Hindi', 'direction': 'ltr', 'tts_supported': True},
    'tr': {'name': 'Turkish', 'direction': 'ltr', 'tts_supported': True},
    'pl': {'name': 'Polish', 'direction': 'ltr', 'tts_supported': True},
    'nl': {'name': 'Dutch', 'direction': 'ltr', 'tts_supported': True},
    'sv': {'name': 'Swedish', 'direction': 'ltr', 'tts_supported': True},
    'da': {'name': 'Danish', 'direction': 'ltr', 'tts_supported': True},
    'no': {'name': 'Norwegian', 'direction': 'ltr', 'tts_supported': True},
    'fi': {'name': 'Finnish', 'direction': 'ltr', 'tts_supported': True},
}

# Voice mappings for each language and TTS service
VOICE_MAPPINGS = {
    'en': {
        'eleven': {'female': 'Rachel', 'male': 'Daniel'},
        'openai': {'female': 'nova', 'male': 'onyx'},
        'azure': {'female': 'en-US-JennyNeural', 'male': 'en-US-GuyNeural'},
    },
    'es': {
        'eleven': {'female': 'Sofia', 'male': 'Alonso'},
        'azure': {'female': 'es-ES-ElviraNeural', 'male': 'es-ES-AlvaroNeural'},
    },
    'fr': {
        'eleven': {'female': 'Charlotte', 'male': 'Henri'},
        'azure': {'female': 'fr-FR-DeniseNeural', 'male': 'fr-FR-HenriNeural'},
    },
    'de': {
        'eleven': {'female': 'Freya', 'male': 'Chris'},
        'azure': {'female': 'de-DE-KatjaNeural', 'male': 'de-DE-ConradNeural'},
    },
    # Add more languages as needed
}


def detect(text: str, sample_size: int = 1000) -> Tuple[str, float, dict]:
    """
    Detect language of input text
    
    Args:
        text: Input text to analyze
        sample_size: Number of characters to sample for detection
        
    Returns:
        Tuple of (language_code, confidence, metadata)
        
    Raises:
        LangDetectException: If language detection fails
    """
    if not text or not text.strip():
        raise ValueError("Text is empty or whitespace only")
    
    with PipelineTimer("Language detection", logger):
        try:
            # Clean text for better detection
            clean_text = clean_text_for_detection(text)
            
            # Use sample if text is very long
            if len(clean_text) > sample_size:
                clean_text = clean_text[:sample_size]
                logger.info(f"Using sample of {sample_size} characters for language detection")
            
            # Detect language with probabilities
            from langdetect import detect as detect_lang, detect_langs
            
            # Get primary language
            primary_lang = detect_lang(clean_text)
            
            # Get all language probabilities
            lang_probs = detect_langs(clean_text)
            confidence = max(prob.prob for prob in lang_probs if prob.lang == primary_lang)
            
            # Handle Chinese variants
            if primary_lang == 'zh':
                primary_lang = detect_chinese_variant(text)
            
            # Get language metadata
            metadata = get_language_metadata(primary_lang, lang_probs)
            
            # Log metrics
            metrics = {
                'detected_language': primary_lang,
                'confidence': confidence,
                'text_sample_length': len(clean_text),
                'all_probabilities': {prob.lang: prob.prob for prob in lang_probs}
            }
            log_pipeline_metrics("language_detection", metrics, logger)
            
            logger.info(f"Detected language: {primary_lang} (confidence: {confidence:.3f})")
            
            return primary_lang, confidence, metadata
            
        except LangDetectException as e:
            logger.error(f"Language detection failed: {e}")
            # Fallback to English
            logger.warning("Falling back to English")
            return 'en', 0.5, get_language_metadata('en', [])
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            raise


def clean_text_for_detection(text: str) -> str:
    """
    Clean text to improve language detection accuracy
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text suitable for language detection
    """
    if not text:
        return ""
    
    # Remove URLs, emails, and special characters that might confuse detection
    import re
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove excessive numbers and special characters
    text = re.sub(r'\d{4,}', '', text)  # Remove long numbers
    text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)  # Keep basic punctuation
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Focus on sentences (take complete sentences)
    sentences = re.split(r'[.!?]+', text)
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if meaningful_sentences:
        text = '. '.join(meaningful_sentences[:10])  # Use first 10 meaningful sentences
    
    return text.strip()


def detect_chinese_variant(text: str) -> str:
    """
    Detect whether Chinese text is Simplified or Traditional
    
    Args:
        text: Chinese text
        
    Returns:
        'zh-cn' for Simplified or 'zh-tw' for Traditional
    """
    # Simple heuristic: count traditional characters
    traditional_chars = set('繁體中文檢測')  # Add more traditional characters as needed
    simplified_chars = set('简体中文检测')   # Add more simplified characters as needed
    
    trad_count = sum(1 for char in text if char in traditional_chars)
    simp_count = sum(1 for char in text if char in simplified_chars)
    
    if trad_count > simp_count:
        return 'zh-tw'
    else:
        return 'zh-cn'


def get_language_metadata(lang_code: str, lang_probs: list) -> dict:
    """
    Get metadata for detected language
    
    Args:
        lang_code: Language code
        lang_probs: List of language probabilities
        
    Returns:
        Dictionary with language metadata
    """
    metadata = LANGUAGE_MAPPINGS.get(lang_code, {
        'name': f'Unknown ({lang_code})',
        'direction': 'ltr',
        'tts_supported': False
    }).copy()
    
    metadata.update({
        'code': lang_code,
        'alternative_languages': [
            {'code': prob.lang, 'probability': prob.prob}
            for prob in lang_probs[:5]  # Top 5 alternatives
        ]
    })
    
    return metadata


def get_voices_for_language(lang_code: str, tts_backend: str = 'eleven') -> Optional[dict]:
    """
    Get available voices for a language and TTS backend
    
    Args:
        lang_code: Language code
        tts_backend: TTS backend ('eleven', 'openai', 'azure', 'coqui')
        
    Returns:
        Dictionary with male and female voice options, or None if not supported
    """
    voices = VOICE_MAPPINGS.get(lang_code, {}).get(tts_backend)
    
    if not voices:
        logger.warning(f"No voices available for {lang_code} with {tts_backend}")
        # Fallback to English voices if available
        if lang_code != 'en':
            logger.info("Falling back to English voices")
            voices = VOICE_MAPPINGS.get('en', {}).get(tts_backend)
    
    return voices


def is_language_supported(lang_code: str, tts_backend: str = 'eleven') -> bool:
    """
    Check if a language is supported by the specified TTS backend
    
    Args:
        lang_code: Language code
        tts_backend: TTS backend name
        
    Returns:
        True if language is supported
    """
    lang_info = LANGUAGE_MAPPINGS.get(lang_code, {})
    has_voices = get_voices_for_language(lang_code, tts_backend) is not None
    
    return lang_info.get('tts_supported', False) and has_voices


def get_text_direction(lang_code: str) -> str:
    """
    Get text direction for language (ltr or rtl)
    
    Args:
        lang_code: Language code
        
    Returns:
        'ltr' for left-to-right or 'rtl' for right-to-left
    """
    return LANGUAGE_MAPPINGS.get(lang_code, {}).get('direction', 'ltr') 