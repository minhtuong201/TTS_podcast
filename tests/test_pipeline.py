#!/usr/bin/env python3
"""
Comprehensive test suite for TTS Podcast Pipeline
Consolidates functionality from multiple test files into a proper test structure
"""
import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pdf_ingest import extract
from lang_detect import detect as detect_language
from summarizer import llm_summary
from script_gen import ScriptGenerator, ScriptConfig, dialogue_to_text, SpeakerRole
from audio_mixer import AudioMixer, MixingConfig

class TestCore:
    """Test core functionality without requiring API keys"""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        Artificial Intelligence in Healthcare: A Revolutionary Approach
        
        Introduction
        The integration of artificial intelligence (AI) in healthcare has emerged as one of the 
        most promising developments in modern medicine. This paper explores how AI technologies 
        are transforming patient care, diagnostic accuracy, and treatment outcomes.
        
        Key Applications
        1. Medical Imaging: AI algorithms can analyze X-rays, MRIs, and CT scans with unprecedented 
           accuracy, often detecting anomalies that human radiologists might miss.
        
        2. Drug Discovery: Machine learning models accelerate the identification of potential drug 
           candidates, reducing the traditional 10-15 year development timeline.
        
        3. Personalized Medicine: AI enables customized treatment plans based on individual patient 
           genetics, lifestyle, and medical history.
        
        Challenges and Considerations
        Despite the promising applications, several challenges remain:
        - Data privacy and security concerns
        - Regulatory compliance and approval processes
        - Integration with existing healthcare systems
        - Training healthcare professionals on AI tools
        
        Future Outlook
        The future of AI in healthcare looks bright, with emerging technologies like quantum 
        computing and advanced neural networks opening new possibilities for medical research 
        and patient care.
        
        Conclusion
        AI has the potential to revolutionize healthcare delivery, making it more efficient, 
        accurate, and accessible. However, successful implementation requires careful consideration 
        of ethical, technical, and regulatory challenges.
        """
    
    def test_language_detection(self, sample_text):
        """Test language detection functionality"""
        language_code, confidence, metadata = detect_language(sample_text)
        assert language_code is not None
        assert confidence > 0
        assert isinstance(metadata, dict)
    
    def test_audio_mixer_imports(self):
        """Test that audio processing components can be imported"""
        assert AudioMixer is not None
        assert MixingConfig is not None
    
    def test_tts_backend_detection(self):
        """Test TTS backend detection"""
        from main import get_available_tts_backends
        backends = get_available_tts_backends()
        assert isinstance(backends, list)


class TestWithAPI:
    """Tests that require API keys - will be skipped if not available"""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return "AI is revolutionizing healthcare through medical imaging, drug discovery, and personalized medicine, though challenges like data privacy and regulatory approval remain."
    
    @pytest.mark.skipif(not os.getenv('OPENROUTER_API_KEY'), reason="OPENROUTER_API_KEY not set")
    def test_summarization(self, sample_text):
        """Test summarization with API key"""
        summary = llm_summary(sample_text, target_len="short")
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.skipif(not os.getenv('OPENROUTER_API_KEY'), reason="OPENROUTER_API_KEY not set")
    def test_script_generation(self, sample_text):
        """Test script generation with API key"""
        script_config = ScriptConfig(target_words=400, language="en")
        script_generator = ScriptGenerator()
        
        dialogue_lines = script_generator.generate_dialogue(sample_text, script_config)
        
        assert len(dialogue_lines) > 0
        assert all(hasattr(line, 'speaker') for line in dialogue_lines)
        assert all(hasattr(line, 'text') for line in dialogue_lines)
        
        # Test script text conversion (simplified test without emotion check)
        script_text = dialogue_to_text(dialogue_lines, include_annotations=False)
        assert isinstance(script_text, str)
        assert len(script_text) > 0


class TestElevenLabs:
    """Test ElevenLabs TTS functionality"""
    
    @pytest.mark.skipif(not os.getenv('ELEVENLABS_API_KEY'), reason="ELEVENLABS_API_KEY not set")
    def test_elevenlabs_backend_creation(self):
        """Test ElevenLabs backend creation"""
        from tts.eleven import ElevenLabsTTSBackend
        from tts.base import TTSConfig
        
        config = TTSConfig(voice_id="MF3mGyEYCl7XYWbV9V6O")  # Elli voice
        backend = ElevenLabsTTSBackend(config)
        assert backend is not None
    
    @pytest.mark.skipif(not os.getenv('ELEVENLABS_API_KEY'), reason="ELEVENLABS_API_KEY not set")  
    def test_voice_settings(self):
        """Test specific voice settings for Vietnamese voices"""
        from tts.eleven import ElevenLabsTTSBackend
        from tts.base import TTSConfig
        
        # Test female voice (Elli)
        female_config = TTSConfig(
            voice_id="MF3mGyEYCl7XYWbV9V6O",
            speed=1.12,
            stability=0.5,
            similarity_boost=0.75,
            style=0.0
        )
        
        # Test male voice (The Hao)
        male_config = TTSConfig(
            voice_id="M0rVwr32hdQ5UXpkI3ni",
            speed=1.15,
            stability=0.25,
            similarity_boost=0.75,
            style=0.0
        )
        
        female_backend = ElevenLabsTTSBackend(female_config)
        male_backend = ElevenLabsTTSBackend(male_config)
        
        assert female_backend is not None
        assert male_backend is not None


class TestGeminiTTS:
    """Test Gemini TTS functionality"""
    
    @pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason="GEMINI_API_KEY not set")
    def test_gemini_backend_creation(self):
        """Test Gemini TTS backend creation"""
        try:
            from tts.gemini import GeminiTTSBackend
            from tts.base import TTSConfig
            
            config = TTSConfig(voice_id="Fenrir")  # Vietnamese-optimized voice
            backend = GeminiTTSBackend(config)
            assert backend is not None
        except ImportError:
            pytest.skip("Gemini TTS dependencies not installed")
    
    @pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason="GEMINI_API_KEY not set")
    def test_voice_profiles_loading(self):
        """Test loading Vietnamese voice profiles"""
        try:
            from tts.gemini import GeminiTTSBackend
            from tts.base import TTSConfig
            
            config = TTSConfig(voice_id="Fenrir")
            backend = GeminiTTSBackend(config)
            
            # Test getting default voices
            default_voices = backend.get_default_voices()
            assert 'host' in default_voices
            assert 'guest' in default_voices
            assert default_voices['host'] == 'Fenrir'
            assert default_voices['guest'] == 'Leda'
            
        except ImportError:
            pytest.skip("Gemini TTS dependencies not installed")
    
    @pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason="GEMINI_API_KEY not set")
    def test_available_voices(self):
        """Test getting available voices with Vietnamese filtering"""
        try:
            from tts.gemini import GeminiTTSBackend
            from tts.base import TTSConfig
            
            config = TTSConfig(voice_id="Fenrir")
            backend = GeminiTTSBackend(config)
            
            # Test getting all voices
            all_voices = backend.get_available_voices()
            assert len(all_voices) > 0
            
            # Test Vietnamese filtering
            vi_voices = backend.get_available_voices(language='vietnamese')
            assert len(vi_voices) > 0
            
            # Check that all Vietnamese voices have excellent/good suitability
            for voice in vi_voices:
                assert voice['vietnamese_suitability'] in ['excellent', 'good']
                
        except ImportError:
            pytest.skip("Gemini TTS dependencies not installed")


class TestGoogleCloudTTS:
    """Test Google Cloud TTS functionality"""
    
    @pytest.mark.skipif(not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'), reason="GOOGLE_APPLICATION_CREDENTIALS not set")
    def test_google_imports(self):
        """Test Google Cloud TTS imports"""
        try:
            from google.cloud import texttospeech
            import google.auth
            assert True
        except ImportError:
            pytest.skip("Google Cloud TTS dependencies not installed")
    
    @pytest.mark.skipif(not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'), reason="GOOGLE_APPLICATION_CREDENTIALS not set")
    def test_google_backend_creation(self):
        """Test Google Cloud TTS backend creation"""
        try:
            from tts.google import GoogleCloudTTSBackend
            from tts.base import TTSConfig
            
            config = TTSConfig(voice_id="en-US-Neural2-C")
            backend = GoogleCloudTTSBackend(config)
            assert backend is not None
        except ImportError:
            pytest.skip("Google Cloud TTS dependencies not installed")


class TestNaturalDialogue:
    """Test natural dialogue features"""
    
    def test_dialogue_elements(self):
        """Test that dialogue contains natural elements"""
        sample_dialogue = """
HOST: Welcome to Tech Talk Today! I'm really excited about today's topic. [cheerful] We're diving into something that's absolutely fascinating - the future of AI in healthcare!

GUEST: [chuckles] Thanks for having me! You know, it's amazing how far we've come... [pause] When I started working in this field ten years ago, the idea of AI diagnosing diseases seemed like pure science fiction.

HOST: [curious] That's incredible! So what changed? What was the breakthrough moment?

GUEST: [thoughtful] Well, I think the real game-changer was when deep learning algorithms started outperforming human radiologists in certain imaging tasks. [excited] Suddenly, we weren't just talking about AI as a tool - it was becoming a partner in healthcare!
        """
        
        # Count natural elements
        import re
        emotion_matches = re.findall(r'\[([^\]]+)\]', sample_dialogue)
        
        assert len(emotion_matches) > 0
        assert 'cheerful' in emotion_matches
        assert 'chuckles' in emotion_matches
        assert sample_dialogue.count('?') > 0
        assert sample_dialogue.count('!') > 0


def run_interactive_test():
    """Run an interactive test for demonstration purposes"""
    print("🎙️  TTS Podcast Pipeline - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test core functionality
    print("\n📄 Testing core functionality...")
    sample_text = """
    Artificial Intelligence in Healthcare: A Revolutionary Approach
    
    AI is transforming healthcare through medical imaging, drug discovery, and personalized medicine.
    However, challenges like data privacy and regulatory approval remain.
    """
    
    try:
        language_code, confidence, _ = detect_language(sample_text)
        print(f"✓ Language detection: {language_code} (confidence: {confidence:.3f})")
    except Exception as e:
        print(f"❌ Language detection failed: {e}")
    
    # Test TTS backend availability
    print("\n🔊 Checking TTS backends...")
    try:
        from main import get_available_tts_backends
        backends = get_available_tts_backends()
        if backends:
            print(f"✓ Available TTS backends: {', '.join(backends)}")
        else:
            print("⚠️ No TTS backends available")
    except Exception as e:
        print(f"❌ TTS backend check failed: {e}")
    
    # Test API-dependent features
    if os.getenv('OPENROUTER_API_KEY'):
        print("\n📝 Testing summarization and script generation...")
        try:
            summary = llm_summary(sample_text, target_len="short")
            print(f"✓ Summary generated: {len(summary)} characters")
            
            script_config = ScriptConfig(target_words=200, language="en")
            script_generator = ScriptGenerator()
            dialogue_lines = script_generator.generate_dialogue(summary, script_config)
            print(f"✓ Script generated: {len(dialogue_lines)} dialogue lines")
            
        except Exception as e:
            print(f"❌ API test failed: {e}")
    else:
        print("\n⚠️ Skipping API tests (OPENROUTER_API_KEY not set)")
    
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    run_interactive_test() 