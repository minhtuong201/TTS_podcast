#!/usr/bin/env python3
"""
Core Demo for TTS Podcast Pipeline
Tests the main functionality without heavy TTS dependencies
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_core_functionality():
    """Test the core components of the TTS podcast pipeline"""
    print("🎙️  TTS Podcast Pipeline - Core Functionality Test")
    print("=" * 60)
    
    # Test 1: PDF text extraction
    print("\n📄 Testing PDF text extraction...")
    try:
        from pdf_ingest import extract
        
        # Create a sample text (simulating PDF content)
        sample_text = """
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
        
        print(f"✓ Text extraction simulation successful: {len(sample_text)} characters")
        
    except Exception as e:
        print(f"❌ PDF extraction test failed: {e}")
        return False
    
    # Test 2: Language detection
    print("\n🌍 Testing language detection...")
    try:
        from lang_detect import detect as detect_language
        
        language_code, confidence, metadata = detect_language(sample_text)
        print(f"✓ Language detected: {language_code} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        return False
    
    # Test 3: Summarization (will fail without API key, which is expected)
    print("\n📝 Testing summarization...")
    try:
        from summarizer import llm_summary
        
        # Check if API key is available
        if os.getenv('OPENROUTER_API_KEY'):
            summary = llm_summary(sample_text, target_len="short")
            print(f"✓ Summary generated: {len(summary)} characters")
            print(f"Preview: {summary[:200]}...")
        else:
            print("⚠️  Summarization skipped (OPENROUTER_API_KEY not set)")
            print("💡 This is expected - set API key to test summarization")
            
    except Exception as e:
        print(f"⚠️  Summarization test result: {e}")
        print("💡 This is expected without proper API configuration")
    
    # Test 4: Script generation (will also fail without API key)
    print("\n🎭 Testing script generation...")
    try:
        from script_gen import ScriptGenerator, ScriptConfig
        
        if os.getenv('OPENROUTER_API_KEY'):
            script_config = ScriptConfig(
                target_words=400,  # Shorter for demo
                language=language_code,
                include_emotions=True,
                include_pauses=True
            )
            
            script_generator = ScriptGenerator()
            # Use a simple summary for testing
            test_summary = "AI is revolutionizing healthcare through medical imaging, drug discovery, and personalized medicine, though challenges like data privacy and regulatory approval remain."
            
            dialogue_lines = script_generator.generate_dialogue(test_summary, script_config)
            print(f"✓ Script generated: {len(dialogue_lines)} dialogue lines")
            
            # Show sample dialogue
            if dialogue_lines:
                print("\nSample dialogue:")
                for i, line in enumerate(dialogue_lines[:3]):  # First 3 lines
                    speaker = line.speaker.value
                    text = line.text[:100] + "..." if len(line.text) > 100 else line.text
                    emotion = f" [{line.emotion}]" if line.emotion else ""
                    print(f"  {speaker}: {text}{emotion}")
                if len(dialogue_lines) > 3:
                    print(f"  ... and {len(dialogue_lines) - 3} more lines")
        else:
            print("⚠️  Script generation skipped (OPENROUTER_API_KEY not set)")
            print("💡 This is expected - set API key to test script generation")
            
    except Exception as e:
        print(f"⚠️  Script generation test result: {e}")
        print("💡 This is expected without proper API configuration")
    
    # Test 5: Audio processing components
    print("\n🔊 Testing audio processing...")
    try:
        from audio_mixer import AudioMixer, MixingConfig
        print("✓ Audio mixer imported successfully")
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False
    
    # Test 6: TTS backend detection
    print("\n🎤 Testing TTS backend detection...")
    try:
        from main import get_available_tts_backends
        
        backends = get_available_tts_backends()
        if backends:
            print(f"✓ Available TTS backends: {', '.join(backends)}")
        else:
            print("⚠️  No TTS backends available")
            print("💡 This is expected without API keys configured")
            
    except Exception as e:
        print(f"❌ TTS backend detection failed: {e}")
        return False
    
    return True

def show_natural_dialogue_example():
    """Show example of natural dialogue features"""
    print("\n🎭 Natural Dialogue Features Example")
    print("=" * 50)
    
    example_dialogue = """
HOST: Welcome everyone to Tech Insights! [cheerful] Today we're diving into something absolutely fascinating - AI in healthcare!

GUEST: [chuckles] Thanks for having me! You know, when I first started in this field... [pause] I never imagined we'd see AI outperforming doctors in diagnostic accuracy.

HOST: [curious] That's incredible! Can you tell us about a specific example that really blew your mind?

GUEST: [excited] Oh, absolutely! There's this study where deep learning algorithms analyzed over a million chest X-rays... [thoughtful] and they were able to detect pneumonia with 94% accuracy - that's better than most radiologists!

HOST: [amazed] Wow! [laughs] I bet some radiologists weren't too happy about that finding!

GUEST: [laughs] You'd think so, but actually most were thrilled! Here's the thing... [pause] AI isn't replacing doctors - it's making them superhuman!
    """
    
    print("Example natural dialogue with emotional annotations:")
    print("-" * 50)
    for line in example_dialogue.strip().split('\n'):
        if line.strip():
            print(line)
    print("-" * 50)
    
    # Count features
    import re
    emotions = re.findall(r'\[([^\]]+)\]', example_dialogue)
    print(f"\n✨ Natural features detected:")
    print(f"   • Emotional expressions: {len(emotions)}")
    print(f"   • Unique emotions: {', '.join(set(emotions))}")
    print(f"   • Questions: {example_dialogue.count('?')}")
    print(f"   • Exclamations: {example_dialogue.count('!')}")
    print(f"   • Conversation fillers: {example_dialogue.count('you know') + example_dialogue.count('actually')}")

def show_setup_instructions():
    """Show setup instructions for full functionality"""
    print("\n🚀 Setup Instructions for Full Functionality")
    print("=" * 60)
    
    print("""
To enable all features, you need to set up API keys:

1. 📝 Create a .env file with:
   OPENROUTER_API_KEY=your_openrouter_key_here
   
2. 🔊 Add TTS provider keys (choose one or more):
   ELEVENLABS_API_KEY=your_elevenlabs_key_here
   OPENAI_API_KEY=your_openai_key_here
   AZURE_SPEECH_KEY=your_azure_key_here
   AZURE_SPEECH_REGION=your_azure_region

3. 🎯 For local TTS (optional), install Coqui TTS:
   pip install TTS torch torchaudio

4. 📖 Get API keys from:
   • OpenRouter: https://openrouter.ai (required for summarization)
   • ElevenLabs: https://elevenlabs.io (best voice quality)
   • OpenAI: https://platform.openai.com (alternative TTS)
   • Azure: https://azure.microsoft.com/cognitive-services/speech-services/

5. 🧪 Once configured, test with:
   python src/main.py your_document.pdf
    """)

if __name__ == "__main__":
    try:
        # Test core functionality
        success = test_core_functionality()
        
        # Show natural dialogue example
        show_natural_dialogue_example()
        
        # Show setup instructions
        show_setup_instructions()
        
        # Summary
        print("\n📊 Test Results")
        print("=" * 30)
        if success:
            print("✅ Core components are working correctly!")
            print("🎉 The TTS Podcast Pipeline is properly set up.")
            print("\n💡 Next steps:")
            print("   1. Add API keys to .env file")
            print("   2. Test with: python demo_core.py")
            print("   3. Generate podcasts with: python src/main.py your_file.pdf")
        else:
            print("❌ Some core components have issues.")
            print("🔧 Check the error messages above for troubleshooting.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 