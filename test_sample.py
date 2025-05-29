#!/usr/bin/env python3
"""
Sample test script for TTS Podcast Pipeline
Demonstrates the complete workflow from PDF to podcast
"""
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_ingest import extract
from lang_detect import detect as detect_language
from summarizer import llm_summary
from script_gen import ScriptGenerator, ScriptConfig, dialogue_to_text

def create_sample_pdf():
    """Create a sample PDF-like text for testing"""
    sample_text = """
# The Future of Artificial Intelligence in Healthcare

## Abstract
Artificial Intelligence (AI) is revolutionizing healthcare by enabling more accurate diagnostics, 
personalized treatment plans, and improved patient outcomes. This paper explores the current 
applications of AI in healthcare and discusses future possibilities.

## Introduction
Healthcare is experiencing a paradigm shift with the integration of artificial intelligence 
technologies. From medical imaging to drug discovery, AI is transforming how healthcare 
professionals diagnose, treat, and prevent diseases.

## Key Applications

### Medical Imaging
AI algorithms can analyze medical images with remarkable accuracy, often surpassing human 
radiologists in detecting certain conditions. Deep learning models trained on millions of 
medical images can identify patterns invisible to the human eye.

### Drug Discovery
Traditional drug discovery takes 10-15 years and costs billions of dollars. AI is accelerating 
this process by predicting molecular behavior, identifying promising compounds, and optimizing 
clinical trial designs.

### Personalized Medicine
AI enables personalized treatment by analyzing patient genetics, lifestyle, and medical history 
to recommend tailored therapies. This approach promises better outcomes with fewer side effects.

## Challenges and Considerations

### Data Privacy
Healthcare data is highly sensitive, requiring robust privacy protections and compliance with 
regulations like HIPAA. AI systems must be designed with privacy-by-design principles.

### Bias and Fairness
AI models can perpetuate biases present in training data, potentially leading to unfair 
treatment of certain patient populations. Ensuring diverse and representative datasets is crucial.

### Regulatory Approval
Medical AI systems require rigorous testing and regulatory approval before clinical deployment. 
The FDA has established frameworks for AI/ML-based medical devices.

## Future Outlook
The future of AI in healthcare looks promising, with emerging technologies like quantum computing 
and advanced neural networks opening new possibilities. However, successful implementation 
requires careful consideration of ethical, technical, and regulatory challenges.

## Conclusion
AI has the potential to transform healthcare delivery, making it more precise, personalized, 
and accessible. As we continue to advance these technologies, we must ensure they serve all 
patients equitably and safely.
    """
    return sample_text

def test_complete_pipeline():
    """Test the complete TTS podcast pipeline"""
    print("🎙️  TTS Podcast Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Create sample content
    print("\n📄 Step 1: Processing document content...")
    text = create_sample_pdf()
    print(f"✓ Document loaded: {len(text)} characters")
    
    # Step 2: Detect language
    print("\n🌍 Step 2: Detecting language...")
    language_code, confidence, lang_metadata = detect_language(text)
    print(f"✓ Language detected: {language_code} (confidence: {confidence:.3f})")
    
    # Step 3: Summarize
    print("\n📝 Step 3: Generating summary...")
    try:
        summary = llm_summary(text, target_len="short")
        print(f"✓ Summary generated: {len(summary)} characters")
        print("\nSummary preview:")
        print("-" * 30)
        print(summary[:300] + "..." if len(summary) > 300 else summary)
        print("-" * 30)
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        print("💡 Make sure you have OPENROUTER_API_KEY set in your .env file")
        return False
    
    # Step 4: Generate script
    print("\n🎭 Step 4: Generating dialogue script...")
    try:
        script_config = ScriptConfig(
            target_words=600,  # Shorter for demo
            language=language_code,
            include_emotions=True,
            include_pauses=True
        )
        
        script_generator = ScriptGenerator()
        dialogue_lines = script_generator.generate_dialogue(summary, script_config)
        
        print(f"✓ Script generated: {len(dialogue_lines)} dialogue lines")
        
        # Show script preview
        print("\nScript preview:")
        print("-" * 40)
        script_text = dialogue_to_text(dialogue_lines, include_annotations=True)
        lines = script_text.split('\n')[:10]  # First 10 lines
        for line in lines:
            if line.strip():
                print(line)
        print("...")
        print("-" * 40)
        
        # Show natural elements
        emotion_count = sum(1 for line in dialogue_lines if line.emotion)
        pause_count = sum(1 for line in dialogue_lines if line.pause_before > 0 or line.pause_after > 0)
        
        print(f"\n✨ Natural dialogue features:")
        print(f"   • Emotional expressions: {emotion_count} lines")
        print(f"   • Natural pauses: {pause_count} lines")
        print(f"   • Host lines: {sum(1 for line in dialogue_lines if line.speaker.value == 'HOST')}")
        print(f"   • Guest lines: {sum(1 for line in dialogue_lines if line.speaker.value == 'GUEST')}")
        
    except Exception as e:
        print(f"❌ Script generation failed: {e}")
        print("💡 Make sure you have OPENROUTER_API_KEY set in your .env file")
        return False
    
    # Step 5: Check available TTS backends
    print("\n🔊 Step 5: Checking TTS backends...")
    from main import get_available_tts_backends
    
    available_backends = get_available_tts_backends()
    if available_backends:
        print(f"✓ Available TTS backends: {', '.join(available_backends)}")
        
        # Suggest next steps
        print(f"\n🚀 Ready for audio synthesis!")
        print(f"To generate the complete podcast, run:")
        print(f"python src/main.py your_document.pdf --tts {available_backends[0]}")
        
    else:
        print("❌ No TTS backends available")
        print("💡 Set up at least one TTS provider:")
        print("   • ElevenLabs: Set ELEVENLABS_API_KEY")
        print("   • OpenAI: Set OPENAI_API_KEY")
        print("   • Azure: Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
        print("   • Coqui: pip install TTS torch torchaudio")
    
    print("\n✅ Pipeline test completed successfully!")
    return True

def test_natural_dialogue_features():
    """Demonstrate the natural dialogue features"""
    print("\n🎭 Natural Dialogue Features Demo")
    print("=" * 40)
    
    # Sample dialogue with natural elements
    sample_script = """
HOST: Welcome to Tech Talk Today! I'm really excited about today's topic. [cheerful] We're diving into something that's absolutely fascinating - the future of AI in healthcare!

GUEST: [chuckles] Thanks for having me! You know, it's amazing how far we've come... [pause] When I started working in this field ten years ago, the idea of AI diagnosing diseases seemed like pure science fiction.

HOST: [curious] That's incredible! So what changed? What was the breakthrough moment?

GUEST: [thoughtful] Well, I think the real game-changer was when deep learning algorithms started outperforming human radiologists in certain imaging tasks. [excited] Suddenly, we weren't just talking about AI as a tool - it was becoming a partner in healthcare!

HOST: [amazed] Wow! That must have been quite a moment. [laughs] I can imagine some radiologists weren't too thrilled about that!

GUEST: [laughs] You're absolutely right! There was definitely some resistance at first. But here's the beautiful thing... [pause] AI isn't replacing doctors - it's making them superhuman!
    """
    
    print("Sample dialogue with natural elements:")
    print("-" * 40)
    print(sample_script.strip())
    print("-" * 40)
    
    # Count natural elements
    import re
    emotion_matches = re.findall(r'\[([^\]]+)\]', sample_script)
    
    print(f"\n✨ Natural elements detected:")
    print(f"   • Emotional expressions: {len(emotion_matches)}")
    print(f"   • Unique emotions: {', '.join(set(emotion_matches))}")
    print(f"   • Conversational fillers: {sample_script.count('you know')}")
    print(f"   • Questions: {sample_script.count('?')}")
    print(f"   • Exclamations: {sample_script.count('!')}")

if __name__ == "__main__":
    try:
        # Test natural dialogue features
        test_natural_dialogue_features()
        
        # Test complete pipeline
        success = test_complete_pipeline()
        
        if success:
            print(f"\n🎉 All tests passed! The TTS Podcast Pipeline is ready to use.")
        else:
            print(f"\n⚠️  Some tests failed. Check your environment configuration.")
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 