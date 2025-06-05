#!/usr/bin/env python3
"""
TTS Pipeline Customization Examples
Shows how to modify emotions, voices, script style, and character personalities
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from script_gen import ScriptConfig, ScriptGenerator, SpeakerRole
from tts.base import TTSConfig

# ═════════════════════════════════════════════════════════════════════════════
# 1. 🎭 CUSTOMIZING SCRIPT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

# Removed modify_emotion_responses function

# ═════════════════════════════════════════════════════════════════════════════
# 2. 🎙️ CUSTOMIZING VOICES
# ═════════════════════════════════════════════════════════════════════════════

def customize_default_voices():
    """Example: How to change default voices"""
    
    print("🎙️  Customizing Default Voices:")
    print("Edit src/main.py lines 111-120:")
    print()
    
    # ElevenLabs voices (examples)
    elevenlabs_voices = {
        'female_voices': {
            'Rachel': '21m00Tcm4TlvDq8ikWAM',
            'Domi': 'AZnzlk1XvdvUeBnXmlld', 
            'Bella': 'EXAVITQu4vr4xnSDxMaL',
            'Elli': 'MF3mGyEYCl7XYWbV9V6O',
            'Sarah': 'EXAVITQu4vr4xnSDxMaL'
        },
        'male_voices': {
            'Adam': 'pNInz6obpgDQGcFmaJgB',
            'Josh': 'TxGEqnHWrfWFTfGW9XjX',
            'Brian': 'nPczCjzI2devNBz1zQrb',
            'Sam': 'yoZ06aMxZJJ28mfd3POQ',
            'Callum': 'N2lVS1w4EtoT3dr4eOWO'
        }
    }
    
    print("Available ElevenLabs voices:")
    for gender, voices in elevenlabs_voices.items():
        print(f"  {gender}:")
        for name, voice_id in voices.items():
            print(f"    • {name}: '{voice_id}'")
    print()
    
    # OpenAI voices
    openai_voices = {
        'female': ['alloy', 'nova', 'shimmer'],
        'male': ['echo', 'fable', 'onyx']
    }
    
    print("Available OpenAI voices:")
    for gender, voices in openai_voices.items():
        print(f"  {gender}: {', '.join(voices)}")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# 3. 📝 CUSTOMIZING SCRIPT STYLE
# ═════════════════════════════════════════════════════════════════════════════

def create_custom_script_configs():
    """Example: Different script configurations for different podcast styles"""
    
    print("📝 Custom Script Configurations:")
    print()
    
    # Academic interview style
    academic_config = ScriptConfig(
        target_words=1200,
        target_minutes=10,
        host_gender="female",
        guest_gender="male", 
        conversation_style="professional"
    )
    
    # Casual podcast style
    casual_config = ScriptConfig(
        target_words=800,
        target_minutes=7,
        host_gender="male",
        guest_gender="female",
        conversation_style="casual"
    )
    
    # Science communication style  
    science_config = ScriptConfig(
        target_words=1000,
        target_minutes=8,
        host_gender="female", 
        guest_gender="male",
        conversation_style="friendly"
    )
    
    configs = {
        "Academic Interview": academic_config,
        "Casual Chat": casual_config,
        "Science Communication": science_config
    }
    
    for style_name, config in configs.items():
        print(f"  {style_name}:")
        print(f"    • Words: {config.target_words}")
        print(f"    • Style: {config.conversation_style}")
        print(f"    • Host: {config.host_gender}")
        print(f"    • Guest: {config.guest_gender}")
        print()

def customize_character_personalities():
    """Example: How to modify character personalities"""
    
    print("👥 Customizing Character Personalities:")
    print("Edit src/script_gen.py lines 155-185 in _get_system_prompt():")
    print()
    
    print("Current setup:")
    print("  • HOST: Curious, engaging, asks thoughtful questions")
    print("  • GUEST: Knowledgeable expert, explains clearly")
    print()
    
    print("Alternative personality examples:")
    print("  HOST personalities:")
    print("    • Skeptical interviewer who challenges ideas")
    print("    • Enthusiastic fan who gets excited about details")
    print("    • Casual friend having a relaxed conversation")
    print("    • Professional journalist asking probing questions")
    print()
    
    print("  GUEST personalities:")
    print("    • Passionate researcher who loves sharing discoveries")
    print("    • Practical expert focused on real-world applications")
    print("    • Storyteller who explains through anecdotes")
    print("    • Cautious scientist who considers all angles")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. 🎯 PRACTICAL CUSTOMIZATION EXAMPLES
# ═════════════════════════════════════════════════════════════════════════════

def example_comedy_podcast_config():
    """Example: Configuration for a comedy-style podcast"""
    
    print("😄 Comedy Podcast Configuration:")
    print()
    
    # This would go in your script
    comedy_config = ScriptConfig(
        target_words=1000,
        conversation_style="casual",
        host_gender="male",
        guest_gender="female"
    )
    
    print("Configuration:")
    print(f"  • Style: {comedy_config.conversation_style}")
    print(f"  • Target length: {comedy_config.target_words} words")
    print()
    
    print("Enhanced emotions for comedy:")
    print("  • Add to emotion map: 'sarcastic', 'deadpan', 'silly'")
    print("  • Increase laugh frequency in _enhance_dialogue()")
    print("  • Remove timing for comedic pauses (no longer supported)")
    print()

def example_educational_podcast_config():
    """Example: Configuration for educational content"""
    
    print("🎓 Educational Podcast Configuration:")
    print()
    
    educational_config = ScriptConfig(
        target_words=1500,
        conversation_style="friendly",
        host_gender="female",
        guest_gender="male",
        preserve_technical=True
    )
    
    print("Configuration:")
    print(f"  • Longer format: {educational_config.target_words} words")
    print(f"  • Style: {educational_config.conversation_style}")
    print("  • Focus on clear explanations")
    print()
    
    print("Educational enhancements:")
    print("  • Add 'explaining' emotion for complex concepts")
    print("  • Increase 'thoughtful' pauses")
    print("  • Add 'understanding' reactions from host")
    print()

def show_file_locations():
    """Show exactly which files to edit for each customization"""
    
    print("📁 File Locations for Customizations:")
    print("=" * 50)
    print()
    
    locations = {
        "🎙️ Default Voices": [
            "src/main.py (lines 111-120) - voice ID assignment",
            "src/main.py (lines 108-120) - voice selection logic"
        ],
        "📝 Script Style": [
            "src/script_gen.py (lines 37-47) - ScriptConfig class",
            "src/script_gen.py (lines 155-185) - system prompt",
            "src/script_gen.py (lines 189-213) - dialogue prompt"
        ],
        "👥 Characters": [
            "src/script_gen.py (lines 155-185) - personality descriptions",
            "src/script_gen.py (lines 289-321) - dialogue enhancement"
        ]
    }
    
    for category, files in locations.items():
        print(f"{category}:")
        for file_info in files:
            print(f"  • {file_info}")
        print()

# ═════════════════════════════════════════════════════════════════════════════
# 5. 🚀 QUICK START EXAMPLES
# ═════════════════════════════════════════════════════════════════════════════

def quick_customization_guide():
    """Quick guide for common customizations"""
    
    print("🚀 Quick Customization Guide:")
    print("=" * 40)
    print()
    
    print("✅ To change default voices:")
    print("   Edit src/main.py lines 111-120")
    print("   Replace voice IDs with your preferred voices")
    print()
    
    print("✅ To customize character personalities:")
    print("   Edit src/script_gen.py lines 155-185")
    print("   Modify the CHARACTER SETUP section")
    print()

if __name__ == "__main__":
    print("🎛️  TTS Pipeline Customization Guide")
    print("=" * 50)
    print()
    
    create_custom_script_configs()
    print("─" * 50)
    
    customize_character_personalities()
    print("─" * 50)
    
    example_comedy_podcast_config()
    print("─" * 50)
    
    example_educational_podcast_config()
    print("─" * 50)
    
    show_file_locations()
    print("─" * 50)
    
    quick_customization_guide() 