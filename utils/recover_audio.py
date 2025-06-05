#!/usr/bin/env python3
"""
Recovery script to generate partial audio from successful segments
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_ingest import extract
from lang_detect import detect as detect_language
from summarizer import llm_summary
from script_gen import ScriptGenerator, ScriptConfig
from tts.base import TTSConfig
from tts.eleven import ElevenLabsTTSBackend
from audio_mixer import AudioMixer, AudioSegmentInfo, MixingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recover_partial_audio():
    """Generate audio from first 16 dialogue lines only"""
    
    pdf_path = Path("pdf_folder/SpeechRecognition.pdf")
    
    # Step 1: Re-extract and process content (fast operations)
    logger.info("Re-extracting PDF content...")
    text = extract(pdf_path)
    
    logger.info("Re-detecting language...")
    language_code, confidence, _ = detect_language(text)
    
    logger.info("Re-generating summary...")
    summary = llm_summary(text, target_len="short")
    
    logger.info("Re-generating script...")
    script_config = ScriptConfig(target_words=400, language=language_code)
    script_generator = ScriptGenerator()
    dialogue_lines = script_generator.generate_dialogue(summary, script_config)
    
    # Take only first 16 lines (the ones that were successfully synthesized)
    successful_lines = dialogue_lines[:16]
    logger.info(f"Processing first {len(successful_lines)} dialogue lines")
    
    # Step 2: Re-synthesize only the successful segments
    voice_female = "9BWtsMINqrJLrRacOk9x"  # Aria
    voice_male = "TX3LPaxmHKxFdv7VOQHJ"    # Liam
    
    audio_segments = []
    total_cost = 0.0
    
    try:
        for i, line in enumerate(successful_lines):
            logger.info(f"Synthesizing line {i+1}/{len(successful_lines)}: {line.speaker.value}")
            
            # Determine voice based on speaker
            voice_id = voice_female if line.speaker.value == "HOST" else voice_male
            
            # Create TTS config for this line
            tts_config = TTSConfig(
                voice_id=voice_id
            )
            
            # Synthesize the line
            backend = ElevenLabsTTSBackend(tts_config)
            result = backend.synthesize(line.text, tts_config)
            
            # Create audio segment info
            segment_info = AudioSegmentInfo(
                audio_data=result.audio_data,
                duration_seconds=result.duration_seconds,
                speaker=line.speaker.value.lower(),
                text=line.text
            )
            
            audio_segments.append(segment_info)
            total_cost += result.cost_estimate
            
            logger.info(f"Synthesized {result.duration_seconds:.1f}s audio (cost: ${result.cost_estimate:.3f})")
            
    except Exception as e:
        logger.error(f"Hit quota limit at segment {len(audio_segments)+1}: {e}")
        logger.info(f"Successfully created {len(audio_segments)} segments before quota limit")
    
    if audio_segments:
        # Step 3: Mix the available segments
        output_path = Path("output/partial_podcast.mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Mixing {len(audio_segments)} audio segments...")
        mixer_config = MixingConfig()
        mixer = AudioMixer(mixer_config)
        
        mix_result = mixer.mix_segments(audio_segments, output_path)
        
        logger.info(f"✅ Partial podcast created: {output_path}")
        logger.info(f"Duration: {mix_result['duration_seconds']:.1f}s")
        logger.info(f"File size: {mix_result['file_size_bytes']/1024/1024:.1f}MB")
        logger.info(f"Total cost: ${total_cost:.3f}")
        
        return str(output_path)
    else:
        logger.error("No audio segments were successfully created")
        return None

if __name__ == "__main__":
    result = recover_partial_audio()
    if result:
        print(f"\n🎉 Partial podcast saved to: {result}")
    else:
        print("\n❌ Could not create partial podcast") 