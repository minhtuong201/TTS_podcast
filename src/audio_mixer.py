"""
Audio mixing and processing module for TTS Podcast Pipeline
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import tempfile
from math import log10
import io

from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

from utils.log_cfg import PipelineTimer, log_pipeline_metrics

logger = logging.getLogger(__name__)


@dataclass
class AudioSegmentInfo:
    """Information about an audio segment"""
    audio_data: bytes
    duration_seconds: float
    speaker: str  # 'host', 'guest', 'male', 'female', etc.
    text: str


@dataclass
class MixingConfig:
    """Configuration for audio mixing"""
    output_format: str = "mp3"
    bitrate: str = "192k"
    sample_rate: int = 44100
    channels: int = 1  # Mono for podcasts
    normalize_audio: bool = True
    intro_silence: float = 0.5
    outro_silence: float = 1.0
    speaker_volumes: Dict[str, float] = None  # Speaker-specific volume multipliers (e.g., {"male": 1.1, "female": 1.0})
    
    def __post_init__(self):
        if self.speaker_volumes is None:
            self.speaker_volumes = {}


class AudioMixer:
    """Audio mixing and processing class"""
    
    def __init__(self, config: Optional[MixingConfig] = None):
        self.config = config or MixingConfig()
    
    def mix_segments(self, segments: List[AudioSegmentInfo], output_path: Path) -> Dict[str, Any]:
        """Mix audio segments into a single podcast file"""
        
        if not segments:
            raise ValueError("No audio segments provided")
        
        with PipelineTimer("Audio mixing", logger):
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                try:
                    # Process segments
                    audio_segments = []
                    total_duration = 0.0
                    total_characters = 0
                    
                    logger.info(f"Processing {len(segments)} audio segments")
                    
                    # Combine all audio segments
                    combined_audio = AudioSegment.empty()
                    
                    for i, segment_info in enumerate(segments):
                        logger.debug(f"Processing segment {i+1}: {segment_info.speaker} - {len(segment_info.text)} chars")
                        
                        # Load and process the audio segment
                        segment_audio = AudioSegment.from_file(
                            io.BytesIO(segment_info.audio_data),
                            format="mp3"
                        )
                        
                        # Apply speaker-specific volume adjustments
                        if segment_info.speaker in self.config.speaker_volumes:
                            volume_multiplier = self.config.speaker_volumes[segment_info.speaker]
                            if volume_multiplier != 1.0:
                                # Convert to dB change
                                db_change = 20 * log10(volume_multiplier)
                                segment_audio = segment_audio + db_change
                                logger.debug(f"Applied {volume_multiplier}x volume ({db_change:+.1f}dB) to {segment_info.speaker}")
                        
                        # Add the segment to combined audio
                        combined_audio += segment_audio
                        
                        total_duration += len(segment_audio) / 1000.0
                        total_characters += len(segment_info.text)
                    
                    # Add intro/outro silence
                    if self.config.intro_silence > 0:
                        intro_silence = AudioSegment.silent(
                            duration=int(self.config.intro_silence * 1000)
                        )
                        combined_audio = intro_silence + combined_audio
                    
                    if self.config.outro_silence > 0:
                        outro_silence = AudioSegment.silent(
                            duration=int(self.config.outro_silence * 1000)
                        )
                        combined_audio = combined_audio + outro_silence
                    
                    # Apply audio processing
                    if self.config.normalize_audio:
                        logger.info("Normalizing audio")
                        combined_audio = normalize(combined_audio)
                    
                    # Export final audio
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_path.suffix.lower() != f'.{self.config.output_format}':
                        output_path = output_path.with_suffix(f'.{self.config.output_format}')
                    
                    logger.info(f"Exporting final audio to {output_path}")
                    combined_audio.export(
                        str(output_path),
                        format=self.config.output_format,
                        bitrate=self.config.bitrate
                    )
                    
                    # Calculate metrics
                    final_duration = len(combined_audio) / 1000.0
                    file_size = output_path.stat().st_size
                    
                    metrics = {
                        'segments_count': len(segments),
                        'total_characters': total_characters,
                        'final_duration_seconds': final_duration,
                        'output_file_size_bytes': file_size,
                        'output_format': self.config.output_format
                    }
                    
                    log_pipeline_metrics("audio_mixing", metrics, logger)
                    
                    logger.info(f"Audio mixing completed: {final_duration:.1f}s, {file_size/1024/1024:.1f}MB")
                    
                    return {
                        'output_path': str(output_path),
                        'duration_seconds': final_duration,
                        'file_size_bytes': file_size,
                        'segments_processed': len(segments),
                        'metrics': metrics
                    }
                    
                except Exception as e:
                    logger.error(f"Audio mixing failed: {e}")
                    raise
    
    def _load_audio_from_bytes(self, audio_data: bytes, temp_path: Path) -> AudioSegment:
        """Load audio from bytes data"""
        
        # Save to temporary file first
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        # Load with pydub
        try:
            audio = AudioSegment.from_file(str(temp_path))
            
            # Ensure consistent format
            audio = audio.set_frame_rate(self.config.sample_rate)
            audio = audio.set_channels(self.config.channels)
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio from bytes: {e}")
            raise ValueError(f"Could not load audio data: {e}")
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink() 