"""
Dialogue script generation module for TTS Podcast Pipeline
Creates natural 2-person conversations with emotional expressions
"""
import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

from utils.log_cfg import PipelineTimer, log_pipeline_metrics

load_dotenv()
logger = logging.getLogger(__name__)


class SpeakerRole(Enum):
    HOST = "HOST"
    GUEST = "GUEST"


@dataclass
class DialogueLine:
    """Represents a single line of dialogue"""
    speaker: SpeakerRole
    text: str
    emotion: Optional[str] = None  # happy, surprised, thoughtful, etc.
    pause_before: float = 0.0  # seconds of pause before this line
    pause_after: float = 0.0   # seconds of pause after this line


@dataclass
class ScriptConfig:
    """Configuration for script generation"""
    target_words: int = 900
    target_minutes: int = 8
    host_gender: str = "female"  # female, male
    guest_gender: str = "male"   # female, male
    conversation_style: str = "friendly"  # friendly, professional, casual
    include_emotions: bool = True
    include_pauses: bool = True
    language: str = "en"
    model: str = "openai/gpt-4o-mini"


class ScriptGenerator:
    """Generate conversational scripts from summaries"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tts-podcast-pipeline.local",
            "X-Title": "TTS Podcast Pipeline"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def _make_request(self, payload: Dict) -> Dict:
        """Make API request with retry logic"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=90
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def generate_dialogue(self, summary: str, config: Optional[ScriptConfig] = None) -> List[DialogueLine]:
        """
        Generate dialogue script from summary
        
        Args:
            summary: Text summary to convert to dialogue
            config: Script generation configuration
            
        Returns:
            List of DialogueLine objects
        """
        config = config or ScriptConfig()
        
        with PipelineTimer("Script generation", logger):
            # Build the dialogue prompt
            prompt = self._build_dialogue_prompt(summary, config)
            
            payload = {
                "model": config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(config)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.8,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            }
            
            try:
                logger.info(f"Generating {config.target_words}-word dialogue script")
                
                response = self._make_request(payload)
                script_text = response['choices'][0]['message']['content'].strip()
                
                # Parse script into dialogue lines
                dialogue_lines = self._parse_script(script_text, config)
                
                # Validate and enhance the dialogue
                dialogue_lines = self._enhance_dialogue(dialogue_lines, config)
                
                # Log metrics
                usage = response.get('usage', {})
                total_words = sum(len(line.text.split()) for line in dialogue_lines)
                
                metrics = {
                    'summary_chars': len(summary),
                    'script_lines': len(dialogue_lines),
                    'total_words': total_words,
                    'target_words': config.target_words,
                    'word_accuracy': total_words / config.target_words if config.target_words > 0 else 0,
                    'host_lines': sum(1 for line in dialogue_lines if line.speaker == SpeakerRole.HOST),
                    'guest_lines': sum(1 for line in dialogue_lines if line.speaker == SpeakerRole.GUEST),
                    'emotional_lines': sum(1 for line in dialogue_lines if line.emotion),
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                }
                log_pipeline_metrics("script_generation", metrics, logger)
                
                logger.info(f"Generated script with {len(dialogue_lines)} lines, {total_words} words")
                
                return dialogue_lines
                
            except Exception as e:
                logger.error(f"Script generation failed: {e}")
                raise
    
    def _get_system_prompt(self, config: ScriptConfig) -> str:
        """Get system prompt based on configuration"""
        
        host_pronoun = "she/her" if config.host_gender == "female" else "he/him"
        guest_pronoun = "she/her" if config.guest_gender == "female" else "he/him"
        
        return f"""You are an expert podcast script writer. Your task is to create engaging, natural dialogue between two people discussing interesting content.

CHARACTER SETUP:
- HOST ({config.host_gender.upper()}): A curious, engaging podcast host who asks thoughtful questions and guides the conversation. Pronouns: {host_pronoun}
- GUEST ({config.guest_gender.upper()}): A knowledgeable expert who explains concepts clearly and enthusiastically. Pronouns: {guest_pronoun}

DIALOGUE REQUIREMENTS:
- Write in {config.language}
- Target: {config.target_words} words total (~{config.target_minutes} minutes of audio)
- Use natural, conversational language with contractions
- Include emotional expressions in [brackets] like [laughs], [chuckles], [pause], [thoughtful], [excited]
- Add natural conversation elements: "Well...", "You know...", "That's fascinating!", "Hmm..."
- Use varied sentence lengths and speaking patterns
- Include questions, reactions, and back-and-forth exchanges
- Each line should start with "HOST:" or "GUEST:"

CONVERSATION STYLE: {config.conversation_style}
- Make it sound like a real conversation between intelligent people
- Include moments of discovery, surprise, and engagement
- Use appropriate pauses and emotional beats
- Ensure both speakers contribute meaningfully"""
    
    def _build_dialogue_prompt(self, summary: str, config: ScriptConfig) -> str:
        """Build the main dialogue generation prompt"""
        
        return f"""Convert the following summary into an engaging {config.target_minutes}-minute podcast dialogue between a HOST and GUEST.

The conversation should:
- Cover all main points from the summary naturally
- Include genuine reactions, questions, and discoveries
- Have a clear introduction, development, and conclusion
- Feel spontaneous while being informative
- Include natural speaking elements and emotions

SUMMARY TO CONVERT:
{summary}

Please write the complete dialogue script with emotional annotations and natural conversation flow."""
    
    def _parse_script(self, script_text: str, config: ScriptConfig) -> List[DialogueLine]:
        """Parse generated script text into DialogueLine objects"""
        
        lines = []
        raw_lines = script_text.split('\n')
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse speaker
            if line.startswith('HOST:'):
                speaker = SpeakerRole.HOST
                text = line[5:].strip()
            elif line.startswith('GUEST:'):
                speaker = SpeakerRole.GUEST
                text = line[6:].strip()
            else:
                # Skip lines without speaker tags
                continue
            
            if not text:
                continue
            
            # Extract emotions and pauses from text
            emotion, pause_before, pause_after, clean_text = self._extract_annotations(text)
            
            dialogue_line = DialogueLine(
                speaker=speaker,
                text=clean_text,
                emotion=emotion,
                pause_before=pause_before,
                pause_after=pause_after
            )
            
            lines.append(dialogue_line)
        
        return lines
    
    def _extract_annotations(self, text: str) -> Tuple[Optional[str], float, float, str]:
        """Extract emotion and pause annotations from text"""
        
        emotion = None
        pause_before = 0.0
        pause_after = 0.0
        
        # Extract emotions in brackets
        emotion_pattern = r'\[([^\]]+)\]'
        emotions = re.findall(emotion_pattern, text)
        
        if emotions:
            # Use the first emotion found
            emotion = emotions[0].lower()
            
            # Convert some emotions to pauses
            if emotion in ['pause', 'long pause', 'thoughtful pause']:
                if 'long' in emotion:
                    pause_before = 1.5
                else:
                    pause_before = 0.8
                emotion = None
            elif emotion in ['end pause', 'trailing off']:
                pause_after = 1.0
                emotion = None
        
        # Remove all bracketed annotations from text
        clean_text = re.sub(emotion_pattern, '', text).strip()
        
        # Clean up extra spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return emotion, pause_before, pause_after, clean_text
    
    def _enhance_dialogue(self, dialogue_lines: List[DialogueLine], config: ScriptConfig) -> List[DialogueLine]:
        """Enhance dialogue with better timing and flow"""
        
        enhanced_lines = []
        
        for i, line in enumerate(dialogue_lines):
            # Add natural pauses between speakers
            if i > 0 and dialogue_lines[i-1].speaker != line.speaker:
                # Different speaker, add small pause
                if line.pause_before == 0.0:
                    line.pause_before = 0.3
            
            # Add pauses for longer sentences
            word_count = len(line.text.split())
            if word_count > 25 and line.pause_after == 0.0:
                line.pause_after = 0.5
            
            # Ensure we have some emotional variety
            if config.include_emotions and not line.emotion and len(line.text.split()) > 15:
                # Add subtle emotions based on content
                text_lower = line.text.lower()
                if any(word in text_lower for word in ['interesting', 'fascinating', 'amazing', 'incredible']):
                    line.emotion = 'interested'
                elif any(word in text_lower for word in ['funny', 'humor', 'joke']):
                    line.emotion = 'amused'
                elif line.text.endswith('?'):
                    line.emotion = 'curious'
            
            enhanced_lines.append(line)
        
        return enhanced_lines


def dialogue(summary: str, 
             lang: str = "en",
             target_words: int = 900,
             host_gender: str = "female",
             guest_gender: str = "male",
             api_key: Optional[str] = None) -> List[DialogueLine]:
    """
    Convenient function for dialogue generation
    
    Args:
        summary: Text summary to convert
        lang: Language code
        target_words: Target word count
        host_gender: Host gender (female/male)
        guest_gender: Guest gender (female/male)
        api_key: Optional API key override
        
    Returns:
        List of DialogueLine objects
    """
    config = ScriptConfig(
        target_words=target_words,
        host_gender=host_gender,
        guest_gender=guest_gender,
        language=lang
    )
    
    generator = ScriptGenerator(api_key)
    return generator.generate_dialogue(summary, config)


def dialogue_to_text(dialogue_lines: List[DialogueLine], include_annotations: bool = True) -> str:
    """
    Convert dialogue lines back to text format
    
    Args:
        dialogue_lines: List of DialogueLine objects
        include_annotations: Whether to include emotion/pause annotations
        
    Returns:
        Text representation of dialogue
    """
    lines = []
    
    for line in dialogue_lines:
        speaker_tag = line.speaker.value
        text = line.text
        
        if include_annotations:
            # Add pause annotations
            if line.pause_before > 0.5:
                text = f"[pause] {text}"
            
            # Add emotion annotations
            if line.emotion:
                text = f"[{line.emotion}] {text}"
            
            if line.pause_after > 0.5:
                text = f"{text} [pause]"
        
        lines.append(f"{speaker_tag}: {text}")
    
    return '\n'.join(lines) 