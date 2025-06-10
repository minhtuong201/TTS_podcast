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
    """Represents a single line of dialogue in the podcast script"""
    speaker: SpeakerRole
    text: str


@dataclass
class ScriptConfig:
    """Configuration for script generation"""
    target_words: int = 900
    target_minutes: int = 8
    host_gender: str = "female"  # female, male
    guest_gender: str = "male"   # female, male
    conversation_style: str = "friendly"  # friendly, professional, casual
    language: str = "en"
    model: str = "openai/gpt-4o"
    max_tokens: int = 4000
    temperature: float = 0.7
    preserve_technical: bool = True
    target_duration: int = 300  # seconds


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
            payload = {
                "model": config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """

Bạn là người biên tập nội dung podcast. Tôi sẽ cung cấp cho bạn phần **phân tích chi tiết nội dung một tài liệu học thuật**, và bạn cần chuyển nội dung đó thành **một đoạn hội thoại podcast tự nhiên giữa hai người**:

---

### Nhân vật
- **Host**: là người dẫn dắt podcast, nói chuyện thoải mái, dễ gần, thường mở đầu, chuyển ý và tổng hợp nội dung.
- **Guest**: là người viết tài liệu hoặc người giảng dạy, trả lời, giải thích và bổ sung thêm thông tin chuyên sâu.

---

### 1. Phong cách ngôn ngữ phải đúng kiểu văn nói đời thực

Hội thoại phải nghe thật tự nhiên, như hai người đang trò chuyện thoải mái. Sử dụng khéo léo các khẩu ngữ đặc trưng sau để tái tạo cảm giác nói thật:

#### Âm đệm, ngập ngừng, kéo dài chữ:
- Ờmm..., Hừmm..., Àa..., Ồ..., Chàa.., Ôii...
- 
#### Từ cảm thán, bộc lộ cảm xúc:
- Hay quá, nghe có vẻ hay đấy, cái này thú vị này, vâng, đúng rồi.

#### Từ làm mềm câu (cuối câu hỏi hoặc trần thuật):
- nhỉ, nha. (e.g. đúng không "nhỉ", hay quá "nha")

#### Từ nối, giữ mạch:
- cái, nó, kiểu, ý nó là, cái mấu chốt là, nói chung là, đơn giản là, hình dung là, ví dụ là, đơn giản như là,...

#### **QUAN TRỌNG NHẤT**: Script phải chứa nhiều cụm: ờm, cái, nó, là.
---

### 2. Mẫu chen ngang ngắn - phản ứng tự nhiên giữa chừng

Những câu rất ngắn, dùng để:
- phản ứng nhanh khi người kia đang nói dở,
- xác nhận lại thông tin,
- thể hiện sự hứng thú, đồng tình, ngạc nhiên,…

Mỗi câu là một dòng độc lập:
- À ra thế...
- Vâng.
- Ồ...
- Chàa... cái này nghe thú vị đấy ha.
- Ờ đúng rồi.
- Hay đấy nha!
- À hiểu rồi...
- Ờ cái này nghe quen quen...

### 3. Cách trình bày hội thoại

- Ghi rõ tên người nói ở đầu dòng:
  - `HOST: ` cho người dẫn chương trình
  - `GUEST: ` cho người được mời
- Mỗi câu một dòng.
- Viết toàn bộ nội dung dưới dạng văn nói liền mạch, như một đoạn hội thoại tự nhiên. Không dùng gạch đầu dòng, không kí tự đặc biệt, không markdown, không chia mục, giữ nguyên dạng raw text.
- Chèn hợp lý thẻ "[cười nhẹ]" để biểu thị cảm xúc, tiếng cười của người nói.
---

### 4. Nội dung phải dựa hoàn toàn vào phân tích PDF

- Tuân theo đúng **trình tự nội dung** trong bản phân tích PDF.
- Không tự chế, không thêm thông tin mới.
- Có thể diễn đạt lại, đơn giản hóa, mở rộng tự nhiên nhưng **không rời xa nội dung gốc**.

---

Ví dụ một đoạn script tự nhiên:
<sample script>
HOST: Tôi, ờm, thực ra chỉ biết dùng thôi. Chứ... cũng chưa thực sự hiểu định nghĩa của cụm LLM này là gì ấy?
GUEST: [Cười nhẹ] À, cái LLM này thực ra, ờmm, nó là viết tắt của Large Language Model.
HOST: Hm ra thế...
GUEST: Vâng, trong đó thì Large nó là viết tắt của Large-scale. 
HOST: Àaa hiểu rồi... Nó là, ờmm... Nhiều tham số, nhiều dữ liệu, nhiều tính toán... có đúng không nhỉ? 
GUEST: [Cười nhẹ] Ừ đúng rồi đấy! Còn cái Language Model này, thì bạn cứ hình dung nó giống như là một cái mô hình dự đoán chữ tiếp theo, ờmm, dựa trên các chữ trước đấy ấy.
HOST: Ồô...
GUEST: Đấy mới là về định nghĩa của LLM thôi. Còn bây giờ, ờmm, cái mấu chốt là, mình phải hiểu nó hoạt động như thế nào, được training ra sao, và... 
HOST: Ừm... Và ứng dụng của nó nữa...
GUEST: Chính xác! [Cười nhẹ]
<\sample script>

**Tôi sẽ cung cấp phần nội dung phân tích. Dựa trên đó, hãy tạo ra hội thoại theo định dạng và yêu cầu trên.**"""
                    },
                    {
                        "role": "user",
                        "content": summary
                    }
                ],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            }
            
            try:
                logger.info(f"Generating {config.target_words}-word dialogue script")
                
                response = self._make_request(payload)
                script_text = response['choices'][0]['message']['content'].strip()
                
                # Debug: Log the raw script text
                logger.debug(f"Raw script response: {script_text[:500]}...")
                
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
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                }
                log_pipeline_metrics("script_generation", metrics, logger)
                
                logger.info(f"Generated script with {len(dialogue_lines)} lines, {total_words} words")
                
                return dialogue_lines
                
            except Exception as e:
                logger.error(f"Script generation failed: {e}")
                raise
    
    def _parse_script(self, script_text: str, config: ScriptConfig) -> List[DialogueLine]:
        """Parse generated script text into DialogueLine objects"""
        
        lines = []
        raw_lines = script_text.split('\n')
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract basic dialogue structure
            dialogue_match = re.match(r'(HOST|GUEST):\s*(.+)', line)
            if dialogue_match:
                role_str, text = dialogue_match.groups()
                speaker = SpeakerRole(role_str)
                
                # Clean and process text (removed emotion extraction)
                clean_text = self._clean_dialogue_text(text)
                
                dialogue_line = DialogueLine(
                    speaker=speaker,
                    text=clean_text
                )
                
                lines.append(dialogue_line)
        
        return lines
    
    def _clean_dialogue_text(self, text: str) -> str:
        """Clean dialogue text by removing unwanted characters and formatting"""
        # Remove all annotation patterns from text
        clean_text = re.sub(r'\([^)]*\)', '', text).strip()
        
        return clean_text
    
    def _enhance_dialogue(self, dialogue_lines: List[DialogueLine], config: ScriptConfig) -> List[DialogueLine]:
        """Enhance dialogue with better timing and flow"""
        enhanced_lines = []
        
        for i, line in enumerate(dialogue_lines):
            enhanced_line = DialogueLine(
                speaker=line.speaker,
                text=line.text
            )
            enhanced_lines.append(enhanced_line)
        
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
            # Add emotion annotations
            if line.emotion:
                text = f"[{line.emotion}] {text}"
        
        lines.append(f"{speaker_tag}: {text}")
    
    return '\n'.join(lines) 