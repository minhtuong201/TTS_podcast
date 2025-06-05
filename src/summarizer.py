"""
Text summarization module using OpenRouter API for TTS Podcast Pipeline
"""
import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

from utils.log_cfg import PipelineTimer, log_pipeline_metrics

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class SummarizerConfig:
    """Configuration for PDF summarization"""
    model: str = "openai/gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 60
    max_retries: int = 3
    target_length: str = "short"  # short, medium, long
    preserve_technical: bool = True


class OpenRouterSummarizer:
    """OpenRouter API client for text summarization"""
    
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
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with retry logic"""
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, will retry...")
                raise
            elif response.status_code == 401:
                logger.error("Authentication failed - check your OpenRouter API key")
                raise ValueError("Invalid API key")
            elif response.status_code >= 500:
                logger.warning(f"Server error {response.status_code}, will retry...")
                raise
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                raise
        except requests.exceptions.Timeout:
            logger.warning("Request timeout, will retry...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            raise
    
    def summarize(self, text: str, config: Optional[SummarizerConfig] = None) -> str:
        """
        Summarize text using OpenRouter API
        
        Args:
            text: Input text to summarize
            config: Summarization configuration
            
        Returns:
            Summarized text
            
        Raises:
            ValueError: If text is empty or API call fails
            Exception: For other errors
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty")
        
        config = config or SummarizerConfig()
        
        with PipelineTimer("Text summarization", logger):
            payload = {
                "model": config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý học thuật có năng lực phân tích và giải thích tài liệu giảng dạy từ nhiều lĩnh vực khác nhau. Dưới đây là nội dung trích xuất từ một tài liệu PDF. Hãy đọc kỹ và thực hiện phân tích một cách **chuyên sâu**, **chính xác**, **khách quan**, **theo đúng mạch của tài liệu**.

Tiến hành theo trình tự sau:

---

### 1. Giới thiệu tổng quan

* Mở đầu bằng câu:
  **"Chúng ta sẽ cùng tìm hiểu về [tên hoặc mô tả lĩnh vực, chủ đề] với người trình bày là [tên giảng viên nếu có – nếu không, bỏ qua]."**
* Mô tả ngắn gọn lĩnh vực này thuộc ngành nào.
* Nêu lý do tại sao chủ đề này quan trọng.
* Trình bày ứng dụng của chủ đề.

---

### 2. Phân tích nội dung

* Phân tích từng phần nội dung theo đúng trình tự tài liệu, không bỏ sót phần nào
* Quá trình phân tích phải chuyên sâu, chi tiết. Không chỉ đưa ra thông tin mà còn phải giải thích cặn kẽ tại sao, như thế nào
* Giải thích rõ ràng các khái niệm, định nghĩa, công thức, mô hình, quy trình, thuật ngữ.
* Với phần phức tạp, khó hiểu nên dùng ví dụ minh họa để giải thích. Chú thích rõ **ELI5**"""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            try:
                logger.info(f"Summarizing {len(text)} characters with {config.model}")
                
                response = self._make_request(payload)
                
                # Extract summary from response
                summary = response['choices'][0]['message']['content'].strip()
                
                if not summary:
                    raise ValueError("Received empty summary from API")
                
                # Log metrics
                usage = response.get('usage', {})
                metrics = {
                    'input_chars': len(text),
                    'input_words': len(text.split()),
                    'summary_chars': len(summary),
                    'summary_words': len(summary.split()),
                    'compression_ratio': len(text) / len(summary) if summary else 0,
                    'model': config.model,
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
                log_pipeline_metrics("summarization", metrics, logger)
                
                logger.info(f"Generated summary: {len(summary)} characters ({metrics['compression_ratio']:.1f}x compression)")
                
                return summary
                
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                raise


def llm_summary(text: str, 
                target_len: str = "short",
                preserve_technical: bool = True,
                api_key: Optional[str] = None) -> str:
    """
    Convenient function for text summarization
    
    Args:
        text: Input text to summarize
        target_len: Target length ("short", "medium", "long")
        preserve_technical: Whether to preserve technical terms
        api_key: Optional API key override
        
    Returns:
        Summarized text
    """
    config = SummarizerConfig(
        target_length=target_len,
        preserve_technical=preserve_technical
    )
    
    summarizer = OpenRouterSummarizer(api_key)
    return summarizer.summarize(text, config)


def validate_summary(summary: str, original_text: str, min_compression: float = 2.0) -> bool:
    """
    Validate that summary meets quality criteria
    
    Args:
        summary: Generated summary
        original_text: Original text
        min_compression: Minimum compression ratio required
        
    Returns:
        True if summary is valid
    """
    if not summary or not summary.strip():
        logger.warning("Summary is empty")
        return False
    
    # Check compression ratio
    compression_ratio = len(original_text) / len(summary)
    if compression_ratio < min_compression:
        logger.warning(f"Summary compression ratio too low: {compression_ratio:.1f}")
        return False
    
    # Check that summary has meaningful content
    word_count = len(summary.split())
    if word_count < 50:
        logger.warning(f"Summary too short: {word_count} words")
        return False
    
    # Check for common summary issues
    if summary.lower().startswith(("sorry", "i cannot", "i'm unable")):
        logger.warning("Summary indicates processing failure")
        return False
    
    return True


def chunk_text_for_summarization(text: str, max_chunk_size: int = 15000) -> list[str]:
    """
    Split large text into chunks for summarization
    
    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If paragraphs are too long, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split long chunks by sentences
            sentences = chunk.split('. ')
            current_sentence_chunk = ""
            
            for sentence in sentences:
                if len(current_sentence_chunk) + len(sentence) + 2 <= max_chunk_size:
                    current_sentence_chunk += sentence + ". "
                else:
                    if current_sentence_chunk:
                        final_chunks.append(current_sentence_chunk.strip())
                    current_sentence_chunk = sentence + ". "
            
            if current_sentence_chunk:
                final_chunks.append(current_sentence_chunk.strip())
    
    logger.info(f"Split text into {len(final_chunks)} chunks for summarization")
    return final_chunks 