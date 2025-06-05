"""
PDF text extraction module for TTS Podcast Pipeline
"""
import logging
from pathlib import Path
from typing import Optional
from io import StringIO

from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

from utils.log_cfg import PipelineTimer, log_pipeline_metrics

logger = logging.getLogger(__name__)


def extract(pdf_path: str | Path, 
           max_chars: Optional[int] = None,
           encoding: str = 'utf-8') -> str:
    """
    Extract text from PDF file
    
    Args:
        pdf_path: Path to PDF file
        max_chars: Optional maximum characters to extract (for very large PDFs)
        encoding: Text encoding for output
        
    Returns:
        Extracted text as string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    with PipelineTimer("PDF text extraction", logger):
        try:
            # Configure layout analysis parameters
            laparams = LAParams(
                boxes_flow=0.5,    # Flow threshold for text grouping
                word_margin=0.1,   # Margin for word separation
                char_margin=2.0,   # Margin for character separation
                line_margin=0.5,   # Margin for line separation
                detect_vertical=True,  # Detect vertical text
            )
            
            logger.info(f"Extracting text from: {pdf_path}")
            
            # Extract text with layout analysis
            text = extract_text(
                str(pdf_path),
                laparams=laparams,
                maxpages=0,  # Extract all pages
                password='',
                caching=True
            )
            
            if not text or not text.strip():
                raise ValueError("No extractable text found in PDF")
            
            # Clean up text
            text = clean_extracted_text(text)
            
            # Truncate if requested
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + "..."
                logger.warning(f"Text truncated to {max_chars} characters")
            
            # Log metrics
            metrics = {
                'file_size_bytes': pdf_path.stat().st_size,
                'text_length_chars': len(text),
                'text_length_words': len(text.split()),
                'pages_processed': 'all'
            }
            log_pipeline_metrics("pdf_extraction", metrics, logger)
            
            logger.info(f"Successfully extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise


def clean_extracted_text(text: str) -> str:
    """
    Clean up extracted PDF text
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive newlines but preserve paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_extracted_text(text: str, min_words: int = 100) -> bool:
    """
    Validate that extracted text is suitable for processing
    
    Args:
        text: Extracted text
        min_words: Minimum number of words required
        
    Returns:
        True if text is valid for processing
    """
    if not text or not text.strip():
        return False
        
    words = text.split()
    if len(words) < min_words:
        logger.warning(f"Text too short: {len(words)} words (minimum: {min_words})")
        return False
    
    # Check for reasonable character distribution
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / len(text) < 0.5:
        logger.warning("Text appears to have low alphabetic content")
        return False
    
    return True


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Get basic information about PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with PDF information
    """
    pdf_path = Path(pdf_path)
    
    try:
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        
        with open(pdf_path, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            
            # Count pages
            pages = list(PDFPage.create_pages(doc))
            
            info = {
                'file_path': str(pdf_path),
                'file_size_bytes': pdf_path.stat().st_size,
                'page_count': len(pages),
                'title': getattr(doc.info[0] if doc.info else {}, 'get', lambda x: None)('Title'),
                'author': getattr(doc.info[0] if doc.info else {}, 'get', lambda x: None)('Author'),
            }
            
            return info
            
    except Exception as e:
        logger.warning(f"Could not extract PDF metadata: {e}")
        return {
            'file_path': str(pdf_path),
            'file_size_bytes': pdf_path.stat().st_size,
            'page_count': 'unknown',
            'title': None,
            'author': None,
        } 