"""
Logging configuration for TTS Podcast Pipeline
"""
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured analysis"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)
            
        return json.dumps(log_obj)


def setup_logging(level=logging.INFO, use_json=False):
    """Setup logging configuration"""
    
    formatter = JSONFormatter() if use_json else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


class PipelineTimer:
    """Context manager for timing pipeline steps"""
    
    def __init__(self, step_name: str, logger: logging.Logger = None):
        self.step_name = step_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.step_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(
                f"Completed {self.step_name}",
                extra={'extra_data': {'duration_seconds': duration}}
            )
        else:
            self.logger.error(
                f"Failed {self.step_name}: {exc_val}",
                extra={'extra_data': {'duration_seconds': duration}}
            )


def log_pipeline_metrics(step: str, metrics: Dict[str, Any], logger: logging.Logger = None):
    """Log pipeline metrics in structured format"""
    logger = logger or logging.getLogger(__name__)
    logger.info(
        f"Pipeline metrics for {step}",
        extra={'extra_data': {'step': step, 'metrics': metrics}}
    ) 