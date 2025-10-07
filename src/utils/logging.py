"""
Beautiful progress tracking and logging
"""
import logging
from datetime import datetime

def setup_logging():
    """Setup beautiful logging with colors and progress tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('volatility_lab.log'),
            logging.StreamHandler()
        ]
    )

def log_step(step_name, emoji="📝"):
    """Log a pipeline step with emoji"""
    logging.info(f"{emoji} {step_name}")

def log_success(message, emoji="✅"):
    """Log success with emoji"""
    logging.info(f"{emoji} {message}")

def log_error(message, emoji="❌"):
    """Log error with emoji"""
    logging.error(f"{emoji} {message}")