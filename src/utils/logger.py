"""
Centralized logging configuration for the entire project.
All modules will use this logger.
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (usually __name__ from calling module)
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
    
    Returns:
        logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (writes to file)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger for the project
def get_logger(name):
    """
    Get a logger for a module.
    Logs will be saved to logs/pipeline.log
    """
    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    return setup_logger(name, log_file=log_file)