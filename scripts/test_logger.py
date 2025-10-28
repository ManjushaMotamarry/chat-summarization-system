"""
Test the logging setup.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

# Test different log levels
logger.info("✅ INFO: Logger is working!")
logger.warning("⚠️ WARNING: This is a warning message")
logger.error("❌ ERROR: This is an error message")
logger.debug("🔍 DEBUG: This debug message won't show (level too low)")

print("\n📁 Check logs/pipeline_YYYYMMDD.log file for saved logs!")