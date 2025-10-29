"""
Run pipeline manually to debug issues.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import DatasetLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

print("🧪 Testing pipeline components manually...\n")

# Test 1: Download
print("1️⃣ Testing download...")
try:
    loader = DatasetLoader('samsum')
    dataset = loader.load()
    print(f"✅ Downloaded: {len(dataset['train'])} samples\n")
except Exception as e:
    print(f"❌ Download failed: {e}\n")
    exit(1)

print("✅ Manual test passed!")
print("\nIf this works but Airflow times out, the issue is Airflow configuration.")