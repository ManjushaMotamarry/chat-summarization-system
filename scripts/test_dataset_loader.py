"""
Test the generic dataset loader.
"""

import sys
import os

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import DatasetLoader

print("🧪 Testing Generic Dataset Loader\n")

# Test 1: Load SAMSum using config
print("1️⃣ Loading SAMSum (from config)...")
loader = DatasetLoader('samsum')
dataset = loader.load()
print()

# Test 2: Check field mappings
print("2️⃣ Field Mappings:")
print(f"   Dialogue field: {loader.get_dialogue_field()}")
print(f"   Summary field: {loader.get_summary_field()}")
print(f"   Separator: {repr(loader.get_separator())}")
print()

# Test 3: Parse a sample dialogue
print("3️⃣ Testing Dialogue Parser:")
sample = dataset['train'][0]
dialogue_text = sample[loader.get_dialogue_field()]
print(f"   Raw dialogue: {dialogue_text[:100]}...")

messages = loader.parse_dialogue(dialogue_text)
print(f"   Parsed into {len(messages)} messages:")
for sender, msg in messages[:3]:
    print(f"      {sender}: {msg[:50]}...")
print()

print("✅ Generic dataset loader working correctly!")