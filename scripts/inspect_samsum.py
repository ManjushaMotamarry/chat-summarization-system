"""
Inspect SAMSum dialogue format to understand structure.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

dataset = load_dataset("knkarthick/samsum")

# Look at first dialogue
sample = dataset['train'][0]

print("📝 Sample Dialogue:")
print("="*60)
print(sample['dialogue'])
print("="*60)
print(f"\nType: {type(sample['dialogue'])}")
print(f"\nRepr: {repr(sample['dialogue'][:200])}")

print("\n\n📝 Second Sample:")
print("="*60)
print(dataset['train'][1]['dialogue'])
print("="*60)