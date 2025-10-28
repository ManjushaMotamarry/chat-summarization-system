"""
Script to download SAMSum dataset from HuggingFace.
This dataset contains ~16k conversations with human-written summaries.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset


def download_samsum():
    print("📥 Downloading SAMSum dataset from HuggingFace...\n")
    
    # Download the dataset
    # This will cache it locally in ~/.cache/huggingface/datasets
    dataset = load_dataset("knkarthick/samsum")
    
    print("✅ Download complete!\n")
    
    # Show dataset structure
    print("📊 Dataset structure:")
    print(dataset)
    
    # Show a sample
    print("\n📝 Sample conversation:")
    sample = dataset['train'][0]
    print(f"\nDialogue:\n{sample['dialogue']}")
    print(f"\nSummary:\n{sample['summary']}")
    
    # Show statistics
    print("\n📈 Dataset statistics:")
    print(f"   Training samples: {len(dataset['train'])}")
    print(f"   Validation samples: {len(dataset['validation'])}")
    print(f"   Test samples: {len(dataset['test'])}")
    print(f"   Total: {len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])}")
    
    return dataset


if __name__ == "__main__":
    dataset = download_samsum()
    print("\n✅ Dataset ready to use!")