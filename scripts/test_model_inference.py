"""
Test trained model by generating summaries for sample conversations
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_model(model_path='models/best_model'):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    print("✅ Model loaded successfully!")
    return tokenizer, model

def generate_summary(conversation, tokenizer, model, max_length=128):
    """
    Generate summary for a conversation
    
    Args:
        conversation: Dialogue text
        tokenizer: Model tokenizer
        model: Trained model
        max_length: Maximum summary length
        
    Returns:
        Generated summary
    """
    # Tokenize input
    inputs = tokenizer(
        conversation,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def test_sample_conversations(tokenizer, model, n_samples=5):
    """Test model on random samples from test set"""
    print("\n" + "="*80)
    print("TESTING ON RANDOM SAMPLES FROM TEST SET")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Get random samples
    samples = test_df.sample(n=n_samples, random_state=42)
    
    for idx, row in samples.iterrows():
        print("\n" + "-"*80)
        print(f"\n📝 CONVERSATION {idx}:")
        print("-"*80)
        print(row['dialogue'][:300] + "..." if len(row['dialogue']) > 300 else row['dialogue'])
        
        print("\n🎯 HUMAN REFERENCE SUMMARY:")
        print(row['summary'])
        
        print("\n🤖 MODEL GENERATED SUMMARY:")
        generated = generate_summary(row['dialogue'], tokenizer, model)
        print(generated)
        
        print("-"*80)

def test_custom_conversation(tokenizer, model):
    """Test model on a custom conversation"""
    print("\n" + "="*80)
    print("TESTING ON CUSTOM CONVERSATION")
    print("="*80)
    
    # Example custom conversation
    conversation = """Amanda: Hey! I just finished baking chocolate chip cookies!
Jerry: Oh wow, they smell amazing from here!
Amanda: Haha, thanks! Want to try some?
Jerry: Absolutely! When can I come over?
Amanda: How about tomorrow morning? I'll bring them to your office.
Jerry: Perfect! Can't wait. Thanks so much!
Amanda: No problem, see you then!"""
    
    print("\n📝 CONVERSATION:")
    print(conversation)
    
    print("\n🤖 MODEL GENERATED SUMMARY:")
    summary = generate_summary(conversation, tokenizer, model)
    print(summary)

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTING TRAINED SUMMARIZATION MODEL")
    print("="*80)
    
    # Load model
    tokenizer, model = load_model()
    
    # Test on samples from test set
    test_sample_conversations(tokenizer, model, n_samples=5)
    
    # Test on custom conversation
    test_custom_conversation(tokenizer, model)
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)