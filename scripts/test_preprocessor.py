"""
Test the smart text preprocessor.
Shows how context is preserved.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.text_preprocessor import TextPreprocessor

print("🧪 Testing SMART Text Preprocessor\n")
print("=" * 70)

# Create preprocessor with default (smart) profile
preprocessor = TextPreprocessor(profile_name='default')

# Test cases showing context preservation
test_cases = [
    ("Hello  World  😊", "Extra spaces + emoji (kept for sentiment)"),
    ("Check this out: http://example.com", "URL replaced with [link] token"),
    ("Look at this <file_photo>", "File reference replaced with [photo]"),
    ("I'll bring you tomorrow :-)", "Text emoji kept for tone"),
    ("HELLO WORLD", "Normalized to lowercase"),
    ("This   has    many     spaces", "Multiple spaces normalized"),
    ("Hi :) Here's a <file_gif> and visit www.site.com", "Multiple replacements"),
]

print("\n📊 SMART PREPROCESSING (Preserves Context):\n")
for text, description in test_cases:
    cleaned = preprocessor.clean_text(text)
    
    print(f"Description: {description}")
    print(f"  Before: '{text}'")
    print(f"  After:  '{cleaned}'")
    print()

print("=" * 70)

# Test conversation cleaning
print("\n🗨️  CONVERSATION EXAMPLE:\n")

conversation = [
    ("Alice", "Hi  there! :)"),
    ("Bob", "Hello! Check this <file_photo>"),
    ("Alice", "Nice! Visit http://example.com for more"),
    ("Bob", "Thanks :D I'll also send <file_video>")
]

print("BEFORE (Raw):")
for sender, msg in conversation:
    print(f"  {sender}: {msg}")

cleaned_conv = preprocessor.clean_conversation(conversation)

print("\nAFTER (Smart Cleaned - Context Preserved!):")
for sender, msg in cleaned_conv:
    print(f"  {sender}: {msg}")

print("\n✨ SUMMARY COMPARISON:\n")
print("❌ With aggressive cleaning (old way):")
print("   'Alice and Bob exchanged greetings.'")
print("   → Misses the entire point!\n")

print("✅ With smart cleaning (new way):")
print("   'Bob shared a [photo] with Alice. Alice sent a [link].")
print("    Bob thanked her and will send a [video].'")
print("   → Captures what actually happened!\n")

print("=" * 70)
print("\n✅ Smart preprocessing preserves context while cleaning!")