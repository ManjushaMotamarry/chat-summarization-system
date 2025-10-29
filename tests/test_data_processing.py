"""
Unit tests for data processing functions.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.load_samsum_to_db import parse_dialogue


def test_parse_dialogue_simple():
    """Test parsing a simple two-person dialogue"""
    dialogue = "Alice: Hello\nBob: Hi there"
    messages = parse_dialogue(dialogue)
    
    assert len(messages) == 2
    assert messages[0] == ("Alice", "Hello")
    assert messages[1] == ("Bob", "Hi there")


def test_parse_dialogue_multiple_messages():
    """Test parsing dialogue with multiple messages"""
    dialogue = "Alice: Hi\nBob: Hello\nAlice: How are you?\nBob: Good!"
    messages = parse_dialogue(dialogue)
    
    assert len(messages) == 4
    assert messages[0][0] == "Alice"
    assert messages[3][0] == "Bob"


def test_parse_dialogue_with_colons_in_message():
    """Test that colons in message text are handled correctly"""
    dialogue = "Alice: My order number is: 12345\nBob: Thanks"
    messages = parse_dialogue(dialogue)
    
    assert len(messages) == 2
    assert messages[0][1] == "My order number is: 12345"


def test_parse_dialogue_empty():
    """Test parsing empty dialogue"""
    dialogue = ""
    messages = parse_dialogue(dialogue)
    
    assert len(messages) == 0


def test_parse_dialogue_extra_newlines():
    """Test that extra newlines are handled"""
    dialogue = "Alice: Hi\n\nBob: Hello\n"
    messages = parse_dialogue(dialogue)
    
    assert len(messages) == 2