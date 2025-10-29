"""
Text preprocessing for conversation data.
Smart preprocessing that preserves context.
"""

import re
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Preprocesses conversation text for training.
    Uses configuration-based smart preprocessing.
    """
    
    def __init__(self, profile_name=None):
        """
        Initialize preprocessor with config profile.
        
        Args:
            profile_name: Name of preprocessing profile (None = active)
        """
        config_loader = get_config()
        self.config = config_loader.get_preprocessing_config(profile_name)
        
        logger.info(f"📋 Loaded preprocessing profile: {self.config['name']}")
    
    def clean_text(self, text):
        """
        Apply smart cleaning to text.
        Preserves context while normalizing.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Step 1: Handle file references (replace with tokens)
        if self.config['file_references']['action'] == 'replace':
            text = self._replace_file_references(text)
        elif self.config['file_references']['action'] == 'remove':
            text = self._remove_file_references(text)
        
        # Step 2: Handle URLs (replace with token)
        if self.config['urls']['action'] == 'replace':
            text = self._replace_urls(text)
        elif self.config['urls']['action'] == 'remove':
            text = self._remove_urls(text)
        
        # Step 3: Handle emojis (keep them for sentiment!)
        # We keep both text and unicode emojis as configured
        
        # Step 4: Normalize whitespace
        if self.config.get('normalize_whitespace', True):
            text = self._normalize_whitespace(text)
        
        # Step 5: Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Check minimum length
        if len(text.strip()) < self.config.get('min_length', 1):
            logger.warning(f"Text too short after cleaning: '{original_text[:50]}...'")
            return ""
        
        return text.strip()
    
    def _replace_file_references(self, text):
        """Replace file references with descriptive tokens"""
        replacements = self.config['file_references']['patterns']
        
        for pattern, replacement in replacements.items():
            text = text.replace(pattern, replacement)
        
        return text
    
    def _remove_file_references(self, text):
        """Remove file references completely"""
        text = re.sub(r'<file_[^>]+>', '', text)
        return text
    
    def _replace_urls(self, text):
        """Replace URLs with [link] token"""
        replacement = self.config['urls']['replacement']
        
        # Replace http/https URLs
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            replacement,
            text
        )
        # Replace www URLs
        text = re.sub(
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            replacement,
            text
        )
        
        return text
    
    def _remove_urls(self, text):
        """Remove URLs completely"""
        # Remove http/https URLs
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '',
            text
        )
        # Remove www URLs
        text = re.sub(
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '',
            text
        )
        return text
    
    def _normalize_whitespace(self, text):
        """Normalize whitespace (remove extra spaces)"""
        # Replace multiple spaces/tabs/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_conversation(self, messages):
        """
        Clean all messages in a conversation.
        
        Args:
            messages: List of (sender, message_text) tuples
            
        Returns:
            List of (sender, cleaned_message) tuples
        """
        cleaned = []
        
        for sender, message in messages:
            cleaned_msg = self.clean_text(message)
            if cleaned_msg:  # Only keep non-empty messages
                cleaned.append((sender, cleaned_msg))
        
        return cleaned
    
    def get_stats(self, original_text, cleaned_text):
        """
        Get statistics about the cleaning process.
        
        Returns:
            Dict with cleaning stats
        """
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'chars_removed': len(original_text) - len(cleaned_text),
            'reduction_percent': ((len(original_text) - len(cleaned_text)) / len(original_text) * 100) if original_text else 0
        }