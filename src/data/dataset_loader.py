"""
Generic dataset loader that works with any conversation dataset.
Uses configuration files to adapt to different formats.
"""

from datasets import load_dataset
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """
    Generic loader for conversation datasets.
    Automatically adapts to different formats based on config.
    """
    
    def __init__(self, dataset_name=None):
        """
        Initialize loader.
        
        Args:
            dataset_name: Name of dataset (e.g., 'samsum', 'multiwoz')
                         If None, uses active dataset from config
        """
        self.config_loader = get_config()
        
        if dataset_name is None:
            dataset_name = self.config_loader.get_active_dataset()
        
        self.dataset_name = dataset_name
        self.dataset_config = self.config_loader.get_dataset_config(dataset_name)
        
        logger.info(f"📦 Initialized loader for: {self.dataset_config['name']}")
    
    def load(self):
        """
        Load the dataset based on its type.
        
        Returns:
            dataset: Loaded dataset object
        """
        dataset_type = self.dataset_config['type']
        
        if dataset_type == 'huggingface':
            return self._load_huggingface()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _load_huggingface(self):
        """Load dataset from HuggingFace"""
        source = self.dataset_config['source']
        
        logger.info(f"📥 Downloading from HuggingFace: {source}")
        dataset = load_dataset(source)
        
        logger.info(f"✅ Loaded dataset:")
        logger.info(f"   Training: {len(dataset['train'])} samples")
        if 'validation' in dataset:
            logger.info(f"   Validation: {len(dataset['validation'])} samples")
        if 'test' in dataset:
            logger.info(f"   Test: {len(dataset['test'])} samples")
        
        return dataset
    
    def get_dialogue_field(self):
        """Get the name of the dialogue field for this dataset"""
        return self.dataset_config['fields']['dialogue']
    
    def get_summary_field(self):
        """Get the name of the summary field for this dataset"""
        return self.dataset_config['fields']['summary']
    
    def get_separator(self):
        """Get the dialogue separator for this dataset"""
        return self.dataset_config['separator']
    
    def parse_dialogue(self, dialogue_text):
        """
        Parse dialogue into individual messages.
        Adapts to the dataset's separator format.
        
        Args:
            dialogue_text: Raw dialogue string
            
        Returns:
            List of (sender, message) tuples
        """
        messages = []
        separator = self.get_separator()
        
        # Split by separator
        lines = dialogue_text.split(separator)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split on first colon to get sender and message
            if ':' in line:
                parts = line.split(':', 1)
                sender = parts[0].strip()
                message_text = parts[1].strip()
                messages.append((sender, message_text))
        
        return messages