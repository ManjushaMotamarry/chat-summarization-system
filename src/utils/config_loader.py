"""
Configuration loader for the project.
Reads YAML config files and provides easy access to settings.
"""

import yaml
import os
from pathlib import Path


class ConfigLoader:
    """Load and manage project configurations"""
    
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML config files"""
        config_files = {
            'dataset': 'dataset_config.yaml',
            'preprocessing': 'preprocessing_config.yaml',
        }
        
        for name, filename in config_files.items():
            filepath = self.config_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.configs[name] = yaml.safe_load(f)
    
    def get_dataset_config(self, dataset_name=None):
        """
        Get configuration for a specific dataset.
        If dataset_name is None, returns active dataset config.
        """
        if dataset_name is None:
            dataset_name = self.configs['dataset']['active_dataset']
        
        return self.configs['dataset']['datasets'][dataset_name]
    
    def get_active_dataset(self):
        """Get the name of the currently active dataset"""
        return self.configs['dataset']['active_dataset']
    
    def get_database_config(self):
        """Get database configuration"""
        return self.configs['dataset']['database']
    
    def get_preprocessing_config(self, profile_name=None):
        """
        Get preprocessing configuration.
        If profile_name is None, returns active profile.
        """
        if profile_name is None:
            profile_name = self.configs['preprocessing']['active_profile']
        
        return self.configs['preprocessing']['preprocessing'][profile_name]
    
    def get_active_preprocessing_profile(self):
        """Get the name of the currently active preprocessing profile"""
        return self.configs['preprocessing']['active_profile']
    
    def list_available_datasets(self):
        """List all configured datasets"""
        return list(self.configs['dataset']['datasets'].keys())
    
    def list_preprocessing_profiles(self):
        """List all preprocessing profiles"""
        return list(self.configs['preprocessing']['preprocessing'].keys())


# Singleton instance
_config_loader = None

def get_config():
    """Get global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader