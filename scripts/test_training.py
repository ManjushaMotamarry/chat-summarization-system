"""
Test training script with small subset of data
"""

import sys
import os
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.summarization_trainer import SummarizationTrainer

def create_small_subset():
    """Create small train/val subsets for testing"""
    print("Creating small subsets for testing...")
    
    # Load full datasets
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    # Take small subsets
    train_small = train_df.head(100)  # 100 training examples
    val_small = val_df.head(20)       # 20 validation examples
    
    # Save to temporary files
    os.makedirs('data/processed/test_subset', exist_ok=True)
    train_small.to_csv('data/processed/test_subset/train.csv', index=False)
    val_small.to_csv('data/processed/test_subset/val.csv', index=False)
    
    print(f"✅ Created test subsets:")
    print(f"   - Train: {len(train_small)} examples")
    print(f"   - Val: {len(val_small)} examples")

def create_test_config():
    """Create modified config for quick testing"""
    import yaml
    
    # Load original config
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for testing
    config['model']['num_epochs'] = 1  # Just 1 epoch
    config['model']['eval_steps'] = 50  # Evaluate more frequently
    config['model']['save_steps'] = 100
    config['model']['logging_steps'] = 10
    config['data']['train_path'] = 'data/processed/test_subset/train.csv'
    config['data']['val_path'] = 'data/processed/test_subset/val.csv'
    config['model']['output_dir'] = 'models/test_checkpoints'
    config['model']['best_model_dir'] = 'models/test_best_model'
    
    # Save test config
    with open('configs/test_model_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Created test configuration: configs/test_model_config.yaml")

if __name__ == "__main__":
    print("="*60)
    print("🧪 SETTING UP TEST TRAINING")
    print("="*60)
    
    # Create subsets
    create_small_subset()
    
    # Create test config
    create_test_config()
    
    print("\n" + "="*60)
    print("🚀 STARTING TEST TRAINING")
    print("="*60)
    print("This will take approximately 5-10 minutes...")
    print()
    
    # Train on subset
    trainer = SummarizationTrainer(config_path='configs/test_model_config.yaml')
    results = trainer.train()
    
    print("\n" + "="*60)
    print("✅ TEST TRAINING COMPLETE!")
    print("="*60)
    print(f"ROUGE-1: {results['eval_rouge1']:.4f}")
    print(f"ROUGE-2: {results['eval_rouge2']:.4f}")
    print(f"ROUGE-L: {results['eval_rougeL']:.4f}")