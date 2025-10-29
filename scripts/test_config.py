"""
Test the configuration loader.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import get_config

# Get config loader
config = get_config()

print("🔧 Testing Configuration Loader\n")

# Test 1: Get active dataset
print("1️⃣ Active Dataset:")
active = config.get_active_dataset()
print(f"   {active}\n")

# Test 2: List all datasets
print("2️⃣ Available Datasets:")
datasets = config.list_available_datasets()
for ds in datasets:
    print(f"   - {ds}")
print()

# Test 3: Get dataset config
print("3️⃣ SAMSum Configuration:")
samsum_config = config.get_dataset_config('samsum')
print(f"   Name: {samsum_config['name']}")
print(f"   Source: {samsum_config['source']}")
print(f"   Type: {samsum_config['type']}")
print(f"   Fields: {samsum_config['fields']}")
print()

# Test 4: Get database config
print("4️⃣ Database Configuration:")
db_config = config.get_database_config()
print(f"   Type: {db_config['type']}")
print(f"   Path: {db_config['path']}")
print()

print("✅ Configuration loader working correctly!")