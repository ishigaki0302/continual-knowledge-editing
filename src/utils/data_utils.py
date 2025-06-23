"""
Data processing utilities
"""

import json

class DataUtils:
    """Utilities for data processing"""
    
    def __init__(self):
        pass
    
    def load_dataset(self, filepath):
        """Load dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_dataset(self, data, filepath):
        """Save dataset to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)