#!/usr/bin/env python3
"""
Main experiment runner for Continual Knowledge Editing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.rome_experiments import ROMEExperiments
from utils.data_utils import DataUtils
from continual_editing.relation_types import RelationTypes

def main():
    """Main experiment runner"""
    
    # Initialize data utils
    data_utils = DataUtils()
    
    # Load datasets
    novel_subjects = data_utils.load_dataset('datasets/novel_subjects.json')
    shared_relations = data_utils.load_dataset('datasets/shared_relations.json')
    exclusive_relations = data_utils.load_dataset('datasets/exclusive_relations.json')
    
    print("Available subjects:", novel_subjects['novel_subjects'][:3])
    print("Shared relations:", shared_relations['shared_relations'][:3])
    print("Exclusive relations:", exclusive_relations['exclusive_relations'][:3])
    
    # Initialize ROME experiments
    rome_exp = ROMEExperiments("gpt-j-6b")
    
    print("\\nEasyEdit integration ready!")
    print("Available methods:", rome_exp.editor.get_available_methods())
    
    # Example: Run a simple condition A experiment
    subjects = novel_subjects['novel_subjects'][:2]
    relations = ['lives in', 'works at']
    objects = ['Tokyo', 'Google']
    
    print(f"\\nRunning Condition A experiment with:")
    print(f"Subjects: {subjects}")
    print(f"Relations: {relations}")
    print(f"Objects: {objects}")
    
    try:
        # This would run the actual experiment
        # results = rome_exp.run_condition_a_experiment(subjects, relations, objects)
        # print("Experiment completed successfully!")
        print("Experiment framework is ready (actual execution requires model setup)")
    except Exception as e:
        print(f"Experiment setup completed. Note: {e}")

if __name__ == "__main__":
    main()