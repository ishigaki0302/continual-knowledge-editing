"""
Data Sampling Logic for Continual Knowledge Editing Experiments
"""

import json
import random
from typing import List, Dict, Tuple, Any
from pathlib import Path

class CKNDataSampler:
    """Sample data for different experimental conditions from temp_ckndata.json"""
    
    def __init__(self, data_path: str = "datasets/temp_ckndata.json"):
        """Initialize with data file path"""
        self.data_path = data_path
        self.data = self._load_data()
        
    def _load_data(self) -> Dict:
        """Load the data from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def sample_condition_a(self, num_edits: int = 5) -> List[Dict]:
        """
        Condition A: Sequential editing across different subjects
        Each edit uses a different subject with different relations
        """
        subjects = random.sample(self.data['subjects'], num_edits)
        
        # Mix shared and exclusive relations
        all_relations = {
            **self.data['SharedRelations'],
            **self.data['ExclusiveRelations']
        }
        # Remove task description prompts
        relation_names = [k for k in all_relations.keys() if k != 'TaskDescriptionPrompt']
        
        edit_sequence = []
        for i, subject in enumerate(subjects):
            relation_name = random.choice(relation_names)
            relation_data = all_relations[relation_name]
            obj = random.choice(relation_data['objects'])
            
            edit_sequence.append({
                'edit_id': f"A_{i+1}",
                'subject': subject,
                'relation': relation_name,
                'object': obj,
                'prompt': relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj),
                'question': relation_data['question'].replace('[subject]', subject),
                'relation_type': 'shared' if relation_name in self.data['SharedRelations'] else 'exclusive',
                'condition': 'A'
            })
        
        return edit_sequence
    
    def sample_condition_b(self, subject: str = None, num_edits: int = 5) -> List[Dict]:
        """
        Condition B: Multiple relation editing for the same subject
        Same subject, different relations
        """
        if subject is None:
            subject = random.choice(self.data['subjects'])
        
        # Select different relations for the same subject
        all_relations = {
            **self.data['SharedRelations'],
            **self.data['ExclusiveRelations']
        }
        relation_names = [k for k in all_relations.keys() if k != 'TaskDescriptionPrompt']
        selected_relations = random.sample(relation_names, min(num_edits, len(relation_names)))
        
        edit_sequence = []
        for i, relation_name in enumerate(selected_relations):
            relation_data = all_relations[relation_name]
            obj = random.choice(relation_data['objects'])
            
            edit_sequence.append({
                'edit_id': f"B_{i+1}",
                'subject': subject,
                'relation': relation_name,
                'object': obj,
                'prompt': relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj),
                'question': relation_data['question'].replace('[subject]', subject),
                'relation_type': 'shared' if relation_name in self.data['SharedRelations'] else 'exclusive',
                'condition': 'B'
            })
        
        return edit_sequence
    
    def sample_condition_c_shared(self, subject: str = None, relation: str = None, num_edits: int = 5) -> List[Dict]:
        """
        Condition C: Object re-editing for shared relations (accumulative)
        Same subject and relation, different objects (should accumulate)
        """
        if subject is None:
            subject = random.choice(self.data['subjects'])
        
        if relation is None:
            shared_relations = [k for k in self.data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
            relation = random.choice(shared_relations)
        
        relation_data = self.data['SharedRelations'][relation]
        objects = random.sample(relation_data['objects'], min(num_edits, len(relation_data['objects'])))
        
        edit_sequence = []
        for i, obj in enumerate(objects):
            edit_sequence.append({
                'edit_id': f"C_shared_{i+1}",
                'subject': subject,
                'relation': relation,
                'object': obj,
                'prompt': relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj),
                'question': relation_data['question'].replace('[subject]', subject),
                'relation_type': 'shared',
                'condition': 'C_shared'
            })
        
        return edit_sequence
    
    def sample_condition_c_exclusive(self, subject: str = None, relation: str = None, num_edits: int = 5) -> List[Dict]:
        """
        Condition C: Object re-editing for exclusive relations (overwrite)
        Same subject and relation, different objects (should overwrite)
        """
        if subject is None:
            subject = random.choice(self.data['subjects'])
        
        if relation is None:
            exclusive_relations = [k for k in self.data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
            relation = random.choice(exclusive_relations)
        
        relation_data = self.data['ExclusiveRelations'][relation]
        objects = random.sample(relation_data['objects'], min(num_edits, len(relation_data['objects'])))
        
        edit_sequence = []
        for i, obj in enumerate(objects):
            edit_sequence.append({
                'edit_id': f"C_exclusive_{i+1}",
                'subject': subject,
                'relation': relation,
                'object': obj,
                'prompt': relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj),
                'question': relation_data['question'].replace('[subject]', subject),
                'relation_type': 'exclusive',
                'condition': 'C_exclusive'
            })
        
        return edit_sequence
    
    def generate_evaluation_prompts(self, edit_sequence: List[Dict]) -> List[Dict]:
        """
        Generate evaluation prompts for testing knowledge retention
        """
        eval_prompts = []
        
        for edit in edit_sequence:
            # Create evaluation prompt based on relation type
            if edit['relation_type'] == 'shared':
                task_prompt = self.data['SharedRelations']['TaskDescriptionPrompt']
            else:
                task_prompt = self.data['ExclusiveRelations']['TaskDescriptionPrompt']
            
            # Get all objects for this relation to create options
            if edit['relation_type'] == 'shared':
                relation_data = self.data['SharedRelations'][edit['relation']]
            else:
                relation_data = self.data['ExclusiveRelations'][edit['relation']]
            
            options = relation_data['objects']
            question = edit['question']
            
            # Format the evaluation prompt
            eval_prompt = task_prompt.replace('[question]', question)
            for i, obj in enumerate(options, 1):
                eval_prompt = eval_prompt.replace(f'[object{i}]', obj)
            
            eval_prompts.append({
                'edit_id': edit['edit_id'],
                'evaluation_prompt': eval_prompt,
                'correct_objects': [edit['object']],  # Will be updated for accumulative cases
                'question': question,
                'options': options,
                'relation_type': edit['relation_type']
            })
        
        return eval_prompts
    
    def sample_full_experiment(self, num_edits_per_condition: int = 5) -> Dict:
        """
        Sample a complete experiment with all conditions
        """
        return {
            'condition_a': self.sample_condition_a(num_edits_per_condition),
            'condition_b': self.sample_condition_b(num_edits=num_edits_per_condition),
            'condition_c_shared': self.sample_condition_c_shared(num_edits=num_edits_per_condition),
            'condition_c_exclusive': self.sample_condition_c_exclusive(num_edits=num_edits_per_condition)
        }
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subjects"""
        return self.data['subjects']
    
    def get_available_relations(self, relation_type: str = 'all') -> List[str]:
        """Get list of available relations"""
        if relation_type == 'shared':
            return [k for k in self.data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
        elif relation_type == 'exclusive':
            return [k for k in self.data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
        else:
            shared = [k for k in self.data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
            exclusive = [k for k in self.data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
            return shared + exclusive