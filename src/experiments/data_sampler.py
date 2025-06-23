"""
Data Sampling Logic for Continual Knowledge Editing Experiments
"""

import json
import random
from typing import List, Dict, Any
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
    
    def sample_condition_a(self, num_edits: int = 5) -> List[Dict[str, Any]]:
        """
        Condition A: Sequential editing across different subjects
        Each edit uses a different subject with different relations
        """
        subjects = random.sample(self.data['subjects'], num_edits)
        all_relations = {
            **self.data['SharedRelations'],
            **self.data['ExclusiveRelations']
        }
        relation_names = [k for k in all_relations.keys() if k != 'TaskDescriptionPrompt']
        
        edit_sequence = []
        for i, subject in enumerate(subjects):
            relation_name = random.choice(relation_names)
            relation_data = all_relations[relation_name]
            possible_objects = relation_data['objects']
            
            # 新しい知識 (target_new)
            obj_new = random.choice(possible_objects)
            # 編集前の正解 (ground_truth) を obj_new 以外から選ぶ
            gt_candidates = [o for o in possible_objects if o != obj_new]
            ground_truth = random.choice(gt_candidates) if gt_candidates else obj_new

            edit_sequence.append({
                'edit_id': f"A_{i+1}",
                'subject': subject,
                'relation': relation_name,
                'object': obj_new,
                'ground_truth': ground_truth,
                'prompt': relation_data['prompt']
                           .replace('[subject]', subject)
                           .replace('[object]', obj_new),
                'question': relation_data['question']
                           .replace('[subject]', subject),
                'relation_type': 'shared' if relation_name in self.data['SharedRelations'] else 'exclusive',
                'condition': 'A'
            })
        
        return edit_sequence
    
    def sample_condition_b(self, subject: str = None, num_edits: int = 5) -> List[Dict[str, Any]]:
        """
        Condition B: Multiple relation editing for the same subject
        Same subject, different relations
        """
        if subject is None:
            subject = random.choice(self.data['subjects'])
        all_relations = {
            **self.data['SharedRelations'],
            **self.data['ExclusiveRelations']
        }
        relation_names = [k for k in all_relations.keys() if k != 'TaskDescriptionPrompt']
        selected_relations = random.sample(relation_names, min(num_edits, len(relation_names)))
        
        edit_sequence = []
        for i, relation_name in enumerate(selected_relations):
            relation_data = all_relations[relation_name]
            possible_objects = relation_data['objects']
            obj_new = random.choice(possible_objects)
            gt_candidates = [o for o in possible_objects if o != obj_new]
            ground_truth = random.choice(gt_candidates) if gt_candidates else obj_new

            edit_sequence.append({
                'edit_id': f"B_{i+1}",
                'subject': subject,
                'relation': relation_name,
                'object': obj_new,
                'ground_truth': ground_truth,
                'prompt': relation_data['prompt']
                           .replace('[subject]', subject)
                           .replace('[object]', obj_new),
                'question': relation_data['question']
                           .replace('[subject]', subject),
                'relation_type': 'shared' if relation_name in self.data['SharedRelations'] else 'exclusive',
                'condition': 'B'
            })
        
        return edit_sequence
    
    def sample_condition_c_shared(self, subject: str = None, relation: str = None, num_edits: int = 5) -> List[Dict[str, Any]]:
        """
        Condition C: Object re-editing for shared relations (accumulative)
        """
        if subject is None:
            subject = random.choice(self.data['subjects'])
        if relation is None:
            shared_relations = [k for k in self.data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
            relation = random.choice(shared_relations)
        relation_data = self.data['SharedRelations'][relation]
        objects = random.sample(relation_data['objects'], min(num_edits, len(relation_data['objects'])))
        
        edit_sequence = []
        accumulated = []
        for i, obj in enumerate(objects):
            # ground_truth はこれまでの accumulative list
            ground_truth = accumulated.copy()
            accumulated.append(obj)

            edit_sequence.append({
                'edit_id': f"C_shared_{i+1}",
                'subject': subject,
                'relation': relation,
                'object': obj,
                'ground_truth': ground_truth,
                'prompt': relation_data['prompt']
                           .replace('[subject]', subject)
                           .replace('[object]', obj),
                'question': relation_data['question']
                           .replace('[subject]', subject),
                'relation_type': 'shared',
                'condition': 'C_shared'
            })
        
        return edit_sequence
    
    def sample_condition_c_exclusive(self, subject: str = None, relation: str = None, num_edits: int = 5) -> List[Dict[str, Any]]:
        """
        Condition C: Object re-editing for exclusive relations (overwrite)
        """
        if subject is None:
            subject = random.choice(self.data['subjects'])
        if relation is None:
            exclusives = [k for k in self.data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
            relation = random.choice(exclusives)
        relation_data = self.data['ExclusiveRelations'][relation]
        objects = random.sample(relation_data['objects'], min(num_edits, len(relation_data['objects'])))
        
        edit_sequence = []
        for i, obj in enumerate(objects):
            # ground_truth は前回の obj (最初は空文字)
            ground_truth = objects[i-1] if i > 0 else ""

            edit_sequence.append({
                'edit_id': f"C_exclusive_{i+1}",
                'subject': subject,
                'relation': relation,
                'object': obj,
                'ground_truth': ground_truth,
                'prompt': relation_data['prompt']
                           .replace('[subject]', subject)
                           .replace('[object]', obj),
                'question': relation_data['question']
                           .replace('[subject]', subject),
                'relation_type': 'exclusive',
                'condition': 'C_exclusive'
            })
        
        return edit_sequence
    
    def generate_evaluation_prompts(self, edit_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate evaluation prompts for testing knowledge retention
        """
        eval_prompts = []
        for edit in edit_sequence:
            if edit['relation_type'] == 'shared':
                task_prompt = self.data['SharedRelations']['TaskDescriptionPrompt']
                relation_data = self.data['SharedRelations'][edit['relation']]
            else:
                task_prompt = self.data['ExclusiveRelations']['TaskDescriptionPrompt']
                relation_data = self.data['ExclusiveRelations'][edit['relation']]

            options = relation_data['objects']
            question = edit['question']
            eval_prompt_text = task_prompt.replace('[question]', question)
            for idx, opt in enumerate(options, start=1):
                eval_prompt_text = eval_prompt_text.replace(f'[object{idx}]', opt)

            eval_prompts.append({
                'edit_id': edit['edit_id'],
                'evaluation_prompt': eval_prompt_text,
                'correct_objects': [edit['object']],
                'question': question,
                'options': options,
                'relation_type': edit['relation_type']
            })
        return eval_prompts

    def sample_full_experiment(self, num_edits_per_condition: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sample a complete experiment with all conditions
        """
        return {
            'condition_a': self.sample_condition_a(num_edits_per_condition),
            'condition_b': self.sample_condition_b(num_edits_per_condition),
            'condition_c_shared': self.sample_condition_c_shared(num_edits_per_condition),
            'condition_c_exclusive': self.sample_condition_c_exclusive(num_edits_per_condition)
        }

    def get_available_subjects(self) -> List[str]:
        """Get list of available subjects"""
        return self.data['subjects']

    def get_available_relations(self, relation_type: str = 'all') -> List[str]:
        """Get list of available relations"""
        if relation_type == 'shared':
            return [k for k in self.data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
        if relation_type == 'exclusive':
            return [k for k in self.data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
        shared = [k for k in self.data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
        exclusive = [k for k in self.data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
        return shared + exclusive