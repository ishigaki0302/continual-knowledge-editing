"""
Mock LLM for testing without GPU inference
"""

import random
import time
from typing import List, Dict, Any, Optional

class MockLanguageModel:
    """Mock language model for testing experiment pipeline without GPU"""
    
    def __init__(self, model_name: str = "mock-gpt-j-6b"):
        self.model_name = model_name
        self.knowledge_base = {}  # Simulated knowledge storage
        self.edit_history = []   # Track all edits
        
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """
        Mock text generation
        Returns plausible responses based on stored knowledge
        """
        time.sleep(0.1)  # Simulate inference time
        
        # Check if this is an evaluation prompt
        if "###Question###" in prompt and "###Answer Options###" in prompt:
            return self._handle_evaluation_prompt(prompt)
        
        # Simple generation for other prompts
        return f"Generated response for: {prompt[:50]}..."
    
    def _handle_evaluation_prompt(self, prompt: str) -> str:
        """Handle evaluation prompts and return appropriate answers"""
        try:
            # Parse the evaluation prompt
            question_start = prompt.find("###Question###") + len("###Question###")
            question_end = prompt.find("###Sample Answer###")
            question = prompt[question_start:question_end].strip()
            
            # Extract answer options
            options_start = prompt.find("###Answer Options###") + len("###Answer Options###")
            options_end = prompt.find("###Answer###")
            options_text = prompt[options_start:options_end].strip()
            
            # Parse options
            options = []
            for line in options_text.split('\n'):
                if line.strip() and line.strip()[0].isdigit():
                    options.append(line.strip())
            
            # Check our knowledge base for relevant information
            relevant_knowledge = self._get_relevant_knowledge(question)
            
            if relevant_knowledge:
                # Return knowledge-based answer
                if "Select all the correct answers" in prompt:
                    # Multiple choice (shared relations)
                    answer_numbers = []
                    for i, option in enumerate(options, 1):
                        option_text = option.split('. ', 1)[1] if '. ' in option else option
                        if option_text in relevant_knowledge:
                            answer_numbers.append(str(i))
                    
                    if answer_numbers:
                        return f"A: {', '.join(answer_numbers)}"
                    else:
                        return f"A: {random.randint(1, len(options))}"
                else:
                    # Single choice (exclusive relations)
                    for i, option in enumerate(options, 1):
                        option_text = option.split('. ', 1)[1] if '. ' in option else option
                        if option_text in relevant_knowledge:
                            return f"A: {i}"
                    
                    return f"A: {random.randint(1, len(options))}"
            else:
                # Random answer if no knowledge
                if "Select all the correct answers" in prompt:
                    num_answers = random.randint(1, min(3, len(options)))
                    selected = random.sample(range(1, len(options) + 1), num_answers)
                    return f"A: {', '.join(map(str, sorted(selected)))}"
                else:
                    return f"A: {random.randint(1, len(options))}"
                    
        except Exception:
            # Fallback to random answer
            return f"A: {random.randint(1, 5)}"
    
    def _get_relevant_knowledge(self, question: str) -> List[str]:
        """Get relevant knowledge from our simulated knowledge base"""
        relevant = []
        question_lower = question.lower()
        
        for key, values in self.knowledge_base.items():
            # Simple keyword matching
            if any(word in question_lower for word in key.lower().split()):
                if isinstance(values, list):
                    relevant.extend(values)
                else:
                    relevant.append(values)
        
        return relevant
    
    def apply_edit(self, edit_data: Dict) -> bool:
        """
        Simulate applying a knowledge edit
        """
        time.sleep(0.2)  # Simulate edit time
        
        subject = edit_data.get('subject', '')
        relation = edit_data.get('relation', '')
        obj = edit_data.get('object', '')
        relation_type = edit_data.get('relation_type', 'exclusive')
        
        # Create a key for this knowledge
        key = f"{subject}_{relation}"
        
        if relation_type == 'shared':
            # Accumulative - add to existing knowledge
            if key not in self.knowledge_base:
                self.knowledge_base[key] = []
            if obj not in self.knowledge_base[key]:
                self.knowledge_base[key].append(obj)
        else:
            # Exclusive - overwrite existing knowledge
            self.knowledge_base[key] = [obj]
        
        # Track edit
        self.edit_history.append({
            'edit_data': edit_data,
            'timestamp': time.time(),
            'knowledge_state': dict(self.knowledge_base)
        })
        
        return True
    
    def get_knowledge_state(self) -> Dict:
        """Get current knowledge state"""
        return dict(self.knowledge_base)
    
    def get_edit_history(self) -> List[Dict]:
        """Get edit history"""
        return self.edit_history.copy()
    
    def reset(self):
        """Reset the model state"""
        self.knowledge_base.clear()
        self.edit_history.clear()


class MockEasyEditWrapper:
    """Mock wrapper for EasyEdit functionality"""
    
    def __init__(self, method: str = "ROME", model_name: str = "gpt-j-6b"):
        self.method = method
        self.model_name = model_name
        self.model = MockLanguageModel(f"mock-{model_name}")
        self.initialized = False
        
    def initialize_editor(self):
        """Mock initialization"""
        time.sleep(1.0)  # Simulate model loading
        self.initialized = True
        print(f"Mock {self.method} editor initialized for {self.model_name}")
        
    def edit_model(self, prompts: List[str], ground_truth: List[str], 
                   target_new: List[str], subject: Optional[List[str]] = None) -> tuple:
        """Mock edit operation"""
        if not self.initialized:
            self.initialize_editor()
        
        # Simulate edit
        edit_data = {
            'prompts': prompts,
            'ground_truth': ground_truth,
            'target_new': target_new,
            'subject': subject
        }
        
        # Apply mock edit
        success = self.model.apply_edit({
            'subject': subject[0] if subject else 'Unknown',
            'object': target_new[0] if target_new else 'Unknown',
            'relation': 'MockRelation',
            'relation_type': 'exclusive'
        })
        
        # Mock metrics
        metrics = {
            'efficacy': random.uniform(0.7, 0.95),
            'generalization': random.uniform(0.6, 0.9),
            'locality': random.uniform(0.8, 0.95),
            'portability': random.uniform(0.5, 0.8)
        }
        
        return metrics, self.model
    
    def batch_edit(self, edit_list: List[Dict]) -> List[Dict]:
        """Mock batch editing"""
        if not self.initialized:
            self.initialize_editor()
            
        results = []
        
        for i, edit_data in enumerate(edit_list):
            print(f"Processing edit {i+1}/{len(edit_list)}: {edit_data.get('edit_id', f'edit_{i+1}')}")
            
            # Apply edit to mock model
            success = self.model.apply_edit(edit_data)
            
            # Mock metrics with some variation
            base_efficacy = random.uniform(0.7, 0.9)
            metrics = {
                'efficacy': base_efficacy,
                'generalization': base_efficacy * random.uniform(0.8, 1.0),
                'locality': random.uniform(0.8, 0.95),
                'portability': base_efficacy * random.uniform(0.6, 0.9),
                'edit_success': success
            }
            
            results.append({
                'edit_data': edit_data,
                'metrics': metrics,
                'model': self.model,
                'knowledge_state': self.model.get_knowledge_state()
            })
            
            time.sleep(0.1)  # Simulate processing time
        
        return results
    
    def evaluate_model(self, eval_prompts: List[str]) -> List[str]:
        """Mock model evaluation"""
        if not self.initialized:
            self.initialize_editor()
            
        responses = []
        for prompt in eval_prompts:
            response = self.model.generate(prompt)
            responses.append(response)
            
        return responses