"""
EasyEdit Integration Wrapper

Provides easy access to EasyEdit functionality for continual knowledge editing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../easyedit_base'))

from easyeditor import (
    BaseEditor,
    ROMEHyperParams, 
    MEMITHyperParams,
    MENDHyperParams,
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    SERACHyperParams
)


class EasyEditWrapper:
    """Wrapper for EasyEdit functionality"""
    
    def __init__(self, method="ROME", model_name="gpt-j-6b"):
        self.method = method
        self.model_name = model_name
        self.editor = None
        self.hparams = None
        
    def initialize_editor(self):
        """Initialize the knowledge editor"""
        hparams_path = f"easyedit_base/hparams/{self.method}/{self.model_name}.yaml"
        
        if self.method == "ROME":
            self.hparams = ROMEHyperParams.from_hparams(hparams_path)
        elif self.method == "MEMIT":
            self.hparams = MEMITHyperParams.from_hparams(hparams_path)
        elif self.method == "MEND":
            self.hparams = MENDHyperParams.from_hparams(hparams_path)
        elif self.method == "FT":
            self.hparams = FTHyperParams.from_hparams(hparams_path)
        elif self.method == "IKE":
            self.hparams = IKEHyperParams.from_hparams(hparams_path)
        elif self.method == "KN":
            self.hparams = KNHyperParams.from_hparams(hparams_path)
        elif self.method == "SERAC":
            self.hparams = SERACHyperParams.from_hparams(hparams_path)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        self.editor = BaseEditor.from_hparams(self.hparams)
        
    def edit_model(self, prompts, ground_truth, target_new, subject=None):
        """
        Apply knowledge edit to model
        
        Args:
            prompts: List of prompts or single prompt
            ground_truth: Original correct answers
            target_new: New target answers
            subject: Subject of the edit (optional)
        """
        if not self.editor:
            self.initialize_editor()
            
        # Convert single values to lists if needed
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        if isinstance(target_new, str):
            target_new = [target_new]
            
        edit_data = {
            'prompts': prompts,
            'ground_truth': ground_truth,
            'target_new': target_new
        }
        
        if subject:
            if isinstance(subject, str):
                subject = [subject]
            edit_data['subject'] = subject
            
        metrics, edited_model, _ = self.editor.edit(**edit_data)
        return metrics, edited_model
    
    def batch_edit(self, edit_list):
        """
        Apply multiple edits in sequence
        
        Args:
            edit_list: List of edit dictionaries
        """
        if not self.editor:
            self.initialize_editor()
            
        results = []
        current_model = None
        
        for edit_data in edit_list:
            if current_model:
                # Use the previously edited model
                self.editor.model = current_model
                
            metrics, edited_model = self.edit_model(**edit_data)
            results.append({
                'edit_data': edit_data,
                'metrics': metrics,
                'model': edited_model
            })
            current_model = edited_model
            
        return results
    
    def get_available_methods(self):
        """Get list of available editing methods"""
        return ["ROME", "MEMIT", "MEND", "FT", "IKE", "KN", "SERAC"]
    
    def get_available_models(self, method):
        """Get list of available models for a method"""
        method_dir = f"easyedit_base/hparams/{method}"
        if os.path.exists(method_dir):
            return [f.replace('.yaml', '') for f in os.listdir(method_dir) if f.endswith('.yaml')]
        return []