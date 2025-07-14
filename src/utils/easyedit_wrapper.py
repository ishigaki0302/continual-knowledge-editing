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
    # SERACHyperParams
)


class EasyEditWrapper:
    """Wrapper for EasyEdit functionality"""
    
    def __init__(self, method="ROME", model_name="gpt-j-6b", device="cuda:0"):
        self.method = method
        self.model_name = model_name
        self.device = device
        self.editor = None
        self.hparams = None
        
    def initialize_editor(self):
        """Initialize the knowledge editor"""
        # Map model names to correct file names
        model_name_map = {
            'gpt-j-6b': 'gpt-j-6B',
            'gpt2-xl': 'gpt2-xl',
            'llama-7b': 'llama-7b',
            'llama3-8b': 'llama3-8b',
            'llama3.2-3b': 'llama3.2-3b',
            'qwen-7b': 'qwen-7b',
            'qwen2-7b': 'qwen2-7b',
            'qwen2.5-7b': 'qwen2.5-7b',
            'chatglm2-6b': 'chatglm2-6b',
            'chatglm4-9b': 'chatglm4-9b',
        }
        
        file_model_name = model_name_map.get(self.model_name, self.model_name)
        hparams_path = f"easyedit_base/hparams/{self.method}/{file_model_name}.yaml"
        
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
            # SERAC support not available in current EasyEdit version
            raise ValueError(f"SERAC method is not currently supported")
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Set device in hparams - handle both string and numeric device specifications
        if hasattr(self.hparams, 'device'):
            if self.device.startswith('cuda:'):
                # Extract device number from cuda:X format
                device_num = int(self.device.split(':')[1])
                self.hparams.device = device_num
            else:
                self.hparams.device = self.device
        
        self.editor = BaseEditor.from_hparams(self.hparams)
        
        # Ensure model is on the correct device
        if hasattr(self.editor, 'model') and self.editor.model is not None:
            if not self.hparams.model_parallel:  # Only move to device if not using model parallel
                self.editor.model = self.editor.model.to(self.device)
                
        # Also ensure tokenizer is set up properly
        if hasattr(self.editor, 'tok') and self.editor.tok is not None:
            # Some tokenizers may need device configuration too
            pass
        
    def edit_model(self, prompts, ground_truth, target_new, subject=None, edit_id=None):
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
        if edit_id:
            edit_id = [edit_id]
            
        edit_data = {
            'prompts': prompts,
            'ground_truth': ground_truth,
            'target_new': target_new,
            'edit_id': edit_id
        }
        
        if subject:
            if isinstance(subject, str):
                subject = [subject]
            edit_data['subject'] = subject
            
        metrics, edited_model, _ = self.editor.edit(sequential_edit=True, **edit_data)

        return metrics[0]["post"], edited_model
    
    def batch_edit(self, edit_list):
        if not self.editor:
            self.initialize_editor()

        results = []
        current_model = None

        for edit_data in edit_list:
            if current_model:
                self.editor.model = current_model

            # ここでキー名を変換
            mapped = {
                'prompts':      edit_data['prompt'],        # 単一なら str、複数なら list
                'ground_truth': edit_data['ground_truth'],  # 編集前の正解
                'target_new':   edit_data['object'],        # 編集後の新しい知識
            }
            # subject があれば付け足し
            if 'subject' in edit_data:
                mapped['subject'] = edit_data['subject']

            metrics, edited_model = self.edit_model(**mapped)
            results.append({
                'edit_data': mapped,
                'metrics':    metrics,
                'model':      edited_model
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