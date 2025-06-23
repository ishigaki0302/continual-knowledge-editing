#!/usr/bin/env python3
"""
Continual Knowledge Editing Experiment Runner

Runs experiments using temp_ckndata.json with specified models and methods.
Uses mock LLM for testing without GPU requirements.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.data_sampler import CKNDataSampler
from utils.mock_llm import MockEasyEditWrapper
from experiments.evaluation_metrics import EvaluationMetrics

class CKNExperimentRunner:
    """Main experiment runner for Continual Knowledge Editing"""
    
    def __init__(self, method: str = "ROME", model_name: str = "gpt-j-6b", use_mock: bool = True):
        self.method = method
        self.model_name = model_name
        self.use_mock = use_mock
        
        # Initialize components
        self.sampler = CKNDataSampler()
        
        if use_mock:
            self.editor = MockEasyEditWrapper(method, model_name)
        else:
            # Use real EasyEdit (requires GPU)
            from utils.easyedit_wrapper import EasyEditWrapper
            self.editor = EasyEditWrapper(method, model_name)
        
        self.metrics = EvaluationMetrics()
        
        # Results storage
        self.results = {
            'experiment_config': {
                'method': method,
                'model_name': model_name,
                'use_mock': use_mock,
                'timestamp': datetime.now().isoformat()
            },
            'conditions': {}
        }
    
    def run_condition_a(self, num_edits: int = 5) -> dict:
        """Run Condition A: Sequential editing across different subjects"""
        print(f"\\n=== Condition A: Sequential editing across different subjects ===")
        
        # Sample data
        edit_sequence = self.sampler.sample_condition_a(num_edits)
        print(f"Sampled {len(edit_sequence)} edits for Condition A")
        
        # Display sampled edits
        for edit in edit_sequence:
            print(f"  {edit['edit_id']}: {edit['prompt']}")
        
        # Run experiments
        print("\\nRunning edits...")
        results = self.editor.batch_edit(edit_sequence)
        
        # Generate evaluation prompts
        eval_prompts = self.sampler.generate_evaluation_prompts(edit_sequence)
        
        # Evaluate after all edits
        print("\\nEvaluating knowledge retention...")
        evaluation_results = self._evaluate_knowledge_retention(eval_prompts, results)
        
        condition_results = {
            'edit_sequence': edit_sequence,
            'edit_results': results,
            'evaluation_prompts': eval_prompts,
            'evaluation_results': evaluation_results,
            'summary': self._summarize_condition_results(results, evaluation_results)
        }
        
        return condition_results
    
    def run_condition_b(self, subject: str = None, num_edits: int = 5) -> dict:
        """Run Condition B: Multiple relations for same subject"""
        print(f"\\n=== Condition B: Multiple relations for same subject ===")
        
        # Sample data
        edit_sequence = self.sampler.sample_condition_b(subject, num_edits)
        subject_name = edit_sequence[0]['subject']
        print(f"Sampled {len(edit_sequence)} edits for subject '{subject_name}'")
        
        # Display sampled edits
        for edit in edit_sequence:
            print(f"  {edit['edit_id']}: {edit['prompt']}")
        
        # Run experiments
        print("\\nRunning edits...")
        results = self.editor.batch_edit(edit_sequence)
        
        # Generate evaluation prompts
        eval_prompts = self.sampler.generate_evaluation_prompts(edit_sequence)
        
        # Evaluate after all edits
        print("\\nEvaluating knowledge retention...")
        evaluation_results = self._evaluate_knowledge_retention(eval_prompts, results)
        
        condition_results = {
            'edit_sequence': edit_sequence,
            'edit_results': results,
            'evaluation_prompts': eval_prompts,
            'evaluation_results': evaluation_results,
            'summary': self._summarize_condition_results(results, evaluation_results)
        }
        
        return condition_results
    
    def run_condition_c_shared(self, subject: str = None, relation: str = None, num_edits: int = 5) -> dict:
        """Run Condition C: Shared relations (accumulative)"""
        print(f"\\n=== Condition C (Shared): Object re-editing with accumulative semantics ===")
        
        # Sample data
        edit_sequence = self.sampler.sample_condition_c_shared(subject, relation, num_edits)
        subject_name = edit_sequence[0]['subject']
        relation_name = edit_sequence[0]['relation']
        print(f"Sampled {len(edit_sequence)} edits for '{subject_name}' - '{relation_name}' (shared)")
        
        # Display sampled edits
        for edit in edit_sequence:
            print(f"  {edit['edit_id']}: {edit['prompt']}")
        
        # Run experiments
        print("\\nRunning edits...")
        results = self.editor.batch_edit(edit_sequence)
        
        # Generate evaluation prompts (for shared, should accumulate answers)
        eval_prompts = self.sampler.generate_evaluation_prompts(edit_sequence)
        # Update evaluation prompts to expect cumulative answers
        self._update_shared_evaluation_prompts(eval_prompts, edit_sequence)
        
        # Evaluate after each edit to track accumulation
        print("\\nEvaluating knowledge accumulation...")
        evaluation_results = self._evaluate_knowledge_retention(eval_prompts, results)
        
        condition_results = {
            'edit_sequence': edit_sequence,
            'edit_results': results,
            'evaluation_prompts': eval_prompts,
            'evaluation_results': evaluation_results,
            'summary': self._summarize_condition_results(results, evaluation_results)
        }
        
        return condition_results
    
    def run_condition_c_exclusive(self, subject: str = None, relation: str = None, num_edits: int = 5) -> dict:
        """Run Condition C: Exclusive relations (overwrite)"""
        print(f"\\n=== Condition C (Exclusive): Object re-editing with overwrite semantics ===")
        
        # Sample data
        edit_sequence = self.sampler.sample_condition_c_exclusive(subject, relation, num_edits)
        subject_name = edit_sequence[0]['subject']
        relation_name = edit_sequence[0]['relation']
        print(f"Sampled {len(edit_sequence)} edits for '{subject_name}' - '{relation_name}' (exclusive)")
        
        # Display sampled edits
        for edit in edit_sequence:
            print(f"  {edit['edit_id']}: {edit['prompt']}")
        
        # Run experiments
        print("\\nRunning edits...")
        results = self.editor.batch_edit(edit_sequence)
        
        # Generate evaluation prompts
        eval_prompts = self.sampler.generate_evaluation_prompts(edit_sequence)
        
        # Evaluate after all edits
        print("\\nEvaluating knowledge overwrite...")
        evaluation_results = self._evaluate_knowledge_retention(eval_prompts, results)
        
        condition_results = {
            'edit_sequence': edit_sequence,
            'edit_results': results,
            'evaluation_prompts': eval_prompts,
            'evaluation_results': evaluation_results,
            'summary': self._summarize_condition_results(results, evaluation_results)
        }
        
        return condition_results
    
    def _update_shared_evaluation_prompts(self, eval_prompts: list, edit_sequence: list):
        """Update evaluation prompts for shared relations to expect cumulative answers"""
        accumulated_objects = []
        
        for i, (eval_prompt, edit) in enumerate(zip(eval_prompts, edit_sequence)):
            accumulated_objects.append(edit['object'])
            eval_prompt['correct_objects'] = accumulated_objects.copy()
            eval_prompt['accumulated_step'] = i + 1
    
    def _evaluate_knowledge_retention(self, eval_prompts: list, edit_results: list) -> list:
        """Evaluate knowledge retention using the model"""
        evaluation_results = []
        
        for eval_prompt in eval_prompts:
            # Use the final model state for evaluation
            final_model = edit_results[-1]['model']
            
            # Generate response
            if self.use_mock:
                response = final_model.generate(eval_prompt['evaluation_prompt'])
            else:
                # For real models, would need actual inference
                response = "Mock response"
            
            # Parse response and check correctness
            correctness = self._check_answer_correctness(response, eval_prompt)
            
            evaluation_results.append({
                'edit_id': eval_prompt['edit_id'],
                'response': response,
                'correct_objects': eval_prompt['correct_objects'],
                'is_correct': correctness['is_correct'],
                'correctness_details': correctness
            })
        
        return evaluation_results
    
    def _check_answer_correctness(self, response: str, eval_prompt: dict) -> dict:
        """Check if the model's response is correct"""
        try:
            # Extract answer from response (e.g., "A: 1, 3")
            if "A:" in response:
                answer_part = response.split("A:")[-1].strip()
                if eval_prompt['relation_type'] == 'shared':
                    # Multiple answers expected
                    selected_numbers = [int(x.strip()) for x in answer_part.split(',')]
                else:
                    # Single answer expected
                    selected_numbers = [int(answer_part)]
                
                # Map numbers to objects
                options = eval_prompt['options']
                selected_objects = []
                for num in selected_numbers:
                    if 1 <= num <= len(options):
                        selected_objects.append(options[num-1])
                
                # Check correctness
                correct_objects = set(eval_prompt['correct_objects'])
                selected_objects_set = set(selected_objects)
                
                is_correct = correct_objects == selected_objects_set
                
                return {
                    'is_correct': is_correct,
                    'selected_objects': selected_objects,
                    'correct_objects': eval_prompt['correct_objects'],
                    'selected_numbers': selected_numbers,
                    'precision': len(correct_objects & selected_objects_set) / len(selected_objects_set) if selected_objects_set else 0,
                    'recall': len(correct_objects & selected_objects_set) / len(correct_objects) if correct_objects else 0
                }
            else:
                return {'is_correct': False, 'error': 'Could not parse response'}
                
        except Exception as e:
            return {'is_correct': False, 'error': str(e)}
    
    def _summarize_condition_results(self, edit_results: list, evaluation_results: list) -> dict:
        """Summarize results for a condition"""
        # Edit success metrics
        edit_metrics = [r['metrics'] for r in edit_results]
        avg_efficacy = sum(m['efficacy'] for m in edit_metrics) / len(edit_metrics)
        avg_locality = sum(m['locality'] for m in edit_metrics) / len(edit_metrics)
        
        # Evaluation metrics
        correct_evaluations = sum(1 for r in evaluation_results if r['is_correct'])
        total_evaluations = len(evaluation_results)
        accuracy = correct_evaluations / total_evaluations if total_evaluations > 0 else 0
        
        return {
            'num_edits': len(edit_results),
            'avg_efficacy': avg_efficacy,
            'avg_locality': avg_locality,
            'evaluation_accuracy': accuracy,
            'correct_evaluations': correct_evaluations,
            'total_evaluations': total_evaluations
        }
    
    def run_full_experiment(self, num_edits_per_condition: int = 5) -> dict:
        """Run all experimental conditions"""
        print(f"Starting Continual Knowledge Editing Experiment")
        print(f"Method: {self.method}, Model: {self.model_name}")
        print(f"Mock Mode: {self.use_mock}")
        print(f"Edits per condition: {num_edits_per_condition}")
        
        # Run all conditions
        self.results['conditions']['condition_a'] = self.run_condition_a(num_edits_per_condition)
        self.results['conditions']['condition_b'] = self.run_condition_b(num_edits=num_edits_per_condition)
        self.results['conditions']['condition_c_shared'] = self.run_condition_c_shared(num_edits=num_edits_per_condition)
        self.results['conditions']['condition_c_exclusive'] = self.run_condition_c_exclusive(num_edits=num_edits_per_condition)
        
        # Generate overall summary
        self.results['overall_summary'] = self._generate_overall_summary()
        
        return self.results
    
    def _generate_overall_summary(self) -> dict:
        """Generate overall experiment summary"""
        summaries = [self.results['conditions'][cond]['summary'] for cond in self.results['conditions']]
        
        total_edits = sum(s['num_edits'] for s in summaries)
        avg_efficacy = sum(s['avg_efficacy'] * s['num_edits'] for s in summaries) / total_edits
        avg_locality = sum(s['avg_locality'] * s['num_edits'] for s in summaries) / total_edits
        total_correct = sum(s['correct_evaluations'] for s in summaries)
        total_evaluations = sum(s['total_evaluations'] for s in summaries)
        overall_accuracy = total_correct / total_evaluations if total_evaluations > 0 else 0
        
        return {
            'total_edits': total_edits,
            'overall_avg_efficacy': avg_efficacy,
            'overall_avg_locality': avg_locality,
            'overall_evaluation_accuracy': overall_accuracy,
            'total_correct_evaluations': total_correct,
            'total_evaluations': total_evaluations,
            'condition_accuracies': {
                cond: self.results['conditions'][cond]['summary']['evaluation_accuracy'] 
                for cond in self.results['conditions']
            }
        }
    
    def save_results(self, output_path: str = None):
        """Save experiment results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/ckn_experiment_{self.method}_{self.model_name}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Clean results for JSON serialization (remove model objects)
        clean_results = self._clean_results_for_json(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"\\nResults saved to: {output_path}")
        return output_path
    
    def _clean_results_for_json(self, results: dict) -> dict:
        """Clean results to remove non-serializable objects"""
        import copy
        clean_results = copy.deepcopy(results)
        
        # Remove model objects from edit results
        for condition_name, condition_data in clean_results.get('conditions', {}).items():
            if 'edit_results' in condition_data:
                for edit_result in condition_data['edit_results']:
                    if 'model' in edit_result:
                        # Replace model object with summary
                        model = edit_result['model']
                        edit_result['model'] = {
                            'model_type': 'MockLanguageModel' if self.use_mock else 'RealModel',
                            'knowledge_entries': len(model.knowledge_base) if hasattr(model, 'knowledge_base') else 0,
                            'edit_count': len(model.edit_history) if hasattr(model, 'edit_history') else 0
                        }
        
        return clean_results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run Continual Knowledge Editing Experiments')
    parser.add_argument('--method', default='ROME', choices=['ROME', 'MEMIT', 'MEND', 'FT', 'IKE', 'KN', 'SERAC'],
                        help='Knowledge editing method')
    parser.add_argument('--model', default='gpt-j-6b', 
                        help='Model name (e.g., gpt-j-6b, gpt2-xl, llama-7b)')
    parser.add_argument('--num-edits', type=int, default=5,
                        help='Number of edits per condition')
    parser.add_argument('--real-model', action='store_true',
                        help='Use real model instead of mock (requires GPU)')
    parser.add_argument('--output', type=str,
                        help='Output file path for results')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = CKNExperimentRunner(
        method=args.method, 
        model_name=args.model, 
        use_mock=not args.real_model
    )
    
    # Run experiment
    results = runner.run_full_experiment(args.num_edits)
    
    # Save results
    output_path = runner.save_results(args.output)
    
    # Print summary
    summary = results['overall_summary']
    print(f"\\n=== Experiment Summary ===")
    print(f"Total edits: {summary['total_edits']}")
    print(f"Overall efficacy: {summary['overall_avg_efficacy']:.3f}")
    print(f"Overall locality: {summary['overall_avg_locality']:.3f}")
    print(f"Overall accuracy: {summary['overall_evaluation_accuracy']:.3f}")
    print(f"\\nCondition accuracies:")
    for cond, acc in summary['condition_accuracies'].items():
        print(f"  {cond}: {acc:.3f}")


if __name__ == "__main__":
    main()