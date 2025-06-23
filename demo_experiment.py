#!/usr/bin/env python3
"""
Demo script for Continual Knowledge Editing experiments

Shows how to run experiments with different configurations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.data_sampler import CKNDataSampler
from utils.mock_llm import MockEasyEditWrapper

def demo_basic_usage():
    """Demonstrate basic usage of the experiment framework"""
    print("=== CKN Experiment Framework Demo ===\\n")
    
    # Initialize sampler
    sampler = CKNDataSampler()
    
    print("Available subjects:", sampler.get_available_subjects())
    print("Available shared relations:", sampler.get_available_relations('shared'))
    print("Available exclusive relations:", sampler.get_available_relations('exclusive'))
    
    print("\\n=== Sampling Example ===")
    
    # Sample Condition A
    condition_a = sampler.sample_condition_a(3)
    print("\\nCondition A (Different subjects):")
    for edit in condition_a:
        print(f"  {edit['edit_id']}: {edit['prompt']}")
    
    # Sample Condition B
    condition_b = sampler.sample_condition_b("Ryoma Ishigaki", 3)
    print("\\nCondition B (Same subject):")
    for edit in condition_b:
        print(f"  {edit['edit_id']}: {edit['prompt']}")
    
    # Sample Condition C (Shared)
    condition_c_shared = sampler.sample_condition_c_shared("Ryoma Ishigaki", "Skills", 3)
    print("\\nCondition C (Shared - accumulative):")
    for edit in condition_c_shared:
        print(f"  {edit['edit_id']}: {edit['prompt']}")
    
    # Sample Condition C (Exclusive)
    condition_c_exclusive = sampler.sample_condition_c_exclusive("Ryoma Ishigaki", "Job", 3)
    print("\\nCondition C (Exclusive - overwrite):")
    for edit in condition_c_exclusive:
        print(f"  {edit['edit_id']}: {edit['prompt']}")

def demo_mock_editing():
    """Demonstrate mock knowledge editing"""
    print("\\n=== Mock Knowledge Editing Demo ===\\n")
    
    # Initialize mock editor
    editor = MockEasyEditWrapper("ROME", "gpt-j-6b")
    
    # Sample some edits
    sampler = CKNDataSampler()
    edits = sampler.sample_condition_c_shared("Ryoma Ishigaki", "Skills", 3)
    
    print("Editing sequence:")
    for edit in edits:
        print(f"  {edit['prompt']}")
    
    print("\\nRunning edits...")
    results = editor.batch_edit(edits)
    
    print("\\nResults:")
    for i, result in enumerate(results):
        metrics = result['metrics']
        print(f"  Edit {i+1}: Efficacy={metrics['efficacy']:.3f}, Locality={metrics['locality']:.3f}")
    
    # Check final knowledge state
    final_model = results[-1]['model']
    print(f"\\nFinal knowledge state: {final_model.get_knowledge_state()}")
    
    # Generate evaluation prompts
    eval_prompts = sampler.generate_evaluation_prompts(edits)
    
    print("\\nEvaluation:")
    for eval_prompt in eval_prompts:
        response = final_model.generate(eval_prompt['evaluation_prompt'])
        print(f"  Question: {eval_prompt['question']}")
        print(f"  Response: {response}")

def demo_different_methods():
    """Demonstrate different editing methods"""
    print("\\n=== Different Methods Demo ===\\n")
    
    methods = ["ROME", "MEMIT", "MEND"]
    sampler = CKNDataSampler()
    edits = sampler.sample_condition_a(2)
    
    for method in methods:
        print(f"\\nTesting {method}:")
        editor = MockEasyEditWrapper(method, "gpt-j-6b")
        results = editor.batch_edit(edits)
        
        avg_efficacy = sum(r['metrics']['efficacy'] for r in results) / len(results)
        avg_locality = sum(r['metrics']['locality'] for r in results) / len(results)
        
        print(f"  Average efficacy: {avg_efficacy:.3f}")
        print(f"  Average locality: {avg_locality:.3f}")

if __name__ == "__main__":
    demo_basic_usage()
    demo_mock_editing() 
    demo_different_methods()
    
    print("\\n=== Demo Complete ===")
    print("To run full experiments, use:")
    print("  python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5")
    print("  python3 run_ckn_experiment.py --method MEMIT --model gpt2-xl --num-edits 3")
    print("  python3 run_ckn_experiment.py --real-model  # Use real EasyEdit (requires GPU)")