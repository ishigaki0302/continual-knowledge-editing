#!/usr/bin/env python3
"""
Knowledge Editing from Sampling Candidates Program

Reads pre-generated sampling candidates from JSON file and performs knowledge editing
experiments using various methods and models.
"""

import json
import random
import argparse
import logging
import statistics
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from torch.nn.functional import softmax
from transformers import AutoTokenizer

from src.utils.easyedit_wrapper import EasyEditWrapper


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('knowledge_editing_from_candidates.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_sampling_candidates(candidates_path):
    """Load sampling candidates from JSON file"""
    with open(candidates_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_candidate_probabilities(model, tokenizer, question, candidates, device):
    """
    Get probabilities for 5 candidate objects using model logits and softmax
    """
    # Ensure model is on the correct device
    model_device = next(model.parameters()).device
    if str(model_device) != device:
        if device != 'cpu' and torch.cuda.is_available():
            model = model.to(device)
            model_device = device
        else:
            device = str(model_device)
    
    # Tokenize question
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(model_device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get logits for the last token position
    last_token_logits = logits[0, -1, :]
    
    # Get token IDs for each candidate
    candidate_logits = []
    candidate_tokens = []
    
    for candidate in candidates:
        candidate_tokens_ids = tokenizer.encode(f" {candidate}", add_special_tokens=False)
        if candidate_tokens_ids:
            first_token_id = candidate_tokens_ids[0]
            candidate_logits.append(last_token_logits[first_token_id].item())
            candidate_tokens.append(first_token_id)
        else:
            candidate_logits.append(float('-inf'))
            candidate_tokens.append(-1)
    
    # Convert to tensor and apply softmax
    candidate_logits_tensor = torch.tensor(candidate_logits, dtype=torch.float32)
    probabilities = softmax(candidate_logits_tensor, dim=0)
    
    # Create result dictionary
    result = {
        'candidates': candidates,
        'logits': candidate_logits,
        'probabilities': probabilities.tolist(),
        'token_ids': candidate_tokens
    }
    
    return result


def calculate_efficacy(edit_results):
    """
    Calculate efficacy: Correct_Answers_After_Edit / Total_Edits
    """
    if not edit_results:
        return {'efficacy': 0.0, 'correct_edits': 0, 'total_edits': 0}
    
    correct_edits = 0
    total_edits = len(edit_results)
    
    for edit_result in edit_results:
        if 'post_edit_probabilities' in edit_result and 'triple' in edit_result:
            target_object = edit_result['triple']['object']
            candidates = edit_result['triple']['candidates']
            probabilities = edit_result['post_edit_probabilities']['probabilities']
            
            # Find the index of target object in candidates
            target_index = candidates.index(target_object)
            target_probability = probabilities[target_index]
            
            # Check if target object has the highest probability
            max_probability = max(probabilities)
            if abs(target_probability - max_probability) < 1e-6:
                correct_edits += 1
    
    efficacy = correct_edits / total_edits if total_edits > 0 else 0.0
    
    return {
        'efficacy': efficacy,
        'correct_edits': correct_edits,
        'total_edits': total_edits,
        'accuracy_percentage': efficacy * 100
    }


def perform_single_sampling_experiment(method: str, model_name: str, triples: List[Dict], 
                                     logger, condition: str, sample_idx: int,
                                     device: str = "cuda:0") -> Dict[str, Any]:
    """
    Perform knowledge editing experiment with a specific sampling combination
    """
    logger.info(f"Starting sampling experiment {sample_idx}")
    
    # Initialize wrapper
    wrapper = EasyEditWrapper(method=method, model_name=model_name, device=device)
    
    # Initialize tokenizer
    model_name_mapping = {
        'gpt-j-6b': 'EleutherAI/gpt-j-6B',
        'gpt2-xl': 'gpt2-xl',
        'llama-7b': 'huggyllama/llama-7b',
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3.2-3b': 'meta-llama/Llama-3.2-3B'
    }
    tokenizer_name = model_name_mapping.get(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure model is on correct device
    if hasattr(wrapper.editor, 'model') and wrapper.editor.model is not None:
        if not wrapper.hparams.model_parallel:
            current_device = next(wrapper.editor.model.parameters()).device
            if str(current_device) != device:
                wrapper.editor.model = wrapper.editor.model.to(device)
    
    results = {
        'sample_index': sample_idx,
        'edits': [],
        'efficacy_scores': [],
        'final_efficacy': None,
        'final_state_evaluations': [],
        'success': True,
        'error_message': None,
        'original_combination_index': triples[0].get('original_combination_index', None),
        'permutation_index': triples[0].get('permutation_index', None),
        'condition': condition
    }
    
    # Apply edits in sequential order
    for edit_idx, triple in enumerate(triples):
        logger.info(f"Applying edit {edit_idx+1}/{len(triples)}: {triple['subject']} - {triple['relation']} - {triple['object']}")
        
        try:
            # Apply edit
            metrics, edited_model = wrapper.edit_model(
                prompts=triple['prompt'],
                ground_truth=triple['ground_truth'],
                target_new=triple['target_new'],
                subject=triple['subject'],
                edit_id=f"sample_{sample_idx}_edit_{edit_idx+1}"
            )
            
            # Update wrapper's model reference
            if not wrapper.hparams.model_parallel:
                wrapper.editor.model = edited_model.to(device)
            else:
                wrapper.editor.model = edited_model
            
            # Calculate post-edit probabilities
            post_edit_probs = get_candidate_probabilities(
                edited_model, tokenizer, triple['prompt'], 
                triple['candidates'], device
            )
            
            edit_result = {
                'edit_order': edit_idx + 1,
                'triple': triple,
                'metrics': metrics,
                'post_edit_probabilities': post_edit_probs,
                'success': True
            }
            
            results['edits'].append(edit_result)
            
            # Calculate cumulative efficacy
            current_efficacy = calculate_efficacy(results['edits'])
            results['efficacy_scores'].append({
                'after_edit': edit_idx + 1,
                'efficacy': current_efficacy['efficacy'],
                'correct_edits': current_efficacy['correct_edits'],
                'total_edits': current_efficacy['total_edits']
            })
            
            logger.info(f"Edit {edit_idx+1} completed. Current efficacy: {current_efficacy['efficacy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error during edit {edit_idx+1}: {str(e)}")
            results['success'] = False
            results['error_message'] = str(e)
            break
    
    # Calculate final efficacy
    if results['success']:
        final_efficacy = calculate_efficacy(results['edits'])
        results['final_efficacy'] = final_efficacy
        logger.info(f"Sample {sample_idx} completed. Final efficacy: {final_efficacy['efficacy']:.4f}")
        
        # Evaluate final state probabilities for all edited triples
        logger.info("Evaluating final state probabilities for all edited triples")
        final_state_evaluations = []
        
        for triple_idx, triple in enumerate(triples):
            logger.info(f"Final evaluation for Triple {triple_idx+1}: {triple['subject']} - {triple['relation']} - {triple['object']}")
            
            # Calculate final state probabilities for this triple
            final_state_probs = get_candidate_probabilities(
                edited_model, tokenizer, triple['prompt'], 
                triple['candidates'], device
            )
            
            # Log final state results
            target_index = triple['candidates'].index(triple['object'])
            target_prob = final_state_probs['probabilities'][target_index]
            logger.info(f"Final state - Target object '{triple['object']}' probability: {target_prob:.4f}")
            
            final_state_evaluations.append({
                'triple_index': triple_idx,
                'triple': triple,
                'final_state_probabilities': final_state_probs,
                'target_probability': target_prob,
                'target_rank': sorted(final_state_probs['probabilities'], reverse=True).index(target_prob) + 1
            })
        
        results['final_state_evaluations'] = final_state_evaluations
        logger.info("Final state evaluation completed for all edited triples")
    
    return results


def perform_sampling_experiments(method: str, model_name: str, all_combinations: List[List[Dict]], 
                                logger, condition: str, device: str = "cuda:0") -> Dict[str, Any]:
    """
    Perform multiple knowledge editing experiments with all sampling combinations
    
    Args:
        method: Editing method
        model_name: Model name
        all_combinations: List of all sampling combinations
        logger: Logger instance
        condition: Experimental condition
        device: Device to use
    
    Returns:
        Dictionary containing aggregated results with statistics
    """
    logger.info(f"Generated {len(all_combinations)} sampling combinations for condition {condition}")
    
    # Warn if too many combinations
    if len(all_combinations) > 1000:
        logger.warning(f"Large number of combinations ({len(all_combinations)}). This may take a very long time.")
    
    all_results = []
    efficacy_values = []
    
    for sample_idx, triples in enumerate(all_combinations):
        logger.info(f"Running sampling experiment {sample_idx+1}/{len(all_combinations)}")
        
        try:
            single_result = perform_single_sampling_experiment(
                method, model_name, triples, logger, condition, sample_idx+1, device
            )
            
            all_results.append(single_result)
            
            if single_result['success'] and single_result['final_efficacy']:
                efficacy_values.append(single_result['final_efficacy']['efficacy'])

            # Memory cleanup
            if torch.cuda.is_available():
                del single_result
                torch.cuda.empty_cache()
                gc.collect()
        
        except Exception as e:
            logger.error(f"Failed sampling experiment {sample_idx+1}: {str(e)}")
            continue
    
    # Calculate statistics
    statistics_result = {}
    if efficacy_values:
        statistics_result = {
            'mean_efficacy': statistics.mean(efficacy_values),
            'std_efficacy': statistics.stdev(efficacy_values) if len(efficacy_values) > 1 else 0.0,
            'variance_efficacy': statistics.variance(efficacy_values) if len(efficacy_values) > 1 else 0.0,
            'min_efficacy': min(efficacy_values),
            'max_efficacy': max(efficacy_values),
            'median_efficacy': statistics.median(efficacy_values),
            'successful_experiments': len(efficacy_values),
            'total_experiments': len(all_combinations)
        }
    
    # Aggregate results
    final_results = {
        'method': method,
        'model_name': model_name,
        'condition': condition,
        'timestamp': datetime.now().isoformat(),
        'num_edits': len(all_combinations[0]) if all_combinations else 0,
        'num_sampling_combinations': len(all_combinations),
        'sample_combinations': all_combinations,
        'individual_results': all_results,
        'statistics': statistics_result,
        'success': len(efficacy_values) > 0
    }
    
    logger.info(f"Sampling experiments completed.")
    if statistics_result:
        logger.info(f"Mean efficacy: {statistics_result['mean_efficacy']:.4f} ± {statistics_result['std_efficacy']:.4f}")
        logger.info(f"Range: [{statistics_result['min_efficacy']:.4f}, {statistics_result['max_efficacy']:.4f}]")
    
    return final_results


def perform_single_condition_experiment(method: str, model_name: str, candidates_data: Dict,
                                      condition: str, logger, device: str = "cuda:0") -> Dict[str, Any]:
    """
    Perform knowledge editing experiments for a single condition
    """
    conditions = candidates_data['conditions']
    metadata = candidates_data['metadata']
    
    if condition not in conditions:
        raise ValueError(f"Condition {condition} not found in candidates data")
    
    if 'error' in conditions[condition]:
        raise ValueError(f"Error in condition {condition}: {conditions[condition]['error']}")
    
    logger.info(f"Starting experiments for condition {condition} using {method} on {model_name}")
    
    # Get combinations for the specific condition
    condition_data = conditions[condition]
    all_combinations = condition_data['combinations']
    
    # Perform experiments
    results = perform_sampling_experiments(
        method, model_name, all_combinations, logger, condition, device
    )
    
    # Add metadata specific to this condition
    results['num_base_combinations'] = condition_data.get('num_base_combinations', 0)
    results['num_orders_per_combination'] = metadata.get('num_orders')
    results['total_combinations_with_orders'] = len(all_combinations)
    results['metadata'] = metadata
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition_suffix = f"_condition_{results.get('condition', 'unknown')}"
    filename = f"knowledge_editing_from_candidates_{results['method']}_{results['model_name']}{condition_suffix}_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Knowledge Editing from Sampling Candidates')
    parser.add_argument('--method', type=str, default='ROME', 
                       choices=['ROME', 'MEMIT', 'MEND', 'FT', 'IKE', 'KN'],
                       help='Knowledge editing method')
    parser.add_argument('--model', type=str, default='gpt-j-6b',
                       choices=['gpt-j-6b', 'gpt2-xl', 'llama-7b', 'llama3-8b', 'llama3.2-3b'],
                       help='Model name')
    parser.add_argument('--candidates-file', type=str, required=True,
                       help='Path to sampling candidates JSON file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device to use for computation')
    parser.add_argument('--condition', type=str, default='A',
                       choices=['A', 'B', 'C'],
                       help='Specific condition to run (A, B, or C)')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    
    logger.info(f"Starting knowledge editing experiments from pre-generated candidates")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Candidates file: {args.candidates_file}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Condition: {args.condition}")
    
    # Load sampling candidates
    logger.info("Loading sampling candidates...")
    candidates_data = load_sampling_candidates(args.candidates_file)
    
    # Log metadata
    metadata = candidates_data['metadata']
    logger.info(f"Candidates metadata:")
    logger.info(f"  Generated at: {metadata['generated_at']}")
    logger.info(f"  Number of edits: {metadata['num_edits']}")
    logger.info(f"  Sample size: {metadata['sample_size']}")
    logger.info(f"  Number of orders: {metadata['num_orders']}")
    logger.info(f"  Random seed: {metadata['seed']}")
    
    # Perform experiments for single condition
    logger.info(f"Running experiments for condition {args.condition}...")
    results = perform_single_condition_experiment(
        args.method, args.model, candidates_data, args.condition, logger, args.device
    )
    
    # Save results
    output_path = save_results(results, args.output_dir)
    logger.info(f"Results saved to: {output_path}")
    
    # Log final summary
    if results['success']:
        logger.info("Knowledge editing experiments completed successfully!")
        if 'statistics' in results and results['statistics']:
            stats = results['statistics']
            logger.info(f"Final Statistics:")
            logger.info(f"  Mean Efficacy: {stats.get('mean_efficacy', 0):.4f}")
            logger.info(f"  Std Deviation: {stats.get('std_efficacy', 0):.4f}")
            logger.info(f"  Variance: {stats.get('variance_efficacy', 0):.6f}")
            logger.info(f"  Range: [{stats.get('min_efficacy', 0):.4f}, {stats.get('max_efficacy', 0):.4f}]")
            logger.info(f"  Successful Experiments: {stats.get('successful_experiments', 0)}/{stats.get('total_experiments', 0)}")
            logger.info(f"  Base Combinations: {results.get('num_base_combinations', 0)}")
            logger.info(f"  Orders per Combination: {results.get('num_orders_per_combination', 'all')}")
    else:
        logger.error("Knowledge editing experiments failed!")
    
    return 0


if __name__ == "__main__":
    exit(main())