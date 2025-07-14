#!/usr/bin/env python3
"""
Knowledge Editing Order Sampling Program

Performs knowledge editing with sampling combinations for SharedRelations
across conditions A, B, and C, with the ability to specify both the number
of sampling combinations and the number of order permutations to test.
"""

import json
import random
import argparse
import logging
import itertools
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
            logging.FileHandler('knowledge_editing_order_sampling.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_dataset(dataset_path):
    """Load knowledge dataset from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_all_combinations_condition_a(data, num_edits=3):
    """
    Condition A: Different subjects - Each edit uses a different subject
    Generate all possible combinations of (s,r,o) where each s is different
    """
    combinations = []
    subjects = data['subjects']
    shared_relations = data['SharedRelations']
    
    if num_edits > len(subjects):
        raise ValueError(f"Cannot generate {num_edits} edits with different subjects. Only {len(subjects)} subjects available.")
    
    # Get all SharedRelations (excluding TaskDescriptionPrompt)
    relations = {k: v for k, v in shared_relations.items() if k != 'TaskDescriptionPrompt'}
    
    # Generate all possible subject combinations
    for subject_combo in itertools.combinations(subjects, num_edits):
        # For each subject combination, generate all possible (r,o) combinations
        relation_object_combinations = []
        for relation_name, relation_data in relations.items():
            for obj in relation_data['objects']:
                relation_object_combinations.append((relation_name, obj))
        
        # Generate all possible ways to assign (r,o) to each subject
        for ro_assignment in itertools.product(relation_object_combinations, repeat=num_edits):
            triples = []
            for i, (subject, (relation, obj)) in enumerate(zip(subject_combo, ro_assignment)):
                relation_data = relations[relation]
                prompt = relation_data['prompt'].replace('[subject]', subject).replace(' [object].', "")
                question = relation_data['question'].replace('[subject]', subject)
                all_candidates = relation_data['objects']
                
                triples.append({
                    'subject': subject,
                    'relation': relation,
                    'object': obj,
                    'prompt': prompt,
                    'question': question,
                    'ground_truth': 'unknown',
                    'target_new': obj,
                    'candidates': all_candidates,
                    'relation_type': 'SharedRelations',
                    'condition': 'A'
                })
            
            combinations.append(triples)
    
    return combinations


def generate_all_combinations_condition_b(data, num_edits=3):
    """
    Condition B: Same subject, different relations - All edits use the same subject with different relations
    Generate all possible combinations of (s,r,o) where s is fixed and each r is different
    """
    combinations = []
    subjects = data['subjects']
    shared_relations = data['SharedRelations']
    
    # Get all SharedRelations (excluding TaskDescriptionPrompt)
    relations = {k: v for k, v in shared_relations.items() if k != 'TaskDescriptionPrompt'}
    
    if num_edits > len(relations):
        raise ValueError(f"Cannot generate {num_edits} edits with different relations. Only {len(relations)} relations available.")
    
    # For each subject
    for subject in subjects:
        # Generate all possible relation combinations
        for relation_combo in itertools.combinations(relations.keys(), num_edits):
            # For each relation combination, generate all possible object assignments
            object_combinations = []
            for relation in relation_combo:
                object_combinations.append(relations[relation]['objects'])
            
            # Generate all possible ways to assign objects to relations
            for obj_assignment in itertools.product(*object_combinations):
                triples = []
                for relation, obj in zip(relation_combo, obj_assignment):
                    relation_data = relations[relation]
                    prompt = relation_data['prompt'].replace('[subject]', subject).replace(' [object].', "")
                    question = relation_data['question'].replace('[subject]', subject)
                    all_candidates = relation_data['objects']
                    
                    triples.append({
                        'subject': subject,
                        'relation': relation,
                        'object': obj,
                        'prompt': prompt,
                        'question': question,
                        'ground_truth': 'unknown',
                        'target_new': obj,
                        'candidates': all_candidates,
                        'relation_type': 'SharedRelations',
                        'condition': 'B'
                    })
                
                combinations.append(triples)
    
    return combinations


def generate_all_combinations_condition_c(data, num_edits=3):
    """
    Condition C: Same subject-relation, different objects - All edits use the same (subject, relation) with different objects
    Generate all possible combinations of (s,r,o) where (s,r) is fixed and each o is different
    """
    combinations = []
    subjects = data['subjects']
    shared_relations = data['SharedRelations']
    
    # Get all SharedRelations (excluding TaskDescriptionPrompt)
    relations = {k: v for k, v in shared_relations.items() if k != 'TaskDescriptionPrompt'}
    
    # For each subject and relation combination
    for subject in subjects:
        for relation_name, relation_data in relations.items():
            objects = relation_data['objects']
            
            if num_edits > len(objects):
                continue  # Skip if not enough objects for this relation
            
            # Generate all possible object combinations
            for obj_combo in itertools.combinations(objects, num_edits):
                triples = []
                for obj in obj_combo:
                    prompt = relation_data['prompt'].replace('[subject]', subject).replace(' [object].', "")
                    question = relation_data['question'].replace('[subject]', subject)
                    all_candidates = relation_data['objects']
                    
                    triples.append({
                        'subject': subject,
                        'relation': relation_name,
                        'object': obj,
                        'prompt': prompt,
                        'question': question,
                        'ground_truth': 'unknown',
                        'target_new': obj,
                        'candidates': all_candidates,
                        'relation_type': 'SharedRelations',
                        'condition': 'C'
                    })
                
                combinations.append(triples)
    
    return combinations


def generate_all_sampling_combinations(data, num_edits=3, condition='A'):
    """
    Generate all possible sampling combinations for the specified condition
    """
    if condition == 'A':
        return generate_all_combinations_condition_a(data, num_edits)
    elif condition == 'B':
        return generate_all_combinations_condition_b(data, num_edits)
    elif condition == 'C':
        return generate_all_combinations_condition_c(data, num_edits)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def sample_random_combinations(data, num_edits=3, condition='A', sample_size=1000, seed=42):
    """
    全組み合わせから sample_size 件をランダムに重複なくサンプリングする関数
    
    Args:
        data: JSONデータ（subjects, SharedRelations含む）
        num_edits: 編集ステップ数
        condition: A, B, or C
        sample_size: ランダムに取り出す組み合わせ数
        seed: ランダムシード
    
    Returns:
        List of sampled combinations
    """
    random.seed(seed)

    # 条件に基づいて全組み合わせを生成
    all_combinations = generate_all_sampling_combinations(data, num_edits, condition)

    if sample_size >= len(all_combinations):
        print(f"[INFO] 要求された件数（{sample_size}）が全組み合わせ数（{len(all_combinations)}）以上のため、全件を返します。")
        return all_combinations

    # 重複なしでランダムサンプリング
    sampled_combinations = random.sample(all_combinations, sample_size)
    return sampled_combinations


def generate_order_permutations(combinations, num_orders=None, seed=42):
    """
    Generate order permutations for the given combinations
    
    Args:
        combinations: List of combinations (each containing ordered triples)
        num_orders: Number of order permutations to generate per combination
                   If None, generates all possible permutations
        seed: Random seed for reproducibility
    
    Returns:
        List of combinations with different orders
    """
    random.seed(seed)
    ordered_combinations = []
    
    for combo_idx, combination in enumerate(combinations):
        # Generate all possible permutations for this combination
        all_permutations = list(itertools.permutations(combination))
        
        if num_orders is None or num_orders >= len(all_permutations):
            # Use all permutations if num_orders is None or larger than available
            selected_permutations = all_permutations
        else:
            # Randomly sample num_orders permutations
            selected_permutations = random.sample(all_permutations, num_orders)
        
        # Convert back to list format and add to results
        for perm_idx, permutation in enumerate(selected_permutations):
            ordered_combo = list(permutation)
            # Add metadata about the original combination and permutation
            for triple in ordered_combo:
                triple['original_combination_index'] = combo_idx
                triple['permutation_index'] = perm_idx
            ordered_combinations.append(ordered_combo)
    
    return ordered_combinations


def sample_random_combinations_with_orders(data, num_edits=3, condition='A', sample_size=1000, num_orders=None, seed=42):
    """
    Generate sampling combinations with specified number of order permutations
    
    Args:
        data: JSONデータ（subjects, SharedRelations含む）
        num_edits: 編集ステップ数
        condition: A, B, or C
        sample_size: ランダムに取り出す組み合わせ数
        num_orders: 各組み合わせに対する順序数（None の場合は全順序）
        seed: ランダムシード
    
    Returns:
        List of combinations with different orders
    """
    # First, sample the base combinations
    base_combinations = sample_random_combinations(data, num_edits, condition, sample_size, seed)
    
    # Then, generate order permutations for each combination
    ordered_combinations = generate_order_permutations(base_combinations, num_orders, seed)
    
    return ordered_combinations


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
    
    Args:
        method: Editing method (ROME, MEMIT, etc.)
        model_name: Model name
        triples: List of knowledge triples
        logger: Logger instance
        condition: Experimental condition
        sample_idx: Sample index for logging
        device: Device to use
    
    Returns:
        Dictionary containing experiment results
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
        'permutation_index': triples[0].get('permutation_index', None)
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


def save_results(results, output_dir):
    """Save results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition_suffix = f"_condition_{results.get('condition', 'unknown')}"
    filename = f"knowledge_editing_order_sampling_{results['method']}_{results['model_name']}{condition_suffix}_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Knowledge Editing Order Sampling Program')
    parser.add_argument('--method', type=str, default='ROME', 
                       choices=['ROME', 'MEMIT', 'MEND', 'FT', 'IKE', 'KN'],
                       help='Knowledge editing method')
    parser.add_argument('--model', type=str, default='gpt-j-6b',
                       choices=['gpt-j-6b', 'gpt2-xl', 'llama-7b', 'llama3-8b', 'llama3.2-3b'],
                       help='Model name')
    parser.add_argument('--num-edits', type=int, default=3,
                       help='Number of knowledge edits to perform (default: 3)')
    parser.add_argument('--condition', type=str, default='A',
                       choices=['A', 'B', 'C'],
                       help='Sampling condition (A, B, or C)')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of base sampling combinations to use (default: 100)')
    parser.add_argument('--num-orders', type=int, default=None,
                       help='Number of order permutations per combination (default: all permutations)')
    parser.add_argument('--dataset', type=str, default='datasets/temp_ckndata.json',
                       help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default="cuda:0",
                       help='Device to use for computation')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    random.seed(args.seed)
    
    logger.info(f"Starting order sampling-based knowledge editing with {args.method} on {args.model}")
    logger.info(f"Number of edits: {args.num_edits}")
    logger.info(f"Sampling condition: {args.condition}")
    logger.info(f"Base sample size: {args.sample_size}")
    logger.info(f"Number of orders per combination: {args.num_orders if args.num_orders is not None else 'all'}")
    logger.info(f"Dataset: {args.dataset}")
    
    # Load dataset
    logger.info("Loading dataset...")
    data = load_dataset(args.dataset)
    
    # Generate sampling combinations with order permutations
    logger.info(f"Generating sampling combinations with order permutations for condition {args.condition}...")
    all_combinations = sample_random_combinations_with_orders(
        data, args.num_edits, args.condition, args.sample_size, args.num_orders, args.seed
    )
    
    # Log the number of combinations
    logger.info(f"Generated {len(all_combinations)} total sampling combinations (with order permutations)")
    
    # Calculate base combinations and permutations info
    base_combinations = sample_random_combinations(data, args.num_edits, args.condition, args.sample_size, args.seed)
    num_base_combinations = len(base_combinations)
    expected_permutations_per_combo = args.num_orders if args.num_orders is not None else max(1, len(base_combinations[0]) if base_combinations else 1)
    
    logger.info(f"Base combinations: {num_base_combinations}")
    logger.info(f"Expected permutations per combination: {expected_permutations_per_combo}")
    
    # Show a few examples
    if all_combinations:
        logger.info("Example combinations:")
        for i, combo in enumerate(all_combinations[:3]):  # Show first 3 combinations
            logger.info(f"  Combination {i+1} (orig: {combo[0].get('original_combination_index', 'N/A')}, perm: {combo[0].get('permutation_index', 'N/A')}):")
            for j, triple in enumerate(combo):
                logger.info(f"    {j+1}. Subject: {triple['subject']}, Relation: {triple['relation']}, Object: {triple['object']}")
    
    # Perform sampling experiments
    logger.info("Starting order sampling-based knowledge editing experiments...")
    results = perform_sampling_experiments(
        args.method, args.model, all_combinations, logger, args.condition, args.device
    )
    
    # Add order sampling specific metadata
    results['num_base_combinations'] = num_base_combinations
    results['num_orders_per_combination'] = args.num_orders
    results['total_combinations_with_orders'] = len(all_combinations)
    
    # Save results
    output_path = save_results(results, args.output_dir)
    logger.info(f"Results saved to: {output_path}")
    
    if results['success']:
        logger.info("Order sampling-based knowledge editing completed successfully!")
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
        logger.error("Order sampling-based knowledge editing failed!")
    
    return 0


if __name__ == "__main__":
    exit(main())