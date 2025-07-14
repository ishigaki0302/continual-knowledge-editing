#!/usr/bin/env python3
"""
Sampling Candidates Generation Program

Generates all sampling combinations for conditions A, B, and C with specified
number of edits, sampling size, and order permutations, then saves to JSON.
"""

import json
import random
import argparse
import logging
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('generate_sampling_candidates.log'),
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
    """
    # First, sample the base combinations
    base_combinations = sample_random_combinations(data, num_edits, condition, sample_size, seed)
    
    # Then, generate order permutations for each combination
    ordered_combinations = generate_order_permutations(base_combinations, num_orders, seed)
    
    return ordered_combinations


def generate_all_conditions_sampling(data, num_edits=3, sample_size=100, num_orders=None, seed=42):
    """
    Generate sampling combinations for all conditions (A, B, C)
    """
    all_conditions_data = {}
    
    for condition in ['A', 'B', 'C']:
        print(f"Generating sampling combinations for condition {condition}...")
        
        try:
            combinations = sample_random_combinations_with_orders(
                data, num_edits, condition, sample_size, num_orders, seed
            )
            
            # Calculate base combinations for metadata
            base_combinations = sample_random_combinations(data, num_edits, condition, sample_size, seed)
            
            all_conditions_data[condition] = {
                'condition': condition,
                'num_edits': num_edits,
                'sample_size': sample_size,
                'num_orders': num_orders,
                'num_base_combinations': len(base_combinations),
                'total_combinations_with_orders': len(combinations),
                'combinations': combinations
            }
            
            print(f"Condition {condition}: {len(base_combinations)} base combinations, {len(combinations)} total with orders")
            
        except Exception as e:
            print(f"Error generating condition {condition}: {str(e)}")
            all_conditions_data[condition] = {
                'condition': condition,
                'error': str(e),
                'combinations': []
            }
    
    return all_conditions_data


def save_sampling_candidates(data, conditions_data, output_dir, num_edits, sample_size, num_orders, seed):
    """Save sampling candidates to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sampling_candidates_edits{num_edits}_samples{sample_size}_orders{num_orders if num_orders else 'all'}_seed{seed}_{timestamp}.json"
    output_path = output_dir / filename
    
    # Prepare complete data structure
    complete_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'num_edits': num_edits,
            'sample_size': sample_size,
            'num_orders': num_orders,
            'seed': seed,
            'dataset_info': {
                'subjects': data['subjects'],
                'shared_relations': list(data['SharedRelations'].keys())
            }
        },
        'conditions': conditions_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate Sampling Candidates for Knowledge Editing')
    parser.add_argument('--num-edits', type=int, default=3,
                       help='Number of knowledge edits to perform (default: 3)')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of base sampling combinations to generate per condition (default: 100)')
    parser.add_argument('--num-orders', type=int, default=None,
                       help='Number of order permutations per combination (default: all permutations)')
    parser.add_argument('--dataset', type=str, default='datasets/temp_ckndata.json',
                       help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='sampling_candidates',
                       help='Output directory for sampling candidates')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    random.seed(args.seed)
    
    logger.info(f"Starting sampling candidates generation")
    logger.info(f"Number of edits: {args.num_edits}")
    logger.info(f"Sample size per condition: {args.sample_size}")
    logger.info(f"Number of orders per combination: {args.num_orders if args.num_orders is not None else 'all'}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load dataset
    logger.info("Loading dataset...")
    data = load_dataset(args.dataset)
    
    # Generate sampling combinations for all conditions
    logger.info("Generating sampling combinations for all conditions (A, B, C)...")
    conditions_data = generate_all_conditions_sampling(
        data, args.num_edits, args.sample_size, args.num_orders, args.seed
    )
    
    # Save results
    output_path = save_sampling_candidates(
        data, conditions_data, args.output_dir, args.num_edits, args.sample_size, args.num_orders, args.seed
    )
    logger.info(f"Sampling candidates saved to: {output_path}")
    
    # Log summary
    logger.info("Summary of generated sampling combinations:")
    for condition, condition_data in conditions_data.items():
        if 'error' in condition_data:
            logger.error(f"Condition {condition}: Error - {condition_data['error']}")
        else:
            logger.info(f"Condition {condition}: {condition_data['num_base_combinations']} base combinations, {condition_data['total_combinations_with_orders']} total with orders")
    
    logger.info("Sampling candidates generation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())