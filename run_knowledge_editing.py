#!/usr/bin/env python3
"""
Knowledge Editing Execution Program

Extracts knowledge triples from dataset and performs knowledge editing
using EasyEdit with real GPU models.
"""

import json
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path
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
            logging.FileHandler('knowledge_editing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_dataset(dataset_path):
    """Load knowledge dataset from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_knowledge_triples_condition_a(data, num_edits=5):
    """
    Condition A: Different subjects - Each edit uses a different subject
    
    Args:
        data: Dataset containing subjects, SharedRelations, ExclusiveRelations
        num_edits: Number of edits to extract (max 5 for different subjects)
    
    Returns:
        List of knowledge triples with different subjects
    """
    triples = []
    subjects = data['subjects']
    
    if num_edits > len(subjects):
        raise ValueError(f"Cannot generate {num_edits} edits with different subjects. Only {len(subjects)} subjects available.")
    
    # Combine all relations
    all_relations = {}
    relation_types = {}
    for rel_type in ['SharedRelations', 'ExclusiveRelations']:
        if rel_type in data:
            for rel_name, rel_data in data[rel_type].items():
                if rel_name != 'TaskDescriptionPrompt':
                    all_relations[rel_name] = rel_data
                    relation_types[rel_name] = rel_type
    
    # Use different subjects for each edit
    selected_subjects = random.sample(subjects, num_edits)
    
    for i in range(num_edits):
        subject = selected_subjects[i]
        relation = random.choice(list(all_relations.keys()))
        relation_data = all_relations[relation]
        obj = random.choice(relation_data['objects'])
        
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
            'relation_type': relation_types[relation],
            'condition': 'A'
        })
    
    return triples


def extract_knowledge_triples_condition_b(data, num_edits=5):
    """
    Condition B: Same subject, different relations - All edits use the same subject with different relations
    
    Args:
        data: Dataset containing subjects, SharedRelations, ExclusiveRelations
        num_edits: Number of edits to extract
    
    Returns:
        List of knowledge triples with same subject, different relations
    """
    triples = []
    subjects = data['subjects']
    
    # Combine all relations
    all_relations = {}
    relation_types = {}
    for rel_type in ['SharedRelations', 'ExclusiveRelations']:
        if rel_type in data:
            for rel_name, rel_data in data[rel_type].items():
                if rel_name != 'TaskDescriptionPrompt':
                    all_relations[rel_name] = rel_data
                    relation_types[rel_name] = rel_type
    
    if num_edits > len(all_relations):
        raise ValueError(f"Cannot generate {num_edits} edits with different relations. Only {len(all_relations)} relations available.")
    
    # Select one subject and different relations
    subject = random.choice(subjects)
    selected_relations = random.sample(list(all_relations.keys()), num_edits)
    
    for i in range(num_edits):
        relation = selected_relations[i]
        relation_data = all_relations[relation]
        obj = random.choice(relation_data['objects'])
        
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
            'relation_type': relation_types[relation],
            'condition': 'B'
        })
    
    return triples


def extract_knowledge_triples_condition_c(data, num_edits=5):
    """
    Condition C: Same subject-relation, different objects - All edits use the same (subject, relation) with different objects
    
    Args:
        data: Dataset containing subjects, SharedRelations, ExclusiveRelations
        num_edits: Number of edits to extract (max 5 for different objects)
    
    Returns:
        List of knowledge triples with same subject-relation, different objects
    """
    triples = []
    subjects = data['subjects']
    
    # Combine all relations
    all_relations = {}
    relation_types = {}
    for rel_type in ['SharedRelations', 'ExclusiveRelations']:
        if rel_type in data:
            for rel_name, rel_data in data[rel_type].items():
                if rel_name != 'TaskDescriptionPrompt':
                    all_relations[rel_name] = rel_data
                    relation_types[rel_name] = rel_type
    
    # Select one subject and one relation
    subject = random.choice(subjects)
    relation = random.choice(list(all_relations.keys()))
    relation_data = all_relations[relation]
    
    if num_edits > len(relation_data['objects']):
        raise ValueError(f"Cannot generate {num_edits} edits with different objects. Only {len(relation_data['objects'])} objects available for relation '{relation}'.")
    
    # Select different objects for the same (subject, relation)
    selected_objects = random.sample(relation_data['objects'], num_edits)
    
    for i in range(num_edits):
        obj = selected_objects[i]
        
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
            'relation_type': relation_types[relation],
            'condition': 'C'
        })
    
    return triples


def extract_knowledge_triples(data, num_edits=5, condition='random'):
    """
    Extract knowledge triples from dataset using specified condition
    
    Args:
        data: Dataset containing subjects, SharedRelations, ExclusiveRelations
        num_edits: Number of edits to extract
        condition: Sampling strategy ('A', 'B', 'C', or 'random')
    
    Returns:
        List of knowledge triples based on the specified condition
    """
    if condition == 'A':
        return extract_knowledge_triples_condition_a(data, num_edits)
    elif condition == 'B':
        return extract_knowledge_triples_condition_b(data, num_edits)
    elif condition == 'C':
        return extract_knowledge_triples_condition_c(data, num_edits)
    else:
        # Original random sampling (backward compatibility)
        return extract_knowledge_triples_random(data, num_edits)


def extract_knowledge_triples_random(data, num_edits=5):
    """
    Original random sampling method - kept for backward compatibility
    
    Args:
        data: Dataset containing subjects, SharedRelations, ExclusiveRelations
        num_edits: Number of edits to extract
    
    Returns:
        List of knowledge triples (subject, relation, object) with all 5 candidates
    """
    triples = []
    subjects = data['subjects']
    
    # Combine all relations
    all_relations = {}
    relation_types = {}  # Track whether relation is shared or exclusive
    for rel_type in ['SharedRelations', 'ExclusiveRelations']:
        if rel_type in data:
            for rel_name, rel_data in data[rel_type].items():
                if rel_name != 'TaskDescriptionPrompt':
                    all_relations[rel_name] = rel_data
                    relation_types[rel_name] = rel_type
    
    # Generate random triples
    for _ in range(num_edits):
        subject = random.choice(subjects)
        relation = random.choice(list(all_relations.keys()))
        relation_data = all_relations[relation]
        obj = random.choice(relation_data['objects'])
        
        prompt = relation_data['prompt'].replace('[subject]', subject).replace(' [object].', "")
        question = relation_data['question'].replace('[subject]', subject)
        
        # Get all 5 candidates for this relation
        all_candidates = relation_data['objects']
        
        triples.append({
            'subject': subject,
            'relation': relation,
            'object': obj,
            'prompt': prompt,
            'question': question,
            'ground_truth': 'unknown',  # Original state is unknown
            'target_new': obj,
            'candidates': all_candidates,  # All 5 possible objects
            'relation_type': relation_types[relation],
            'condition': 'random'
        })
    
    return triples


def get_candidate_probabilities(model, tokenizer, question, candidates, device):
    """
    Get probabilities for 5 candidate objects using model logits and softmax
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        question: Question prompt (e.g., "Which skills does John have?")
        candidates: List of 5 candidate objects
        device: Device to run inference on
    
    Returns:
        Dictionary with candidate probabilities
    """
    # Tokenize question
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    # Get logits for the last token position (where answer would be generated)
    last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
    
    # Get token IDs for each candidate
    candidate_logits = []
    candidate_tokens = []
    
    for candidate in candidates:
        # Tokenize each candidate and get first token ID
        candidate_tokens_ids = tokenizer.encode(f" {candidate}", add_special_tokens=False)
        if candidate_tokens_ids:
            first_token_id = candidate_tokens_ids[0]
            candidate_logits.append(last_token_logits[first_token_id].item())
            candidate_tokens.append(first_token_id)
        else:
            candidate_logits.append(float('-inf'))  # Invalid candidate
            candidate_tokens.append(-1)
    
    # Convert to tensor and apply softmax (on CPU to avoid device issues)
    candidate_logits_tensor = torch.tensor(candidate_logits, dtype=torch.float32)
    probabilities = softmax(candidate_logits_tensor, dim=0)
    
    # Create result dictionary
    result = {
        'candidates': candidates,
        'logits': candidate_logits,
        'probabilities': probabilities.tolist(),
        'token_ids': candidate_tokens
    }
    print(result)

    return result


def calculate_efficacy(edit_results):
    """
    Calculate efficacy: Correct_Answers_After_Edit / Total_Edits
    
    Args:
        edit_results: List of edit results with probability measurements
    
    Returns:
        Dictionary with efficacy metrics
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
            if abs(target_probability - max_probability) < 1e-6:  # Account for floating point precision
                correct_edits += 1
    
    efficacy = correct_edits / total_edits if total_edits > 0 else 0.0
    
    return {
        'efficacy': efficacy,
        'correct_edits': correct_edits,
        'total_edits': total_edits,
        'accuracy_percentage': efficacy * 100
    }


def perform_knowledge_editing(method, model_name, triples, logger, condition='random', device="cuda:0"):
    """
    Perform knowledge editing using EasyEdit with efficacy measurement
    
    Args:
        method: Editing method (ROME, MEMIT, etc.)
        model_name: Model name (gpt-j-6b, gpt2-xl, etc.)
        triples: List of knowledge triples
        logger: Logger instance
    
    Returns:
        Dictionary containing results with efficacy measurements
    """
    logger.info(f"Initializing {method} editor with {model_name}")
    
    if not torch.cuda.is_available():
        logger.warning("GPU not available, but continuing with CPU (may be very slow)")
    
    # Initialize wrapper
    wrapper = EasyEditWrapper(method=method, model_name=model_name)
    
    results = {
        'method': method,
        'model_name': model_name,
        'condition': condition,
        'timestamp': datetime.now().isoformat(),
        'num_edits': len(triples),
        'edits': [],
        'efficacy_scores': [],
        'final_efficacy': None,
        'success': True,
        'error_message': None
    }
    
    # Initialize tokenizer for probability calculation
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
    logger.info(f"Loaded tokenizer for {tokenizer_name}")
    
    # Perform sequential editing
    for i, triple in enumerate(triples):
        logger.info(f"Performing edit {i+1}/{len(triples)}: {triple['subject']} - {triple['relation']} - {triple['object']}")
        
        # Apply edit
        metrics, edited_model = wrapper.edit_model(
            prompts=triple['prompt'],
            ground_truth=triple['ground_truth'],
            target_new=triple['target_new'],
            subject=triple['subject'],
            edit_id=f"edit_{i+1}"
        )
        
        # Update wrapper's model reference for sequential editing
        wrapper.editor.model = edited_model
        
        edit_result = {
            'edit_id': i + 1,
            'triple': triple,
            'metrics': metrics,
            'success': True
        }
        
        # Calculate post-edit probabilities for the 5 candidates
        logger.info(f"Calculating probabilities for 5 candidates after edit {i+1}")
        # questionは，直接編集した文章じゃないので，ここでの評価はprtmpt（編集時の文章）を用いる．ただし，汎化性を測るのに使えるので，今後これも使う．
        # post_edit_probs = get_candidate_probabilities(
        #     edited_model, tokenizer, triple['question'], 
        #     triple['candidates'], device
        # )
        post_edit_probs = get_candidate_probabilities(
            edited_model, tokenizer, triple['prompt'], 
            triple['candidates'], device
        )
        edit_result['post_edit_probabilities'] = post_edit_probs
        
        # Log probability results
        target_index = triple['candidates'].index(triple['object'])
        target_prob = post_edit_probs['probabilities'][target_index]
        logger.info(f"Target object '{triple['object']}' probability: {target_prob:.4f}")
    
        results['edits'].append(edit_result)
    
        # Calculate cumulative efficacy after each edit
        current_efficacy = calculate_efficacy(results['edits'])
        results['efficacy_scores'].append({
            'after_edit': i + 1,
            'efficacy': current_efficacy['efficacy'],
            'correct_edits': current_efficacy['correct_edits'],
            'total_edits': current_efficacy['total_edits']
        })
        
        logger.info(f"Edit {i+1} completed. Current efficacy: {current_efficacy['efficacy']:.4f} ({current_efficacy['correct_edits']}/{current_efficacy['total_edits']})")
        
    # Calculate final efficacy
    final_efficacy = calculate_efficacy(results['edits'])
    results['final_efficacy'] = final_efficacy
    logger.info(f"Final efficacy: {final_efficacy['efficacy']:.4f} ({final_efficacy['accuracy_percentage']:.2f}%)")
    
    # Evaluate final state probabilities for all edited sro triples
    logger.info("Evaluating final state probabilities for all edited triples")
    final_state_evaluations = []
    
    for i, triple in enumerate(triples):
        logger.info(f"Final evaluation {i+1}/{len(triples)}: {triple['subject']} - {triple['relation']} - {triple['object']}")
        
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
            'edit_id': i + 1,
            'triple': triple,
            'final_state_probabilities': final_state_probs,
            'target_probability': target_prob,
            'target_rank': sorted(final_state_probs['probabilities'], reverse=True).index(target_prob) + 1
        })
    
    results['final_state_evaluations'] = final_state_evaluations
    logger.info("Final state evaluation completed for all edited triples")
    
    return results


def save_results(results, output_dir):
    """Save results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition_suffix = f"_condition_{results.get('condition', 'random')}" if results.get('condition') != 'random' else ""
    filename = f"knowledge_editing_{results['method']}_{results['model_name']}{condition_suffix}_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Knowledge Editing Execution Program')
    parser.add_argument('--method', type=str, default='ROME', 
                       choices=['ROME', 'MEMIT', 'MEND', 'FT', 'IKE', 'KN'],
                       help='Knowledge editing method')
    parser.add_argument('--model', type=str, default='gpt-j-6b',
                       choices=['gpt-j-6b', 'gpt2-xl', 'llama-7b', 'llama3-8b', 'llama3.2-3b'],
                       help='Model name')
    parser.add_argument('--num-edits', type=int, default=5,
                       help='Number of knowledge edits to perform')
    parser.add_argument('--condition', type=str, default='random',
                       choices=['A', 'B', 'C', 'random'],
                       help='Sampling condition: A (different subjects), B (same subject, different relations), C (same subject-relation, different objects), random (original random sampling)')
    parser.add_argument('--dataset', type=str, default='datasets/temp_ckndata.json',
                       help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default="cuda:0",)
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    random.seed(args.seed)
    
    logger.info(f"Starting knowledge editing with {args.method} on {args.model}")
    logger.info(f"Number of edits: {args.num_edits}")
    logger.info(f"Sampling condition: {args.condition}")
    logger.info(f"Dataset: {args.dataset}")
    
    # Load dataset
    logger.info("Loading dataset...")
    data = load_dataset(args.dataset)
    
    # Extract knowledge triples
    logger.info(f"Extracting knowledge triples using condition {args.condition}...")
    triples = extract_knowledge_triples(data, args.num_edits, args.condition)
    
    # Log the generated triples for verification
    logger.info("Generated triples:")
    for i, triple in enumerate(triples):
        logger.info(f"  {i+1}. Subject: {triple['subject']}, Relation: {triple['relation']}, Object: {triple['object']}")
    
    # Perform knowledge editing
    logger.info("Starting knowledge editing...")
    results = perform_knowledge_editing(args.method, args.model, triples, logger, args.condition, args.device)
    
    # Save results
    output_path = save_results(results, args.output_dir)
    logger.info(f"Results saved to: {output_path}")
    
    if results['success']:
        logger.info("Knowledge editing completed successfully!")
        logger.info(f"Processed {len(results['edits'])} edits")
    else:
        logger.error(f"Knowledge editing failed: {results['error_message']}")
    
    return 0


if __name__ == "__main__":
    exit(main())