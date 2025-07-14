#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_knowledge_editing_order.py

Knowledge Editing Order Control の結果 JSON から、
順序制御実験の統計的な結果を可視化するスクリプト。
各編集ステップの確率を平均値とエラーバー（分散）で表示し、
使用された知識トリプル(s,r,o)を図の下部に表示。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {path}: {e}")
        raise


def extract_probability_statistics(data: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], int]:
    """
    Extract probability statistics from order experiment results
    
    Args:
        data: JSON data from order experiment
    
    Returns:
        Tuple of (mean_probabilities, std_probabilities, candidate_labels, num_candidates)
    """
    individual_results = data.get('individual_results', [])
    if not individual_results:
        raise ValueError("No individual results found in data")
    
    # Get basic info from first successful result
    first_successful = None
    for result in individual_results:
        if result.get('success', False) and result.get('edits'):
            first_successful = result
            break
    
    if not first_successful:
        raise ValueError("No successful results found")
    
    num_edits = len(first_successful['edits'])
    candidates = first_successful['edits'][0]['triple']['candidates']
    num_candidates = len(candidates)
    
    # Collect probabilities for each edit step across all experiments
    all_probabilities = []  # [edit_step][experiment][candidate]
    
    for edit_step in range(num_edits):
        step_probabilities = []
        for result in individual_results:
            if result.get('success', False) and len(result.get('edits', [])) > edit_step:
                edit = result['edits'][edit_step]
                if 'post_edit_probabilities' in edit:
                    probs = edit['post_edit_probabilities']['probabilities']
                    step_probabilities.append(probs)
        all_probabilities.append(step_probabilities)
    
    # Calculate statistics
    mean_probabilities = []
    std_probabilities = []
    
    for step_probs in all_probabilities:
        if step_probs:
            step_array = np.array(step_probs)  # [experiment, candidate]
            mean_probs = np.mean(step_array, axis=0)  # [candidate]
            std_probs = np.std(step_array, axis=0)    # [candidate]
            mean_probabilities.append(mean_probs)
            std_probabilities.append(std_probs)
        else:
            # Fallback if no data for this step
            mean_probabilities.append(np.zeros(num_candidates))
            std_probabilities.append(np.zeros(num_candidates))
    
    return mean_probabilities, std_probabilities, candidates, num_candidates


def extract_post_edit_and_final_statistics(data: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], int]:
    """
    Extract both post-edit and final-state probability statistics from order experiment results
    
    Args:
        data: JSON data from order experiment
    
    Returns:
        Tuple of (post_edit_means, post_edit_stds, final_state_means, final_state_stds, candidate_labels, num_candidates)
    """
    individual_results = data.get('individual_results', [])
    if not individual_results:
        raise ValueError("No individual results found in data")
    
    # Get basic info from first successful result
    first_successful = None
    for result in individual_results:
        if result.get('success', False) and result.get('edits'):
            first_successful = result
            break
    
    if not first_successful:
        raise ValueError("No successful results found")
    
    num_edits = len(first_successful['edits'])
    candidates = first_successful['edits'][0]['triple']['candidates']
    num_candidates = len(candidates)
    
    # Collect post-edit probabilities for each edit step across all experiments
    all_post_edit_probabilities = []  # [edit_step][experiment][candidate]
    all_final_state_probabilities = []  # [edit_step][experiment][candidate]
    
    for edit_step in range(num_edits):
        step_post_edit_probs = []
        step_final_state_probs = []
        
        for result in individual_results:
            if result.get('success', False) and len(result.get('edits', [])) > edit_step:
                edit = result['edits'][edit_step]
                
                # Post-edit probabilities
                if 'post_edit_probabilities' in edit:
                    post_probs = edit['post_edit_probabilities']['probabilities']
                    step_post_edit_probs.append(post_probs)
                
                # Final-state probabilities (check various possible locations)
                final_probs = None
                if 'final_state_probabilities' in edit:
                    final_probs = edit['final_state_probabilities']['probabilities']
                elif 'final_state_evaluations' in result and edit_step < len(result['final_state_evaluations']):
                    final_eval = result['final_state_evaluations'][edit_step]
                    if 'final_state_probabilities' in final_eval:
                        final_probs = final_eval['final_state_probabilities']['probabilities']
                
                if final_probs:
                    step_final_state_probs.append(final_probs)
                else:
                    # Fallback: use post-edit probabilities if final-state not available
                    if 'post_edit_probabilities' in edit:
                        step_final_state_probs.append(edit['post_edit_probabilities']['probabilities'])
        
        all_post_edit_probabilities.append(step_post_edit_probs)
        all_final_state_probabilities.append(step_final_state_probs)
    
    # Calculate statistics for post-edit probabilities
    post_edit_means = []
    post_edit_stds = []
    
    for step_probs in all_post_edit_probabilities:
        if step_probs:
            step_array = np.array(step_probs)  # [experiment, candidate]
            mean_probs = np.mean(step_array, axis=0)  # [candidate]
            std_probs = np.std(step_array, axis=0)    # [candidate]
            post_edit_means.append(mean_probs)
            post_edit_stds.append(std_probs)
        else:
            post_edit_means.append(np.zeros(num_candidates))
            post_edit_stds.append(np.zeros(num_candidates))
    
    # Calculate statistics for final-state probabilities
    final_state_means = []
    final_state_stds = []
    
    for step_probs in all_final_state_probabilities:
        if step_probs:
            step_array = np.array(step_probs)  # [experiment, candidate]
            mean_probs = np.mean(step_array, axis=0)  # [candidate]
            std_probs = np.std(step_array, axis=0)    # [candidate]
            final_state_means.append(mean_probs)
            final_state_stds.append(std_probs)
        else:
            final_state_means.append(np.zeros(num_candidates))
            final_state_stds.append(np.zeros(num_candidates))
    
    return post_edit_means, post_edit_stds, final_state_means, final_state_stds, candidates, num_candidates


def extract_permutation_statistics(data: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float], int, int]:
    """
    Extract permutation-level statistics for each step
    
    Args:
        data: JSON data from order experiment
    
    Returns:
        Tuple of (post_edit_means, post_edit_stds, final_state_means, final_state_stds, num_permutations, num_edits)
    """
    individual_results = data.get('individual_results', [])
    if not individual_results:
        raise ValueError("No individual results found in data")
    
    # Get basic info from first successful result
    first_successful = None
    for result in individual_results:
        if result.get('success', False) and result.get('edits'):
            first_successful = result
            break
    
    if not first_successful:
        raise ValueError("No successful results found")
    
    num_edits = len(first_successful['edits'])
    num_permutations = len(individual_results)
    
    # For each step, collect statistics across permutations
    post_edit_step_means = []
    post_edit_step_stds = []
    final_state_step_means = []
    final_state_step_stds = []
    
    for step in range(num_edits):
        # Collect efficacy or target probability for this step across all permutations
        step_post_edit_values = []
        step_final_state_values = []
        
        for result in individual_results:
            if result.get('success', False) and len(result.get('edits', [])) > step:
                edit = result['edits'][step]
                
                # Extract target object probability for post-edit
                if 'post_edit_probabilities' in edit:
                    post_probs = edit['post_edit_probabilities']['probabilities']
                    triple = edit['triple']
                    candidates = triple['candidates']
                    target_object = triple['object']
                    
                    if target_object in candidates:
                        target_idx = candidates.index(target_object)
                        target_prob_post = post_probs[target_idx] if len(post_probs) > target_idx else 0.0
                        step_post_edit_values.append(target_prob_post)
                
                # Extract target object probability for final-state
                final_prob = None
                triple = edit['triple']
                candidates = triple['candidates']
                target_object = triple['object']
                target_idx = candidates.index(target_object) if target_object in candidates else 0
                
                if 'final_state_probabilities' in edit:
                    final_probs = edit['final_state_probabilities']['probabilities']
                    final_prob = final_probs[target_idx] if len(final_probs) > target_idx else 0.0
                elif 'final_state_evaluations' in result and step < len(result['final_state_evaluations']):
                    final_eval = result['final_state_evaluations'][step]
                    if 'final_state_probabilities' in final_eval:
                        final_probs = final_eval['final_state_probabilities']['probabilities']
                        final_prob = final_probs[target_idx] if len(final_probs) > target_idx else 0.0
                
                if final_prob is not None:
                    step_final_state_values.append(final_prob)
                elif len(step_post_edit_values) > 0:
                    step_final_state_values.append(step_post_edit_values[-1])
        
        # Calculate mean and std for this step
        if step_post_edit_values:
            post_edit_step_means.append(np.mean(step_post_edit_values))
            post_edit_step_stds.append(np.std(step_post_edit_values))
        else:
            post_edit_step_means.append(0.0)
            post_edit_step_stds.append(0.0)
            
        if step_final_state_values:
            final_state_step_means.append(np.mean(step_final_state_values))
            final_state_step_stds.append(np.std(step_final_state_values))
        else:
            final_state_step_means.append(0.0)
            final_state_step_stds.append(0.0)
    
    return post_edit_step_means, post_edit_step_stds, final_state_step_means, final_state_step_stds, num_permutations, num_edits


def get_knowledge_triples(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract knowledge triples used in the experiment
    
    Args:
        data: JSON data from order experiment
    
    Returns:
        List of dictionaries with subject, relation, object
    """
    triples = data.get('triples', [])
    return [{
        'subject': t.get('subject', ''),
        'relation': t.get('relation', ''),
        'object': t.get('object', ''),
        'condition': t.get('condition', 'unknown'),
        'relation_type': t.get('relation_type', 'unknown')
    } for t in triples]


def plot_order_results(data: Dict[str, Any], output_path: Optional[str] = None):
    """
    Plot order-controlled experiment results with target object probabilities across steps
    
    Args:
        data: JSON data from order experiment
        output_path: Path to save figure (optional)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting to plot order results")
    
    # Validate data structure
    if 'individual_results' not in data:
        raise ValueError("Data does not contain 'individual_results' field")
    
    # Extract target object statistics
    try:
        target_means, target_stds, _, _, num_edits = extract_target_object_statistics(data)
        triples = get_knowledge_triples(data)
    except Exception as e:
        logger.error(f"Error extracting statistics: {e}")
        raise
    
    # Get experiment info
    method = data.get('method', 'Unknown')
    model_name = data.get('model_name', 'Unknown')
    condition = data.get('condition', 'Unknown')
    order_strategy = data.get('order_strategy', 'Unknown')
    num_experiments = data.get('num_order_sequences', 0)
    
    # Statistics summary
    stats = data.get('statistics', {})
    mean_efficacy = stats.get('mean_efficacy', 0)
    std_efficacy = stats.get('std_efficacy', 0)
    
    # Create figure with individual subplots for each step
    # 2 rows: top row for histograms, bottom row for triple information
    fig, axes = plt.subplots(nrows=2, ncols=num_edits, 
                            figsize=(4 * num_edits, 10),
                            gridspec_kw={'height_ratios': [3, 1]})
    
    # Ensure axes is 2D array even for single edit
    if num_edits == 1:
        axes = axes.reshape(2, 1)
    
    # Target object statistics are already extracted
    # target_means and target_stds are already the correct values
    
    # Create single plot showing target object statistics across steps
    ax = axes[0, :]
    if num_edits == 1:
        ax = [axes[0, 0]]
    
    # Plot target object statistics for all steps
    if num_edits > 1:
        ax_main = axes[0, :]
        fig.delaxes(axes[0, 0])
        for i in range(1, num_edits):
            fig.delaxes(axes[0, i])
        ax_main = fig.add_subplot(2, 1, 1)
    else:
        ax_main = axes[0, 0]
    
    # Plot bars with error bars for target object across steps
    steps = range(1, num_edits + 1)
    bars = ax_main.bar(steps, target_means, 
                      yerr=target_stds,
                      color='orange',
                      edgecolor='red',
                      linewidth=2,
                      alpha=0.7,
                      capsize=5,
                      error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Customize main plot
    ax_main.set_title(f'Target Object (Object 1) Probability Across Steps', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Edit Steps', fontsize=12)
    ax_main.set_ylabel('Target Object Probability (Mean ± Std)', fontsize=12)
    ax_main.set_xticks(steps)
    ax_main.set_xticklabels([f'Step {i}' for i in steps])
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, max([m + s for m, s in zip(target_means, target_stds)]) * 1.1 if target_means else 1.0)
    
    # Add statistical information
    stats_text = f"6 Permutation orders\nMean±Std across {num_experiments} experiments\nTarget object statistics only"
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
               fontsize=10, ha='left', va='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))
    
    # Add experiment details in the bottom row
    for edit_idx in range(num_edits):
        ax_info = axes[1, edit_idx]
        ax_info.axis('off')
        
        if edit_idx == 0:
            # Add general experiment info in the first bottom subplot
            details_text = f"Method: {method}\nModel: {model_name}\n"
            details_text += f"Condition: {condition}\nOrder: {order_strategy}\n"
            details_text += f"Experiments: {num_experiments}\n"
            details_text += f"Mean Efficacy: {mean_efficacy:.3f} ± {std_efficacy:.3f}"
            
            ax_info.text(0.5, 0.5, details_text, transform=ax_info.transAxes,
                        fontsize=9, ha='center', va='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        elif edit_idx == num_edits - 1 and stats:
            # Add statistics in the last bottom subplot
            stats_text = f"Efficacy Range:\n[{stats.get('min_efficacy', 0):.3f}, {stats.get('max_efficacy', 0):.3f}]\n"
            stats_text += f"Variance: {stats.get('variance_efficacy', 0):.6f}"
            
            ax_info.text(0.5, 0.5, stats_text, transform=ax_info.transAxes,
                        fontsize=9, ha='center', va='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
    
    # Overall title
    fig.suptitle(f'Order-Controlled Knowledge Editing Results\n'
                f'{method} on {model_name} | Condition {condition} | '
                f'{order_strategy.capitalize()} Order ({num_experiments} experiments)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Figure saved to: {output_path}")
    else:
        plt.show()


def plot_efficacy_progression(data: Dict[str, Any], output_path: Optional[str] = None):
    """
    Plot efficacy progression across edit steps with error bars
    
    Args:
        data: JSON data from order experiment
        output_path: Path to save figure (optional)
    """
    logger = logging.getLogger(__name__)
    individual_results = data.get('individual_results', [])
    if not individual_results:
        logger.warning("No individual results found for efficacy progression plot")
        return
    
    # Collect efficacy scores across experiments
    all_efficacy_scores = []  # [experiment][edit_step]
    
    for result in individual_results:
        if result.get('success', False) and 'efficacy_scores' in result:
            efficacy_sequence = [score['efficacy'] for score in result['efficacy_scores']]
            all_efficacy_scores.append(efficacy_sequence)
    
    if not all_efficacy_scores:
        logger.warning("No efficacy scores found")
        return
    
    # Calculate statistics
    efficacy_array = np.array(all_efficacy_scores)  # [experiment, edit_step]
    mean_efficacy = np.mean(efficacy_array, axis=0)
    std_efficacy = np.std(efficacy_array, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    edit_steps = np.arange(1, len(mean_efficacy) + 1)
    ax.errorbar(edit_steps, mean_efficacy, yerr=std_efficacy, 
               marker='o', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel('Edit Step', fontsize=12)
    ax.set_ylabel('Efficacy (Mean ± Std)', fontsize=12)
    ax.set_title(f'Efficacy Progression Across Edit Steps\n'
                f'{data.get("method", "Unknown")} on {data.get("model_name", "Unknown")} | '
                f'Condition {data.get("condition", "Unknown")}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_path:
        efficacy_path = output_path.replace('.png', '_efficacy.png')
        plt.savefig(efficacy_path, dpi=200, bbox_inches='tight')
        logger.info(f"Efficacy progression figure saved to: {efficacy_path}")
    else:
        plt.show()


def plot_multi_condition_results(data: Dict[str, Dict[str, Any]], output_path: Optional[str] = None):
    """
    Plot order-controlled experiment results for multiple conditions (A, B, C)
    Layout: Upper row = Post-edit probabilities, Lower row = Final-state probabilities
    Compatible with the structure from run_knowledge_editing_order.py
    
    Args:
        data: Dictionary with keys 'A', 'B', 'C' mapping to condition data
        output_path: Path to save figure (optional)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting multi-condition plotting")
    
    # Validate that we have at least one condition
    if not data:
        raise ValueError("No condition data provided")
    
    # Get the first available condition to extract basic structure
    first_condition = list(data.keys())[0]
    first_data = data[first_condition]
    
    # Validate data structure
    if 'individual_results' not in first_data:
        raise ValueError(f"Condition {first_condition} data does not contain 'individual_results' field")
    
    try:
        post_edit_means_first, _, final_state_means_first, _, candidates, num_candidates = extract_post_edit_and_final_statistics(first_data)
        num_edits = len(post_edit_means_first)
    except Exception as e:
        logger.error(f"Error extracting statistics from condition {first_condition}: {e}")
        raise
    
    # Extract statistics for all conditions using target object specific extraction
    conditions_stats = {}
    for cond in ['A', 'B', 'C']:
        post_edit_means, post_edit_stds, final_state_means, final_state_stds, num_edits = extract_target_object_statistics(data[cond])
        triples = get_knowledge_triples(data[cond])
        conditions_stats[cond] = {
            'post_edit_means': post_edit_means,
            'post_edit_stds': post_edit_stds,
            'final_state_means': final_state_means,
            'final_state_stds': final_state_stds,
            'triples': triples,
            'method': data[cond].get('method', 'Unknown'),
            'model_name': data[cond].get('model_name', 'Unknown'),
            'order_strategy': data[cond].get('order_strategy', 'Unknown'),
            'num_experiments': data[cond].get('num_order_sequences', 0),
            'stats': data[cond].get('statistics', {})
        }
    
    # Create figure: 3 rows × 2 columns (post-edit and final-state side by side)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
    
    # Plot each condition (1 row per condition, 2 columns: Post-edit + Final-state)
    for cond_idx, cond in enumerate(['A', 'B', 'C']):
        cond_data = conditions_stats[cond]
        post_edit_means = cond_data['post_edit_means']
        post_edit_stds = cond_data['post_edit_stds']
        final_state_means = cond_data['final_state_means']
        final_state_stds = cond_data['final_state_stds']
        
        # Target object statistics are already extracted by extract_target_object_statistics
        target_post_means = post_edit_means
        target_post_stds = post_edit_stds
        target_final_means = final_state_means
        target_final_stds = final_state_stds
        
        steps = range(1, num_edits + 1)
        
        # Left column: Post-edit probabilities
        ax_post = axes[cond_idx, 0]
        bars_post = ax_post.bar(steps, target_post_means, 
                               yerr=target_post_stds,
                               color='skyblue', edgecolor='red', linewidth=2,
                               alpha=0.7, capsize=5,
                               error_kw={'elinewidth': 2, 'capthick': 2})
        
        # Customize post-edit plot
        ax_post.set_ylabel(f"Condition {cond}\nTarget Object Probability", fontsize=12)
        ax_post.set_xlabel("Edit Steps", fontsize=10)
        ax_post.set_xticks(steps)
        ax_post.set_xticklabels([f'Step {i}' for i in steps])
        ax_post.grid(True, alpha=0.3)
        ax_post.set_title(f"Post-edit", fontsize=12, fontweight='bold')
        
        # Add statistical information for post-edit
        stats_text = f"6 Permutation orders\nMean±Std across {cond_data['num_experiments']} experiments"
        ax_post.text(0.02, 0.98, stats_text, transform=ax_post.transAxes,
                   fontsize=8, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))
        
        # Right column: Final-state probabilities
        ax_final = axes[cond_idx, 1]
        bars_final = ax_final.bar(steps, target_final_means,
                                 yerr=target_final_stds,
                                 color='lightcoral', edgecolor='red', linewidth=2,
                                 alpha=0.7, capsize=5,
                                 error_kw={'elinewidth': 2, 'capthick': 2})
        
        # Customize final-state plot
        ax_final.set_ylabel(f"Target Object Probability", fontsize=12)
        ax_final.set_xlabel("Edit Steps", fontsize=10)
        ax_final.set_xticks(steps)
        ax_final.set_xticklabels([f'Step {i}' for i in steps])
        ax_final.grid(True, alpha=0.3)
        ax_final.set_title(f"Final-state", fontsize=12, fontweight='bold')
        
        # Add experiment details in final-state column for last condition
        if cond_idx == 2:  # Condition C (last condition)
            details_text = f"Method: {cond_data['method']}\n"
            details_text += f"Model: {cond_data['model_name']}\n"
            details_text += f"Order: {cond_data['order_strategy']}\n"
            details_text += f"Experiments: {cond_data['num_experiments']}"
            
            stats = cond_data['stats']
            if stats:
                details_text += f"\nMean Efficacy: {stats.get('mean_efficacy', 0):.3f}"
            
            ax_final.text(0.98, 0.02, details_text, transform=ax_final.transAxes,
                         fontsize=8, ha='right', va='bottom', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Overall title
    method = conditions_stats['A']['method']
    model_name = conditions_stats['A']['model_name']
    order_strategy = conditions_stats['A']['order_strategy']
    
    fig.suptitle(f'Order-Controlled Knowledge Editing: Post-edit vs Final-state Probabilities\n'
                f'{method} on {model_name} | {order_strategy.capitalize()} Order', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"Multi-condition figure saved to: {output_path}")
    else:
        plt.show()


def calculate_efficacy_from_results(edit_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate efficacy: Correct_Answers_After_Edit / Total_Edits
    Based on the implementation in run_knowledge_editing_order.py
    
    Args:
        edit_results: List of edit results from individual experiments
    
    Returns:
        Dictionary containing efficacy metrics
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
            if target_object in candidates:
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


def validate_order_experiment_data(data: Dict[str, Any], file_path: str) -> bool:
    """
    Validate that the data is from an order experiment
    
    Args:
        data: JSON data to validate
        file_path: Path to the file (for error messages)
    
    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    required_fields = ['order_strategy', 'individual_results', 'num_order_sequences']
    for field in required_fields:
        if field not in data:
            logger.warning(f"File {file_path} is missing required field '{field}' for order experiment data")
            return False
    
    if not data.get('individual_results'):
        logger.warning(f"File {file_path} has empty 'individual_results'")
        return False
    
    return True


def main():
    # Setup logging
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Plot order-controlled knowledge editing results (conditions A/B/C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Plot single condition
  python plot_knowledge_editing_order.py --fileA results/condition_A.json
  
  # Plot multiple conditions
  python plot_knowledge_editing_order.py --fileA results/condition_A.json --fileB results/condition_B.json --fileC results/condition_C.json
  
  # Save plot to file
  python plot_knowledge_editing_order.py --fileA results/condition_A.json --out output.png
  
  # Include efficacy progression plot
  python plot_knowledge_editing_order.py --fileA results/condition_A.json --efficacy
        """
    )
    parser.add_argument("--fileA", 
                       help="Condition A order experiment result JSON file")
    parser.add_argument("--fileB", 
                       help="Condition B order experiment result JSON file")
    parser.add_argument("--fileC", 
                       help="Condition C order experiment result JSON file")
    parser.add_argument("--out", default=None,
                       help="Path to save figure (optional, shows plot if not specified)")
    parser.add_argument("--efficacy", action='store_true',
                       help="Also plot efficacy progression")
    parser.add_argument("--verbose", "-v", action='store_true',
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting order-controlled knowledge editing results plotting")
    
    # Collect provided files
    provided_files = {}
    if args.fileA:
        provided_files['A'] = args.fileA
    if args.fileB:
        provided_files['B'] = args.fileB
    if args.fileC:
        provided_files['C'] = args.fileC
    
    if not provided_files:
        parser.error("At least one of --fileA, --fileB, or --fileC must be provided")
    
    logger.info(f"Processing {len(provided_files)} condition file(s): {list(provided_files.keys())}")
    
    # If only one file is provided, use single file mode
    if len(provided_files) == 1:
        condition, file_path = list(provided_files.items())[0]
        
        try:
            data = load_json(Path(file_path))
            logger.info(f"Successfully loaded data from {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return 1
            
        # Validate order experiment data
        if not validate_order_experiment_data(data, file_path):
            logger.warning(f"File {file_path} may not be valid order experiment data, but proceeding anyway...")
        
        # Generate output path if not specified
        output_path = args.out
        if output_path is None and args.efficacy:
            # Create default output path for efficacy plot
            input_path = Path(file_path)
            output_path = str(input_path.parent / f"{input_path.stem}_plot.png")
        
        try:
            # Plot main results
            plot_order_results(data, output_path)
            
            # Plot efficacy progression if requested
            if args.efficacy:
                plot_efficacy_progression(data, output_path)
                
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return 1
    
    else:
        # Multi-condition mode
        try:
            # Load provided condition data
            data = {}
            for cond, path in provided_files.items():
                logger.info(f"Loading condition {cond} from {path}")
                data[cond] = load_json(Path(path))
                
                # Validate order experiment data
                if not validate_order_experiment_data(data[cond], path):
                    logger.warning(f"Condition {cond} file may not be valid order experiment data, but proceeding anyway...")
            
            # Plot multi-condition results
            plot_multi_condition_results(data, args.out)
            
        except Exception as e:
            logger.error(f"Error creating multi-condition plot: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return 1
    
    logger.info("Order-controlled knowledge editing results plotting completed successfully")
    return 0


def extract_target_object_statistics(data: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """
    Extract target object statistics across permutations for each step
    
    Args:
        data: JSON data from order experiment
    
    Returns:
        Tuple of (post_edit_means, post_edit_stds, final_state_means, final_state_stds, num_edits)
    """
    individual_results = data.get('individual_results', [])
    if not individual_results:
        raise ValueError("No individual results found in data")
    
    # Get basic info from first successful result
    first_successful = None
    for result in individual_results:
        if result.get('success', False) and result.get('edits'):
            first_successful = result
            break
    
    if not first_successful:
        raise ValueError("No successful results found")
    
    num_edits = len(first_successful['edits'])
    
    # Collect target object probabilities for each step across all permutations
    post_edit_step_values = [[] for _ in range(num_edits)]
    final_state_step_values = [[] for _ in range(num_edits)]
    
    for result in individual_results:
        if result.get('success', False):
            edits = result.get('edits', [])
            
            for step in range(min(num_edits, len(edits))):
                edit = edits[step]
                triple = edit['triple']
                candidates = triple['candidates']
                target_object = triple['object']
                
                if target_object in candidates:
                    target_idx = candidates.index(target_object)
                    
                    # Extract post-edit probability
                    if 'post_edit_probabilities' in edit:
                        post_probs = edit['post_edit_probabilities']['probabilities']
                        if len(post_probs) > target_idx:
                            post_edit_step_values[step].append(post_probs[target_idx])
                    
                    # Extract final-state probability
                    final_prob = None
                    if 'final_state_probabilities' in edit:
                        final_probs = edit['final_state_probabilities']['probabilities']
                        if len(final_probs) > target_idx:
                            final_prob = final_probs[target_idx]
                    elif 'final_state_evaluations' in result and step < len(result['final_state_evaluations']):
                        final_eval = result['final_state_evaluations'][step]
                        if 'final_state_probabilities' in final_eval:
                            final_probs = final_eval['final_state_probabilities']['probabilities']
                            if len(final_probs) > target_idx:
                                final_prob = final_probs[target_idx]
                    
                    if final_prob is not None:
                        final_state_step_values[step].append(final_prob)
                    elif len(post_edit_step_values[step]) > 0:
                        final_state_step_values[step].append(post_edit_step_values[step][-1])
    
    # Calculate statistics for each step
    post_edit_means = []
    post_edit_stds = []
    final_state_means = []
    final_state_stds = []
    
    for step in range(num_edits):
        if post_edit_step_values[step]:
            post_edit_means.append(np.mean(post_edit_step_values[step]))
            post_edit_stds.append(np.std(post_edit_step_values[step]))
        else:
            post_edit_means.append(0.0)
            post_edit_stds.append(0.0)
            
        if final_state_step_values[step]:
            final_state_means.append(np.mean(final_state_step_values[step]))
            final_state_stds.append(np.std(final_state_step_values[step]))
        else:
            final_state_means.append(0.0)
            final_state_stds.append(0.0)
    
    return post_edit_means, post_edit_stds, final_state_means, final_state_stds, num_edits


def extract_detailed_target_probabilities(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Extract target object probabilities for each condition, step, and order
    
    Args:
        data: Dictionary with keys 'A', 'B', 'C' mapping to condition data
    
    Returns:
        Dictionary structured as:
        {
            condition: {
                step: {
                    order_idx: {
                        'post_edit': probability,
                        'final_state': probability
                    }
                }
            }
        }
    """
    detailed_probs = {}
    
    for condition, condition_data in data.items():
        detailed_probs[condition] = {}
        
        individual_results = condition_data.get('individual_results', [])
        if not individual_results:
            continue
            
        # Get number of steps from first successful result
        first_successful = None
        for result in individual_results:
            if result.get('success', False) and result.get('edits'):
                first_successful = result
                break
        
        if not first_successful:
            continue
            
        num_steps = len(first_successful['edits'])
        
        # Initialize structure
        for step in range(num_steps):
            detailed_probs[condition][f'step_{step+1}'] = {}
        
        # Extract probabilities for each order (permutation)
        for order_idx, result in enumerate(individual_results):
            if not result.get('success', False):
                continue
                
            edits = result.get('edits', [])
            
            for step in range(min(num_steps, len(edits))):
                edit = edits[step]
                step_key = f'step_{step+1}'
                
                # Initialize order entry
                detailed_probs[condition][step_key][f'order_{order_idx+1}'] = {
                    'post_edit': 0.0,
                    'final_state': 0.0
                }
                
                # Extract post-edit probability for target object
                if 'post_edit_probabilities' in edit:
                    post_probs = edit['post_edit_probabilities']['probabilities']
                    triple = edit['triple']
                    candidates = triple['candidates']
                    target_object = triple['object']
                    
                    if target_object in candidates:
                        target_idx = candidates.index(target_object)
                        detailed_probs[condition][step_key][f'order_{order_idx+1}']['post_edit'] = post_probs[target_idx]
                
                # Extract final-state probability for target object
                final_prob = None
                triple = edit['triple']
                candidates = triple['candidates']
                target_object = triple['object']
                target_idx = candidates.index(target_object) if target_object in candidates else 0
                
                if 'final_state_probabilities' in edit:
                    final_probs = edit['final_state_probabilities']['probabilities']
                    if len(final_probs) > target_idx:
                        final_prob = final_probs[target_idx]
                elif 'final_state_evaluations' in result and step < len(result['final_state_evaluations']):
                    final_eval = result['final_state_evaluations'][step]
                    if 'final_state_probabilities' in final_eval:
                        final_probs = final_eval['final_state_probabilities']['probabilities']
                        if len(final_probs) > target_idx:
                            final_prob = final_probs[target_idx]
                
                if final_prob is not None:
                    detailed_probs[condition][step_key][f'order_{order_idx+1}']['final_state'] = final_prob
                else:
                    # Fallback to post-edit probability
                    detailed_probs[condition][step_key][f'order_{order_idx+1}']['final_state'] = \
                        detailed_probs[condition][step_key][f'order_{order_idx+1}']['post_edit']
    
    return detailed_probs


def print_detailed_probabilities(detailed_probs: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]):
    """
    Print detailed probabilities in a readable format
    
    Args:
        detailed_probs: Output from extract_detailed_target_probabilities
    """
    for condition in ['A', 'B', 'C']:
        if condition not in detailed_probs:
            continue
            
        print(f"\n=== Condition {condition} ===")
        condition_data = detailed_probs[condition]
        
        # Get all steps
        steps = sorted([k for k in condition_data.keys() if k.startswith('step_')])
        
        for step in steps:
            print(f"\n{step.replace('_', ' ').title()}:")
            step_data = condition_data[step]
            
            # Get all orders
            orders = sorted([k for k in step_data.keys() if k.startswith('order_')])
            
            print("  Order | Post-edit | Final-state")
            print("  ------|-----------|------------")
            
            for order in orders:
                order_data = step_data[order]
                post_edit = order_data['post_edit']
                final_state = order_data['final_state']
                order_num = order.replace('order_', '')
                print(f"    {order_num:2s}  |   {post_edit:.4f}  |   {final_state:.4f}")


if __name__ == "__main__":
    exit(main())