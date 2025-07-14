#!/usr/bin/env python3
"""
Simplified IRT analysis script that focuses on core functionality.
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_experimental_results(results_dir: str = "/app/EasyEdit/results") -> Dict:
    """Load experimental results from JSON files."""
    results = {}
    results_path = Path(results_dir)
    
    for result_file in results_path.glob("knowledge_editing_from_candidates_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            # Extract experiment identifier
            filename = result_file.stem
            parts = filename.split('_')
            method = parts[4]  # ROME, MEMIT, etc.
            model = parts[5]   # gpt-j-6b, gpt2-xl
            condition = parts[6]  # condition_A, condition_B, condition_C
            
            exp_id = f"{method}_{model}_{condition}"
            results[exp_id] = data
            
            logger.info(f"Loaded {exp_id}: {len(data.get('individual_results', []))} samples")
            
        except Exception as e:
            logger.error(f"Error loading {result_file}: {e}")
            continue
    
    return results

def extract_analysis_data(results: Dict) -> pd.DataFrame:
    """Extract data for analysis."""
    data = []
    
    for exp_id, exp_data in results.items():
        if not exp_data.get('success', False):
            continue
            
        method, model, condition = exp_id.split('_')
        individual_results = exp_data.get('individual_results', [])
        
        for sample_idx, sample in enumerate(individual_results):
            edits = sample.get('edits', [])
            final_evaluations = sample.get('final_state_evaluations', [])
            
            for edit_idx, edit in enumerate(edits):
                triple = edit.get('triple', {})
                
                # Extract immediate response
                post_edit_probs = edit.get('post_edit_probabilities', {})
                candidates = post_edit_probs.get('candidates', [])
                probabilities = post_edit_probs.get('probabilities', [])
                
                if not candidates or not probabilities:
                    continue
                
                target_object = triple.get('object', '')
                immediate_prob = 0.0
                immediate_rank = len(candidates) + 1
                
                if target_object in candidates:
                    target_idx = candidates.index(target_object)
                    immediate_prob = probabilities[target_idx]
                    sorted_probs = sorted(probabilities, reverse=True)
                    immediate_rank = sorted_probs.index(immediate_prob) + 1
                
                # Extract cumulative response
                cumulative_prob = 0.0
                cumulative_rank = len(candidates) + 1
                
                for final_eval in final_evaluations:
                    if final_eval.get('triple_index') == edit_idx:
                        cumulative_prob = final_eval.get('target_probability', 0.0)
                        cumulative_rank = final_eval.get('target_rank', len(candidates) + 1)
                        break
                
                data.append({
                    'method': method,
                    'model': model,
                    'condition': condition,
                    'sample_index': sample_idx,
                    'edit_order': edit.get('edit_order', edit_idx + 1),
                    'subject': triple.get('subject', ''),
                    'relation': triple.get('relation', ''),
                    'object': triple.get('object', ''),
                    'immediate_probability': immediate_prob,
                    'cumulative_probability': cumulative_prob,
                    'immediate_rank': immediate_rank,
                    'cumulative_rank': cumulative_rank,
                    'immediate_correct': 1 if immediate_rank == 1 else 0,
                    'cumulative_correct': 1 if cumulative_rank == 1 else 0,
                    'probability_change': immediate_prob - cumulative_prob,
                    'rank_change': cumulative_rank - immediate_rank
                })
    
    return pd.DataFrame(data)

def create_basic_visualizations(df: pd.DataFrame, output_dir: str = "/app/EasyEdit/irt_evaluation/output"):
    """Create basic visualizations."""
    output_path = Path(output_dir)
    figures_path = output_path / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # Color scheme
    colors = {'immediate': '#4CAF50', 'cumulative': '#FF7043'}
    
    # 1. Success rates by edit type (condition)
    plt.figure(figsize=(12, 8))
    
    success_rates = df.groupby('condition').agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean'
    }).reset_index()
    
    x = np.arange(len(success_rates))
    width = 0.35
    
    plt.bar(x - width/2, success_rates['immediate_correct'], width, 
            label='即時反応 (Immediate)', alpha=0.8, color=colors['immediate'])
    plt.bar(x + width/2, success_rates['cumulative_correct'], width, 
            label='累積反応 (Cumulative)', alpha=0.8, color=colors['cumulative'])
    
    plt.xlabel('編集タイプ (Edit Type)')
    plt.ylabel('成功率 (Success Rate)')
    plt.title('編集タイプ別成功率 (Success Rates by Edit Type)')
    plt.xticks(x, [f'タイプ{c}' for c in success_rates['condition']])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_path / 'success_rates_by_edit_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Success rates by method
    plt.figure(figsize=(12, 8))
    
    method_success = df.groupby('method').agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean'
    }).reset_index()
    
    x = np.arange(len(method_success))
    
    plt.bar(x - width/2, method_success['immediate_correct'], width, 
            label='即時反応 (Immediate)', alpha=0.8, color=colors['immediate'])
    plt.bar(x + width/2, method_success['cumulative_correct'], width, 
            label='累積反応 (Cumulative)', alpha=0.8, color=colors['cumulative'])
    
    plt.xlabel('編集手法 (Method)')
    plt.ylabel('成功率 (Success Rate)')
    plt.title('手法別成功率 (Success Rates by Method)')
    plt.xticks(x, method_success['method'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_path / 'success_rates_by_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Success rates by model
    plt.figure(figsize=(12, 8))
    
    model_success = df.groupby('model').agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean'
    }).reset_index()
    
    x = np.arange(len(model_success))
    
    plt.bar(x - width/2, model_success['immediate_correct'], width, 
            label='即時反応 (Immediate)', alpha=0.8, color=colors['immediate'])
    plt.bar(x + width/2, model_success['cumulative_correct'], width, 
            label='累積反応 (Cumulative)', alpha=0.8, color=colors['cumulative'])
    
    plt.xlabel('モデル (Model)')
    plt.ylabel('成功率 (Success Rate)')
    plt.title('モデル別成功率 (Success Rates by Model)')
    plt.xticks(x, model_success['model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_path / 'success_rates_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Detailed comparison matrix (Method x Model)
    plt.figure(figsize=(14, 10))
    
    # Create method-model combinations
    method_model_stats = df.groupby(['method', 'model']).agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean'
    }).reset_index()
    
    methods = method_model_stats['method'].unique()
    models = method_model_stats['model'].unique()
    
    # Create subplots for immediate and cumulative
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Immediate response matrix
    immediate_matrix = method_model_stats.pivot(index='method', columns='model', values='immediate_correct')
    im1 = ax1.imshow(immediate_matrix.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('即時反応成功率 (Immediate Response Success Rate)')
    ax1.set_xlabel('モデル (Model)')
    ax1.set_ylabel('手法 (Method)')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(models)):
            text = ax1.text(j, i, f'{immediate_matrix.iloc[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Cumulative response matrix
    cumulative_matrix = method_model_stats.pivot(index='method', columns='model', values='cumulative_correct')
    im2 = ax2.imshow(cumulative_matrix.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('累積反応成功率 (Cumulative Response Success Rate)')
    ax2.set_xlabel('モデル (Model)')
    ax2.set_ylabel('手法 (Method)')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(models)):
            text = ax2.text(j, i, f'{cumulative_matrix.iloc[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(figures_path / 'method_model_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed edit type analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Edit type x Method matrix for immediate response
    edit_method_immediate = df.groupby(['condition', 'method'])['immediate_correct'].mean().unstack()
    im1 = axes[0,0].imshow(edit_method_immediate.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0,0].set_title('編集タイプ×手法 即時反応 (Edit Type × Method: Immediate)')
    axes[0,0].set_xlabel('手法 (Method)')
    axes[0,0].set_ylabel('編集タイプ (Edit Type)')
    axes[0,0].set_xticks(range(len(edit_method_immediate.columns)))
    axes[0,0].set_xticklabels(edit_method_immediate.columns)
    axes[0,0].set_yticks(range(len(edit_method_immediate.index)))
    axes[0,0].set_yticklabels([f'タイプ{c}' for c in edit_method_immediate.index])
    
    # Add text annotations
    for i in range(len(edit_method_immediate.index)):
        for j in range(len(edit_method_immediate.columns)):
            axes[0,0].text(j, i, f'{edit_method_immediate.iloc[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Edit type x Method matrix for cumulative response
    edit_method_cumulative = df.groupby(['condition', 'method'])['cumulative_correct'].mean().unstack()
    im2 = axes[0,1].imshow(edit_method_cumulative.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0,1].set_title('編集タイプ×手法 累積反応 (Edit Type × Method: Cumulative)')
    axes[0,1].set_xlabel('手法 (Method)')
    axes[0,1].set_ylabel('編集タイプ (Edit Type)')
    axes[0,1].set_xticks(range(len(edit_method_cumulative.columns)))
    axes[0,1].set_xticklabels(edit_method_cumulative.columns)
    axes[0,1].set_yticks(range(len(edit_method_cumulative.index)))
    axes[0,1].set_yticklabels([f'タイプ{c}' for c in edit_method_cumulative.index])
    
    # Add text annotations
    for i in range(len(edit_method_cumulative.index)):
        for j in range(len(edit_method_cumulative.columns)):
            axes[0,1].text(j, i, f'{edit_method_cumulative.iloc[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Edit type x Model matrix for immediate response
    edit_model_immediate = df.groupby(['condition', 'model'])['immediate_correct'].mean().unstack()
    im3 = axes[1,0].imshow(edit_model_immediate.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1,0].set_title('編集タイプ×モデル 即時反応 (Edit Type × Model: Immediate)')
    axes[1,0].set_xlabel('モデル (Model)')
    axes[1,0].set_ylabel('編集タイプ (Edit Type)')
    axes[1,0].set_xticks(range(len(edit_model_immediate.columns)))
    axes[1,0].set_xticklabels(edit_model_immediate.columns, rotation=45)
    axes[1,0].set_yticks(range(len(edit_model_immediate.index)))
    axes[1,0].set_yticklabels([f'タイプ{c}' for c in edit_model_immediate.index])
    
    # Add text annotations
    for i in range(len(edit_model_immediate.index)):
        for j in range(len(edit_model_immediate.columns)):
            axes[1,0].text(j, i, f'{edit_model_immediate.iloc[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Edit type x Model matrix for cumulative response
    edit_model_cumulative = df.groupby(['condition', 'model'])['cumulative_correct'].mean().unstack()
    im4 = axes[1,1].imshow(edit_model_cumulative.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1,1].set_title('編集タイプ×モデル 累積反応 (Edit Type × Model: Cumulative)')
    axes[1,1].set_xlabel('モデル (Model)')
    axes[1,1].set_ylabel('編集タイプ (Edit Type)')
    axes[1,1].set_xticks(range(len(edit_model_cumulative.columns)))
    axes[1,1].set_xticklabels(edit_model_cumulative.columns, rotation=45)
    axes[1,1].set_yticks(range(len(edit_model_cumulative.index)))
    axes[1,1].set_yticklabels([f'タイプ{c}' for c in edit_model_cumulative.index])
    
    # Add text annotations
    for i in range(len(edit_model_cumulative.index)):
        for j in range(len(edit_model_cumulative.columns)):
            axes[1,1].text(j, i, f'{edit_model_cumulative.iloc[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    plt.colorbar(im3, ax=axes[1,0], shrink=0.8)
    plt.colorbar(im4, ax=axes[1,1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(figures_path / 'detailed_analysis_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Probability distributions
    plt.figure(figsize=(12, 8))
    
    plt.hist(df['immediate_probability'], bins=30, alpha=0.7, 
             label='即時反応 (Immediate)', color=colors['immediate'])
    plt.hist(df['cumulative_probability'], bins=30, alpha=0.7, 
             label='累積反応 (Cumulative)', color=colors['cumulative'])
    
    plt.xlabel('確率 (Probability)')
    plt.ylabel('頻度 (Frequency)')
    plt.title('確率分布 (Probability Distributions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_path / 'probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Probability changes
    plt.figure(figsize=(12, 8))
    
    plt.hist(df['probability_change'], bins=50, alpha=0.7, color='gold')
    plt.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    plt.xlabel('確率変化 (Probability Change: Immediate - Cumulative)')
    plt.ylabel('頻度 (Frequency)')
    plt.title('確率変化分布 (Probability Change Distribution)')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    positive_changes = (df['probability_change'] > 0).sum()
    negative_changes = (df['probability_change'] < 0).sum()
    plt.text(0.02, plt.ylim()[1] * 0.9, f'向上: {positive_changes}\n劣化: {negative_changes}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(figures_path / 'probability_changes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Performance by edit order
    plt.figure(figsize=(12, 8))
    
    order_stats = df.groupby('edit_order').agg({
        'immediate_correct': ['mean', 'std'],
        'cumulative_correct': ['mean', 'std']
    }).reset_index()
    
    order_stats.columns = ['edit_order', 'imm_mean', 'imm_std', 'cum_mean', 'cum_std']
    
    plt.errorbar(order_stats['edit_order'], order_stats['imm_mean'], 
                yerr=order_stats['imm_std'], marker='o', label='即時反応 (Immediate)', 
                color=colors['immediate'], capsize=5, linewidth=2, markersize=8)
    plt.errorbar(order_stats['edit_order'], order_stats['cum_mean'], 
                yerr=order_stats['cum_std'], marker='s', label='累積反応 (Cumulative)', 
                color=colors['cumulative'], capsize=5, linewidth=2, markersize=8)
    
    plt.xlabel('編集順序 (Edit Order)')
    plt.ylabel('成功率 (Success Rate)')
    plt.title('編集順序による性能変化 (Performance by Edit Order)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_path / 'performance_by_edit_order.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created visualizations in {figures_path}")

def create_summary_report(df: pd.DataFrame, output_dir: str = "/app/EasyEdit/irt_evaluation/output"):
    """Create a summary report."""
    output_path = Path(output_dir)
    reports_path = output_path / 'reports'
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    summary_stats = {
        'total_observations': len(df),
        'unique_methods': df['method'].nunique(),
        'unique_models': df['model'].nunique(),
        'unique_conditions': df['condition'].nunique(),
        'overall_immediate_success': df['immediate_correct'].mean(),
        'overall_cumulative_success': df['cumulative_correct'].mean(),
        'performance_degradation': df['immediate_correct'].mean() - df['cumulative_correct'].mean(),
        'mean_probability_change': df['probability_change'].mean(),
        'methods': df['method'].unique().tolist(),
        'models': df['model'].unique().tolist(),
        'conditions': df['condition'].unique().tolist()
    }
    
    # Method comparison
    method_stats = df.groupby('method').agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean',
        'probability_change': 'mean'
    }).round(3)
    
    # Model comparison
    model_stats = df.groupby('model').agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean',
        'probability_change': 'mean'
    }).round(3)
    
    # Edit type comparison
    edit_type_stats = df.groupby('condition').agg({
        'immediate_correct': 'mean',
        'cumulative_correct': 'mean',
        'probability_change': 'mean'
    }).round(3)
    
    # Response type comparison
    response_stats = {
        'immediate': {
            'mean_success': df['immediate_correct'].mean(),
            'mean_probability': df['immediate_probability'].mean()
        },
        'cumulative': {
            'mean_success': df['cumulative_correct'].mean(),
            'mean_probability': df['cumulative_probability'].mean()
        }
    }
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>継続的知識編集 評価レポート</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Yu Gothic', 'Hiragino Sans', Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007acc; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .highlight {{ background-color: #fff3cd; padding: 5px; border-radius: 3px; }}
            .matrix {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>継続的知識編集 評価レポート</h1>
            <p>Continual Knowledge Editing Performance Analysis</p>
        </div>
        
        <div class="section">
            <h2>📊 分析サマリー</h2>
            <div class="metric">
                <p><strong>総観測数:</strong> {summary_stats['total_observations']:,}</p>
                <p><strong>分析手法:</strong> {', '.join(summary_stats['methods'])}</p>
                <p><strong>分析モデル:</strong> {', '.join(summary_stats['models'])}</p>
                <p><strong>編集タイプ:</strong> {', '.join([f'タイプ{c}' for c in summary_stats['conditions']])}</p>
            </div>
            
            <div class="metric">
                <h3>🎯 全体的性能</h3>
                <p><strong>即時反応成功率:</strong> {summary_stats['overall_immediate_success']:.3f} ({summary_stats['overall_immediate_success']:.1%})</p>
                <p><strong>累積反応成功率:</strong> {summary_stats['overall_cumulative_success']:.3f} ({summary_stats['overall_cumulative_success']:.1%})</p>
                <p class="highlight"><strong>性能劣化度:</strong> {summary_stats['performance_degradation']:.3f} ({summary_stats['performance_degradation']:.1%})</p>
                <p><strong>平均確率変化:</strong> {summary_stats['mean_probability_change']:.3f}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔧 手法別比較</h2>
            <table>
                <tr>
                    <th>手法 (Method)</th>
                    <th>即時反応成功率</th>
                    <th>累積反応成功率</th>
                    <th>確率変化</th>
                </tr>"""
    
    for method, row in method_stats.iterrows():
        html_content += f"""
                <tr>
                    <td><strong>{method}</strong></td>
                    <td>{row['immediate_correct']:.3f}</td>
                    <td>{row['cumulative_correct']:.3f}</td>
                    <td>{row['probability_change']:.3f}</td>
                </tr>"""
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>🤖 モデル別比較</h2>
            <table>
                <tr>
                    <th>モデル (Model)</th>
                    <th>即時反応成功率</th>
                    <th>累積反応成功率</th>
                    <th>確率変化</th>
                </tr>"""
    
    for model, row in model_stats.iterrows():
        html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{row['immediate_correct']:.3f}</td>
                    <td>{row['cumulative_correct']:.3f}</td>
                    <td>{row['probability_change']:.3f}</td>
                </tr>"""
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>📝 編集タイプ別比較</h2>
            <table>
                <tr>
                    <th>編集タイプ (Edit Type)</th>
                    <th>即時反応成功率</th>
                    <th>累積反応成功率</th>
                    <th>確率変化</th>
                </tr>"""
    
    for condition, row in edit_type_stats.iterrows():
        html_content += f"""
                <tr>
                    <td><strong>タイプ{condition}</strong></td>
                    <td>{row['immediate_correct']:.3f}</td>
                    <td>{row['cumulative_correct']:.3f}</td>
                    <td>{row['probability_change']:.3f}</td>
                </tr>"""
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>📈 反応タイプ別比較</h2>
            <div class="metric">
                <h3>即時反応 (Immediate Response)</h3>
                <p><strong>平均成功率:</strong> {response_stats['immediate']['mean_success']:.3f}</p>
                <p><strong>平均確率:</strong> {response_stats['immediate']['mean_probability']:.3f}</p>
            </div>
            <div class="metric">
                <h3>累積反応 (Cumulative Response)</h3>
                <p><strong>平均成功率:</strong> {response_stats['cumulative']['mean_success']:.3f}</p>
                <p><strong>平均確率:</strong> {response_stats['cumulative']['mean_probability']:.3f}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 主要な発見</h2>
            <ul>
                <li><strong>即時反応成功率:</strong> {summary_stats['overall_immediate_success']:.1%}</li>
                <li><strong>累積反応成功率:</strong> {summary_stats['overall_cumulative_success']:.1%}</li>
                <li><strong>性能劣化:</strong> {summary_stats['performance_degradation']:.1%}</li>
                <li><strong>最良手法:</strong> {method_stats['cumulative_correct'].idxmax()}</li>
                <li><strong>最困難編集タイプ:</strong> タイプ{edit_type_stats['cumulative_correct'].idxmin()}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📊 生成された可視化</h2>
            <ul>
                <li>編集タイプ別成功率 (success_rates_by_edit_type.png)</li>
                <li>手法別成功率 (success_rates_by_method.png)</li>
                <li>モデル別成功率 (success_rates_by_model.png)</li>
                <li>手法×モデル 性能マトリックス (method_model_matrix.png)</li>
                <li>詳細分析マトリックス (detailed_analysis_matrices.png)</li>
                <li>確率分布 (probability_distributions.png)</li>
                <li>確率変化分布 (probability_changes.png)</li>
                <li>編集順序効果 (performance_by_edit_order.png)</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(reports_path / 'simple_analysis_report.html', 'w') as f:
        f.write(html_content)
    
    # Save data as CSV
    df.to_csv(output_path / 'analysis_data.csv', index=False)
    
    # Save summary as JSON
    with open(output_path / 'summary_statistics.json', 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'method_stats': method_stats.to_dict(),
            'model_stats': model_stats.to_dict(),
            'edit_type_stats': edit_type_stats.to_dict(),
            'response_stats': response_stats
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created report and data files in {output_path}")

def main():
    """Main analysis function."""
    print("🚀 Starting Simple IRT-based Knowledge Editing Analysis")
    print("=" * 60)
    
    # Load data
    print("📊 Loading experimental results...")
    results = load_experimental_results()
    
    if not results:
        print("❌ No experimental results found!")
        return
    
    # Extract analysis data
    print("🔄 Extracting analysis data...")
    df = extract_analysis_data(results)
    
    if df.empty:
        print("❌ No data extracted!")
        return
    
    print(f"✅ Extracted {len(df)} observations")
    
    # Create visualizations
    print("📈 Creating visualizations...")
    create_basic_visualizations(df)
    
    # Create summary report
    print("📋 Creating summary report...")
    create_summary_report(df)
    
    print("\n✅ Analysis completed successfully!")
    print("📁 Results saved to: /app/EasyEdit/irt_evaluation/output/")
    print("📊 Visualizations: /app/EasyEdit/irt_evaluation/output/figures/")
    print("📋 Report: /app/EasyEdit/irt_evaluation/output/reports/simple_analysis_report.html")

if __name__ == "__main__":
    main()