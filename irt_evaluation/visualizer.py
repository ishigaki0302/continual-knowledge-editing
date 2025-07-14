#!/usr/bin/env python3
"""
Visualization Module for IRT-based Knowledge Editing Evaluation

This module creates comprehensive visualizations for IRT analysis results.
Includes ICC plots, parameter distributions, model comparisons, and research-ready figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class IRTVisualizer:
    """
    Creates comprehensive visualizations for IRT analysis results.
    
    Features:
    - Item Characteristic Curves (ICC)
    - Parameter distribution plots
    - Model comparison visualizations
    - Person-item maps
    - Research-ready publication figures
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Try to set style, fallback if not available
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            warnings.warn(f"Style '{style}' not available, using default")
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C7B7F'
        }
        
        self.method_colors = {
            'ROME': '#2E86AB',
            'MEMIT': '#A23B72',
            'MEND': '#F18F01',
            'FT': '#C73E1D',
            'IKE': '#6C7B7F',
            'KN': '#8B5A2B'
        }
        
        self.condition_colors = {
            'A': '#2E86AB',
            'B': '#A23B72',
            'C': '#F18F01'
        }
    
    def plot_item_characteristic_curves(self, 
                                      icc_data: Dict[str, Any],
                                      items: Optional[List[str]] = None,
                                      group_by: Optional[str] = None,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Item Characteristic Curves
        
        Args:
            icc_data: ICC data from fit_irt module
            items: Specific items to plot (if None, plots all)
            group_by: Group items by condition ('condition', 'relation_type', etc.)
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Plotting Item Characteristic Curves")
        
        theta = icc_data['theta']
        items_to_plot = items if items is not None else list(icc_data['items'].keys())
        
        # Determine subplot layout
        if group_by:
            # Group items and create subplots
            groups = self._group_items(items_to_plot, group_by)
            n_groups = len(groups)
            
            fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 6))
            if n_groups == 1:
                axes = [axes]
            
            for i, (group_name, group_items) in enumerate(groups.items()):
                ax = axes[i]
                self._plot_icc_group(ax, theta, icc_data, group_items, group_name)
        else:
            # Single plot with all items
            fig, ax = plt.subplots(figsize=self.figsize)
            self._plot_icc_group(ax, theta, icc_data, items_to_plot, "All Items")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ICC plot saved to: {save_path}")
        
        return fig
    
    def _group_items(self, items: List[str], group_by: str) -> Dict[str, List[str]]:
        """Group items by specified criterion"""
        groups = {}
        
        for item in items:
            # Parse item_id to extract grouping criterion
            if group_by == 'condition':
                group_key = item.split('_')[0]  # First part is condition
            elif group_by == 'edit_order':
                parts = item.split('_')
                if len(parts) >= 3:
                    edit_order = int(parts[2])
                    if edit_order <= 2:
                        group_key = 'Early (1-2)'
                    elif edit_order <= 4:
                        group_key = 'Middle (3-4)'
                    else:
                        group_key = 'Late (5+)'
                else:
                    group_key = 'Unknown'
            else:
                group_key = 'All'
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        
        return groups
    
    def _plot_icc_group(self, ax: plt.Axes, theta: np.ndarray, 
                       icc_data: Dict[str, Any], items: List[str], 
                       title: str):
        """Plot ICC for a group of items"""
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(items)))
        
        for i, item in enumerate(items):
            if item in icc_data['items']:
                probabilities = icc_data['items'][item]['probabilities']
                beta = icc_data['items'][item]['beta']
                alpha = icc_data['items'][item]['alpha']
                
                # Plot ICC curve
                ax.plot(theta, probabilities, color=colors[i], 
                       linewidth=2, alpha=0.8, label=f'{item} (β={beta:.2f}, α={alpha:.2f})')
                
                # Mark item difficulty
                ax.axvline(x=beta, color=colors[i], linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Person Ability (θ)', fontsize=12)
        ax.set_ylabel('Probability of Correct Response', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(theta.min(), theta.max())
        ax.set_ylim(0, 1)
        
        # Add legend if not too many items
        if len(items) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def plot_parameter_distributions(self, 
                                   results: Dict[str, Any],
                                   irt_data: Optional[pd.DataFrame] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distributions of IRT parameters
        
        Args:
            results: IRT model results
            irt_data: Original IRT data for grouping
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Plotting parameter distributions")
        
        # Determine number of subplots based on model type
        params = []
        if 'theta' in results:
            params.append('theta')
        if 'beta' in results:
            params.append('beta')
        if 'alpha' in results and results['alpha'] is not None:
            params.append('alpha')
        if 'gamma' in results and results['gamma'] is not None:
            params.append('gamma')
        
        n_params = len(params)
        fig, axes = plt.subplots(2, n_params, figsize=(4*n_params, 8))
        
        if n_params == 1:
            axes = axes.reshape(2, 1)
        
        for i, param in enumerate(params):
            values = results[param]
            
            # Top row: histogram
            ax1 = axes[0, i]
            ax1.hist(values, bins=20, alpha=0.7, color=self.colors['primary'], edgecolor='black')
            ax1.set_xlabel(self._get_param_label(param))
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{self._get_param_label(param)} Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax1.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Bottom row: boxplot or violin plot
            ax2 = axes[1, i]
            
            if param == 'theta' and irt_data is not None:
                # Group theta by method
                self._plot_grouped_parameter(ax2, values, results['persons'], 
                                           irt_data, param, 'method')
            elif param == 'beta' and irt_data is not None:
                # Group beta by condition
                self._plot_grouped_parameter(ax2, values, results['items'], 
                                           irt_data, param, 'condition')
            else:
                # Simple boxplot
                ax2.boxplot(values, vert=True)
                ax2.set_ylabel(self._get_param_label(param))
                ax2.set_title(f'{self._get_param_label(param)} Boxplot')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Parameter distribution plot saved to: {save_path}")
        
        return fig
    
    def _get_param_label(self, param: str) -> str:
        """Get formatted parameter label"""
        labels = {
            'theta': 'Person Ability (θ)',
            'beta': 'Item Difficulty (β)',
            'alpha': 'Item Discrimination (α)',
            'gamma': 'Guessing Parameter (γ)'
        }
        return labels.get(param, param)
    
    def _plot_grouped_parameter(self, ax: plt.Axes, values: np.ndarray, 
                              ids: List[str], irt_data: pd.DataFrame, 
                              param: str, group_by: str):
        """Plot parameter values grouped by a criterion"""
        
        # Create mapping from IDs to groups
        if param == 'theta':
            id_col = 'person_id'
        elif param == 'beta':
            id_col = 'item_id'
        else:
            # Simple boxplot
            ax.boxplot(values, vert=True)
            ax.set_ylabel(self._get_param_label(param))
            return
        
        # Group values
        groups = {}
        for i, id_val in enumerate(ids):
            group_data = irt_data[irt_data[id_col] == id_val]
            if len(group_data) > 0:
                group_val = group_data[group_by].iloc[0]
                if group_val not in groups:
                    groups[group_val] = []
                groups[group_val].append(values[i])
        
        # Plot grouped boxplots
        group_names = list(groups.keys())
        group_values = [groups[name] for name in group_names]
        
        positions = range(1, len(group_names) + 1)
        bp = ax.boxplot(group_values, positions=positions, patch_artist=True)
        
        # Color boxes
        colors = [self.method_colors.get(name, self.colors['neutral']) 
                 for name in group_names]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(group_names, rotation=45)
        ax.set_ylabel(self._get_param_label(param))
        ax.set_title(f'{self._get_param_label(param)} by {group_by.title()}')
        ax.grid(True, alpha=0.3)
    
    def plot_person_item_map(self, 
                           results: Dict[str, Any],
                           irt_data: Optional[pd.DataFrame] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot person-item map (Wright map)
        
        Args:
            results: IRT model results
            irt_data: Original IRT data for labels
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Plotting person-item map")
        
        theta = results['theta']
        beta = results['beta']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        
        # Left side: Person ability distribution
        ax1.hist(theta, bins=20, orientation='horizontal', alpha=0.7, 
                color=self.colors['primary'], edgecolor='black')
        ax1.set_ylabel('Ability (θ)', fontsize=12)
        ax1.set_xlabel('Number of Persons', fontsize=12)
        ax1.set_title('Person Ability Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right side: Item difficulty distribution
        ax2.hist(beta, bins=20, orientation='horizontal', alpha=0.7, 
                color=self.colors['secondary'], edgecolor='black')
        ax2.set_ylabel('Difficulty (β)', fontsize=12)
        ax2.set_xlabel('Number of Items', fontsize=12)
        ax2.set_title('Item Difficulty Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Align y-axes
        y_min = min(theta.min(), beta.min()) - 0.5
        y_max = max(theta.max(), beta.max()) + 0.5
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        # Add mean lines
        ax1.axhline(y=np.mean(theta), color='red', linestyle='--', 
                   label=f'Mean θ = {np.mean(theta):.2f}')
        ax2.axhline(y=np.mean(beta), color='red', linestyle='--', 
                   label=f'Mean β = {np.mean(beta):.2f}')
        
        ax1.legend()
        ax2.legend()
        
        # Add interpretation text
        fig.text(0.5, 0.02, 
                'Left: Person abilities (higher = more capable)\n'
                'Right: Item difficulties (higher = more difficult)\n'
                'Ideal: Person abilities > Item difficulties', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Person-item map saved to: {save_path}")
        
        return fig
    
    def plot_model_comparison(self, 
                            comparison_results: Dict[str, Any],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model comparison results
        
        Args:
            comparison_results: Model comparison results
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Plotting model comparison")
        
        models = comparison_results['models']
        
        # Create DataFrame for plotting
        df = pd.DataFrame(models)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Information criteria
        ax1 = axes[0]
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['aic'], width, label='AIC', 
               color=self.colors['primary'], alpha=0.7)
        ax1.bar(x + width/2, df['bic'], width, label='BIC', 
               color=self.colors['secondary'], alpha=0.7)
        
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Information Criterion')
        ax1.set_title('Model Comparison: Information Criteria')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['model_type'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-likelihood
        ax2 = axes[1]
        bars = ax2.bar(df['model_type'], df['log_likelihood'], 
                      color=self.colors['accent'], alpha=0.7)
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('Log-Likelihood')
        ax2.set_title('Model Comparison: Log-Likelihood')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best model
        best_model_type = comparison_results['best_model']['model_type']
        for i, bar in enumerate(bars):
            if df.iloc[i]['model_type'] == best_model_type:
                bar.set_color(self.colors['success'])
                bar.set_alpha(1.0)
        
        # Plot 3: Number of parameters
        ax3 = axes[2]
        ax3.bar(df['model_type'], df['n_parameters'], 
               color=self.colors['neutral'], alpha=0.7)
        ax3.set_xlabel('Model Type')
        ax3.set_ylabel('Number of Parameters')
        ax3.set_title('Model Complexity')
        ax3.grid(True, alpha=0.3)
        
        # Add text annotation for best model
        fig.text(0.5, 0.02, 
                f'Best Model: {best_model_type} (AIC: {comparison_results["best_model"]["aic"]:.2f})', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to: {save_path}")
        
        return fig
    
    def plot_method_performance(self, 
                              irt_data: pd.DataFrame,
                              results: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot performance comparison across methods
        
        Args:
            irt_data: IRT formatted data
            results: IRT model results
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Plotting method performance comparison")
        
        # Create method performance summary
        method_stats = []
        
        for i, person_id in enumerate(results['persons']):
            person_data = irt_data[irt_data['person_id'] == person_id]
            if len(person_data) > 0:
                method = person_data['method'].iloc[0]
                model = person_data['model_name'].iloc[0]
                
                method_stats.append({
                    'person_id': person_id,
                    'method': method,
                    'model': model,
                    'theta': results['theta'][i],
                    'mean_response': person_data['response'].mean(),
                    'accuracy': person_data['is_correct'].mean()
                })
        
        df = pd.DataFrame(method_stats)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Theta by method
        ax1 = axes[0, 0]
        methods = df['method'].unique()
        theta_by_method = [df[df['method'] == method]['theta'].values for method in methods]
        
        bp1 = ax1.boxplot(theta_by_method, labels=methods, patch_artist=True)
        colors = [self.method_colors.get(method, self.colors['neutral']) for method in methods]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Person Ability (θ)')
        ax1.set_title('Person Ability by Method')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy by method
        ax2 = axes[0, 1]
        accuracy_by_method = [df[df['method'] == method]['accuracy'].values for method in methods]
        
        bp2 = ax2.boxplot(accuracy_by_method, labels=methods, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy by Method')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Theta vs Accuracy scatter
        ax3 = axes[1, 0]
        for method in methods:
            method_data = df[df['method'] == method]
            ax3.scatter(method_data['theta'], method_data['accuracy'], 
                       label=method, alpha=0.7, s=50,
                       color=self.method_colors.get(method, self.colors['neutral']))
        
        ax3.set_xlabel('Person Ability (θ)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Ability vs Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Method rankings
        ax4 = axes[1, 1]
        method_means = df.groupby('method')['theta'].mean().sort_values(ascending=False)
        
        bars = ax4.bar(method_means.index, method_means.values, 
                      color=[self.method_colors.get(method, self.colors['neutral']) 
                            for method in method_means.index], alpha=0.7)
        
        ax4.set_ylabel('Mean Person Ability (θ)')
        ax4.set_title('Method Rankings')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, method_means.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Method performance plot saved to: {save_path}")
        
        return fig
    
    def plot_condition_analysis(self, 
                              irt_data: pd.DataFrame,
                              results: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot analysis by experimental conditions
        
        Args:
            irt_data: IRT formatted data
            results: IRT model results
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Plotting condition analysis")
        
        # Create condition-based item analysis
        item_stats = []
        
        for i, item_id in enumerate(results['items']):
            item_data = irt_data[irt_data['item_id'] == item_id]
            if len(item_data) > 0:
                condition = item_data['condition'].iloc[0]
                edit_order = item_data['edit_order'].iloc[0]
                relation_type = item_data.get('relation_type', ['unknown']).iloc[0]
                
                item_stats.append({
                    'item_id': item_id,
                    'condition': condition,
                    'edit_order': edit_order,
                    'relation_type': relation_type,
                    'beta': results['beta'][i],
                    'alpha': results['alpha'][i] if results['alpha'] is not None else 1.0,
                    'mean_response': item_data['response'].mean(),
                    'accuracy': item_data['is_correct'].mean()
                })
        
        df = pd.DataFrame(item_stats)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Difficulty by condition
        ax1 = axes[0, 0]
        conditions = df['condition'].unique()
        difficulty_by_condition = [df[df['condition'] == cond]['beta'].values for cond in conditions]
        
        bp1 = ax1.boxplot(difficulty_by_condition, labels=conditions, patch_artist=True)
        colors = [self.condition_colors.get(cond, self.colors['neutral']) for cond in conditions]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Item Difficulty (β)')
        ax1.set_title('Item Difficulty by Condition')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difficulty by edit order
        ax2 = axes[0, 1]
        edit_orders = sorted(df['edit_order'].unique())
        difficulty_by_order = [df[df['edit_order'] == order]['beta'].values for order in edit_orders]
        
        bp2 = ax2.boxplot(difficulty_by_order, labels=edit_orders, patch_artist=True)
        order_colors = plt.cm.viridis(np.linspace(0, 1, len(edit_orders)))
        for patch, color in zip(bp2['boxes'], order_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Edit Order')
        ax2.set_ylabel('Item Difficulty (β)')
        ax2.set_title('Item Difficulty by Edit Order')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relation type comparison
        ax3 = axes[1, 0]
        if 'relation_type' in df.columns:
            relation_types = df['relation_type'].unique()
            for rel_type in relation_types:
                type_data = df[df['relation_type'] == rel_type]
                ax3.scatter(type_data['beta'], type_data['alpha'], 
                          label=rel_type, alpha=0.7, s=50)
        
        ax3.set_xlabel('Item Difficulty (β)')
        ax3.set_ylabel('Item Discrimination (α)')
        ax3.set_title('Difficulty vs Discrimination by Relation Type')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Condition performance heatmap
        ax4 = axes[1, 1]
        condition_order_means = df.groupby(['condition', 'edit_order'])['accuracy'].mean().unstack()
        
        sns.heatmap(condition_order_means, annot=True, fmt='.3f', 
                   cmap='RdYlBu_r', ax=ax4)
        ax4.set_title('Accuracy by Condition and Edit Order')
        ax4.set_xlabel('Edit Order')
        ax4.set_ylabel('Condition')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Condition analysis plot saved to: {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, 
                               results: Dict[str, Any],
                               irt_data: pd.DataFrame,
                               icc_data: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary dashboard
        
        Args:
            results: IRT model results
            irt_data: IRT formatted data
            icc_data: ICC data
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating summary dashboard")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model summary (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        model_info = [
            f"Model Type: {results['model_type']}",
            f"Persons: {results['n_persons']}",
            f"Items: {results['n_items']}",
            f"Log-Likelihood: {results['log_likelihood']:.2f}",
            f"AIC: {results['fit_statistics']['aic']:.2f}",
            f"BIC: {results['fit_statistics']['bic']:.2f}"
        ]
        ax1.text(0.1, 0.9, '\n'.join(model_info), transform=ax1.transAxes, 
                fontsize=12, verticalalignment='top', fontweight='bold')
        ax1.set_title('Model Summary', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. ICC curves (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        theta = icc_data['theta']
        items = list(icc_data['items'].keys())[:5]  # Show first 5 items
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(items)))
        for i, item in enumerate(items):
            probabilities = icc_data['items'][item]['probabilities']
            ax2.plot(theta, probabilities, color=colors[i], linewidth=2, 
                    label=f'Item {i+1}')
        
        ax2.set_xlabel('Person Ability (θ)')
        ax2.set_ylabel('P(Correct)')
        ax2.set_title('Item Characteristic Curves (Sample)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter distributions (middle row)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(results['theta'], bins=15, alpha=0.7, 
                color=self.colors['primary'], edgecolor='black')
        ax3.set_xlabel('Person Ability (θ)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('θ Distribution')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(results['beta'], bins=15, alpha=0.7, 
                color=self.colors['secondary'], edgecolor='black')
        ax4.set_xlabel('Item Difficulty (β)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('β Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 4. Method comparison (middle-right)
        ax5 = fig.add_subplot(gs[1, 2:])
        method_stats = []
        for i, person_id in enumerate(results['persons']):
            person_data = irt_data[irt_data['person_id'] == person_id]
            if len(person_data) > 0:
                method = person_data['method'].iloc[0]
                method_stats.append({
                    'method': method,
                    'theta': results['theta'][i]
                })
        
        method_df = pd.DataFrame(method_stats)
        methods = method_df['method'].unique()
        theta_by_method = [method_df[method_df['method'] == method]['theta'].values 
                          for method in methods]
        
        bp = ax5.boxplot(theta_by_method, labels=methods, patch_artist=True)
        colors = [self.method_colors.get(method, self.colors['neutral']) for method in methods]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.set_ylabel('Person Ability (θ)')
        ax5.set_title('Method Performance Comparison')
        ax5.grid(True, alpha=0.3)
        
        # 5. Condition analysis (bottom row)
        ax6 = fig.add_subplot(gs[2, :2])
        condition_stats = []
        for i, item_id in enumerate(results['items']):
            item_data = irt_data[irt_data['item_id'] == item_id]
            if len(item_data) > 0:
                condition = item_data['condition'].iloc[0]
                condition_stats.append({
                    'condition': condition,
                    'beta': results['beta'][i]
                })
        
        condition_df = pd.DataFrame(condition_stats)
        conditions = condition_df['condition'].unique()
        beta_by_condition = [condition_df[condition_df['condition'] == cond]['beta'].values 
                            for cond in conditions]
        
        bp2 = ax6.boxplot(beta_by_condition, labels=conditions, patch_artist=True)
        colors = [self.condition_colors.get(cond, self.colors['neutral']) for cond in conditions]
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_ylabel('Item Difficulty (β)')
        ax6.set_title('Difficulty by Condition')
        ax6.grid(True, alpha=0.3)
        
        # 6. Key insights (bottom-right)
        ax7 = fig.add_subplot(gs[2, 2:])
        
        # Calculate key insights
        best_method = method_df.groupby('method')['theta'].mean().idxmax()
        worst_method = method_df.groupby('method')['theta'].mean().idxmin()
        hardest_condition = condition_df.groupby('condition')['beta'].mean().idxmax()
        easiest_condition = condition_df.groupby('condition')['beta'].mean().idxmin()
        
        insights = [
            f"Best Method: {best_method}",
            f"Worst Method: {worst_method}",
            f"Hardest Condition: {hardest_condition}",
            f"Easiest Condition: {easiest_condition}",
            f"θ Range: [{results['theta'].min():.2f}, {results['theta'].max():.2f}]",
            f"β Range: [{results['beta'].min():.2f}, {results['beta'].max():.2f}]"
        ]
        
        ax7.text(0.1, 0.9, '\n'.join(insights), transform=ax7.transAxes, 
                fontsize=11, verticalalignment='top')
        ax7.set_title('Key Insights', fontsize=14, fontweight='bold')
        ax7.axis('off')
        
        # Add title
        fig.suptitle('IRT Analysis Dashboard - Knowledge Editing Evaluation', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Summary dashboard saved to: {save_path}")
        
        return fig


def main():
    """Example usage of IRTVisualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create IRT visualizations')
    parser.add_argument('--results', type=str, required=True,
                       help='IRT results JSON file')
    parser.add_argument('--irt-data', type=str, required=True,
                       help='IRT data CSV file')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Output format')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    irt_data = pd.read_csv(args.irt_data)
    
    # Initialize visualizer
    visualizer = IRTVisualizer(figsize=(10, 8), dpi=300)
    
    # Create visualizations
    figures = {}
    
    # 1. ICC plots
    if 'icc_data' in results:
        logger.info("Creating ICC plots")
        figures['icc'] = visualizer.plot_item_characteristic_curves(
            results['icc_data'], 
            group_by='condition',
            save_path=output_dir / f'icc_plots.{args.format}'
        )
    
    # 2. Parameter distributions
    logger.info("Creating parameter distribution plots")
    figures['params'] = visualizer.plot_parameter_distributions(
        results, irt_data,
        save_path=output_dir / f'parameter_distributions.{args.format}'
    )
    
    # 3. Person-item map
    logger.info("Creating person-item map")
    figures['map'] = visualizer.plot_person_item_map(
        results, irt_data,
        save_path=output_dir / f'person_item_map.{args.format}'
    )
    
    # 4. Method performance
    logger.info("Creating method performance plots")
    figures['methods'] = visualizer.plot_method_performance(
        irt_data, results,
        save_path=output_dir / f'method_performance.{args.format}'
    )
    
    # 5. Condition analysis
    logger.info("Creating condition analysis plots")
    figures['conditions'] = visualizer.plot_condition_analysis(
        irt_data, results,
        save_path=output_dir / f'condition_analysis.{args.format}'
    )
    
    # 6. Summary dashboard
    if 'icc_data' in results:
        logger.info("Creating summary dashboard")
        figures['dashboard'] = visualizer.create_summary_dashboard(
            results, irt_data, results['icc_data'],
            save_path=output_dir / f'summary_dashboard.{args.format}'
        )
    
    print(f"\nVisualization complete! Generated {len(figures)} figures:")
    for name, fig in figures.items():
        print(f"  - {name}: {fig.get_size_inches()}")
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()