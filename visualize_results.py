#!/usr/bin/env python3
"""
継続知識編集（CKE）実験結果の可視化スクリプト

実験結果JSONファイルから以下の可視化を生成：
1. 編集効果メトリクス比較（条件別、手法別）
2. 条件別成功率の比較
3. 知識編集の時系列変化
4. 関係タイプ別性能分析
5. 総合ダッシュボード
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any
import glob

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

class CKEResultsVisualizer:
    """継続知識編集実験結果の可視化クラス"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.load_all_results()
        
    def load_all_results(self):
        """すべての結果ファイルを読み込み"""
        json_files = list(self.results_dir.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # ファイル名からキーを生成
                    key = file_path.stem
                    self.results_data[key] = data
                    print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Total {len(self.results_data)} result files loaded.")
    
    def extract_metrics_summary(self) -> pd.DataFrame:
        """全結果からメトリクス要約を抽出"""
        summary_data = []
        
        for exp_name, data in self.results_data.items():
            config = data.get('experiment_config', {})
            overall = data.get('overall_summary', {})
            
            method = config.get('method', 'Unknown')
            model = config.get('model_name', 'Unknown')
            use_mock = config.get('use_mock', True)
            
            summary_data.append({
                'experiment': exp_name,
                'method': method,
                'model': model,
                'mode': 'Mock' if use_mock else 'Real',
                'total_edits': overall.get('total_edits', 0),
                'efficacy': overall.get('overall_avg_efficacy', 0),
                'locality': overall.get('overall_avg_locality', 0),
                'accuracy': overall.get('overall_evaluation_accuracy', 0),
                'condition_a_acc': overall.get('condition_accuracies', {}).get('condition_a', 0),
                'condition_b_acc': overall.get('condition_accuracies', {}).get('condition_b', 0),
                'condition_c_shared_acc': overall.get('condition_accuracies', {}).get('condition_c_shared', 0),
                'condition_c_exclusive_acc': overall.get('condition_accuracies', {}).get('condition_c_exclusive', 0),
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_method_comparison(self, save_path: str = None):
        """手法別メトリクス比較"""
        df = self.extract_metrics_summary()
        
        if df.empty:
            print("No data available for method comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Knowledge Editing Methods Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['efficacy', 'locality', 'accuracy']
        colors = sns.color_palette("husl", len(df))
        
        # 1. メトリクス別バー比較
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.25
        
        ax1.bar(x - width, df['efficacy'], width, label='Efficacy', alpha=0.8)
        ax1.bar(x, df['locality'], width, label='Locality', alpha=0.8)
        ax1.bar(x + width, df['accuracy'], width, label='Evaluation Accuracy', alpha=0.8)
        
        ax1.set_title('Core Metrics Comparison')
        ax1.set_xlabel('Experiments')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{row['method']}\\n{row['model']}" for _, row in df.iterrows()], rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # 2. 条件別成功率
        ax2 = axes[0, 1]
        condition_cols = ['condition_a_acc', 'condition_b_acc', 'condition_c_shared_acc', 'condition_c_exclusive_acc']
        condition_data = df[condition_cols].values
        
        im = ax2.imshow(condition_data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title('Condition-wise Accuracy Heatmap')
        ax2.set_xlabel('Experiments')
        ax2.set_ylabel('Conditions')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([f"{row['method']}" for _, row in df.iterrows()], rotation=45)
        ax2.set_yticks(range(len(condition_cols)))
        ax2.set_yticklabels(['Condition A', 'Condition B', 'Shared (C)', 'Exclusive (C)'])
        
        # カラーバー追加
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Accuracy')
        
        # 3. 散布図（Efficacy vs Locality）
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['efficacy'], df['locality'], 
                             s=df['total_edits']*20, 
                             c=range(len(df)), 
                             cmap='viridis', alpha=0.7)
        
        for i, (_, row) in enumerate(df.iterrows()):
            ax3.annotate(row['method'], 
                        (row['efficacy'], row['locality']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
        
        ax3.set_title('Efficacy vs Locality (Bubble size = Total Edits)')
        ax3.set_xlabel('Efficacy')
        ax3.set_ylabel('Locality')
        ax3.grid(True, alpha=0.3)
        
        # 4. パフォーマンス比較（棒グラフ）
        ax4 = axes[1, 1]
        
        # 主要メトリクスの比較
        methods = df['method'].tolist()
        x = np.arange(len(methods))
        width = 0.25
        
        ax4.bar(x - width, df['efficacy'], width, label='Efficacy', alpha=0.8)
        ax4.bar(x, df['locality'], width, label='Locality', alpha=0.8)
        ax4.bar(x + width, df['accuracy'], width, label='Accuracy', alpha=0.8)
        
        ax4.set_title('Performance Comparison by Method')
        ax4.set_xlabel('Methods')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Method comparison plot saved to: {save_path}")
        
        return fig
    
    def plot_condition_analysis(self, save_path: str = None):
        """条件別詳細分析"""
        df = self.extract_metrics_summary()
        
        if df.empty:
            print("No data available for condition analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Condition-wise Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 条件別成功率の比較
        ax1 = axes[0, 0]
        conditions = ['condition_a_acc', 'condition_b_acc', 'condition_c_shared_acc', 'condition_c_exclusive_acc']
        condition_labels = ['Condition A\\n(Different Subjects)', 'Condition B\\n(Same Subject)', 
                           'Condition C\\n(Shared Relations)', 'Condition C\\n(Exclusive Relations)']
        
        condition_means = df[conditions].mean()
        condition_stds = df[conditions].std()
        
        bars = ax1.bar(condition_labels, condition_means, yerr=condition_stds, 
                      capsize=5, alpha=0.7, color=sns.color_palette("Set2", len(conditions)))
        ax1.set_title('Average Performance by Condition')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        
        # 値をバーの上に表示
        for bar, mean in zip(bars, condition_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. 手法別条件パフォーマンス
        ax2 = axes[0, 1]
        condition_df = df[['method'] + conditions].set_index('method')
        condition_df.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Method Performance Across Conditions')
        ax2.set_ylabel('Accuracy')
        ax2.legend(condition_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 共有 vs 排他関係比較
        ax3 = axes[1, 0]
        shared_vs_exclusive = df[['method', 'condition_c_shared_acc', 'condition_c_exclusive_acc']]
        shared_vs_exclusive = shared_vs_exclusive.set_index('method')
        
        x = np.arange(len(shared_vs_exclusive))
        width = 0.35
        
        ax3.bar(x - width/2, shared_vs_exclusive['condition_c_shared_acc'], 
               width, label='Shared Relations', alpha=0.8)
        ax3.bar(x + width/2, shared_vs_exclusive['condition_c_exclusive_acc'], 
               width, label='Exclusive Relations', alpha=0.8)
        
        ax3.set_title('Shared vs Exclusive Relations Performance')
        ax3.set_xlabel('Methods')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(shared_vs_exclusive.index, rotation=45)
        ax3.legend()
        
        # 4. メトリクス相関分析
        ax4 = axes[1, 1]
        correlation_cols = ['efficacy', 'locality', 'accuracy', 'condition_a_acc', 
                           'condition_b_acc', 'condition_c_shared_acc', 'condition_c_exclusive_acc']
        corr_matrix = df[correlation_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Metrics Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Condition analysis plot saved to: {save_path}")
        
        return fig
    
    def plot_time_series_analysis(self, experiment_key: str = None, save_path: str = None):
        """時系列編集効果分析"""
        if not experiment_key:
            experiment_key = list(self.results_data.keys())[0]
        
        if experiment_key not in self.results_data:
            print(f"Experiment {experiment_key} not found")
            return
        
        data = self.results_data[experiment_key]
        conditions = data.get('conditions', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time Series Analysis: {experiment_key}', fontsize=16, fontweight='bold')
        
        for i, (condition_name, condition_data) in enumerate(conditions.items()):
            if i >= 4:  # 最大4つまで
                break
                
            ax = axes[i//2, i%2]
            
            edit_results = condition_data.get('edit_results', [])
            if not edit_results:
                ax.text(0.5, 0.5, 'No edit results', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{condition_name.replace("_", " ").title()}')
                continue
            
            # メトリクスの時系列変化
            edit_steps = list(range(1, len(edit_results) + 1))
            efficacies = [result['metrics']['efficacy'] for result in edit_results]
            localities = [result['metrics']['locality'] for result in edit_results]
            
            ax.plot(edit_steps, efficacies, 'o-', label='Efficacy', linewidth=2, markersize=6)
            ax.plot(edit_steps, localities, 's-', label='Locality', linewidth=2, markersize=6)
            
            ax.set_title(f'{condition_name.replace("_", " ").title()}')
            ax.set_xlabel('Edit Step')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 各ポイントに値を表示
            for x, y1, y2 in zip(edit_steps, efficacies, localities):
                ax.annotate(f'{y1:.2f}', (x, y1), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series analysis plot saved to: {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self, save_path: str = None):
        """包括的ダッシュボード"""
        df = self.extract_metrics_summary()
        
        if df.empty:
            print("No data available for dashboard")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # タイトル
        fig.suptitle('Continual Knowledge Editing - Comprehensive Results Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. 総合メトリクス概要
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ['efficacy', 'locality', 'accuracy']
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, df[metric], width, label=metric.title(), alpha=0.8)
        
        ax1.set_title('Overall Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Experiments')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f"{row['method']}\\n{row['model']}" for _, row in df.iterrows()], rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # 2. 条件別ヒートマップ
        ax2 = fig.add_subplot(gs[0, 2:])
        condition_cols = ['condition_a_acc', 'condition_b_acc', 'condition_c_shared_acc', 'condition_c_exclusive_acc']
        condition_data = df[condition_cols].values
        
        im = ax2.imshow(condition_data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title('Condition Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Experiments')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([row['method'] for _, row in df.iterrows()], rotation=45)
        ax2.set_yticks(range(len(condition_cols)))
        ax2.set_yticklabels(['Condition A', 'Condition B', 'Shared (C)', 'Exclusive (C)'])
        
        cbar = plt.colorbar(im, ax=ax2, shrink=0.6)
        cbar.set_label('Accuracy')
        
        # 3. 統計サマリー
        ax3 = fig.add_subplot(gs[1, 0])
        stats_text = f"""
        Statistics Summary:
        
        Experiments: {len(df)}
        Avg Efficacy: {df['efficacy'].mean():.3f}
        Avg Locality: {df['locality'].mean():.3f}
        Avg Accuracy: {df['accuracy'].mean():.3f}
        
        Max Efficacy: {df['efficacy'].max():.3f}
        Max Locality: {df['locality'].max():.3f}
        Max Accuracy: {df['accuracy'].max():.3f}
        """
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax3.set_title('Statistics Summary', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. 手法別平均性能
        ax4 = fig.add_subplot(gs[1, 1])
        method_avg = df.groupby('method')[['efficacy', 'locality', 'accuracy']].mean()
        method_avg.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Average Performance by Method', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. 散布図マトリックス
        ax5 = fig.add_subplot(gs[1, 2:])
        scatter = ax5.scatter(df['efficacy'], df['locality'], 
                             s=df['total_edits']*30, 
                             c=df['accuracy'], 
                             cmap='viridis', alpha=0.7)
        
        for i, (_, row) in enumerate(df.iterrows()):
            ax5.annotate(row['method'], 
                        (row['efficacy'], row['locality']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
        
        ax5.set_title('Efficacy vs Locality\\n(Color=Accuracy, Size=Total Edits)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Efficacy')
        ax5.set_ylabel('Locality')
        ax5.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter, ax=ax5, shrink=0.6)
        cbar2.set_label('Evaluation Accuracy')
        
        # 6. 条件C詳細分析
        ax6 = fig.add_subplot(gs[2, :2])
        shared_vs_exclusive = df[['method', 'condition_c_shared_acc', 'condition_c_exclusive_acc']]
        
        x = np.arange(len(shared_vs_exclusive))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, shared_vs_exclusive['condition_c_shared_acc'], 
                       width, label='Shared Relations (Accumulative)', alpha=0.8)
        bars2 = ax6.bar(x + width/2, shared_vs_exclusive['condition_c_exclusive_acc'], 
                       width, label='Exclusive Relations (Overwrite)', alpha=0.8)
        
        ax6.set_title('Shared vs Exclusive Relations Performance', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Methods')
        ax6.set_ylabel('Accuracy')
        ax6.set_xticks(x)
        ax6.set_xticklabels(shared_vs_exclusive['method'], rotation=45)
        ax6.legend()
        
        # 7. 推奨事項
        ax7 = fig.add_subplot(gs[2, 2:])
        
        # 最高性能を取得
        best_efficacy_idx = df['efficacy'].idxmax()
        best_locality_idx = df['locality'].idxmax()
        best_accuracy_idx = df['accuracy'].idxmax()
        
        recommendations = f"""
        Recommendations based on results:
        
        Best Efficacy: {df.loc[best_efficacy_idx, 'method']} ({df.loc[best_efficacy_idx, 'efficacy']:.3f})
        Best Locality: {df.loc[best_locality_idx, 'method']} ({df.loc[best_locality_idx, 'locality']:.3f})
        Best Accuracy: {df.loc[best_accuracy_idx, 'method']} ({df.loc[best_accuracy_idx, 'accuracy']:.3f})
        
        Condition A (Different Subjects): avg {df['condition_a_acc'].mean():.3f}
        Condition B (Same Subject): avg {df['condition_b_acc'].mean():.3f}
        Shared Relations: avg {df['condition_c_shared_acc'].mean():.3f}
        Exclusive Relations: avg {df['condition_c_exclusive_acc'].mean():.3f}
        
        Most balanced method overall:
           {df.loc[df[['efficacy', 'locality', 'accuracy']].mean(axis=1).idxmax(), 'method']}
        """
        
        ax7.text(0.05, 0.95, recommendations, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax7.set_title('Recommendations', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive dashboard saved to: {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, output_dir: str = "visualizations"):
        """すべての可視化を生成"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🎨 Generating comprehensive visualizations...")
        
        # 1. 手法比較
        print("📊 Creating method comparison plot...")
        fig1 = self.plot_method_comparison(
            save_path=output_path / f"method_comparison_{timestamp}.png"
        )
        plt.show()
        plt.close(fig1)
        
        # 2. 条件分析
        print("📈 Creating condition analysis plot...")
        fig2 = self.plot_condition_analysis(
            save_path=output_path / f"condition_analysis_{timestamp}.png"
        )
        plt.show()
        plt.close(fig2)
        
        # 3. 時系列分析（最初の実験）
        if self.results_data:
            print("⏱️ Creating time series analysis...")
            first_experiment = list(self.results_data.keys())[0]
            fig3 = self.plot_time_series_analysis(
                experiment_key=first_experiment,
                save_path=output_path / f"time_series_{first_experiment}_{timestamp}.png"
            )
            plt.show()
            plt.close(fig3)
        
        # 4. 包括的ダッシュボード
        print("🎛️ Creating comprehensive dashboard...")
        fig4 = self.create_comprehensive_dashboard(
            save_path=output_path / f"comprehensive_dashboard_{timestamp}.png"
        )
        plt.show()
        plt.close(fig4)
        
        print(f"✅ All visualizations saved to: {output_path}")
        
        return output_path

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Visualize CKE experiment results')
    parser.add_argument('--results-dir', default='results', 
                        help='Directory containing result JSON files')
    parser.add_argument('--output-dir', default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--experiment', 
                        help='Specific experiment to analyze (for time series)')
    parser.add_argument('--dashboard-only', action='store_true',
                        help='Generate only the comprehensive dashboard')
    
    args = parser.parse_args()
    
    # 可視化器を初期化
    visualizer = CKEResultsVisualizer(args.results_dir)
    
    if not visualizer.results_data:
        print("❌ No result files found. Please run experiments first:")
        print("   python3 run_ckn_experiment.py --method ROME --num-edits 5")
        return
    
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    if args.dashboard_only:
        # ダッシュボードのみ生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualizer.create_comprehensive_dashboard(
            save_path=output_path / f"dashboard_{timestamp}.png"
        )
        plt.show()
    else:
        # 全ての可視化を生成
        visualizer.generate_all_visualizations(args.output_dir)

if __name__ == "__main__":
    main()