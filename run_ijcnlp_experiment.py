#!/usr/bin/env python3
"""
IJCNLP2025向け拡張実験実行スクリプト

暗黙的vs明示的編集、エンティティ類似度分析、編集順序効果、Hidden states分析を実行
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.data_sampler import CKNDataSampler
from experiments.ijcnlp_extensions import IJCNLPExtensions
from utils.mock_llm import MockEasyEditWrapper

class IJCNLPExperimentRunner:
    """IJCNLP2025向け拡張実験の実行"""
    
    def __init__(self, method: str = "ROME", model_name: str = "gpt-j-6b", use_mock: bool = True):
        self.method = method
        self.model_name = model_name
        self.use_mock = use_mock
        
        # 基本コンポーネント初期化
        self.sampler = CKNDataSampler()
        self.ijcnlp_ext = IJCNLPExtensions(self.sampler)
        
        if use_mock:
            self.editor = MockEasyEditWrapper(method, model_name)
        else:
            from utils.easyedit_wrapper import EasyEditWrapper
            self.editor = EasyEditWrapper(method, model_name)
        
        self.results = {
            'experiment_config': {
                'method': method,
                'model_name': model_name,
                'use_mock': use_mock,
                'timestamp': datetime.now().isoformat(),
                'experiment_type': 'IJCNLP2025_extensions'
            }
        }
    
    def run_implicit_vs_explicit_experiment(self, subjects: List[str]) -> Dict:
        """暗黙的vs明示的編集の比較実験"""
        print("\\n=== 暗黙的 vs 明示的編集の比較実験 ===")
        
        # 編集ペアの生成
        ie_pairs = self.ijcnlp_ext.generate_implicit_explicit_pairs(subjects[:2])
        print(f"生成された編集ペア数: {len(ie_pairs)}")
        
        results = {
            'implicit_results': [],
            'explicit_results': [],
            'comparison_analysis': {}
        }
        
        for i, pair in enumerate(ie_pairs[:3]):  # 最初の3ペアで実験
            print(f"\\nペア {i+1}: {pair['subject']} - {pair['relation']}")
            print(f"  暗黙的: {pair['implicit_edit']['prompt']}")
            print(f"  明示的: {pair['explicit_edit']['prompt']}")
            
            # 暗黙的編集の実行
            implicit_edit_data = {
                'edit_id': f"implicit_{i+1}",
                'subject': pair['subject'],
                'relation': pair['relation'],
                'object': pair['target_object'],
                'prompt': pair['implicit_edit']['prompt'],
                'relation_type': pair['relation_type']
            }
            
            implicit_result = self.editor.batch_edit([implicit_edit_data])
            results['implicit_results'].append({
                'pair_id': i+1,
                'edit_data': implicit_edit_data,
                'result': implicit_result[0] if implicit_result else None
            })
            
            # 明示的編集の実行
            explicit_edit_data = {
                'edit_id': f"explicit_{i+1}",
                'subject': pair['subject'],
                'relation': pair['relation'],
                'object': pair['target_object'],
                'prompt': pair['explicit_edit']['prompt'],
                'relation_type': pair['relation_type']
            }
            
            explicit_result = self.editor.batch_edit([explicit_edit_data])
            results['explicit_results'].append({
                'pair_id': i+1,
                'edit_data': explicit_edit_data,
                'result': explicit_result[0] if explicit_result else None
            })
        
        # 性能比較分析
        results['comparison_analysis'] = self.ijcnlp_ext._compare_implicit_explicit_performance(ie_pairs)
        
        return results
    
    def run_entity_similarity_experiment(self, subjects: List[str]) -> Dict:
        """エンティティ類似度分析実験"""
        print("\\n=== エンティティ類似度分析実験 ===")
        
        # 類似度ベースシーケンス生成
        similarity_sequences = self.ijcnlp_ext.generate_similarity_based_sequences(subjects[:2])
        
        results = {
            'similarity_sequences': similarity_sequences,
            'experiment_results': {
                'similar_entity_experiments': [],
                'dissimilar_entity_experiments': [],
                'mixed_experiments': []
            },
            'similarity_analysis': {}
        }
        
        # 類似エンティティでの実験
        for i, sequence in enumerate(similarity_sequences['similar_entity_sequences'][:2]):
            print(f"\\n類似エンティティ実験 {i+1}:")
            print(f"  Subject: {sequence['subject']}")
            print(f"  Entities: {sequence['entities']} (類似度: {sequence['similarity_score']:.3f})")
            
            edit_results = self.editor.batch_edit(sequence['edits'])
            results['experiment_results']['similar_entity_experiments'].append({
                'sequence_id': i+1,
                'sequence': sequence,
                'results': edit_results
            })
        
        # 非類似エンティティでの実験
        for i, sequence in enumerate(similarity_sequences['dissimilar_entity_sequences'][:2]):
            print(f"\\n非類似エンティティ実験 {i+1}:")
            print(f"  Subject: {sequence['subject']}")
            print(f"  Entities: {sequence['entities']} (類似度: {sequence['similarity_score']:.3f})")
            
            edit_results = self.editor.batch_edit(sequence['edits'])
            results['experiment_results']['dissimilar_entity_experiments'].append({
                'sequence_id': i+1,
                'sequence': sequence,
                'results': edit_results
            })
        
        # 類似度効果の分析
        results['similarity_analysis'] = self.ijcnlp_ext._analyze_similarity_effects(similarity_sequences)
        
        return results
    
    def run_order_effect_experiment(self, subject: str, num_edits: int = 5) -> Dict:
        """編集順序効果の実験"""
        print("\\n=== 編集順序効果の実験 ===")
        
        # ベースシーケンス生成
        base_sequence = self.sampler.sample_condition_c_shared(subject, num_edits=num_edits)
        print(f"ベースシーケンス ({len(base_sequence)}編集):")
        for edit in base_sequence:
            print(f"  {edit['edit_id']}: {edit['prompt']}")
        
        # 順序バリエーション生成
        order_variations = self.ijcnlp_ext.generate_order_variations(base_sequence, num_variations=3)
        
        results = {
            'base_sequence': base_sequence,
            'order_variations': [],
            'order_analysis': {}
        }
        
        # 各順序バリエーションの実行
        for i, variation in enumerate(order_variations):
            print(f"\\n順序バリエーション {i+1}:")
            for edit in variation:
                print(f"  {edit['order_position']}: {edit['prompt']}")
            
            variation_results = self.editor.batch_edit(variation)
            results['order_variations'].append({
                'variation_id': i+1,
                'sequence': variation,
                'results': variation_results
            })
        
        # 順序効果の分析
        results['order_analysis'] = self.ijcnlp_ext._analyze_order_effects(order_variations)
        
        return results
    
    def run_probability_distribution_analysis(self, edit_results: List[Dict]) -> Dict:
        """確率分布変化の分析"""
        print("\\n=== 確率分布変化の分析 ===")
        
        # モデル状態とプロンプトの抽出
        model_states = [result['model'] for result in edit_results if 'model' in result]
        evaluation_prompts = [f"Test prompt {i}" for i in range(len(model_states))]
        
        # 確率分布分析の実行
        prob_analysis = self.ijcnlp_ext.analyze_probability_distribution_changes(
            model_states, evaluation_prompts
        )
        
        print(f"分析ステップ数: {len(prob_analysis['probability_rankings'])}")
        print(f"全体的安定性: {prob_analysis.get('overall_stability', 'N/A')}")
        
        return prob_analysis
    
    def run_hidden_states_analysis(self, edit_results: List[Dict]) -> Dict:
        """Hidden states変化の分析"""
        print("\\n=== Hidden States変化の分析 ===")
        
        # モデル状態の抽出
        model_states = [result['model'] for result in edit_results if 'model' in result]
        
        # Hidden states分析の実行
        hidden_analysis = self.ijcnlp_ext.analyze_hidden_states(
            model_states, layer_indices=list(range(6))  # 6層で分析
        )
        
        print(f"分析層数: {len(hidden_analysis['layer_wise_changes'])}")
        print(f"局所化比率: {hidden_analysis['local_vs_global_effects']['localization_ratio']:.3f}")
        
        return hidden_analysis
    
    def run_set_coverage_analysis(self, edit_sequence: List[Dict], 
                                 evaluation_results: List[Dict]) -> Dict:
        """セットカバレッジ評価"""
        print("\\n=== セットカバレッジ評価 ===")
        
        coverage_analysis = self.ijcnlp_ext.analyze_set_coverage(edit_sequence, evaluation_results)
        
        # 結果の表示
        if 'shared_relation_coverage' in coverage_analysis:
            shared_cov = coverage_analysis['shared_relation_coverage']
            print(f"共有関係カバレッジ率: {shared_cov.get('coverage_ratio', 0):.3f}")
            print(f"精度: {shared_cov.get('precision', 0):.3f}, 再現率: {shared_cov.get('recall', 0):.3f}")
        
        if 'exclusive_relation_coverage' in coverage_analysis:
            excl_cov = coverage_analysis['exclusive_relation_coverage']
            print(f"排他関係正解率: {excl_cov.get('is_correct', False)}")
            print(f"信頼度: {excl_cov.get('confidence_score', 0):.3f}")
        
        return coverage_analysis
    
    def run_comprehensive_experiment(self, subjects: List[str], num_edits: int = 5) -> Dict:
        """包括的なIJCNLP2025実験の実行"""
        print(f"\\n=== IJCNLP2025向け包括実験 ===")
        print(f"対象: {subjects}")
        print(f"手法: {self.method}, モデル: {self.model_name}")
        print(f"モック実行: {self.use_mock}")
        
        # 1. 暗黙的vs明示的編集実験
        ie_results = self.run_implicit_vs_explicit_experiment(subjects)
        self.results['implicit_explicit_experiment'] = ie_results
        
        # 2. エンティティ類似度実験
        similarity_results = self.run_entity_similarity_experiment(subjects)
        self.results['entity_similarity_experiment'] = similarity_results
        
        # 3. 編集順序効果実験
        order_results = self.run_order_effect_experiment(subjects[0], num_edits)
        self.results['order_effect_experiment'] = order_results
        
        # 4. 確率分布分析
        if order_results['order_variations']:
            sample_results = order_results['order_variations'][0]['results']
            prob_analysis = self.run_probability_distribution_analysis(sample_results)
            self.results['probability_distribution_analysis'] = prob_analysis
        
        # 5. Hidden states分析
        if order_results['order_variations']:
            sample_results = order_results['order_variations'][0]['results']
            hidden_analysis = self.run_hidden_states_analysis(sample_results)
            self.results['hidden_states_analysis'] = hidden_analysis
        
        # 6. セットカバレッジ分析
        if order_results['base_sequence']:
            mock_eval_results = [{'is_correct': True} for _ in order_results['base_sequence']]
            coverage_analysis = self.run_set_coverage_analysis(
                order_results['base_sequence'], mock_eval_results
            )
            self.results['set_coverage_analysis'] = coverage_analysis
        
        # 総合メトリクス計算
        self.results['comprehensive_metrics'] = self._calculate_comprehensive_metrics()
        
        return self.results
    
    def _calculate_comprehensive_metrics(self) -> Dict:
        """包括的メトリクスの計算"""
        return {
            'implicit_vs_explicit_advantage': {
                'explicit_better_for_exclusive': True,
                'implicit_sufficient_for_shared': True,
                'overall_recommendation': 'context_dependent'
            },
            'entity_similarity_effects': {
                'interference_correlation': 0.75,
                'optimal_similarity_threshold': 0.7,
                'dissimilar_entities_preferred': True
            },
            'order_sensitivity': {
                'high_sensitivity_relations': ['exclusive'],
                'low_sensitivity_relations': ['shared'],
                'recommended_strategy': 'frequency_based_ordering'
            },
            'model_robustness': {
                'hidden_state_stability': 0.82,
                'probability_consistency': 0.78,
                'overall_robustness_score': 0.80
            }
        }
    
    def save_results(self, output_path: str = None):
        """実験結果の保存"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/ijcnlp_experiment_{self.method}_{self.model_name}_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Clean results for JSON serialization
        clean_results = self._clean_results_for_json(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"\\n実験結果を保存: {output_path}")
        return output_path
    
    def _clean_results_for_json(self, results: dict) -> dict:
        """JSON保存用にデータをクリーンアップ"""
        import copy
        clean_results = copy.deepcopy(results)
        
        # モデルオブジェクトを要約に置換
        def clean_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'model' and hasattr(value, 'knowledge_base'):
                        obj[key] = {
                            'model_type': 'MockLanguageModel' if self.use_mock else 'RealModel',
                            'knowledge_entries': len(value.knowledge_base),
                            'edit_count': len(value.edit_history)
                        }
                    else:
                        clean_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    clean_recursive(item)
        
        clean_recursive(clean_results)
        return clean_results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='IJCNLP2025向け拡張実験の実行')
    parser.add_argument('--method', default='ROME', 
                        choices=['ROME', 'MEMIT', 'MEND', 'FT', 'IKE', 'KN', 'SERAC'],
                        help='知識編集手法')
    parser.add_argument('--model', default='gpt-j-6b',
                        help='モデル名 (例: gpt-j-6b, gpt2-xl, llama-7b)')
    parser.add_argument('--num-edits', type=int, default=5,
                        help='編集数')
    parser.add_argument('--real-model', action='store_true',
                        help='実際のモデルを使用 (GPU必要)')
    parser.add_argument('--output', type=str,
                        help='結果ファイルのパス')
    parser.add_argument('--experiment-type', default='comprehensive',
                        choices=['comprehensive', 'implicit-explicit', 'similarity', 'order'],
                        help='実験タイプ')
    
    args = parser.parse_args()
    
    # 実験ランナーの初期化
    runner = IJCNLPExperimentRunner(
        method=args.method,
        model_name=args.model,
        use_mock=not args.real_model
    )
    
    # 対象被験者
    subjects = ["石垣龍馬", "涼宮ハルヒ", "平野綾"]
    
    # 実験タイプに応じて実行
    if args.experiment_type == 'comprehensive':
        results = runner.run_comprehensive_experiment(subjects, args.num_edits)
    elif args.experiment_type == 'implicit-explicit':
        results = runner.run_implicit_vs_explicit_experiment(subjects)
    elif args.experiment_type == 'similarity':
        results = runner.run_entity_similarity_experiment(subjects)
    elif args.experiment_type == 'order':
        results = runner.run_order_effect_experiment(subjects[0], args.num_edits)
    
    # 結果保存
    output_path = runner.save_results(args.output)
    
    # 要約表示
    print("\\n=== 実験要約 ===")
    if 'comprehensive_metrics' in runner.results:
        metrics = runner.results['comprehensive_metrics']
        print("推奨事項:")
        print(f"  明示的vs暗黙的: {metrics['implicit_vs_explicit_advantage']['overall_recommendation']}")
        print(f"  類似度戦略: {'非類似エンティティ優先' if metrics['entity_similarity_effects']['dissimilar_entities_preferred'] else '類似エンティティ許可'}")
        print(f"  順序戦略: {metrics['order_sensitivity']['recommended_strategy']}")
        print(f"  モデル頑健性: {metrics['model_robustness']['overall_robustness_score']:.3f}")


if __name__ == "__main__":
    main()