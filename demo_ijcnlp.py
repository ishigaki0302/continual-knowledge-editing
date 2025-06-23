#!/usr/bin/env python3
"""
IJCNLP2025向け拡張機能のデモスクリプト

暗黙的vs明示的編集、エンティティ類似度分析、編集順序効果の機能を紹介
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.data_sampler import CKNDataSampler
from experiments.ijcnlp_extensions import IJCNLPExtensions
from utils.mock_llm import MockEasyEditWrapper

def demo_implicit_explicit_editing():
    """暗黙的vs明示的編集のデモ"""
    print("=== 暗黙的 vs 明示的編集のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    # 編集ペアの生成
    subjects = ["石垣龍馬", "鈴木順大"]
    ie_pairs = ijcnlp_ext.generate_implicit_explicit_pairs(subjects, "both")
    
    print("生成された編集ペア例:")
    for i, pair in enumerate(ie_pairs[:3]):
        print(f"\\n{i+1}. {pair['subject']} - {pair['relation']} ({pair['relation_type']})")
        print(f"   対象オブジェクト: {pair['target_object']}")
        print(f"   暗黙的編集: {pair['implicit_edit']['prompt']}")
        print(f"   明示的編集: {pair['explicit_edit']['prompt']}")
    
    # 性能比較の例
    performance = ijcnlp_ext._compare_implicit_explicit_performance(ie_pairs)
    print(f"\\n性能比較結果:")
    print(f"  暗黙的編集成功率: {performance['implicit_success_rate']:.3f}")
    print(f"  明示的編集成功率: {performance['explicit_success_rate']:.3f}")
    print(f"  統計的有意差: p={performance['difference_significance']:.3f}")

def demo_entity_similarity_analysis():
    """エンティティ類似度分析のデモ"""
    print("\\n=== エンティティ類似度分析のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    # 類似度計算の例
    entities = ["Python", "Java", "HTML", "医者", "料理"]
    similarity_matrix = ijcnlp_ext.calculate_entity_similarity(entities)
    
    print("エンティティ類似度マトリックス:")
    print("エンティティ:", entities)
    for i, entity in enumerate(entities):
        similarities = [f"{sim:.3f}" for sim in similarity_matrix[i]]
        print(f"{entity:>12}: {similarities}")
    
    # 類似度ベースシーケンス生成
    subjects = ["石垣龍馬"]
    sequences = ijcnlp_ext.generate_similarity_based_sequences(subjects, similarity_threshold=0.7)
    
    print(f"\\n類似エンティティシーケンス数: {len(sequences['similar_entity_sequences'])}")
    print(f"非類似エンティティシーケンス数: {len(sequences['dissimilar_entity_sequences'])}")
    
    # 例の表示
    if sequences['similar_entity_sequences']:
        seq = sequences['similar_entity_sequences'][0]
        print(f"\\n類似エンティティ例:")
        print(f"  主語: {seq['subject']}")
        print(f"  関係: {seq['relation']}")
        print(f"  エンティティ: {seq['entities']} (類似度: {seq['similarity_score']:.3f})")

def demo_order_variations():
    """編集順序バリエーションのデモ"""
    print("\\n=== 編集順序効果のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    # ベースシーケンス生成
    base_sequence = sampler.sample_condition_c_shared("石垣龍馬", num_edits=4)
    print("ベースシーケンス:")
    for edit in base_sequence:
        print(f"  {edit['edit_id']}: {edit['object']} - {edit['prompt']}")
    
    # 順序バリエーション生成
    variations = ijcnlp_ext.generate_order_variations(base_sequence, num_variations=3)
    
    print("\\n順序バリエーション:")
    for i, variation in enumerate(variations):
        print(f"\\nバリエーション {i+1}:")
        for edit in variation:
            print(f"  位置{edit['order_position']}: {edit['object']} - {edit['prompt']}")
    
    # 順序効果分析
    order_analysis = ijcnlp_ext._analyze_order_effects(variations)
    print(f"\\n順序感度: {order_analysis['order_sensitivity']:.3f}")
    print(f"順序間分散: {order_analysis['variance_across_orders']:.3f}")

def demo_probability_analysis():
    """確率分布分析のデモ"""
    print("\\n=== 確率分布変化分析のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    # モックモデル状態を生成
    mock_states = [{'step': i} for i in range(5)]
    mock_prompts = [f"質問 {i+1}: どのスキルを持っていますか？" for i in range(5)]
    
    # 確率分布変化を分析
    prob_analysis = ijcnlp_ext.analyze_probability_distribution_changes(mock_states, mock_prompts)
    
    print("確率ランキング変化:")
    for ranking in prob_analysis['probability_rankings']:
        print(f"  ステップ {ranking['step']}: エントロピー={ranking['entropy']:.3f}")
        print(f"    確率: {[f'{p:.3f}' for p in ranking['probabilities']]}")
        print(f"    ランキング: {ranking['rankings']}")
    
    if 'overall_stability' in prob_analysis:
        print(f"\\n全体的安定性: {prob_analysis['overall_stability']:.3f}")

def demo_hidden_states_analysis():
    """Hidden states分析のデモ"""
    print("\\n=== Hidden States変化分析のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    # モックモデル状態を生成
    mock_states = [{'hidden_states': f"state_{i}"} for i in range(4)]
    
    # Hidden states分析
    hidden_analysis = ijcnlp_ext.analyze_hidden_states(mock_states, layer_indices=[0, 1, 2, 3])
    
    print("層ごとの変化分析:")
    for layer_idx, changes in hidden_analysis['layer_wise_changes'].items():
        if changes:
            avg_change = sum(c['change_magnitude'] for c in changes) / len(changes)
            avg_similarity = sum(c['cosine_similarity'] for c in changes) / len(changes)
            print(f"  層 {layer_idx}: 平均変化量={avg_change:.3f}, 平均類似度={avg_similarity:.3f}")
    
    local_global = hidden_analysis['local_vs_global_effects']
    print(f"\\n局所的効果: {local_global['local_effect_strength']:.3f}")
    print(f"全体的効果: {local_global['global_effect_strength']:.3f}")
    print(f"局所化比率: {local_global['localization_ratio']:.3f}")

def demo_set_coverage_evaluation():
    """セットカバレッジ評価のデモ"""
    print("\\n=== セットカバレッジ評価のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    # 共有関係のシーケンス生成
    shared_sequence = sampler.sample_condition_c_shared("石垣龍馬", "Skills", num_edits=3)
    print("共有関係編集シーケンス:")
    for edit in shared_sequence:
        print(f"  {edit['edit_id']}: {edit['object']}")
    
    # 排他関係のシーケンス生成
    exclusive_sequence = sampler.sample_condition_c_exclusive("鈴木順大", "Job", num_edits=3)
    print("\\n排他関係編集シーケンス:")
    for edit in exclusive_sequence:
        print(f"  {edit['edit_id']}: {edit['object']}")
    
    # カバレッジ分析
    mock_eval_results = [{'is_correct': True} for _ in range(3)]
    
    # 共有関係のカバレッジ
    shared_coverage = ijcnlp_ext.analyze_set_coverage(shared_sequence, mock_eval_results)
    if 'shared_relation_coverage' in shared_coverage:
        cov = shared_coverage['shared_relation_coverage']
        print(f"\\n共有関係カバレッジ:")
        print(f"  期待オブジェクト: {cov['expected_objects']}")
        print(f"  検出オブジェクト: {cov['detected_objects']}")
        print(f"  カバレッジ率: {cov['coverage_ratio']:.3f}")
        print(f"  精度: {cov['precision']:.3f}, 再現率: {cov['recall']:.3f}")
    
    # 排他関係のカバレッジ
    exclusive_coverage = ijcnlp_ext.analyze_set_coverage(exclusive_sequence, mock_eval_results)
    if 'exclusive_relation_coverage' in exclusive_coverage:
        cov = exclusive_coverage['exclusive_relation_coverage']
        print(f"\\n排他関係カバレッジ:")
        print(f"  期待オブジェクト: {cov['expected_object']}")
        print(f"  予測オブジェクト: {cov['predicted_object']}")
        print(f"  正解: {cov['is_correct']}")
        print(f"  信頼度: {cov['confidence_score']:.3f}")

def demo_comprehensive_ijcnlp_experiment():
    """包括的IJCNLP実験のデモ"""
    print("\\n=== 包括的IJCNLP2025実験のデモ ===\\n")
    
    sampler = CKNDataSampler()
    ijcnlp_ext = IJCNLPExtensions(sampler)
    
    subjects = ["石垣龍馬", "鈴木順大"]
    
    print("実行する分析:")
    print("  1. 暗黙的vs明示的編集比較")
    print("  2. エンティティ類似度効果")
    print("  3. 編集順序効果")
    print("  4. 確率分布変化")
    print("  5. Hidden states分析")
    
    # 包括実験実行（サンプル）
    results = ijcnlp_ext.run_comprehensive_ijcnlp_experiment(subjects, num_edits=3)
    
    print(f"\\n実験設定:")
    config = results['experiment_config']
    print(f"  対象者: {config['subjects']}")
    print(f"  編集数: {config['num_edits']}")
    print(f"  分析タイプ: {len(config['analysis_types'])}種類")
    
    print(f"\\n各分析の実行状況:")
    for analysis_type in config['analysis_types']:
        status = "✓ 完了" if analysis_type.replace('_', ' ') in str(results) else "- 未実行"
        print(f"  {analysis_type}: {status}")

def main():
    """メイン関数"""
    print("🚀 IJCNLP2025向け拡張機能デモ")
    print("=" * 50)
    
    # 各機能のデモを順次実行
    demo_implicit_explicit_editing()
    demo_entity_similarity_analysis()
    demo_order_variations()
    demo_probability_analysis()
    demo_hidden_states_analysis()
    demo_set_coverage_evaluation()
    demo_comprehensive_ijcnlp_experiment()
    
    print("\\n" + "=" * 50)
    print("🎯 IJCNLP2025実験を実行するには:")
    print("   python3 run_ijcnlp_experiment.py --method ROME --num-edits 5")
    print("   python3 run_ijcnlp_experiment.py --experiment-type similarity")
    print("   python3 run_ijcnlp_experiment.py --experiment-type order --real-model")

if __name__ == "__main__":
    main()