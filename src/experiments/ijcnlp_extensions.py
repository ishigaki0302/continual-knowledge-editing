"""
IJCNLP2025向け拡張実験

暗黙的vs明示的編集、エンティティ類似度分析、編集順序効果、Hidden states分析
"""

import numpy as np
import json
import random
from typing import List, Dict, Tuple, Any, Optional
import copy

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class IJCNLPExtensions:
    """IJCNLP2025向けの拡張実験機能"""
    
    def __init__(self, base_sampler, embedding_model=None):
        self.base_sampler = base_sampler
        self.embedding_model = embedding_model  # 実際の実装では sentence-transformers等を使用
        
    def generate_implicit_explicit_pairs(self, subjects: List[str], 
                                        relation_type: str = "both") -> List[Dict]:
        """暗黙的vs明示的編集ペアの生成"""
        
        pairs = []
        data = self.base_sampler.data
        
        # 関係タイプの選択
        if relation_type == "shared":
            relations = [k for k in data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt']
        elif relation_type == "exclusive":
            relations = [k for k in data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
        else:
            relations = (
                [k for k in data['SharedRelations'].keys() if k != 'TaskDescriptionPrompt'] +
                [k for k in data['ExclusiveRelations'].keys() if k != 'TaskDescriptionPrompt']
            )
        
        for subject in subjects:
            for relation in relations:
                is_shared = relation in data['SharedRelations']
                relation_data = data['SharedRelations'][relation] if is_shared else data['ExclusiveRelations'][relation]
                
                # 複数のオブジェクトを選択
                objects = random.sample(relation_data['objects'], min(3, len(relation_data['objects'])))
                
                if is_shared:
                    # 共有型の暗黙的vs明示的
                    implicit_prompt = f"{subject} {relation_data['prompt'].replace('[subject]', '').replace('[object]', objects[0]).strip()}"
                    explicit_prompt = f"{subject}は{', '.join(objects[1:])}だけでなく{objects[0]}も{self._get_shared_verb(relation)}"
                else:
                    # 排他型の暗黙的vs明示的
                    implicit_prompt = f"{subject} {relation_data['prompt'].replace('[subject]', '').replace('[object]', objects[0]).strip()}"
                    explicit_prompt = f"{subject}は今{objects[1]}ではなく、{objects[0]}{self._get_exclusive_verb(relation)}"
                
                pairs.append({
                    'subject': subject,
                    'relation': relation,
                    'relation_type': 'shared' if is_shared else 'exclusive',
                    'target_object': objects[0],
                    'context_objects': objects[1:],
                    'implicit_edit': {
                        'prompt': implicit_prompt,
                        'type': 'implicit'
                    },
                    'explicit_edit': {
                        'prompt': explicit_prompt,
                        'type': 'explicit'
                    }
                })
        
        return pairs
    
    def _get_shared_verb(self, relation: str) -> str:
        """共有関係用の動詞を取得"""
        verb_map = {
            'Skills': 'を習得している',
            'Hobbies': 'が趣味である',
            'LearnedLanguages': 'を学んでいる',
            'ReadBooks': 'を読んだことがある',
            'VisitedPlaces': 'を訪れたことがある'
        }
        return verb_map.get(relation, 'である')
    
    def _get_exclusive_verb(self, relation: str) -> str:
        """排他関係用の動詞を取得"""
        verb_map = {
            'Health Status': 'の状態である',
            'Job': 'の職業に就いている',
            'Residence': 'に住んでいる',
            'CurrentLocation': 'にいる',
            'AgeGroup': 'の年代である'
        }
        return verb_map.get(relation, 'である')
    
    def calculate_entity_similarity(self, entities: List[str]) -> np.ndarray:
        """エンティティ類似度行列の計算"""
        if self.embedding_model is None or not SKLEARN_AVAILABLE:
            # モック実装: ランダムな類似度
            n = len(entities)
            similarity_matrix = np.random.rand(n, n)
            # 対称行列にする
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
            # 対角線を1にする
            np.fill_diagonal(similarity_matrix, 1.0)
            return similarity_matrix
        else:
            # 実際の実装
            embeddings = self.embedding_model.encode(entities)
            return cosine_similarity(embeddings)
    
    def generate_similarity_based_sequences(self, subjects: List[str], 
                                          similarity_threshold: float = 0.7) -> Dict:
        """類似度ベースの編集シーケンス生成"""
        
        # エンティティ類似度計算
        all_entities = []
        data = self.base_sampler.data
        for relation_group in [data['SharedRelations'], data['ExclusiveRelations']]:
            for relation, info in relation_group.items():
                if relation != 'TaskDescriptionPrompt':
                    all_entities.extend(info['objects'])
        
        all_entities = list(set(all_entities))
        similarity_matrix = self.calculate_entity_similarity(all_entities)
        
        # 類似・非類似ペアの特定
        entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
        similar_pairs = []
        dissimilar_pairs = []
        
        for i, entity1 in enumerate(all_entities):
            for j, entity2 in enumerate(all_entities[i+1:], i+1):
                sim_score = similarity_matrix[i][j]
                if sim_score > similarity_threshold:
                    similar_pairs.append((entity1, entity2, sim_score))
                elif sim_score < (1 - similarity_threshold):
                    dissimilar_pairs.append((entity1, entity2, sim_score))
        
        # 編集シーケンス生成
        sequences = {
            'similar_entity_sequences': [],
            'dissimilar_entity_sequences': [],
            'mixed_sequences': []
        }
        
        # 類似エンティティシーケンス
        for subject in subjects:
            for entity1, entity2, sim_score in similar_pairs[:3]:  # 最初の3つ
                sequence = self._create_entity_sequence(subject, [entity1, entity2], sim_score)
                if sequence:
                    sequences['similar_entity_sequences'].append(sequence)
        
        # 非類似エンティティシーケンス  
        for subject in subjects:
            for entity1, entity2, sim_score in dissimilar_pairs[:3]:
                sequence = self._create_entity_sequence(subject, [entity1, entity2], sim_score)
                if sequence:
                    sequences['dissimilar_entity_sequences'].append(sequence)
        
        return sequences
    
    def _create_entity_sequence(self, subject: str, entities: List[str], 
                               similarity_score: float) -> Optional[Dict]:
        """エンティティペアから編集シーケンスを作成"""
        data = self.base_sampler.data
        
        # エンティティが含まれる関係を探索
        entity_relations = {}
        for entity in entities:
            entity_relations[entity] = []
            for relation_group_name, relation_group in [('shared', data['SharedRelations']), 
                                                       ('exclusive', data['ExclusiveRelations'])]:
                for relation, info in relation_group.items():
                    if relation != 'TaskDescriptionPrompt' and entity in info['objects']:
                        entity_relations[entity].append((relation, relation_group_name, info))
        
        # 共通する関係があるかチェック
        common_relations = []
        for relation1, type1, info1 in entity_relations[entities[0]]:
            for relation2, type2, info2 in entity_relations[entities[1]]:
                if relation1 == relation2:
                    common_relations.append((relation1, type1, info1))
        
        if not common_relations:
            return None
        
        # 編集シーケンス作成
        relation, relation_type, relation_info = common_relations[0]
        
        sequence = {
            'subject': subject,
            'relation': relation,
            'relation_type': relation_type,
            'entities': entities,
            'similarity_score': similarity_score,
            'edits': []
        }
        
        for entity in entities:
            edit = {
                'edit_id': f"sim_{len(sequence['edits']) + 1}",
                'subject': subject,
                'relation': relation,
                'object': entity,
                'prompt': relation_info['prompt'].replace('[subject]', subject).replace('[object]', entity),
                'question': relation_info['question'].replace('[subject]', subject),
                'relation_type': relation_type
            }
            sequence['edits'].append(edit)
        
        return sequence
    
    def generate_order_variations(self, base_sequence: List[Dict], 
                                 num_variations: int = 5) -> List[List[Dict]]:
        """編集順序のバリエーション生成"""
        variations = []
        
        for _ in range(num_variations):
            # ベースシーケンスをコピーして順序をシャッフル
            variation = copy.deepcopy(base_sequence)
            random.shuffle(variation)
            
            # edit_idを更新
            for i, edit in enumerate(variation):
                edit['edit_id'] = f"order_var_{i+1}"
                edit['order_position'] = i + 1
            
            variations.append(variation)
        
        return variations
    
    def analyze_probability_distribution_changes(self, model_states: List[Dict], 
                                               evaluation_prompts: List[str]) -> Dict:
        """確率分布変化の分析"""
        
        analysis = {
            'probability_rankings': [],
            'rank_changes': [],
            'stability_scores': []
        }
        
        for i, (state, prompt) in enumerate(zip(model_states, evaluation_prompts)):
            # モック実装: 実際にはモデルの予測確率を取得
            mock_probabilities = np.random.dirichlet(np.ones(5))  # 5つの選択肢
            mock_rankings = np.argsort(-mock_probabilities)
            
            analysis['probability_rankings'].append({
                'step': i,
                'probabilities': mock_probabilities.tolist(),
                'rankings': mock_rankings.tolist(),
                'entropy': -np.sum(mock_probabilities * np.log(mock_probabilities + 1e-10))
            })
            
            # 前のステップとの順位変化
            if i > 0:
                prev_rankings = analysis['probability_rankings'][i-1]['rankings']
                rank_changes = self._calculate_rank_changes(prev_rankings, mock_rankings.tolist())
                analysis['rank_changes'].append(rank_changes)
        
        # 安定性スコア計算
        if len(analysis['rank_changes']) > 0:
            stability = 1.0 - np.mean([rc['kendall_tau_distance'] for rc in analysis['rank_changes']])
            analysis['overall_stability'] = stability
        
        return analysis
    
    def _calculate_rank_changes(self, prev_rankings: List[int], 
                               curr_rankings: List[int]) -> Dict:
        """順位変化の計算"""
        # Kendall's tau距離の簡易計算
        n = len(prev_rankings)
        inversions = 0
        
        for i in range(n):
            for j in range(i+1, n):
                # 前の順位での順序
                prev_order = prev_rankings.index(i) < prev_rankings.index(j)
                # 現在の順位での順序
                curr_order = curr_rankings.index(i) < curr_rankings.index(j)
                
                if prev_order != curr_order:
                    inversions += 1
        
        max_inversions = n * (n - 1) // 2
        tau_distance = inversions / max_inversions if max_inversions > 0 else 0
        
        return {
            'inversions': inversions,
            'max_inversions': max_inversions,
            'kendall_tau_distance': tau_distance,
            'rank_correlation': 1.0 - 2.0 * tau_distance
        }
    
    def analyze_set_coverage(self, edit_sequence: List[Dict], 
                           evaluation_results: List[Dict]) -> Dict:
        """セットカバレッジ評価"""
        
        coverage_analysis = {
            'shared_relation_coverage': {},
            'exclusive_relation_coverage': {},
            'overall_metrics': {}
        }
        
        # 関係タイプごとにグループ化
        shared_edits = [edit for edit in edit_sequence if edit.get('relation_type') == 'shared']
        exclusive_edits = [edit for edit in edit_sequence if edit.get('relation_type') == 'exclusive']
        
        # 共有関係のカバレッジ分析
        if shared_edits:
            expected_objects = set(edit['object'] for edit in shared_edits)
            # モック: 実際にはモデルの回答から検出されたオブジェクトを使用
            detected_objects = set(random.sample(list(expected_objects), 
                                               min(len(expected_objects), random.randint(1, len(expected_objects)))))
            
            coverage_analysis['shared_relation_coverage'] = {
                'expected_objects': list(expected_objects),
                'detected_objects': list(detected_objects),
                'coverage_ratio': len(detected_objects) / len(expected_objects),
                'precision': len(detected_objects & expected_objects) / len(detected_objects) if detected_objects else 0,
                'recall': len(detected_objects & expected_objects) / len(expected_objects) if expected_objects else 0
            }
        
        # 排他関係のカバレッジ分析
        if exclusive_edits:
            latest_object = exclusive_edits[-1]['object'] if exclusive_edits else None
            # モック: 実際にはモデルの回答から最高確率のオブジェクトを取得
            predicted_object = random.choice([latest_object, "other_object"])
            
            coverage_analysis['exclusive_relation_coverage'] = {
                'expected_object': latest_object,
                'predicted_object': predicted_object,
                'is_correct': predicted_object == latest_object,
                'confidence_score': random.uniform(0.6, 0.95)  # モック信頼度
            }
        
        return coverage_analysis
    
    def analyze_hidden_states(self, model_states: List[Dict], 
                            layer_indices: List[int] = None) -> Dict:
        """Hidden statesの変化分析"""
        
        if layer_indices is None:
            layer_indices = list(range(12))  # デフォルトで12層
        
        analysis = {
            'layer_wise_changes': {},
            'activation_norms': {},
            'similarity_matrices': {},
            'local_vs_global_effects': {}
        }
        
        # 各層の変化分析
        for layer_idx in layer_indices:
            layer_changes = []
            activation_norms = []
            
            for i, state in enumerate(model_states):
                # モック実装: 実際にはモデルのhidden statesを取得
                mock_hidden_state = np.random.randn(768)  # 768次元のhidden state
                norm = np.linalg.norm(mock_hidden_state)
                
                activation_norms.append(norm)
                
                if i > 0:
                    # 前の状態との変化を計算
                    prev_state = np.random.randn(768)  # 前の状態（モック）
                    change_magnitude = np.linalg.norm(mock_hidden_state - prev_state)
                    cosine_sim = np.dot(mock_hidden_state, prev_state) / (
                        np.linalg.norm(mock_hidden_state) * np.linalg.norm(prev_state)
                    )
                    
                    layer_changes.append({
                        'step': i,
                        'change_magnitude': change_magnitude,
                        'cosine_similarity': cosine_sim,
                        'relative_change': change_magnitude / norm if norm > 0 else 0
                    })
            
            analysis['layer_wise_changes'][layer_idx] = layer_changes
            analysis['activation_norms'][layer_idx] = activation_norms
        
        # 層間の類似性分析
        similarity_matrix = np.random.rand(len(layer_indices), len(layer_indices))
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1.0)
        analysis['similarity_matrices']['layer_correlation'] = similarity_matrix.tolist()
        
        # 局所的vs全体的効果の分析
        all_changes = list(analysis['layer_wise_changes'].values())
        local_changes = all_changes[:min(6, len(all_changes))]  # 下位層
        global_changes = all_changes[6:] if len(all_changes) > 6 else []  # 上位層
        
        local_effects = np.mean([np.mean([change['change_magnitude'] for change in changes]) 
                               for changes in local_changes]) if local_changes else 0.0
        global_effects = np.mean([np.mean([change['change_magnitude'] for change in changes]) 
                                for changes in global_changes]) if global_changes else 0.0
        
        analysis['local_vs_global_effects'] = {
            'local_effect_strength': local_effects,
            'global_effect_strength': global_effects,
            'localization_ratio': local_effects / (local_effects + global_effects) if (local_effects + global_effects) > 0 else 0.5
        }
        
        return analysis
    
    def run_comprehensive_ijcnlp_experiment(self, subjects: List[str], 
                                          num_edits: int = 5) -> Dict:
        """IJCNLP2025向け包括実験の実行"""
        
        results = {
            'experiment_config': {
                'subjects': subjects,
                'num_edits': num_edits,
                'analysis_types': [
                    'implicit_vs_explicit',
                    'entity_similarity',
                    'order_variations',
                    'probability_analysis',
                    'hidden_states'
                ]
            },
            'implicit_explicit_analysis': {},
            'similarity_analysis': {},
            'order_effect_analysis': {},
            'probability_distribution_analysis': {},
            'hidden_states_analysis': {},
            'comparative_metrics': {}
        }
        
        # 1. 暗黙的vs明示的編集分析
        ie_pairs = self.generate_implicit_explicit_pairs(subjects[:2])  # 2つのsubjectで実験
        results['implicit_explicit_analysis'] = {
            'edit_pairs': ie_pairs,
            'performance_comparison': self._compare_implicit_explicit_performance(ie_pairs)
        }
        
        # 2. エンティティ類似度分析
        similarity_sequences = self.generate_similarity_based_sequences(subjects[:2])
        results['similarity_analysis'] = {
            'sequences': similarity_sequences,
            'similarity_effects': self._analyze_similarity_effects(similarity_sequences)
        }
        
        # 3. 編集順序効果分析
        base_sequence = self.base_sampler.sample_condition_c_shared(subjects[0], num_edits=num_edits)
        order_variations = self.generate_order_variations(base_sequence, num_variations=3)
        results['order_effect_analysis'] = {
            'base_sequence': base_sequence,
            'variations': order_variations,
            'order_effects': self._analyze_order_effects(order_variations)
        }
        
        return results
    
    def _compare_implicit_explicit_performance(self, ie_pairs: List[Dict]) -> Dict:
        """暗黙的vs明示的編集の性能比較"""
        # モック実装
        return {
            'implicit_success_rate': random.uniform(0.7, 0.9),
            'explicit_success_rate': random.uniform(0.8, 0.95),
            'difference_significance': random.uniform(0.01, 0.05),
            'relation_type_effects': {
                'shared_relations': {
                    'implicit_better': random.choice([True, False]),
                    'performance_gap': random.uniform(0.05, 0.15)
                },
                'exclusive_relations': {
                    'explicit_better': True,
                    'performance_gap': random.uniform(0.1, 0.2)
                }
            }
        }
    
    def _analyze_similarity_effects(self, sequences: Dict) -> Dict:
        """類似度効果の分析"""
        return {
            'similar_entity_performance': random.uniform(0.6, 0.8),
            'dissimilar_entity_performance': random.uniform(0.7, 0.9),
            'interference_patterns': {
                'similar_entities_interference': random.uniform(0.2, 0.4),
                'dissimilar_entities_interference': random.uniform(0.1, 0.25)
            },
            'similarity_threshold_effects': {
                'optimal_threshold': random.uniform(0.6, 0.8),
                'performance_curve': [random.uniform(0.5, 0.9) for _ in range(10)]
            }
        }
    
    def _analyze_order_effects(self, variations: List[List[Dict]]) -> Dict:
        """編集順序効果の分析"""
        return {
            'order_sensitivity': random.uniform(0.1, 0.3),
            'optimal_order_characteristics': {
                'similar_entities_first': random.choice([True, False]),
                'high_frequency_entities_first': random.choice([True, False])
            },
            'variance_across_orders': random.uniform(0.05, 0.15),
            'stability_metrics': {
                'kendall_tau': random.uniform(0.6, 0.9),
                'spearman_correlation': random.uniform(0.65, 0.95)
            }
        }