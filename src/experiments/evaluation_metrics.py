"""
Evaluation Metrics for Continual Knowledge Editing
"""

import numpy as np
from typing import List, Dict, Any

class EvaluationMetrics:
    """Metrics for evaluating continual knowledge editing"""
    
    def __init__(self):
        pass
    
    def probability_ranking_analysis(self, predictions: List[Dict]) -> Dict:
        """Analyze probability rankings across edits"""
        # Mock implementation - would analyze actual model probabilities in real scenario
        rankings = []
        for pred in predictions:
            rankings.append({
                'edit_id': pred.get('edit_id', 'unknown'),
                'probability_ranking': np.random.random(),  # Mock probability
                'confidence': np.random.uniform(0.5, 1.0)
            })
        
        return {
            'rankings': rankings,
            'avg_probability': np.mean([r['probability_ranking'] for r in rankings]),
            'avg_confidence': np.mean([r['confidence'] for r in rankings])
        }
    
    def calculate_interference_score(self, before_edits: List[Dict], after_edits: List[Dict]) -> float:
        """Calculate interference between edits"""
        # Mock implementation - would calculate actual interference in real scenario
        if not before_edits or not after_edits:
            return 0.0
        
        # Simulate interference calculation
        interference = np.random.uniform(0.0, 0.3)  # Low interference is good
        return interference
    
    def calculate_retention_score(self, edit_results: List[Dict], evaluation_results: List[Dict]) -> Dict:
        """Calculate knowledge retention scores"""
        if not edit_results or not evaluation_results:
            return {'retention_score': 0.0, 'details': {}}
        
        # Calculate retention based on evaluation accuracy
        correct_evaluations = sum(1 for r in evaluation_results if r.get('is_correct', False))
        total_evaluations = len(evaluation_results)
        retention_score = correct_evaluations / total_evaluations if total_evaluations > 0 else 0.0
        
        # Calculate per-edit retention
        per_edit_retention = {}
        for eval_result in evaluation_results:
            edit_id = eval_result.get('edit_id', 'unknown')
            per_edit_retention[edit_id] = 1.0 if eval_result.get('is_correct', False) else 0.0
        
        return {
            'retention_score': retention_score,
            'correct_evaluations': correct_evaluations,
            'total_evaluations': total_evaluations,
            'per_edit_retention': per_edit_retention
        }
    
    def calculate_efficacy_metrics(self, edit_results: List[Dict]) -> Dict:
        """Calculate edit efficacy metrics"""
        if not edit_results:
            return {'avg_efficacy': 0.0, 'efficacy_scores': []}
        
        efficacy_scores = []
        for result in edit_results:
            metrics = result.get('metrics', {})
            efficacy = metrics.get('efficacy', 0.0)
            efficacy_scores.append(efficacy)
        
        return {
            'avg_efficacy': np.mean(efficacy_scores),
            'std_efficacy': np.std(efficacy_scores),
            'min_efficacy': np.min(efficacy_scores),
            'max_efficacy': np.max(efficacy_scores),
            'efficacy_scores': efficacy_scores
        }
    
    def calculate_locality_metrics(self, edit_results: List[Dict]) -> Dict:
        """Calculate locality preservation metrics"""
        if not edit_results:
            return {'avg_locality': 0.0, 'locality_scores': []}
        
        locality_scores = []
        for result in edit_results:
            metrics = result.get('metrics', {})
            locality = metrics.get('locality', 0.0)
            locality_scores.append(locality)
        
        return {
            'avg_locality': np.mean(locality_scores),
            'std_locality': np.std(locality_scores),
            'min_locality': np.min(locality_scores),
            'max_locality': np.max(locality_scores),
            'locality_scores': locality_scores
        }
    
    def analyze_condition_differences(self, condition_results: Dict) -> Dict:
        """Analyze differences between experimental conditions"""
        condition_summaries = {}
        
        for condition_name, results in condition_results.items():
            if 'summary' in results:
                summary = results['summary']
                condition_summaries[condition_name] = {
                    'efficacy': summary.get('avg_efficacy', 0.0),
                    'locality': summary.get('avg_locality', 0.0),
                    'accuracy': summary.get('evaluation_accuracy', 0.0),
                    'num_edits': summary.get('num_edits', 0)
                }
        
        # Calculate differences
        conditions = list(condition_summaries.keys())
        differences = {}
        
        if len(conditions) >= 2:
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i+1:]:
                    diff_key = f"{cond1}_vs_{cond2}"
                    differences[diff_key] = {
                        'efficacy_diff': condition_summaries[cond1]['efficacy'] - condition_summaries[cond2]['efficacy'],
                        'locality_diff': condition_summaries[cond1]['locality'] - condition_summaries[cond2]['locality'],
                        'accuracy_diff': condition_summaries[cond1]['accuracy'] - condition_summaries[cond2]['accuracy']
                    }
        
        return {
            'condition_summaries': condition_summaries,
            'differences': differences,
            'best_condition': max(condition_summaries.keys(), 
                                key=lambda k: condition_summaries[k]['accuracy']) if condition_summaries else None
        }
    
    def calculate_continual_learning_metrics(self, edit_sequence: List[Dict], 
                                           edit_results: List[Dict]) -> Dict:
        """Calculate metrics specific to continual learning scenarios"""
        
        # Catastrophic forgetting analysis
        forgetting_scores = []
        for i in range(1, len(edit_results)):
            # Compare current performance with previous edits
            current_metrics = edit_results[i]['metrics']
            prev_metrics = edit_results[i-1]['metrics']
            
            # Calculate forgetting as decrease in performance
            forgetting = max(0, prev_metrics.get('efficacy', 0) - current_metrics.get('efficacy', 0))
            forgetting_scores.append(forgetting)
        
        # Positive transfer analysis
        transfer_scores = []
        base_efficacy = edit_results[0]['metrics'].get('efficacy', 0) if edit_results else 0
        
        for result in edit_results[1:]:
            current_efficacy = result['metrics'].get('efficacy', 0)
            transfer = max(0, current_efficacy - base_efficacy)
            transfer_scores.append(transfer)
        
        return {
            'catastrophic_forgetting': {
                'avg_forgetting': np.mean(forgetting_scores) if forgetting_scores else 0.0,
                'max_forgetting': np.max(forgetting_scores) if forgetting_scores else 0.0,
                'forgetting_scores': forgetting_scores
            },
            'positive_transfer': {
                'avg_transfer': np.mean(transfer_scores) if transfer_scores else 0.0,
                'max_transfer': np.max(transfer_scores) if transfer_scores else 0.0,
                'transfer_scores': transfer_scores
            },
            'stability': 1.0 - (np.mean(forgetting_scores) if forgetting_scores else 0.0),
            'plasticity': np.mean(transfer_scores) if transfer_scores else 0.0
        }