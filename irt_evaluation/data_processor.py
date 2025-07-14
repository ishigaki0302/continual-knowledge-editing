"""
Data processor for converting knowledge editing results to IRT-compatible format.
Handles both immediate and cumulative response extraction from experimental results.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class KnowledgeEditingDataProcessor:
    """
    Processes knowledge editing experimental results for IRT analysis.
    
    Converts raw experimental data into structured format suitable for:
    - Binary IRT analysis (correct/incorrect responses)
    - Continuous probability analysis
    - Immediate vs cumulative response comparison
    """
    
    def __init__(self, results_dir: str = "/app/EasyEdit/results"):
        self.results_dir = Path(results_dir)
        self.raw_data = []
        self.processed_data = {}
        
    def load_all_results(self) -> Dict:
        """
        Load all experimental results from the results directory.
        
        Returns:
            Dict: Mapping of experiment identifiers to result data
        """
        results = {}
        
        for result_file in self.results_dir.glob("knowledge_editing_from_candidates_*.json"):
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
    
    def extract_irt_data(self, results: Dict) -> pd.DataFrame:
        """
        Extract data for IRT analysis from experimental results.
        
        Args:
            results: Dictionary of experimental results
            
        Returns:
            DataFrame with columns: [person_id, item_id, response_immediate, response_cumulative, 
                                   probability_immediate, probability_cumulative, method, model, 
                                   condition, edit_order, subject, relation, object]
        """
        irt_data = []
        
        for exp_id, data in results.items():
            if not data.get('success', False):
                continue
                
            method, model, condition = exp_id.split('_')
            individual_results = data.get('individual_results', [])
            
            for sample_idx, sample in enumerate(individual_results):
                # Process immediate responses (after each edit)
                edits = sample.get('edits', [])
                
                for edit_idx, edit in enumerate(edits):
                    edit_order = edit.get('edit_order', edit_idx + 1)
                    triple = edit.get('triple', {})
                    
                    # Extract immediate response data
                    post_edit_probs = edit.get('post_edit_probabilities', {})
                    candidates = post_edit_probs.get('candidates', [])
                    probabilities = post_edit_probs.get('probabilities', [])
                    
                    if candidates and probabilities:
                        target_object = triple.get('object', '')
                        
                        # Find target probability and rank
                        target_prob_immediate = 0.0
                        target_rank_immediate = len(candidates) + 1  # Worst possible rank
                        
                        if target_object in candidates:
                            target_idx = candidates.index(target_object)
                            target_prob_immediate = probabilities[target_idx]
                            
                            # Calculate rank (1 = best, higher = worse)
                            sorted_probs = sorted(probabilities, reverse=True)
                            target_rank_immediate = sorted_probs.index(target_prob_immediate) + 1
                        
                        # Binary response (1 if target is rank 1, 0 otherwise)
                        response_immediate = 1 if target_rank_immediate == 1 else 0
                        
                        # Process cumulative response
                        final_evaluations = sample.get('final_state_evaluations', [])
                        response_cumulative = 0
                        probability_cumulative = 0.0
                        
                        # Find corresponding final evaluation
                        for final_eval in final_evaluations:
                            if final_eval.get('triple_index') == edit_idx:
                                target_prob_cumulative = final_eval.get('target_probability', 0.0)
                                target_rank_cumulative = final_eval.get('target_rank', len(candidates) + 1)
                                
                                probability_cumulative = target_prob_cumulative
                                response_cumulative = 1 if target_rank_cumulative == 1 else 0
                                break
                        
                        # Create unique identifiers
                        person_id = f"{method}_{model}"
                        item_id = f"{condition}_{edit_order}_{triple.get('subject', '')[:10]}_{triple.get('relation', '')}"
                        
                        irt_data.append({
                            'person_id': person_id,
                            'item_id': item_id,
                            'response_immediate': response_immediate,
                            'response_cumulative': response_cumulative,
                            'probability_immediate': target_prob_immediate,
                            'probability_cumulative': probability_cumulative,
                            'method': method,
                            'model': model,
                            'condition': condition,
                            'edit_order': edit_order,
                            'sample_index': sample_idx,
                            'subject': triple.get('subject', ''),
                            'relation': triple.get('relation', ''),
                            'object': triple.get('object', ''),
                            'relation_type': triple.get('relation_type', ''),
                            'target_rank_immediate': target_rank_immediate,
                            'target_rank_cumulative': target_rank_cumulative if 'target_rank_cumulative' in locals() else 0,
                            'candidates': candidates,
                            'all_probabilities_immediate': probabilities,
                            'all_probabilities_cumulative': final_eval.get('final_state_probabilities', {}).get('probabilities', []) if 'final_eval' in locals() else []
                        })
        
        return pd.DataFrame(irt_data)
    
    def create_person_item_matrix(self, df: pd.DataFrame, response_type: str = 'immediate') -> pd.DataFrame:
        """
        Create person-item response matrix for IRT analysis.
        
        Args:
            df: Processed IRT data
            response_type: 'immediate' or 'cumulative'
            
        Returns:
            DataFrame with persons as rows and items as columns
        """
        response_col = f'response_{response_type}'
        
        # Create pivot table
        matrix = df.pivot_table(
            index='person_id',
            columns='item_id',
            values=response_col,
            aggfunc='mean'  # Average if multiple observations per person-item
        )
        
        return matrix
    
    def calculate_item_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate descriptive statistics for each item.
        
        Args:
            df: Processed IRT data
            
        Returns:
            DataFrame with item statistics
        """
        item_stats = df.groupby('item_id').agg({
            'response_immediate': ['mean', 'std', 'count'],
            'response_cumulative': ['mean', 'std', 'count'],
            'probability_immediate': ['mean', 'std'],
            'probability_cumulative': ['mean', 'std'],
            'condition': 'first',
            'edit_order': 'first',
            'relation': 'first'
        }).round(4)
        
        # Flatten column names
        new_cols = []
        for col in item_stats.columns.values:
            if isinstance(col, tuple):
                new_cols.append('_'.join(col))
            else:
                new_cols.append(str(col))
        item_stats.columns = new_cols
        
        return item_stats
    
    def calculate_person_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate descriptive statistics for each person (method-model combination).
        
        Args:
            df: Processed IRT data
            
        Returns:
            DataFrame with person statistics
        """
        person_stats = df.groupby('person_id').agg({
            'response_immediate': ['mean', 'std', 'count'],
            'response_cumulative': ['mean', 'std', 'count'],
            'probability_immediate': ['mean', 'std'],
            'probability_cumulative': ['mean', 'std'],
            'method': 'first',
            'model': 'first'
        }).round(4)
        
        # Flatten column names
        new_cols = []
        for col in person_stats.columns.values:
            if isinstance(col, tuple):
                new_cols.append('_'.join(col))
            else:
                new_cols.append(str(col))
        person_stats.columns = new_cols
        
        return person_stats
    
    def extract_continuous_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract continuous probability scores for detailed analysis.
        
        Args:
            df: Processed IRT data
            
        Returns:
            DataFrame optimized for continuous score analysis
        """
        continuous_data = df.copy()
        
        # Add derived metrics
        continuous_data['probability_difference'] = (
            continuous_data['probability_immediate'] - continuous_data['probability_cumulative']
        )
        continuous_data['rank_difference'] = (
            continuous_data['target_rank_cumulative'] - continuous_data['target_rank_immediate']
        )
        
        # Performance degradation indicator
        continuous_data['performance_degraded'] = (
            continuous_data['probability_cumulative'] < continuous_data['probability_immediate']
        ).astype(int)
        
        return continuous_data
    
    def process_all_data(self) -> Dict:
        """
        Complete processing pipeline for all experimental data.
        
        Returns:
            Dictionary containing all processed data structures
        """
        # Load raw results
        raw_results = self.load_all_results()
        
        # Extract IRT data
        irt_df = self.extract_irt_data(raw_results)
        
        # Create matrices and statistics
        immediate_matrix = self.create_person_item_matrix(irt_df, 'immediate')
        cumulative_matrix = self.create_person_item_matrix(irt_df, 'cumulative')
        
        item_stats = self.calculate_item_statistics(irt_df)
        person_stats = self.calculate_person_statistics(irt_df)
        
        continuous_scores = self.extract_continuous_scores(irt_df)
        
        processed_data = {
            'raw_results': raw_results,
            'irt_data': irt_df,
            'immediate_matrix': immediate_matrix,
            'cumulative_matrix': cumulative_matrix,
            'item_statistics': item_stats,
            'person_statistics': person_stats,
            'continuous_scores': continuous_scores
        }
        
        logger.info(f"Processed {len(irt_df)} observations across {len(raw_results)} experiments")
        
        return processed_data
    
    def save_processed_data(self, processed_data: Dict, output_dir: str = "/app/EasyEdit/irt_evaluation/output"):
        """
        Save processed data to files.
        
        Args:
            processed_data: Dictionary of processed data
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main dataframes
        processed_data['irt_data'].to_csv(output_path / 'irt_data.csv', index=False)
        processed_data['continuous_scores'].to_csv(output_path / 'continuous_scores.csv', index=False)
        processed_data['item_statistics'].to_csv(output_path / 'item_statistics.csv')
        processed_data['person_statistics'].to_csv(output_path / 'person_statistics.csv')
        
        # Save matrices
        processed_data['immediate_matrix'].to_csv(output_path / 'immediate_response_matrix.csv')
        processed_data['cumulative_matrix'].to_csv(output_path / 'cumulative_response_matrix.csv')
        
        # Save summary statistics
        summary = {
            'total_observations': len(processed_data['irt_data']),
            'unique_persons': processed_data['irt_data']['person_id'].nunique(),
            'unique_items': processed_data['irt_data']['item_id'].nunique(),
            'conditions': processed_data['irt_data']['condition'].unique().tolist(),
            'methods': processed_data['irt_data']['method'].unique().tolist(),
            'models': processed_data['irt_data']['model'].unique().tolist(),
        }
        
        with open(output_path / 'processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Process all data
    processor = KnowledgeEditingDataProcessor()
    processed_data = processor.process_all_data()
    
    # Save results
    processor.save_processed_data(processed_data)
    
    print("Data processing complete!")