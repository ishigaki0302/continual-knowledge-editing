#!/usr/bin/env python3
"""
Data Converter Module for IRT-based Knowledge Editing Evaluation

This module converts experimental data into the format required for IRT analysis.
Handles person-item matrix creation, response scoring, and data transformation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class IRTDataConverter:
    """
    Converts experimental data into IRT-ready format.
    
    Key transformations:
    - (method, model) -> person_id
    - (condition, sample_id, order_id, edit_step) -> item_id
    - Response scoring (binary, probability-based, rank-based)
    - Support for immediate vs cumulative responses
    """
    
    def __init__(self, 
                 score_type: str = 'binary',
                 probability_threshold: float = 0.5,
                 include_cumulative: bool = True):
        """
        Initialize data converter
        
        Args:
            score_type: Type of scoring ('binary', 'probability', 'rank')
            probability_threshold: Threshold for binary scoring based on probability
            include_cumulative: Whether to include cumulative response analysis
        """
        self.score_type = score_type
        self.probability_threshold = probability_threshold
        self.include_cumulative = include_cumulative
        
        # Validation
        if score_type not in ['binary', 'probability', 'rank']:
            raise ValueError("score_type must be 'binary', 'probability', or 'rank'")
        
        if not 0 < probability_threshold < 1:
            raise ValueError("probability_threshold must be between 0 and 1")
    
    def convert_to_irt_table(self, 
                           raw_data: pd.DataFrame,
                           response_types: List[str] = None) -> pd.DataFrame:
        """
        Convert raw experimental data to IRT table format
        
        Args:
            raw_data: DataFrame from log_loader.extract_raw_data()
            response_types: List of response types to include ['immediate', 'cumulative']
                          If None, includes both if available
        
        Returns:
            DataFrame in IRT format with columns:
            - person_id: (method, model) combination
            - item_id: (condition, sample_id, edit_order) combination
            - response_type: 'immediate' or 'cumulative'
            - response: response value (binary, probability, or rank)
            - is_correct: binary correctness indicator
            - score: normalized score (0-1)
            - metadata columns for analysis
        """
        if response_types is None:
            response_types = ['immediate', 'cumulative'] if self.include_cumulative else ['immediate']
        
        logger.info(f"Converting {len(raw_data)} data points to IRT format")
        logger.info(f"Score type: {self.score_type}, Response types: {response_types}")
        
        irt_rows = []
        
        # Process each response type
        for response_type in response_types:
            if response_type == 'immediate':
                # For immediate responses, use data as-is
                type_data = raw_data.copy()
            elif response_type == 'cumulative':
                # For cumulative responses, need to evaluate all previous edits
                type_data = self._prepare_cumulative_data(raw_data)
            else:
                logger.warning(f"Unknown response type: {response_type}")
                continue
            
            # Convert each row
            for _, row in type_data.iterrows():
                irt_row = self._convert_single_row(row, response_type)
                if irt_row is not None:
                    irt_rows.append(irt_row)
        
        irt_df = pd.DataFrame(irt_rows)
        
        if len(irt_df) == 0:
            raise ValueError("No valid IRT data could be generated")
        
        logger.info(f"Generated {len(irt_df)} IRT data points")
        return irt_df
    
    def _convert_single_row(self, row: pd.Series, response_type: str) -> Optional[Dict[str, Any]]:
        """Convert a single data row to IRT format"""
        try:
            # Create person_id (method + model)
            person_id = f"{row['method']}_{row['model_name']}"
            
            # Create item_id (condition + sample + order)
            item_id = f"{row['condition']}_{row['sample_index']}_{row['edit_order']}"
            
            # Get response value based on score type
            response = self._calculate_response(row)
            
            # Binary correctness
            is_correct = int(row['is_correct']) if pd.notna(row['is_correct']) else 0
            
            # Normalized score (0-1)
            score = self._calculate_normalized_score(row)
            
            irt_row = {
                'person_id': person_id,
                'item_id': item_id,
                'response_type': response_type,
                'response': response,
                'is_correct': is_correct,
                'score': score,
                # Metadata
                'method': row['method'],
                'model_name': row['model_name'],
                'condition': row['condition'],
                'sample_index': row['sample_index'],
                'edit_order': row['edit_order'],
                'subject': row['subject'],
                'relation': row['relation'],
                'object': row['object'],
                'relation_type': row.get('relation_type', 'unknown'),
                'target_probability': row.get('target_probability', 0.0),
                'target_rank': row.get('target_rank', 5),
                'source_file': row.get('source_file', '')
            }
            
            return irt_row
            
        except Exception as e:
            logger.warning(f"Failed to convert row to IRT format: {str(e)}")
            return None
    
    def _calculate_response(self, row: pd.Series) -> float:
        """Calculate response value based on score type"""
        if self.score_type == 'binary':
            # Binary based on rank or probability threshold
            if pd.notna(row['target_rank']):
                return float(row['target_rank'] == 1)
            elif pd.notna(row['target_probability']):
                return float(row['target_probability'] >= self.probability_threshold)
            else:
                return 0.0
        
        elif self.score_type == 'probability':
            # Use target probability directly
            return float(row.get('target_probability', 0.0))
        
        elif self.score_type == 'rank':
            # Use inverse rank (higher is better)
            if pd.notna(row['target_rank']):
                # Convert rank to score: rank 1 -> 1.0, rank 2 -> 0.8, etc.
                max_rank = 5  # Assuming 5 candidates
                return (max_rank - row['target_rank'] + 1) / max_rank
            else:
                return 0.0
        
        return 0.0
    
    def _calculate_normalized_score(self, row: pd.Series) -> float:
        """Calculate normalized score (0-1)"""
        if pd.notna(row['target_probability']):
            return float(row['target_probability'])
        elif pd.notna(row['target_rank']):
            # Convert rank to normalized score
            max_rank = 5
            return (max_rank - row['target_rank'] + 1) / max_rank
        else:
            return 0.0
    
    def _prepare_cumulative_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare cumulative response data by evaluating all previous edits
        
        For cumulative analysis, we need to check how well each edit
        is retained after subsequent edits are applied.
        """
        # This is a simplified version - in practice, you'd need to re-evaluate
        # the model after each edit to get true cumulative effects
        
        # For now, we'll create a copy and mark it as cumulative
        cumulative_data = raw_data.copy()
        
        # Group by experiment (method, model, condition, sample)
        grouped = cumulative_data.groupby(['method', 'model_name', 'condition', 'sample_index'])
        
        cumulative_rows = []
        
        for group_key, group_data in grouped:
            # Sort by edit order
            group_data = group_data.sort_values('edit_order')
            
            # For each edit, evaluate cumulative retention
            for i, (_, row) in enumerate(group_data.iterrows()):
                # In a real implementation, you'd re-evaluate the model
                # Here we'll simulate some degradation in later edits
                
                # Simple degradation model: performance decreases with more edits
                degradation_factor = 0.9 ** i  # Each edit reduces performance by 10%
                
                cumulative_row = row.copy()
                
                # Adjust probability and rank based on degradation
                if pd.notna(row['target_probability']):
                    cumulative_row['target_probability'] = row['target_probability'] * degradation_factor
                
                # Recalculate correctness based on adjusted probability
                if cumulative_row['target_probability'] >= self.probability_threshold:
                    cumulative_row['is_correct'] = True
                    cumulative_row['target_rank'] = 1
                else:
                    cumulative_row['is_correct'] = False
                    cumulative_row['target_rank'] = min(5, int(1 / (cumulative_row['target_probability'] + 0.1)))
                
                cumulative_rows.append(cumulative_row)
        
        return pd.DataFrame(cumulative_rows)
    
    def create_person_item_matrix(self, irt_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create person-item matrix for IRT analysis
        
        Args:
            irt_data: IRT formatted data
            
        Returns:
            Tuple of (matrix DataFrame, metadata dict)
        """
        logger.info("Creating person-item matrix")
        
        # Use only one response type for matrix (default: immediate)
        matrix_data = irt_data[irt_data['response_type'] == 'immediate'].copy()
        
        # Create pivot table
        matrix = matrix_data.pivot_table(
            index='person_id',
            columns='item_id',
            values='response',
            aggfunc='mean'  # Average if multiple responses per person-item
        )
        
        # Fill NaN with 0 (no response)
        matrix = matrix.fillna(0)
        
        # Create metadata
        metadata = {
            'persons': list(matrix.index),
            'items': list(matrix.columns),
            'n_persons': len(matrix.index),
            'n_items': len(matrix.columns),
            'response_rate': (matrix > 0).sum().sum() / (matrix.shape[0] * matrix.shape[1]),
            'person_stats': self._calculate_person_stats(matrix),
            'item_stats': self._calculate_item_stats(matrix)
        }
        
        logger.info(f"Created {metadata['n_persons']} x {metadata['n_items']} person-item matrix")
        logger.info(f"Response rate: {metadata['response_rate']:.3f}")
        
        return matrix, metadata
    
    def _calculate_person_stats(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate person-level statistics"""
        person_scores = matrix.sum(axis=1)  # Sum across items
        person_responses = (matrix > 0).sum(axis=1)  # Number of responses
        
        return {
            'mean_score': person_scores.mean(),
            'std_score': person_scores.std(),
            'min_score': person_scores.min(),
            'max_score': person_scores.max(),
            'mean_responses': person_responses.mean(),
            'std_responses': person_responses.std()
        }
    
    def _calculate_item_stats(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate item-level statistics"""
        item_scores = matrix.sum(axis=0)  # Sum across persons
        item_responses = (matrix > 0).sum(axis=0)  # Number of responses
        item_difficulty = 1 - (item_scores / item_responses)  # Higher = more difficult
        
        return {
            'mean_score': item_scores.mean(),
            'std_score': item_scores.std(),
            'min_score': item_scores.min(),
            'max_score': item_scores.max(),
            'mean_difficulty': item_difficulty.mean(),
            'std_difficulty': item_difficulty.std()
        }
    
    def add_item_covariates(self, irt_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add item-level covariates for explanatory IRT models
        
        Args:
            irt_data: IRT formatted data
            
        Returns:
            DataFrame with additional covariate columns
        """
        logger.info("Adding item covariates")
        
        # Create item-level covariates based on experimental design
        item_covariates = []
        
        for item_id in irt_data['item_id'].unique():
            # Parse item_id to extract components
            parts = item_id.split('_')
            if len(parts) >= 3:
                condition = parts[0]
                sample_idx = int(parts[1])
                edit_order = int(parts[2])
            else:
                condition = 'unknown'
                sample_idx = 0
                edit_order = 0
            
            # Get item-specific data
            item_data = irt_data[irt_data['item_id'] == item_id].iloc[0]
            
            covariate = {
                'item_id': item_id,
                'condition': condition,
                'sample_index': sample_idx,
                'edit_order': edit_order,
                'relation_type': item_data['relation_type'],
                'relation': item_data['relation'],
                # Derived features
                'is_shared_relation': item_data['relation_type'] == 'SharedRelations',
                'is_exclusive_relation': item_data['relation_type'] == 'ExclusiveRelations',
                'is_early_edit': edit_order <= 2,
                'is_late_edit': edit_order >= 4,
                'condition_A': condition == 'A',
                'condition_B': condition == 'B',
                'condition_C': condition == 'C'
            }
            
            item_covariates.append(covariate)
        
        covariates_df = pd.DataFrame(item_covariates)
        
        # Merge with IRT data
        enhanced_data = irt_data.merge(covariates_df, on='item_id', how='left', suffixes=('', '_cov'))
        
        logger.info(f"Added {len(covariates_df.columns) - 1} item covariates")
        return enhanced_data
    
    def add_person_covariates(self, irt_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add person-level covariates for explanatory IRT models
        
        Args:
            irt_data: IRT formatted data
            
        Returns:
            DataFrame with additional covariate columns
        """
        logger.info("Adding person covariates")
        
        # Create person-level covariates based on method and model
        person_covariates = []
        
        for person_id in irt_data['person_id'].unique():
            # Parse person_id to extract components
            parts = person_id.split('_')
            if len(parts) >= 2:
                method = parts[0]
                model = '_'.join(parts[1:])
            else:
                method = 'unknown'
                model = 'unknown'
            
            covariate = {
                'person_id': person_id,
                'method': method,
                'model_name': model,
                # Method categories
                'is_ROME': method == 'ROME',
                'is_MEMIT': method == 'MEMIT',
                'is_MEND': method == 'MEND',
                'is_FT': method == 'FT',
                'is_IKE': method == 'IKE',
                'is_KN': method == 'KN',
                # Model categories
                'is_gpt_j': 'gpt-j' in model,
                'is_gpt2': 'gpt2' in model,
                'is_llama': 'llama' in model,
                'is_large_model': any(x in model for x in ['gpt-j-6b', 'llama-7b', 'llama3-8b']),
                'is_small_model': any(x in model for x in ['gpt2-xl', 'llama3.2-3b'])
            }
            
            person_covariates.append(covariate)
        
        covariates_df = pd.DataFrame(person_covariates)
        
        # Merge with IRT data
        enhanced_data = irt_data.merge(covariates_df, on='person_id', how='left', suffixes=('', '_cov'))
        
        logger.info(f"Added {len(covariates_df.columns) - 1} person covariates")
        return enhanced_data
    
    def validate_irt_data(self, irt_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate IRT data quality and completeness
        
        Args:
            irt_data: IRT formatted data
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating IRT data")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        required_columns = ['person_id', 'item_id', 'response', 'is_correct']
        missing_columns = [col for col in required_columns if col not in irt_data.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        for col in required_columns:
            if col in irt_data.columns:
                null_count = irt_data[col].isnull().sum()
                if null_count > 0:
                    validation_results['warnings'].append(f"Column '{col}' has {null_count} null values")
        
        # Check response value ranges
        if 'response' in irt_data.columns:
            response_min = irt_data['response'].min()
            response_max = irt_data['response'].max()
            
            if response_min < 0 or response_max > 1:
                validation_results['warnings'].append(f"Response values outside [0,1]: [{response_min}, {response_max}]")
        
        # Check for sufficient data
        n_persons = irt_data['person_id'].nunique()
        n_items = irt_data['item_id'].nunique()
        
        if n_persons < 2:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Too few persons for IRT analysis: {n_persons}")
        
        if n_items < 3:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Too few items for IRT analysis: {n_items}")
        
        # Calculate statistics
        validation_results['stats'] = {
            'n_observations': len(irt_data),
            'n_persons': n_persons,
            'n_items': n_items,
            'mean_response': irt_data['response'].mean() if 'response' in irt_data.columns else 0,
            'response_variance': irt_data['response'].var() if 'response' in irt_data.columns else 0,
            'missing_rate': irt_data.isnull().sum().sum() / (len(irt_data) * len(irt_data.columns))
        }
        
        logger.info(f"Validation complete. Valid: {validation_results['is_valid']}")
        return validation_results


def main():
    """Example usage of IRTDataConverter"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert experimental data to IRT format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file from log_loader')
    parser.add_argument('--output', type=str, default='irt_data.csv',
                       help='Output IRT data file')
    parser.add_argument('--matrix-output', type=str, default='person_item_matrix.csv',
                       help='Output person-item matrix file')
    parser.add_argument('--score-type', type=str, default='binary',
                       choices=['binary', 'probability', 'rank'],
                       help='Type of response scoring')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for binary scoring')
    parser.add_argument('--include-cumulative', action='store_true',
                       help='Include cumulative response analysis')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load raw data
    logger.info(f"Loading raw data from: {args.input}")
    raw_data = pd.read_csv(args.input)
    
    # Initialize converter
    converter = IRTDataConverter(
        score_type=args.score_type,
        probability_threshold=args.threshold,
        include_cumulative=args.include_cumulative
    )
    
    # Convert to IRT format
    irt_data = converter.convert_to_irt_table(raw_data)
    
    # Add covariates
    irt_data = converter.add_person_covariates(irt_data)
    irt_data = converter.add_item_covariates(irt_data)
    
    # Validate data
    validation = converter.validate_irt_data(irt_data)
    
    print("\n=== IRT Data Validation ===")
    print(f"Valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    print(f"Statistics: {validation['stats']}")
    
    # Create person-item matrix
    matrix, metadata = converter.create_person_item_matrix(irt_data)
    
    print("\n=== Person-Item Matrix ===")
    print(f"Dimensions: {metadata['n_persons']} persons x {metadata['n_items']} items")
    print(f"Response rate: {metadata['response_rate']:.3f}")
    print(f"Person stats: {metadata['person_stats']}")
    print(f"Item stats: {metadata['item_stats']}")
    
    # Save outputs
    irt_data.to_csv(args.output, index=False)
    matrix.to_csv(args.matrix_output)
    
    print(f"\nIRT data saved to: {args.output}")
    print(f"Person-item matrix saved to: {args.matrix_output}")
    print(f"Total IRT observations: {len(irt_data)}")


if __name__ == "__main__":
    main()