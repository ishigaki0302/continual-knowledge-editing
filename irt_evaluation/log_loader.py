#!/usr/bin/env python3
"""
Log Loader Module for IRT-based Knowledge Editing Evaluation

This module handles loading and validating experiment logs from knowledge editing experiments.
Supports both JSON and CSV formats with comprehensive data quality checks.
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class LogLoader:
    """
    Handles loading and validation of experiment logs for IRT analysis.
    
    Supports:
    - JSON format from run_knowledge_editing_new_order_sampling.py
    - CSV format for external data
    - Data quality checks and validation
    - Multiple experiment file loading
    """
    
    def __init__(self, validate_data: bool = True):
        """
        Initialize LogLoader
        
        Args:
            validate_data: Whether to perform data validation checks
        """
        self.validate_data = validate_data
        self.loaded_experiments = []
        self.data_quality_report = {}
        
    def load_single_experiment(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single experiment file
        
        Args:
            file_path: Path to the experiment file
            
        Returns:
            Dictionary containing experiment data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Experiment file not found: {file_path}")
        
        logger.info(f"Loading experiment file: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.json':
                return self._load_json_experiment(file_path)
            elif file_path.suffix.lower() in ['.csv', '.tsv']:
                return self._load_csv_experiment(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load experiment file {file_path}: {str(e)}")
            raise
    
    def load_multiple_experiments(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Load multiple experiment files
        
        Args:
            file_paths: List of paths to experiment files
            
        Returns:
            List of experiment dictionaries
        """
        experiments = []
        
        for file_path in file_paths:
            try:
                experiment = self.load_single_experiment(file_path)
                experiments.append(experiment)
                logger.info(f"Successfully loaded: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {str(e)}")
                continue
        
        if not experiments:
            raise ValueError("No experiments could be loaded successfully")
        
        logger.info(f"Loaded {len(experiments)} experiments successfully")
        return experiments
    
    def load_from_directory(self, directory: Union[str, Path], 
                          pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Load all experiment files from a directory
        
        Args:
            directory: Directory path containing experiment files
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            List of experiment dictionaries
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        file_paths = list(directory.glob(pattern))
        
        if not file_paths:
            raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")
        
        logger.info(f"Found {len(file_paths)} files matching pattern '{pattern}'")
        return self.load_multiple_experiments(file_paths)
    
    def _load_json_experiment(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON experiment file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add file metadata
        data['source_file'] = str(file_path)
        data['loaded_at'] = datetime.now().isoformat()
        
        if self.validate_data:
            self._validate_json_structure(data, file_path)
        
        return data
    
    def _load_csv_experiment(self, file_path: Path) -> Dict[str, Any]:
        """Load CSV experiment file"""
        # Determine separator
        separator = '\t' if file_path.suffix.lower() == '.tsv' else ','
        
        df = pd.read_csv(file_path, sep=separator)
        
        # Convert to experiment format
        experiment_data = {
            'source_file': str(file_path),
            'loaded_at': datetime.now().isoformat(),
            'method': df.get('method', ['unknown']).iloc[0] if 'method' in df.columns else 'unknown',
            'model_name': df.get('model_name', ['unknown']).iloc[0] if 'model_name' in df.columns else 'unknown',
            'condition': df.get('condition', ['unknown']).iloc[0] if 'condition' in df.columns else 'unknown',
            'individual_results': self._convert_csv_to_results(df)
        }
        
        if self.validate_data:
            self._validate_csv_structure(df, file_path)
        
        return experiment_data
    
    def _convert_csv_to_results(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert CSV data to individual results format"""
        results = []
        
        # Group by sample_index if available
        if 'sample_index' in df.columns:
            for sample_idx in df['sample_index'].unique():
                sample_data = df[df['sample_index'] == sample_idx]
                result = {
                    'sample_index': sample_idx,
                    'edits': [],
                    'final_state_evaluations': []
                }
                
                for _, row in sample_data.iterrows():
                    edit = {
                        'edit_order': row.get('edit_order', 1),
                        'triple': {
                            'subject': row.get('subject', ''),
                            'relation': row.get('relation', ''),
                            'object': row.get('object', ''),
                            'candidates': row.get('candidates', '').split(',') if 'candidates' in row else []
                        },
                        'post_edit_probabilities': {
                            'probabilities': [float(x) for x in row.get('probabilities', '').split(',') if x] if 'probabilities' in row else []
                        }
                    }
                    result['edits'].append(edit)
                
                results.append(result)
        else:
            # Single result without sample grouping
            result = {
                'sample_index': 1,
                'edits': [],
                'final_state_evaluations': []
            }
            
            for _, row in df.iterrows():
                edit = {
                    'edit_order': row.get('edit_order', 1),
                    'triple': {
                        'subject': row.get('subject', ''),
                        'relation': row.get('relation', ''),
                        'object': row.get('object', ''),
                        'candidates': row.get('candidates', '').split(',') if 'candidates' in row else []
                    },
                    'post_edit_probabilities': {
                        'probabilities': [float(x) for x in row.get('probabilities', '').split(',') if x] if 'probabilities' in row else []
                    }
                }
                result['edits'].append(edit)
            
            results.append(result)
        
        return results
    
    def _validate_json_structure(self, data: Dict[str, Any], file_path: Path):
        """Validate JSON experiment structure"""
        required_fields = ['method', 'model_name', 'condition']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            warnings.warn(f"Missing required fields in {file_path}: {missing_fields}")
        
        # Check for individual results
        if 'individual_results' not in data:
            warnings.warn(f"No 'individual_results' field found in {file_path}")
            return
        
        results = data['individual_results']
        if not isinstance(results, list) or len(results) == 0:
            warnings.warn(f"Empty or invalid 'individual_results' in {file_path}")
            return
        
        # Validate sample structure
        sample_issues = []
        for i, result in enumerate(results):
            if 'edits' not in result:
                sample_issues.append(f"Sample {i}: missing 'edits' field")
            elif not isinstance(result['edits'], list):
                sample_issues.append(f"Sample {i}: 'edits' is not a list")
            else:
                # Check edit structure
                for j, edit in enumerate(result['edits']):
                    if 'triple' not in edit:
                        sample_issues.append(f"Sample {i}, Edit {j}: missing 'triple' field")
                    if 'post_edit_probabilities' not in edit:
                        sample_issues.append(f"Sample {i}, Edit {j}: missing 'post_edit_probabilities' field")
        
        if sample_issues:
            warnings.warn(f"Structure issues in {file_path}: {sample_issues[:5]}...")  # Show first 5 issues
    
    def _validate_csv_structure(self, df: pd.DataFrame, file_path: Path):
        """Validate CSV experiment structure"""
        required_columns = ['subject', 'relation', 'object']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            warnings.warn(f"Missing required columns in {file_path}: {missing_columns}")
        
        # Check for empty values
        for col in required_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                warnings.warn(f"Column '{col}' has {df[col].isnull().sum()} null values in {file_path}")
    
    def get_data_quality_report(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate data quality report for loaded experiments
        
        Args:
            experiments: List of experiment dictionaries
            
        Returns:
            Dictionary containing data quality metrics
        """
        report = {
            'total_experiments': len(experiments),
            'methods': {},
            'models': {},
            'conditions': {},
            'sample_counts': [],
            'edit_counts': [],
            'issues': []
        }
        
        for exp in experiments:
            # Count methods, models, conditions
            method = exp.get('method', 'unknown')
            model = exp.get('model_name', 'unknown')
            condition = exp.get('condition', 'unknown')
            
            report['methods'][method] = report['methods'].get(method, 0) + 1
            report['models'][model] = report['models'].get(model, 0) + 1
            report['conditions'][condition] = report['conditions'].get(condition, 0) + 1
            
            # Count samples and edits
            if 'individual_results' in exp:
                num_samples = len(exp['individual_results'])
                report['sample_counts'].append(num_samples)
                
                for result in exp['individual_results']:
                    if 'edits' in result:
                        num_edits = len(result['edits'])
                        report['edit_counts'].append(num_edits)
        
        # Calculate statistics
        if report['sample_counts']:
            report['sample_stats'] = {
                'mean': np.mean(report['sample_counts']),
                'std': np.std(report['sample_counts']),
                'min': np.min(report['sample_counts']),
                'max': np.max(report['sample_counts'])
            }
        
        if report['edit_counts']:
            report['edit_stats'] = {
                'mean': np.mean(report['edit_counts']),
                'std': np.std(report['edit_counts']),
                'min': np.min(report['edit_counts']),
                'max': np.max(report['edit_counts'])
            }
        
        self.data_quality_report = report
        return report
    
    def filter_experiments(self, experiments: List[Dict[str, Any]], 
                          method: Optional[str] = None,
                          model: Optional[str] = None,
                          condition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter experiments by method, model, or condition
        
        Args:
            experiments: List of experiment dictionaries
            method: Filter by method (optional)
            model: Filter by model (optional)
            condition: Filter by condition (optional)
            
        Returns:
            Filtered list of experiments
        """
        filtered = experiments
        
        if method:
            filtered = [exp for exp in filtered if exp.get('method') == method]
        
        if model:
            filtered = [exp for exp in filtered if exp.get('model_name') == model]
        
        if condition:
            filtered = [exp for exp in filtered if exp.get('condition') == condition]
        
        logger.info(f"Filtered {len(experiments)} experiments to {len(filtered)} experiments")
        return filtered
    
    def extract_raw_data(self, experiments: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract raw data from experiments into a flat DataFrame
        
        Args:
            experiments: List of experiment dictionaries
            
        Returns:
            DataFrame with raw experimental data
        """
        rows = []
        
        for exp in experiments:
            method = exp.get('method', 'unknown')
            model = exp.get('model_name', 'unknown')
            condition = exp.get('condition', 'unknown')
            
            if 'individual_results' not in exp:
                continue
                
            for result in exp['individual_results']:
                sample_idx = result.get('sample_index', 1)
                
                if 'edits' not in result:
                    continue
                
                for edit in result['edits']:
                    edit_order = edit.get('edit_order', 1)
                    triple = edit.get('triple', {})
                    
                    # Extract probabilities
                    post_edit_probs = edit.get('post_edit_probabilities', {})
                    probabilities = post_edit_probs.get('probabilities', [])
                    candidates = triple.get('candidates', [])
                    
                    # Find target probability and rank
                    target_object = triple.get('object', '')
                    target_prob = 0.0
                    target_rank = len(candidates)
                    
                    if target_object in candidates and len(probabilities) >= len(candidates):
                        target_idx = candidates.index(target_object)
                        target_prob = probabilities[target_idx]
                        # Calculate rank (1-based)
                        sorted_probs = sorted(probabilities, reverse=True)
                        target_rank = sorted_probs.index(target_prob) + 1
                    
                    row = {
                        'method': method,
                        'model_name': model,
                        'condition': condition,
                        'sample_index': sample_idx,
                        'edit_order': edit_order,
                        'subject': triple.get('subject', ''),
                        'relation': triple.get('relation', ''),
                        'object': target_object,
                        'candidates': ','.join(candidates),
                        'probabilities': ','.join(map(str, probabilities)),
                        'target_probability': target_prob,
                        'target_rank': target_rank,
                        'is_correct': target_rank == 1,
                        'source_file': exp.get('source_file', ''),
                        'relation_type': triple.get('relation_type', 'unknown')
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(df)} data points from {len(experiments)} experiments")
        return df


def main():
    """Example usage of LogLoader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and validate experiment logs')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', type=str, default='extracted_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='File pattern for directory loading')
    parser.add_argument('--method', type=str,
                       help='Filter by method')
    parser.add_argument('--model', type=str,
                       help='Filter by model')
    parser.add_argument('--condition', type=str,
                       help='Filter by condition')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = LogLoader(validate_data=True)
    
    # Load experiments
    input_path = Path(args.input)
    if input_path.is_dir():
        experiments = loader.load_from_directory(input_path, args.pattern)
    else:
        experiments = [loader.load_single_experiment(input_path)]
    
    # Filter experiments
    if args.method or args.model or args.condition:
        experiments = loader.filter_experiments(experiments, args.method, args.model, args.condition)
    
    # Generate quality report
    report = loader.get_data_quality_report(experiments)
    print("\n=== Data Quality Report ===")
    print(f"Total experiments: {report['total_experiments']}")
    print(f"Methods: {report['methods']}")
    print(f"Models: {report['models']}")
    print(f"Conditions: {report['conditions']}")
    
    if 'sample_stats' in report:
        print(f"Sample statistics: {report['sample_stats']}")
    if 'edit_stats' in report:
        print(f"Edit statistics: {report['edit_stats']}")
    
    # Extract raw data
    df = loader.extract_raw_data(experiments)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nExtracted data saved to: {args.output}")
    print(f"Total data points: {len(df)}")


if __name__ == "__main__":
    main()