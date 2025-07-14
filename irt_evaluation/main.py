#!/usr/bin/env python3
"""
Main Entry Point for IRT-based Knowledge Editing Evaluation

This module provides a unified command-line interface for the complete IRT evaluation pipeline.
Supports end-to-end analysis from raw experiment logs to publication-ready reports.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import evaluation modules
from log_loader import LogLoader
from data_converter import IRTDataConverter
from fit_irt import IRTModelFitter
from visualizer import IRTVisualizer
from reporter import IRTReporter

# Version information
__version__ = "1.0.0"
__author__ = "IRT Knowledge Editing Evaluation System"


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_dir = Path(config.get('output', {}).get('directories', {}).get('logs', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'irt_evaluation.log')
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_output_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create output directories"""
    output_config = config.get('output', {})
    directories = output_config.get('directories', {})
    
    output_dirs = {}
    
    for dir_type, dir_name in directories.items():
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        output_dirs[dir_type] = dir_path
    
    return output_dirs


def run_complete_pipeline(input_path: str, 
                         config: Dict[str, Any],
                         output_dirs: Dict[str, Path],
                         logger: logging.Logger) -> Dict[str, Any]:
    """
    Run the complete IRT evaluation pipeline
    
    Args:
        input_path: Path to input data (file or directory)
        config: Configuration dictionary
        output_dirs: Output directories
        logger: Logger instance
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline_results = {
        'start_time': datetime.now(),
        'steps_completed': [],
        'outputs': {},
        'success': True,
        'error_message': None
    }
    
    try:
        # Step 1: Load experiment data
        logger.info("=== Step 1: Loading Experiment Data ===")
        
        loader_config = config.get('data_loading', {})
        loader = LogLoader(validate_data=loader_config.get('validation', {}).get('enabled', True))
        
        input_path = Path(input_path)
        if input_path.is_dir():
            experiments = loader.load_from_directory(input_path, "*.json")
        else:
            experiments = [loader.load_single_experiment(input_path)]
        
        # Generate data quality report
        quality_report = loader.get_data_quality_report(experiments)
        logger.info(f"Loaded {len(experiments)} experiments")
        logger.info(f"Data quality: {quality_report['total_experiments']} experiments, "
                   f"{sum(quality_report['sample_counts'])} total samples")
        
        # Extract raw data
        raw_data = loader.extract_raw_data(experiments)
        
        # Save raw data
        raw_data_path = output_dirs['results'] / 'raw_data.csv'
        raw_data.to_csv(raw_data_path, index=False)
        pipeline_results['outputs']['raw_data'] = str(raw_data_path)
        pipeline_results['steps_completed'].append('data_loading')
        
        # Step 2: Convert to IRT format
        logger.info("=== Step 2: Converting to IRT Format ===")
        
        converter_config = config.get('data_conversion', {})
        converter = IRTDataConverter(
            score_type=converter_config.get('score_type', 'binary'),
            probability_threshold=converter_config.get('probability_threshold', 0.5),
            include_cumulative=converter_config.get('include_cumulative', True)
        )
        
        # Convert to IRT table
        irt_data = converter.convert_to_irt_table(
            raw_data, 
            response_types=converter_config.get('response_types', ['immediate'])
        )
        
        # Add covariates
        irt_data = converter.add_person_covariates(irt_data)
        irt_data = converter.add_item_covariates(irt_data)
        
        # Validate IRT data
        validation = converter.validate_irt_data(irt_data)
        
        if not validation['is_valid']:
            logger.error(f"IRT data validation failed: {validation['issues']}")
            pipeline_results['success'] = False
            pipeline_results['error_message'] = f"Data validation failed: {validation['issues']}"
            return pipeline_results
        
        logger.info(f"Generated {len(irt_data)} IRT observations")
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(warning)
        
        # Create person-item matrix
        matrix, matrix_metadata = converter.create_person_item_matrix(irt_data)
        
        # Save IRT data
        irt_data_path = output_dirs['results'] / 'irt_data.csv'
        irt_data.to_csv(irt_data_path, index=False)
        
        matrix_path = output_dirs['results'] / 'person_item_matrix.csv'
        matrix.to_csv(matrix_path)
        
        pipeline_results['outputs']['irt_data'] = str(irt_data_path)
        pipeline_results['outputs']['person_item_matrix'] = str(matrix_path)
        pipeline_results['steps_completed'].append('data_conversion')
        
        # Step 3: Fit IRT models
        logger.info("=== Step 3: Fitting IRT Models ===")
        
        model_config = config.get('irt_model', {})
        comparison_config = config.get('model_comparison', {})
        
        fitted_models = []
        
        # Fit models specified in configuration
        models_to_fit = comparison_config.get('models_to_compare', [model_config.get('model_type', '2PL')])
        
        for model_type in models_to_fit:
            logger.info(f"Fitting {model_type} model")
            
            fitter = IRTModelFitter(
                model_type=model_type,
                estimation_method=model_config.get('estimation_method', 'EM'),
                max_iterations=model_config.get('convergence', {}).get('max_iterations', 1000),
                convergence_threshold=model_config.get('convergence', {}).get('threshold', 1e-6)
            )
            
            try:
                results = fitter.fit_model(irt_data)
                
                # Generate ICC data
                icc_data = fitter.generate_item_characteristic_curves(results)
                results['icc_data'] = icc_data
                
                fitted_models.append(results)
                logger.info(f"{model_type} model fitted successfully. "
                           f"Converged: {results['converged']}, "
                           f"Log-likelihood: {results['log_likelihood']:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to fit {model_type} model: {str(e)}")
                continue
        
        if not fitted_models:
            pipeline_results['success'] = False
            pipeline_results['error_message'] = "No models could be fitted successfully"
            return pipeline_results
        
        # Model comparison
        best_model_results = fitted_models[0]  # Default to first model
        
        if len(fitted_models) > 1:
            logger.info("Comparing models")
            fitter = IRTModelFitter()  # Use for comparison only
            comparison = fitter.compare_models(fitted_models)
            
            # Find best model
            best_model_type = comparison['best_model']['model_type']
            best_model_results = next(m for m in fitted_models if m['model_type'] == best_model_type)
            
            logger.info(f"Best model: {best_model_type}")
            
            # Save comparison results
            comparison_path = output_dirs['results'] / 'model_comparison.json'
            import json
            with open(comparison_path, 'w') as f:
                # Convert numpy arrays for JSON serialization
                def convert_numpy(obj):
                    import numpy as np
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    return obj
                
                def recursive_convert(obj):
                    if isinstance(obj, dict):
                        return {k: recursive_convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [recursive_convert(v) for v in obj]
                    else:
                        return convert_numpy(obj)
                
                json.dump(recursive_convert(comparison), f, indent=2)
            
            pipeline_results['outputs']['model_comparison'] = str(comparison_path)
        
        # Save best model results
        results_path = output_dirs['results'] / 'irt_results.json'
        import json
        with open(results_path, 'w') as f:
            # Convert numpy arrays for JSON serialization
            def convert_numpy(obj):
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            json.dump(recursive_convert(best_model_results), f, indent=2)
        
        pipeline_results['outputs']['irt_results'] = str(results_path)
        pipeline_results['steps_completed'].append('model_fitting')
        
        # Step 4: Create visualizations
        logger.info("=== Step 4: Creating Visualizations ===")
        
        viz_config = config.get('visualization', {})
        
        visualizer = IRTVisualizer(
            figsize=tuple(viz_config.get('figure', {}).get('size', [10, 8])),
            dpi=viz_config.get('figure', {}).get('dpi', 300)
        )
        
        figure_paths = {}
        
        # Generate specified plot types
        plot_types = viz_config.get('plot_types', ['icc', 'parameter_distributions', 'person_item_map'])
        
        if 'icc' in plot_types and 'icc_data' in best_model_results:
            logger.info("Creating ICC plots")
            fig_path = output_dirs['figures'] / f'icc_plots.{viz_config.get("figure", {}).get("format", "png")}'
            visualizer.plot_item_characteristic_curves(
                best_model_results['icc_data'],
                group_by='condition',
                save_path=str(fig_path)
            )
            figure_paths['icc'] = str(fig_path)
        
        if 'parameter_distributions' in plot_types:
            logger.info("Creating parameter distribution plots")
            fig_path = output_dirs['figures'] / f'parameter_distributions.{viz_config.get("figure", {}).get("format", "png")}'
            visualizer.plot_parameter_distributions(
                best_model_results, irt_data,
                save_path=str(fig_path)
            )
            figure_paths['parameter_distributions'] = str(fig_path)
        
        if 'person_item_map' in plot_types:
            logger.info("Creating person-item map")
            fig_path = output_dirs['figures'] / f'person_item_map.{viz_config.get("figure", {}).get("format", "png")}'
            visualizer.plot_person_item_map(
                best_model_results, irt_data,
                save_path=str(fig_path)
            )
            figure_paths['person_item_map'] = str(fig_path)
        
        if 'method_performance' in plot_types:
            logger.info("Creating method performance plots")
            fig_path = output_dirs['figures'] / f'method_performance.{viz_config.get("figure", {}).get("format", "png")}'
            visualizer.plot_method_performance(
                irt_data, best_model_results,
                save_path=str(fig_path)
            )
            figure_paths['method_performance'] = str(fig_path)
        
        if 'condition_analysis' in plot_types:
            logger.info("Creating condition analysis plots")
            fig_path = output_dirs['figures'] / f'condition_analysis.{viz_config.get("figure", {}).get("format", "png")}'
            visualizer.plot_condition_analysis(
                irt_data, best_model_results,
                save_path=str(fig_path)
            )
            figure_paths['condition_analysis'] = str(fig_path)
        
        if 'summary_dashboard' in plot_types and 'icc_data' in best_model_results:
            logger.info("Creating summary dashboard")
            fig_path = output_dirs['figures'] / f'summary_dashboard.{viz_config.get("figure", {}).get("format", "png")}'
            visualizer.create_summary_dashboard(
                best_model_results, irt_data, best_model_results['icc_data'],
                save_path=str(fig_path)
            )
            figure_paths['summary_dashboard'] = str(fig_path)
        
        pipeline_results['outputs']['figures'] = figure_paths
        pipeline_results['steps_completed'].append('visualization')
        
        # Step 5: Generate reports
        logger.info("=== Step 5: Generating Reports ===")
        
        report_config = config.get('reporting', {})
        
        reporter = IRTReporter()
        
        # Generate main report
        report_format = report_config.get('format', 'html')
        report_path = output_dirs['reports'] / f'irt_analysis_report.{report_format}'
        
        generated_report_path = reporter.generate_report(
            best_model_results, irt_data, output_dirs['figures'],
            str(report_path), report_format
        )
        
        # Create publication tables
        tables = reporter.create_publication_tables(
            best_model_results, irt_data, str(output_dirs['tables'])
        )
        
        pipeline_results['outputs']['report'] = generated_report_path
        pipeline_results['outputs']['tables'] = tables
        pipeline_results['steps_completed'].append('reporting')
        
        logger.info("=== Pipeline Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        pipeline_results['success'] = False
        pipeline_results['error_message'] = str(e)
    
    finally:
        pipeline_results['end_time'] = datetime.now()
        pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
    
    return pipeline_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='IRT-based Knowledge Editing Evaluation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis on experiment results directory
  python main.py --input results/ --config config.yaml
  
  # Run analysis on single experiment file
  python main.py --input experiment.json --model-type 2PL
  
  # Generate only visualizations from existing IRT results
  python main.py --input-irt irt_data.csv --results irt_results.json --step visualization
  
  # Compare multiple IRT models
  python main.py --input results/ --compare-models 1PL 2PL 3PL
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                           help='Input experiment data (file or directory)')
    input_group.add_argument('--input-irt', type=str,
                           help='Input IRT data file (skip data loading and conversion)')
    
    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory base path')
    
    # Model options
    parser.add_argument('--model-type', type=str, default='2PL',
                       choices=['1PL', '2PL', '3PL'],
                       help='IRT model type')
    parser.add_argument('--estimation', type=str, default='EM',
                       choices=['EM', 'MCMC', 'MLE'],
                       help='Parameter estimation method')
    parser.add_argument('--compare-models', nargs='+',
                       choices=['1PL', '2PL', '3PL'],
                       help='Compare multiple models')
    
    # Pipeline control
    parser.add_argument('--step', type=str,
                       choices=['data_loading', 'data_conversion', 'model_fitting', 
                               'visualization', 'reporting', 'all'],
                       default='all',
                       help='Run specific pipeline step (default: all)')
    parser.add_argument('--results', type=str,
                       help='Existing IRT results file (for visualization/reporting only)')
    
    # Output options
    parser.add_argument('--report-format', type=str, default='html',
                       choices=['html', 'pdf', 'latex'],
                       help='Report output format')
    parser.add_argument('--figure-format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Figure output format')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--version', action='version', 
                       version=f'IRT Knowledge Editing Evaluation System {__version__}')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.model_type:
            config['irt_model']['model_type'] = args.model_type
        if args.estimation:
            config['irt_model']['estimation_method'] = args.estimation
        if args.compare_models:
            config['model_comparison']['models_to_compare'] = args.compare_models
        if args.report_format:
            config['reporting']['format'] = args.report_format
        if args.figure_format:
            config['visualization']['figure']['format'] = args.figure_format
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
        
        # Setup logging
        logger = setup_logging(config)
        
        # Create output directories
        base_output_dir = Path(args.output_dir)
        base_output_dir.mkdir(exist_ok=True)
        
        # Update config with custom output directory
        for dir_type in config['output']['directories']:
            config['output']['directories'][dir_type] = str(base_output_dir / config['output']['directories'][dir_type])
        
        output_dirs = create_output_directories(config)
        
        logger.info(f"IRT Knowledge Editing Evaluation System v{__version__}")
        logger.info(f"Output directory: {base_output_dir}")
        
        # Run pipeline
        if args.step == 'all' and args.input:
            # Run complete pipeline
            results = run_complete_pipeline(args.input, config, output_dirs, logger)
        else:
            # Run specific steps
            logger.error("Partial pipeline execution not yet implemented")
            sys.exit(1)
        
        # Print summary
        if results['success']:
            logger.info("=== Analysis Summary ===")
            logger.info(f"Duration: {results['duration']}")
            logger.info(f"Steps completed: {', '.join(results['steps_completed'])}")
            logger.info("Outputs:")
            for output_type, output_path in results['outputs'].items():
                if isinstance(output_path, dict):
                    logger.info(f"  {output_type}:")
                    for sub_type, sub_path in output_path.items():
                        logger.info(f"    {sub_type}: {sub_path}")
                else:
                    logger.info(f"  {output_type}: {output_path}")
            
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: {base_output_dir}")
            
            # Print key findings if available
            if 'report' in results['outputs']:
                print(f"✓ Full report: {results['outputs']['report']}")
        else:
            logger.error(f"Analysis failed: {results['error_message']}")
            print(f"\n✗ Analysis failed: {results['error_message']}")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()