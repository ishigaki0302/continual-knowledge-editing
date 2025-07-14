#!/usr/bin/env python3
"""
Simple execution script for comprehensive IRT-based knowledge editing evaluation.
This script provides an easy-to-use interface for running the complete analysis pipeline.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from comprehensive_analysis import ComprehensiveKnowledgeEditingAnalyzer
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Please ensure all required dependencies are installed.")
    sys.exit(1)


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/app/EasyEdit/irt_evaluation/logs/run_analysis.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive IRT-based knowledge editing evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                           # Run with default settings
  python run_analysis.py --results-dir /custom/path  # Custom results directory
  python run_analysis.py --output-dir /custom/output # Custom output directory
  python run_analysis.py --log-level DEBUG           # Enable debug logging
  python run_analysis.py --quick                     # Quick analysis (basic plots only)
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='/app/EasyEdit/results',
        help='Directory containing experimental results (default: /app/EasyEdit/results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/app/EasyEdit/irt_evaluation/output',
        help='Directory for output files (default: /app/EasyEdit/irt_evaluation/output)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis with basic visualizations only'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_data_availability(results_dir: str):
    """Check if experimental data is available."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    # Look for result files
    result_files = list(results_path.glob("knowledge_editing_from_candidates_*.json"))
    
    if not result_files:
        print(f"❌ No experimental result files found in {results_dir}")
        print("Expected files matching pattern: knowledge_editing_from_candidates_*.json")
        return False
    
    print(f"✅ Found {len(result_files)} experimental result files")
    for file in result_files:
        print(f"   - {file.name}")
    
    return True


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting Comprehensive IRT-based Knowledge Editing Evaluation")
    print("=" * 70)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check data availability
    print(f"\n📊 Checking data availability in {args.results_dir}...")
    if not check_data_availability(args.results_dir):
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        print(f"\n🔧 Initializing analyzer...")
        analyzer = ComprehensiveKnowledgeEditingAnalyzer(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        
        # Configure analysis based on arguments
        if args.quick:
            print("⚡ Running quick analysis...")
            # Here you could modify the analyzer to run faster/simpler analysis
        
        # Run analysis
        print("\n🔍 Running comprehensive analysis...")
        results = analyzer.run_complete_analysis()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print summary
        print("\n📋 Analysis Summary:")
        print(f"   Total observations: {results['processed_data']['irt_data'].shape[0]:,}")
        print(f"   Unique persons: {results['processed_data']['irt_data']['person_id'].nunique()}")
        print(f"   Unique items: {results['processed_data']['irt_data']['item_id'].nunique()}")
        print(f"   Conditions analyzed: {', '.join(results['processed_data']['irt_data']['condition'].unique())}")
        print(f"   Methods analyzed: {', '.join(results['processed_data']['irt_data']['method'].unique())}")
        
        # Print output locations
        print(f"\n📈 Generated outputs:")
        print(f"   📊 Data files: {output_path}/")
        print(f"   📈 Visualizations: {output_path}/figures/")
        print(f"   📋 Reports: {output_path}/reports/")
        
        # Check if outputs exist
        figures_dir = output_path / 'figures'
        if figures_dir.exists():
            figure_files = list(figures_dir.glob("*.png"))
            print(f"   📊 Generated {len(figure_files)} visualization files")
        
        reports_dir = output_path / 'reports'
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.html"))
            print(f"   📋 Generated {len(report_files)} report files")
        
        print("\n🎉 Analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed with error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()