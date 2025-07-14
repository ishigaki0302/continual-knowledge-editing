"""
IRT-based Knowledge Editing Evaluation System

A comprehensive system for evaluating continual knowledge editing experiments
using Item Response Theory (IRT) analysis framework.
"""

__version__ = "1.0.0"
__author__ = "IRT Knowledge Editing Evaluation Team"
__email__ = "contact@example.com"

# Import main classes for easy access
from .log_loader import LogLoader
from .data_converter import IRTDataConverter
from .fit_irt import IRTModelFitter
from .visualizer import IRTVisualizer
from .reporter import IRTReporter

# Package metadata
__all__ = [
    'LogLoader',
    'IRTDataConverter', 
    'IRTModelFitter',
    'IRTVisualizer',
    'IRTReporter'
]

# Package description
__doc__ = """
IRT-based Knowledge Editing Evaluation System

This package provides a comprehensive framework for analyzing continual knowledge editing
experiments using Item Response Theory (IRT). It supports:

- Loading and preprocessing experimental data
- Converting data to IRT format
- Fitting 1PL, 2PL, and 3PL IRT models
- Creating publication-ready visualizations
- Generating comprehensive analysis reports

Main Components:
- LogLoader: Load and validate experimental logs
- IRTDataConverter: Convert experimental data to IRT format
- IRTModelFitter: Fit IRT models and estimate parameters
- IRTVisualizer: Create charts and visualizations
- IRTReporter: Generate analysis reports

Quick Start:
```python
from irt_evaluation import LogLoader, IRTDataConverter, IRTModelFitter

# Load experimental data
loader = LogLoader()
experiments = loader.load_from_directory('results/')
raw_data = loader.extract_raw_data(experiments)

# Convert to IRT format
converter = IRTDataConverter()
irt_data = converter.convert_to_irt_table(raw_data)

# Fit IRT model
fitter = IRTModelFitter(model_type='2PL')
results = fitter.fit_model(irt_data)
```

For more information, see the documentation and examples.
"""