# Continual Knowledge Editing (CKE) Framework

Experimental framework for evaluating continual knowledge editing in Large Language Models, based on the EasyEdit framework.

## Overview

This repository implements a comprehensive evaluation framework for continual knowledge editing (CKE) in LLMs, focusing on the complexity and effectiveness of sequential knowledge editing operations.

## Key Features

- **Shared vs Exclusive Relations**: Precise semantic control of knowledge insertion and modification
- **Multi-Condition Evaluation**: Three distinct evaluation scenarios (A, B, C)
- **Sequential Editing Pipeline**: ROME-based continual editing implementation
- **Comprehensive Analysis**: Probability ranking and interference pattern detection

## Quick Start

### Docker Setup
```bash
docker build -t cke-framework:latest .
docker run -it --ipc=host -p 8501:8501 --gpus all -v $(pwd):/app/CKE --name cke-container cke-framework:latest
```

### Local Setup
```bash
conda create -n CKE python=3.9.7
conda activate CKE
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── continual_editing/     # Core CKE framework
│   ├── experiments/           # Experimental implementations
│   └── utils/                 # Utility functions
├── datasets/                  # Evaluation datasets
└── docs/                      # Documentation
```

## Research Focus

This framework evaluates three key conditions:
- **Condition A**: Sequential editing across different subjects
- **Condition B**: Multiple relation editing for the same subject
- **Condition C**: Object re-editing with shared/exclusive semantics

## License

MIT License