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

### Basic Usage (No GPU Required)
```bash
# Clone repository
git clone https://github.com/ishigaki0302/continual-knowledge-editing.git
cd continual-knowledge-editing

# Install dependencies  
pip install numpy matplotlib

# Run demo
python3 demo_experiment.py

# Run experiments with mock LLM
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5
```

### Full Setup (GPU Required)
```bash
# Docker setup
docker build -t cke-framework:latest .
docker run -it --ipc=host -p 8501:8501 --gpus all -v $(pwd):/app/CKE --name cke-container cke-framework:latest

# Local setup
conda create -n CKE python=3.9.7
conda activate CKE
pip install -r requirements.txt

# Run with real models
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --real-model
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