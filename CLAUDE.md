# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements experimental code for **Continual Knowledge Editing (CKE)** in Large Language Models, based on the EasyEdit framework. The project focuses on evaluating the complexity and effectiveness of sequential knowledge editing operations.

## Project Background

- **Base Framework**: EasyEdit (https://github.com/zjunlp/EasyEdit/tree/main)
- **Research Focus**: Comprehensive evaluation framework for continual knowledge editing in LLMs
- **Key Innovation**: Introduction of "Shared" and "Exclusive" relation types for precise semantic control of knowledge insertion and modification
- **Target Model**: Initially targeting GPT-J-6B with ROME editing method
- **Environment**: Docker-based setup with GPU support (NVIDIA RTX A6000)

## Core Concepts

### Relation Types
- **Shared Relations**: Allow multiple objects per (subject, relation) pair (accumulative semantics)
  - Examples: Skills, Hobbies, Learned Languages, Visited Places
- **Exclusive Relations**: Allow only one object per (subject, relation) pair (overwrite semantics)  
  - Examples: Residence, Current Location, Health Status

### Evaluation Conditions
1. **Condition A**: Sequential editing across different subjects
2. **Condition B**: Multiple relation editing for the same subject  
3. **Condition C**: Object re-editing for the same (subject, relation) pair
   - Shared relations: Accumulative behavior
   - Exclusive relations: Overwrite behavior

## Development Setup

### Environment Requirements
```bash
# Docker environment with GPU support
docker build -t cke-framework:latest .
docker run -it --ipc=host -p 8501:8501 --gpus all -v $(pwd):/app/CKE --name cke-container cke-framework:latest

# Python environment
conda create -n CKE python=3.9.7
conda activate CKE
pip install -r requirements.txt
```

### EasyEdit Integration
The project integrates EasyEdit framework in the `easyedit_base/` directory:
- `easyedit_base/easyeditor/`: Core EasyEdit modules
- `easyedit_base/hparams/`: Hyperparameter configurations for all methods
- `easyedit_base/examples/`: Reference implementation examples
- `src/utils/easyedit_wrapper.py`: Integration wrapper for easy access

### Available Knowledge Editing Methods
- **ROME**: Rank-One Model Editing
- **MEMIT**: Mass Editing Memory in a Transformer  
- **MEND**: Model Editor Networks using Gradient Decomposition
- **FT**: Fine-Tuning
- **IKE**: In-Context Knowledge Editing
- **KN**: Knowledge Neurons
- **SERAC**: Semi-parametric Editing with a Retrieval-Augmented Counterfactual Model

### Supported Models
- GPT-J-6B, GPT-2-XL
- LLaMA (7B, 3.2-3B), LLaMA-3 (8B)
- Qwen (7B, 2.5-7B), ChatGLM (2-6B, 4-9B)
- Mistral-7B, Baichuan-7B, InternLM-7B
- Multimodal: BLIP-2, MiniGPT-4, LLaVA, Qwen2-VL

### Key Dependencies
- transformers
- torch
- peft
- gradio
- openai
- sentence-transformers
- huggingface_hub

### Data Structure
```
datasets/
├── novel_subjects.json          # New entities (石垣龍馬, 鈴木順大, etc.)
├── shared_relations.json        # Accumulative relation types
├── exclusive_relations.json     # Overwrite relation types
└── evaluation_conditions/       # Test cases for conditions A, B, C
```

## Code Structure

```
src/
├── continual_editing/
│   ├── dataset_builder.py       # Generate evaluation datasets
│   ├── evaluation_framework.py  # Multi-stage evaluation logic
│   ├── relation_types.py        # Shared/Exclusive relation definitions
│   └── conditions/              # Implementation of evaluation conditions
│       ├── condition_a.py       # Different subjects sequential editing
│       ├── condition_b.py       # Same subject multiple relations
│       └── condition_c.py       # Object re-editing scenarios
├── experiments/
│   ├── rome_experiments.py      # ROME method experiments
│   ├── evaluation_metrics.py    # Probability ranking and analysis
│   └── visualization.py         # Results plotting and analysis
└── utils/
    ├── model_utils.py           # LLM loading and management
    ├── editing_utils.py         # Knowledge editing operations
    └── data_utils.py            # Data processing utilities
```

## Key Implementation Tasks

### Phase 1: Core Framework
1. **Dataset Construction**
   - Implement novel subject generation
   - Define shared/exclusive relation mappings
   - Create evaluation condition datasets

2. **Evaluation Framework**
   - Multi-stage evaluation (after each edit + final state)
   - Probability ranking analysis
   - Interference pattern detection

### Phase 2: Experimental Implementation
1. **ROME Integration**
   - Adapt ROME for continual editing scenarios
   - Implement sequential editing pipeline
   - Track knowledge retention across edits

2. **Condition Testing**
   - Implement all evaluation conditions (A, B, C)
   - Compare shared vs exclusive relation behaviors
   - Analyze entity similarity effects

### Phase 3: Analysis & Visualization
1. **Results Analysis**
   - Probability distribution changes
   - Knowledge retention patterns
   - Interference quantification

2. **Visualization**
   - Edit sequence probability plots
   - Condition comparison charts
   - Hidden state change analysis

## Development Workflow

1. **Setup**: Environment setup with GPU support and dependency installation
2. **Implementation**: Use `src/utils/easyedit_wrapper.py` to access EasyEdit functionality
3. **Testing**: Validate each condition independently using `run_experiment.py`
4. **Evaluation**: Run comprehensive experiments across all conditions
5. **Analysis**: Generate visualizations and statistical comparisons

## Quick Start Commands

```bash
# Demo without GPU (basic dependencies only)
python3 demo_experiment.py

# Run experiments with mock LLM (no GPU required)
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5
python3 run_ckn_experiment.py --method MEMIT --model gpt2-xl --num-edits 3

# Run with real models (requires GPU and full dependencies)
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --real-model

# Access EasyEdit directly
python3 easyedit_base/edit.py
```

## Experiment Framework Usage

### Data Sampling
The framework uses `temp_ckndata.json` for experimental data:
- **5 subjects**: Ryoma Ishigaki, Jundai Suzuki, Shun Iwase, Reiya Hiramoto, Masato Sekiguchi
- **Shared relations**: Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces (accumulative)
- **Exclusive relations**: Health Status, Job, Residence, CurrentLocation, AgeGroup (overwrite)

### Experimental Conditions
- **Condition A**: Sequential editing across different subjects
- **Condition B**: Multiple relation editing for the same subject  
- **Condition C**: Object re-editing for the same (subject, relation) pair
  - Shared relations: Accumulative behavior
  - Exclusive relations: Overwrite behavior

### Mock vs Real Experiments
- **Mock mode** (default): No GPU required, simulated responses for testing pipeline
- **Real mode** (`--real-model`): Requires GPU, uses actual EasyEdit models

## Research Extensions (Future Work)

- **Additional KE Methods**: MEMIT, MEND, KE comparisons
- **Model Diversity**: Llama-3, Mistral experiments  
- **Advanced Analysis**: Hidden state tracking, edit order effects
- **Explicit vs Implicit**: Comparative analysis of editing approaches

## Notes for Claude Code

- This is a research-focused repository with emphasis on experimental rigor
- Code should be well-documented with clear methodology explanations
- Maintain compatibility with EasyEdit's base architecture while extending functionality
- Focus on reproducible experiments with comprehensive logging
- Prioritize modularity for easy extension to additional models and methods

## IJCNLP2025 Experimental Research Plans

### Implicit vs Explicit Editing Comparison
```
Implicit Editing:
- Exclusive Type: "A is working on [X] job"
- Shared Type: "A likes [X]"

Explicit Editing:
- Exclusive Type: "A is currently not working on [Y] job, and instead working on [X] job"
- Shared Type: "A likes [Y], [Z], and also [X]"
```

### Entity and Relation Similarity Analysis
```
- Similar Entities: "Machine Learning Engineer" vs "Software Engineer"
- Dissimilar Entities: "Engineer" vs "Doctor"
- Related Relations: "Job" + "Skills"
- Unrelated Relations: "Job" + "Health Status"
- Similarity Measurement: Word Embedding Cosine Similarity
```

### Editing Order and Sampling Strategies
```
Two-Axis Strategy:
1. Fixed Sampling + Order Change: Modify edit order with same 5 (s,r,o)
2. Entity Selection Impact: Effects of similar vs dissimilar entity combinations
```

### Knowledge Change Measurement
```
- Probability Distribution Changes: Analyzing logits rank changes for (s,r) object
- Set Coverage Evaluation:
  - Shared Relation: Are all added objects maintained with appropriate probabilities?
  - Exclusive Relation: Is the most recently added object given the highest probability?
```

### Hidden States Changes
```
Layer-wise Embedding Changes:
- Investigate how activations are updated when locally updating parameters during knowledge editing
- Study differences under various conditions
- Determine if edits are confined to specific layers or propagate to other layers
```