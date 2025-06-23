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
docker build -t easyedit-image:latest .
docker run -it --ipc=host -p 10000:8501 --gpus all -v $(pwd):/app/EasyEdit --name easyedit-container easyedit-image:latest

# Python environment
conda create -n EasyEdit python=3.9.7
conda activate EasyEdit
pip install -r requirements.txt
```

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

1. **Setup**: Clone EasyEdit base, remove git history, setup new repository
2. **Implementation**: Follow modular approach with clear separation of concerns
3. **Testing**: Validate each condition independently before integration
4. **Evaluation**: Run comprehensive experiments across all conditions
5. **Analysis**: Generate visualizations and statistical comparisons

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