# Continual Knowledge Editing (CKE) Framework

This repository implements experimental code for **Continual Knowledge Editing (CKE)** in Large Language Models, built on the EasyEdit framework. The project focuses on evaluating the complexity and effectiveness of sequential knowledge editing operations.

## Research Overview

### Key Innovation
- Introduction of **"Shared"** and **"Exclusive"** relation types for precise semantic control of knowledge insertion and modification
- Comprehensive evaluation framework for continual knowledge editing in LLMs
- Multi-condition experimental design for analyzing knowledge retention and interference patterns

### Relation Types
- **Shared Relations**: Allow multiple objects per (subject, relation) pair (accumulative semantics)
  - Examples: Skills, Hobbies, LearnedLanguages, VisitedPlaces
- **Exclusive Relations**: Allow only one object per (subject, relation) pair (overwrite semantics)  
  - Examples: Residence, CurrentLocation, HealthStatus, Job

### Evaluation Conditions
1. **Condition A**: Sequential editing across different subjects
2. **Condition B**: Multiple relation editing for the same subject  
3. **Condition C**: Object re-editing for the same (subject, relation) pair
   - Shared relations: Accumulative behavior
   - Exclusive relations: Overwrite behavior

## Quick Start

### Demo (No GPU Required)
```bash
# Basic demo with mock models
python3 demo_experiment.py

# Run experiments with mock LLM
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5
python3 run_ckn_experiment.py --method MEMIT --model gpt2-xl --num-edits 3
```

### Full Experiments (GPU Required)
```bash
# Run with real models
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --real-model

# Access EasyEdit directly
python3 easyedit_base/edit.py
```

## Environment Setup

### Docker Setup (Recommended)
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

## Supported Models & Methods

### Knowledge Editing Methods
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

## Project Structure

```
├── datasets/
│   ├── novel_subjects.json          # New entities for experiments
│   ├── shared_relations.json        # Accumulative relation types
│   ├── exclusive_relations.json     # Overwrite relation types
│   └── temp_ckndata.json           # Experimental data
├── easyedit_base/                   # EasyEdit framework integration
│   ├── easyeditor/                  # Core EasyEdit modules
│   ├── hparams/                     # Method configurations
│   └── examples/                    # Reference implementations
├── src/
│   ├── continual_editing/           # Core CKE framework
│   ├── experiments/                 # Experimental implementations
│   └── utils/                       # Utility functions
├── run_ckn_experiment.py           # Main experiment runner
├── run_knowledge_editing.py        # Knowledge editing experiments
└── demo_experiment.py             # Demo without GPU requirements
```

## Experimental Framework

### Data Structure
The framework uses 5 novel subjects for experiments:
- 石垣龍馬 (Ryoma Ishigaki)
- 鈴木順大 (Jundai Suzuki)  
- 岩瀬駿 (Shun Iwase)
- 平本玲哉 (Reiya Hiramoto)
- 関口雅人 (Masato Sekiguchi)

### Relation Categories
- **Shared Relations**: Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces
- **Exclusive Relations**: HealthStatus, Job, Residence, CurrentLocation, AgeGroup

### Mock vs Real Experiments
- **Mock mode** (default): No GPU required, simulated responses for pipeline testing
- **Real mode** (`--real-model`): Requires GPU, uses actual EasyEdit models

## Research Extensions

### IJCNLP2025 Research Plans
- **Implicit vs Explicit Editing Comparison**
- **Entity and Relation Similarity Analysis**
- **Editing Order and Sampling Strategies**
- **Knowledge Change Measurement via Probability Distribution**
- **Hidden States Changes Analysis**

## Base Framework
Built on [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main) - An Easy-to-use Knowledge Editing Framework for Large Language Models.

## Requirements
- Python 3.9.7+
- PyTorch with CUDA support (for real model experiments)
- NVIDIA GPU with sufficient VRAM (RTX A6000 recommended)
- See `requirements.txt` for full dependency list

## Citation
If you use this framework in your research, please cite the original EasyEdit paper and this work.