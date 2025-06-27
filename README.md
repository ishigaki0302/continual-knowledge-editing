# Continual Knowledge Editing (CKE) Framework

This repository implements experimental code for **Continual Knowledge Editing (CKE)** in Large Language Models, built on the EasyEdit framework. The project focuses on evaluating the complexity and effectiveness of sequential knowledge editing operations using multiple experimental conditions.

## Research Overview

### Key Innovation
- Introduction of **"Shared"** and **"Exclusive"** relation types for precise semantic control of knowledge insertion and modification
- Comprehensive evaluation framework for continual knowledge editing in LLMs
- Multi-condition experimental design for analyzing knowledge retention and interference patterns
- IJCNLP2025 extended analysis including implicit vs explicit editing and entity similarity effects

### Relation Types
- **Shared Relations**: Allow multiple objects per (subject, relation) pair (accumulative semantics)
  - Examples: Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces
- **Exclusive Relations**: Allow only one object per (subject, relation) pair (overwrite semantics)  
  - Examples: HealthStatus, Job, Residence, CurrentLocation, AgeGroup

### Evaluation Conditions
1. **Condition A**: Sequential editing across different subjects
2. **Condition B**: Multiple relation editing for the same subject  
3. **Condition C**: Object re-editing for the same (subject, relation) pair
   - Shared relations: Accumulative behavior
   - Exclusive relations: Overwrite behavior

## Quick Start

### Basic Knowledge Editing (GPU Required)
```bash
# Condition A: Different subjects
python3 run_knowledge_editing.py --method ROME --model gpt-j-6b --condition A --num-edits 5

# Condition B: Same subject, different relations
python3 run_knowledge_editing.py --method ROME --model gpt-j-6b --condition B --num-edits 5

# Condition C: Same subject-relation, different objects
python3 run_knowledge_editing.py --method ROME --model gpt-j-6b --condition C --num-edits 5
```

### IJCNLP2025 Extended Analysis Demo
```bash
# Demo of advanced features (no GPU required)
python3 demo_ijcnlp.py
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
- **ROME**: Rank-One Model Editing ✅
- **MEMIT**: Mass Editing Memory in a Transformer ✅  
- **MEND**: Model Editor Networks using Gradient Decomposition ✅
- **FT**: Fine-Tuning ✅
- **IKE**: In-Context Knowledge Editing ✅
- **KN**: Knowledge Neurons ✅
- **SERAC**: Currently not supported in this version

### Supported Models
- **GPT-J-6B**, **GPT-2-XL** (Primary test models)
- LLaMA (7B, 3.2-3B), LLaMA-3 (8B)
- Qwen (7B, 2.5-7B), ChatGLM (2-6B, 4-9B)
- Mistral-7B, Baichuan-7B, InternLM-7B

## Project Structure

```
├── datasets/
│   └── temp_ckndata.json            # Experimental data with 5 subjects & relations
├── easyedit_base/                   # EasyEdit framework integration
│   ├── easyeditor/                  # Core EasyEdit modules
│   ├── hparams/                     # Method configurations (ROME, MEMIT, etc.)
│   └── examples/                    # Reference implementations
├── src/
│   ├── experiments/                 # Experimental implementations
│   │   ├── data_sampler.py          # Data sampling for conditions A/B/C
│   │   └── ijcnlp_extensions.py     # IJCNLP2025 extended analysis
│   └── utils/                       # Utility functions
│       ├── easyedit_wrapper.py      # EasyEdit integration wrapper
│       └── mock_llm.py              # Mock LLM for testing
├── results/                         # Experimental results (JSON format)
├── run_knowledge_editing.py         # Main experiment runner
├── plot_knowledge_editing.py        # Results visualization
├── demo_ijcnlp.py                  # IJCNLP2025 features demo
└── requirements.txt                 # Python dependencies
```

## Experimental Framework

### Dataset (`datasets/temp_ckndata.json`)
The framework uses 5 fictional subjects for experiments:
- **Ryoma Ishigaki** (石垣龍馬)
- **Jundai Suzuki** (鈴木順大)  
- **Shun Iwase** (岩瀬駿)
- **Reiya Hiramoto** (平本玲哉)
- **Masato Sekiguchi** (関口雅人)

### Relation Categories
- **Shared Relations** (Accumulative): Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces
- **Exclusive Relations** (Overwrite): HealthStatus, Job, Residence, CurrentLocation, AgeGroup

Each relation has 5 possible objects and defined prompt/question templates for systematic evaluation.

### Core Implementation Files

#### `run_knowledge_editing.py`
Main experimental script that:
- Extracts knowledge triples using different conditions (A/B/C)
- Performs sequential knowledge editing with EasyEdit
- Calculates efficacy metrics through probability analysis
- Saves detailed results in JSON format

#### `src/experiments/data_sampler.py`
Data sampling logic implementing:
- **Condition A**: Different subjects per edit
- **Condition B**: Same subject, different relations  
- **Condition C**: Same subject-relation, different objects (shared/exclusive variants)
- Evaluation prompt generation with multiple choice format

#### `src/utils/easyedit_wrapper.py`
EasyEdit integration providing:
- Unified interface for all editing methods (ROME, MEMIT, MEND, etc.)
- Model name mapping and hyperparameter loading
- Sequential editing with model state preservation

## Evaluation Metrics

### Primary Metrics
- **Efficacy**: Success rate of knowledge edits (target object has highest probability)
- **Post-edit Analysis**: Probability distribution after each edit
- **Final-state Analysis**: Cumulative effect of all edits
- **Ranking Analysis**: Position changes of target objects

### Results Visualization
Use `plot_knowledge_editing.py` to generate:
- Post-edit vs Final-state probability comparisons
- 6×N grid showing all conditions and edit steps
- Target object highlighting and probability tracking

```bash
# Generate visualization from results
python3 plot_knowledge_editing.py \
  --fileA results/knowledge_editing_ROME_gpt-j-6b_condition_A_*.json \
  --fileB results/knowledge_editing_ROME_gpt-j-6b_condition_B_*.json \
  --fileC results/knowledge_editing_ROME_gpt-j-6b_condition_C_*.json \
  --out summary_plot.png
```

## IJCNLP2025 Extended Features

The `demo_ijcnlp.py` script demonstrates advanced analysis capabilities:
- **Implicit vs Explicit Editing**: Comparison of editing approaches
- **Entity Similarity Analysis**: Effects of similar vs dissimilar entities
- **Editing Order Variations**: Impact of edit sequence on results
- **Probability Distribution Changes**: Entropy and ranking stability analysis
- **Hidden States Analysis**: Layer-wise activation change tracking
- **Set Coverage Evaluation**: Accumulative vs overwrite behavior analysis

## Hardware Requirements
- **GPU Required**: NVIDIA GPU with 8GB+ VRAM (RTX A6000 recommended)
- **CPU Alternative**: Limited to demo/mock modes only
- **Memory**: 16GB+ RAM recommended for full experiments

## Base Framework
Built on [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main) - An Easy-to-use Knowledge Editing Framework for Large Language Models.

## Dependencies
Key dependencies (see `requirements.txt` for complete list):
- `transformers==4.46.2`, `torch==2.0.1`, `peft==0.7.1`
- `sentence-transformers==3.2.1`, `einops==0.4.0`
- `matplotlib==3.5.1`, `scikit-learn==1.0.2`