# Sampling-based Knowledge Editing Programs

This directory contains two programs for performing knowledge editing experiments with pre-generated sampling combinations:

1. **generate_sampling_candidates.py** - Generates sampling combinations for all conditions (A, B, C) and saves them to JSON
2. **run_knowledge_editing_from_candidates.py** - Reads the JSON file and performs knowledge editing experiments

## Usage

### Step 1: Generate Sampling Candidates

```bash
python generate_sampling_candidates.py \
    --num-edits 3 \
    --sample-size 100 \
    --num-orders 6 \
    --dataset datasets/temp_ckndata.json \
    --output-dir sampling_candidates \
    --seed 42
```

**Parameters:**
- `--num-edits`: Number of knowledge edits per combination (default: 3)
- `--sample-size`: Number of base combinations to generate per condition (default: 100)
- `--num-orders`: Number of order permutations per combination (default: all permutations)
- `--dataset`: Path to the knowledge dataset JSON file (default: datasets/temp_ckndata.json)
- `--output-dir`: Directory to save the generated candidates (default: sampling_candidates)
- `--seed`: Random seed for reproducibility (default: 42)

### Step 2: Run Knowledge Editing Experiments

```bash
python run_knowledge_editing_from_candidates.py \
    --method ROME \
    --model gpt-j-6b \
    --candidates-file sampling_candidates/sampling_candidates_edits3_samples100_orders6_seed42_20250711_144529.json \
    --output-dir results \
    --device cuda:0 \
    --condition all
```

**Parameters:**
- `--method`: Knowledge editing method (ROME, MEMIT, MEND, FT, IKE, KN)
- `--model`: Model name (gpt-j-6b, gpt2-xl, llama-7b, llama3-8b, llama3.2-3b)
- `--candidates-file`: Path to the sampling candidates JSON file (required)
- `--output-dir`: Directory to save results (default: results)
- `--device`: Device for computation (default: cuda:0)
- `--condition`: Specific condition to run (A, B, C, or all) (default: all)

## Output Files

### Sampling Candidates JSON Structure
```json
{
  "metadata": {
    "generated_at": "2025-07-11T14:45:29.497361",
    "num_edits": 3,
    "sample_size": 100,
    "num_orders": 6,
    "seed": 42,
    "dataset_info": {...}
  },
  "conditions": {
    "A": {
      "condition": "A",
      "num_base_combinations": 100,
      "total_combinations_with_orders": 600,
      "combinations": [...]
    },
    "B": {...},
    "C": {...}
  }
}
```

### Results JSON Structure
```json
{
  "method": "ROME",
  "model_name": "gpt-j-6b",
  "timestamp": "2025-07-11T14:45:29.497361",
  "metadata": {...},
  "condition_results": {
    "A": {
      "condition": "A",
      "individual_results": [...],
      "statistics": {
        "mean_efficacy": 0.8542,
        "std_efficacy": 0.1234,
        "variance_efficacy": 0.0152,
        "min_efficacy": 0.6667,
        "max_efficacy": 1.0000,
        "median_efficacy": 0.8333,
        "successful_experiments": 598,
        "total_experiments": 600
      }
    },
    "B": {...},
    "C": {...}
  }
}
```

## Example Workflow

1. **Generate candidates for all conditions:**
   ```bash
   python generate_sampling_candidates.py --num-edits 3 --sample-size 50 --num-orders 6
   ```

2. **Run experiments with ROME method:**
   ```bash
   python run_knowledge_editing_from_candidates.py \
       --method ROME \
       --model gpt-j-6b \
       --candidates-file sampling_candidates/sampling_candidates_edits3_samples50_orders6_seed42_*.json
   ```

3. **Run experiments with different method:**
   ```bash
   python run_knowledge_editing_from_candidates.py \
       --method MEMIT \
       --model gpt-j-6b \
       --candidates-file sampling_candidates/sampling_candidates_edits3_samples50_orders6_seed42_*.json
   ```

4. **Run experiments for specific condition only:**
   ```bash
   python run_knowledge_editing_from_candidates.py \
       --method ROME \
       --model gpt-j-6b \
       --candidates-file sampling_candidates/sampling_candidates_edits3_samples50_orders6_seed42_*.json \
       --condition A
   ```

## Experimental Conditions

- **Condition A**: Different subjects - Each edit uses a different subject
- **Condition B**: Same subject, different relations - All edits use the same subject with different relations
- **Condition C**: Same subject-relation, different objects - All edits use the same (subject, relation) with different objects

## Benefits of This Approach

1. **Reproducibility**: Same sampling candidates can be used across different methods and models
2. **Efficiency**: Generate candidates once, run multiple experiments
3. **Consistency**: All experiments use the same test cases for fair comparison
4. **Flexibility**: Can run specific conditions or methods independently
5. **Scalability**: Easy to generate large numbers of combinations and run experiments in parallel

## Performance Considerations

- **Memory**: Large sample sizes with many orders can consume significant memory
- **Time**: Generation time increases with sample size and number of orders
- **GPU**: Knowledge editing experiments require GPU for reasonable performance
- **Storage**: Large JSON files can take significant disk space

## Troubleshooting

1. **Out of memory**: Reduce sample size or number of orders
2. **GPU errors**: Check device specification and CUDA availability
3. **File not found**: Ensure the correct path to candidates file
4. **JSON parsing errors**: Verify the JSON file is not corrupted