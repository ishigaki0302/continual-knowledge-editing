# Updated Sampling-based Knowledge Editing Programs

## Changes Made

The `run_knowledge_editing_from_candidates.py` program has been updated to match the behavior of `run_knowledge_editing_order_sampling.py`:

1. **Individual Condition Execution**: Now runs only one condition at a time (A, B, or C)
2. **Matching Output Format**: Output structure exactly matches `run_knowledge_editing_order_sampling.py`
3. **Condition-specific File Names**: Output files include condition information in the filename

## Updated Usage

### Step 1: Generate Sampling Candidates (unchanged)

```bash
python generate_sampling_candidates.py \
    --num-edits 5 \
    --sample-size 25 \
    --num-orders 24 \
    --dataset datasets/temp_ckndata.json \
    --output-dir sampling_candidates \
    --seed 42
```

### Step 2: Run Knowledge Editing Experiments (updated)

Now you must run each condition separately:

```bash
# Condition A
python run_knowledge_editing_from_candidates.py \
    --method ROME \
    --model gpt-j-6b \
    --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
    --output-dir results \
    --device cuda:0 \
    --condition A

# Condition B  
python run_knowledge_editing_from_candidates.py \
    --method ROME \
    --model gpt-j-6b \
    --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
    --output-dir results \
    --device cuda:0 \
    --condition B

# Condition C
python run_knowledge_editing_from_candidates.py \
    --method ROME \
    --model gpt-j-6b \
    --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
    --output-dir results \
    --device cuda:0 \
    --condition C
```

## Updated Output Files

Each condition now generates a separate output file with the condition in the filename:

```
results/knowledge_editing_from_candidates_ROME_gpt-j-6b_condition_A_20250711_153000.json
results/knowledge_editing_from_candidates_ROME_gpt-j-6b_condition_B_20250711_153500.json
results/knowledge_editing_from_candidates_ROME_gpt-j-6b_condition_C_20250711_154000.json
```

## Updated Output Format

The output format now exactly matches `run_knowledge_editing_order_sampling.py`:

```json
{
  "method": "ROME",
  "model_name": "gpt-j-6b",
  "condition": "A",
  "timestamp": "2025-07-11T15:30:00.000000",
  "num_edits": 5,
  "num_sampling_combinations": 600,
  "sample_combinations": [...],
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
  },
  "success": true,
  "num_base_combinations": 25,
  "num_orders_per_combination": 24,
  "total_combinations_with_orders": 600,
  "metadata": {...}
}
```

## Key Changes

1. **Removed `--condition all` option**: Must specify A, B, or C
2. **Individual condition execution**: Each run processes only one condition
3. **Condition-specific output files**: Filenames include condition information
4. **Matching data structure**: Output format identical to `run_knowledge_editing_order_sampling.py`
5. **Consistent function naming**: Uses `perform_sampling_experiments` like the original

## Arguments

- `--condition`: **Required** - Must be A, B, or C (no longer accepts 'all')
- All other arguments remain the same

## Example Workflow

```bash
# Generate candidates once
python generate_sampling_candidates.py --num-edits 5 --sample-size 25 --num-orders 24

# Run each condition separately
python run_knowledge_editing_from_candidates.py --method ROME --model gpt-j-6b --condition A --candidates-file sampling_candidates/sampling_candidates_*.json
python run_knowledge_editing_from_candidates.py --method ROME --model gpt-j-6b --condition B --candidates-file sampling_candidates/sampling_candidates_*.json  
python run_knowledge_editing_from_candidates.py --method ROME --model gpt-j-6b --condition C --candidates-file sampling_candidates/sampling_candidates_*.json

# Run with different methods
python run_knowledge_editing_from_candidates.py --method MEMIT --model gpt-j-6b --condition A --candidates-file sampling_candidates/sampling_candidates_*.json
python run_knowledge_editing_from_candidates.py --method MEMIT --model gpt-j-6b --condition B --candidates-file sampling_candidates/sampling_candidates_*.json
python run_knowledge_editing_from_candidates.py --method MEMIT --model gpt-j-6b --condition C --candidates-file sampling_candidates/sampling_candidates_*.json
```

This approach ensures consistency with the original `run_knowledge_editing_order_sampling.py` program while providing the benefits of pre-generated sampling candidates.