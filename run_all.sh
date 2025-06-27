#!/bin/bash

models=("gpt-j-6b" "gpt2-xl")
methods=("ROME" "MEMIT" "MEND")
conditions=("A" "B" "C")
device="cuda:1"
num_edits=5

for model in "${models[@]}"; do
  for method in "${methods[@]}"; do
    for condition in "${conditions[@]}"; do
      python run_knowledge_editing.py \
        --condition "$condition" \
        --method "$method" \
        --model "$model" \
        --num-edits "$num_edits" \
        --device "$device"
    done
  done
done

# python3 plot_knowledge_editing.py \
#   --fileA results/knowledge_editing_ROME_gpt-j-6b_condition_A_20250627_082512.json \
#   --fileB results/knowledge_editing_ROME_gpt-j-6b_condition_B_20250627_082629.json \
#   --fileC results/knowledge_editing_ROME_gpt-j-6b_condition_C_20250627_082806.json \
#   --out results/summary_plot.png