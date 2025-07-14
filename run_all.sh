# !/bin/bash

# models=("gpt-j-6b" "gpt2-xl")
# methods=("ROME" "MEMIT" "MEND")
# conditions=("A" "B" "C")
# device="cuda:0"
# num_edits=5

# for model in "${models[@]}"; do
#   for method in "${methods[@]}"; do
#     # gpt-j-6b に対して MEMIT / MEND をスキップ
#     if [[ "$model" == "gpt-j-6b" && ( "$method" == "MEMIT" || "$method" == "MEND" ) ]]; then
#       echo "Skipping $method for $model"
#       continue
#     fi
#     for condition in "${conditions[@]}"; do
#       python run_knowledge_editing.py \
#         --condition "$condition" \
#         --method "$method" \
#         --model "$model" \
#         --num-edits "$num_edits" \
#         --device "$device"
#     done
#   done
# done

# # 単一ランダム順序での実験
# python run_knowledge_editing_order.py --method ROME --model gpt-j-6b --num-edits 3 --condition A
# --order-strategy single
# # 全順列での実験（3! = 6通り）
# python run_knowledge_editing_order.py --method ROME --model gpt-j-6b --num-edits 3 --condition A
# --order-strategy all
# より多くの編集（注意：4! = 24、5! = 120通り）
# python run_knowledge_editing_order.py --method ROME --model gpt2-xl --num-edits 5 --condition A --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method ROME --model gpt2-xl --num-edits 5 --condition B --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method ROME --model gpt2-xl --num-edits 5 --condition C --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method MEMIT --model gpt2-xl --num-edits 5 --condition A --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method MEMIT --model gpt2-xl --num-edits 5 --condition B --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method MEMIT --model gpt2-xl --num-edits 5 --condition C --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method MEND --model gpt2-xl --num-edits 5 --condition A --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method MEND --model gpt2-xl --num-edits 5 --condition B --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method MEND --model gpt2-xl --num-edits 5 --condition C --order-strategy all --device cuda:0
# python run_knowledge_editing_order.py --method ROME --model gpt-j-6b --num-edits 5 --condition A --order-strategy all --device cuda:1
# python run_knowledge_editing_order.py --method ROME --model gpt-j-6b --num-edits 5 --condition B --order-strategy all --device cuda:1
# python run_knowledge_editing_order.py --method ROME --model gpt-j-6b --num-edits 5 --condition C --order-strategy all --device cuda:1
# python run_knowledge_editing_order.py --method MEMIT --model gpt-j-6b --num-edits 5 --condition A --order-strategy all --device cuda:1
# python run_knowledge_editing_order.py --method MEMIT --model gpt-j-6b --num-edits 5 --condition B --order-strategy all --device cuda:1
# python run_knowledge_editing_order.py --method MEMIT --model gpt-j-6b --num-edits 5 --condition C --order-strategy all --device cuda:1

# python run_knowledge_editing_sampling.py --method ROME --model gpt-j-6b --num-edits 5 --condition A --sample-size 25 --device cuda:0
# python run_knowledge_editing_sampling.py --method ROME --model gpt-j-6b --num-edits 5 --condition B --sample-size 25 --device cuda:0
# python run_knowledge_editing_sampling.py --method ROME --model gpt-j-6b --num-edits 5 --condition C --sample-size 25 --device cuda:0
# python run_knowledge_editing_sampling.py --method MEMIT --model gpt2-xl --num-edits 5 --condition A --sample-size 25 --device cuda:0
# python run_knowledge_editing_sampling.py --method MEMIT --model gpt2-xl --num-edits 5 --condition B --sample-size 25 --device cuda:0
# python run_knowledge_editing_sampling.py --method MEMIT --model gpt2-xl --num-edits 5 --condition C --sample-size 25 --device cuda:0


LOGFILE="execution_times.txt"

# echo "===== Execution started at $(date) =====" > "$LOGFILE"

# echo "[Condition A]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_order_sampling.py --method ROME --model gpt2-xl --num-edits 5 --condition A --sample-size 125 --num-orders 120 --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition A: $((END - START)) seconds" >> "$LOGFILE"

# echo "[Condition B]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_order_sampling.py --method ROME --model gpt2-xl --num-edits 5 --condition B --sample-size 125 --num-orders 120 --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition B: $((END - START)) seconds" >> "$LOGFILE"

# echo "[Condition C]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_order_sampling.py --method ROME --model gpt2-xl --num-edits 5 --condition C --sample-size 25 --num-orders 120 --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition C: $((END - START)) seconds" >> "$LOGFILE"

# echo "===== Execution ended at $(date) =====" >> "$LOGFILE"

# echo "===== Execution started at $(date) =====" > "$LOGFILE"

# echo "[Condition A]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_from_candidates.py \
#       --method ROME \
#       --model gpt2-xl \
#       --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
#       --condition A \
#       --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition A: $((END - START)) seconds" >> "$LOGFILE"

# echo "[Condition B]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_from_candidates.py \
#       --method ROME \
#       --model gpt2-xl \
#       --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
#       --condition B \
#       --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition B: $((END - START)) seconds" >> "$LOGFILE"

# echo "[Condition C]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_from_candidates.py \
#       --method ROME \
#       --model gpt2-xl \
#       --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
#       --condition C \
#       --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition C: $((END - START)) seconds" >> "$LOGFILE"

# echo "===== Execution ended at $(date) =====" >> "$LOGFILE"

# echo "===== Execution started at $(date) =====" > "$LOGFILE"

# echo "[Condition A]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_from_candidates.py \
#       --method ROME \
#       --model gpt-j-6b \
#       --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
#       --condition A \
#       --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition A: $((END - START)) seconds" >> "$LOGFILE"

# echo "[Condition B]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_from_candidates.py \
#       --method ROME \
#       --model gpt-j-6b \
#       --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
#       --condition B \
#       --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition B: $((END - START)) seconds" >> "$LOGFILE"

# echo "[Condition C]" >> "$LOGFILE"
# START=$(date +%s)
# python run_knowledge_editing_from_candidates.py \
#       --method ROME \
#       --model gpt-j-6b \
#       --candidates-file sampling_candidates/sampling_candidates_edits5_samples25_orders24_seed42_20250711_150218.json \
#       --condition C \
#       --device cuda:0
# END=$(date +%s)
# echo "Time taken for condition C: $((END - START)) seconds" >> "$LOGFILE"

# echo "===== Execution ended at $(date) =====" >> "$LOGFILE"

# python3 plot_knowledge_editing.py \
#   --fileA results/knowledge_editing_ROME_gpt-j-6b_condition_A_20250627_082512.json \
#   --fileB results/knowledge_editing_ROME_gpt-j-6b_condition_B_20250627_082629.json \
#   --fileC results/knowledge_editing_ROME_gpt-j-6b_condition_C_20250627_082806.json \
#   --out results/summary_ROME_gpt-j-6b_plot.png

# python3 plot_knowledge_editing.py \
#   --fileA results/knowledge_editing_MEMIT_gpt2-xl_condition_B_20250627_174301.json \
#   --fileB results/knowledge_editing_MEMIT_gpt2-xl_condition_A_20250627_173932.json \
#   --fileC results/knowledge_editing_MEMIT_gpt2-xl_condition_C_20250627_174429.json \
#   --out results/summary_MEMIT_gpt2-xl_plot.png

# python3 plot_knowledge_editing.py \
#   --fileA results/knowledge_editing_MEND_gpt2-xl_condition_A_20250628_031019.json \
#   --fileB results/knowledge_editing_MEND_gpt2-xl_condition_B_20250628_031033.json \
#   --fileC results/knowledge_editing_MEND_gpt2-xl_condition_C_20250628_031048.json \
#   --out results/summary_MEND_gpt2-xl_plot.png

# 素晴らしい
# python plot_knowledge_editing_order.py \
#   --fileA results/knowledge_editing_order_ROME_gpt2-xl_condition_A_order_all_20250630_122515.json \
#   --fileB results/knowledge_editing_order_ROME_gpt2-xl_condition_B_order_all_20250630_131528.json \
#   --fileC results/knowledge_editing_order_ROME_gpt2-xl_condition_C_order_all_20250630_140140.json \
#   --out results/summary_order_ROME_gpt2-xl_plot.png

# python plot_knowledge_editing_order.py \
#   --fileA results/knowledge_editing_order_ROME_gpt-j-6b_condition_A_order_single_20250630_075441.json \
#   --fileB results/knowledge_editing_order_ROME_gpt-j-6b_condition_B_order_all_20250630_170133.json \
#   --fileC results/knowledge_editing_order_ROME_gpt-j-6b_condition_C_order_all_20250630_195749.json \
#   --out results/summary_order_ROME_gpt-j-6b_plot.png

# python plot_knowledge_editing_order.py \
#   --fileA results/knowledge_editing_order_MEND_gpt2-xl_condition_A_order_all_20250630_192356.json \
#   --fileB results/knowledge_editing_order_MEMIT_gpt2-xl_condition_B_order_all_20250630_181359.json \
#   --fileC results/knowledge_editing_order_MEMIT_gpt2-xl_condition_C_order_all_20250630_191215.json \
#   --out results/summary_order_MEND_gpt2-xl_plot.png

# python plot_knowledge_editing_order.py \
#   --fileA results/knowledge_editing_order_MEMIT_gpt2-xl_condition_A_order_all_20250630_170814.json \
#   --fileB results/knowledge_editing_order_MEMIT_gpt2-xl_condition_B_order_all_20250630_181359.json \
#   --fileC results/knowledge_editing_order_MEMIT_gpt2-xl_condition_C_order_all_20250630_191215.json \
#   --out results/summary_order_MEMIT_gpt2-xl_plot.png

# python plot_knowledge_editing.py \
#    --fileA results/knowledge_editing_from_candidates_ROME_gpt2-xl_condition_A_20250711_195417.json \
#    --fileB results/knowledge_editing_from_candidates_ROME_gpt2-xl_condition_B_20250711_235337.json \
#    --fileC results/knowledge_editing_from_candidates_ROME_gpt2-xl_condition_C_20250712_035009.json \
#    --out results/summary_new_ROME_gpt2-xl_plot.png

python plot_knowledge_editing.py \
   --fileA results/knowledge_editing_from_candidates_ROME_gpt-j-6b_condition_A_20250712_205611.json \
   --fileB results/knowledge_editing_from_candidates_ROME_gpt-j-6b_condition_B_20250713_094647.json \
   --fileC results/knowledge_editing_from_candidates_ROME_gpt-j-6b_condition_C_20250713_231554.json \
   --out results/summary_new_ROME_gpt-j-6b_plot.png