#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_knowledge_editing.py

Knowledge Editing の結果 JSON (条件 A/B/C) から、
各編集ステップの Post-edit と Final-state 確率を
6行×5列のグリッドで可視化するスクリプト。
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_results(paths, output_path=None):
    """
    paths: dict with keys 'A','B','C' mapping to JSON file paths
    output_path: 図を保存するパス (指定なければ表示のみ)
    """
    # 各条件のデータ読み込み
    data = {cond: load_json(Path(fp)) for cond, fp in paths.items()}
    n_steps = data['A']['num_edits']  # 各条件とも同じステップ数を想定

    # 描画領域：6行×ステップ数列
    fig, axes = plt.subplots(nrows=6, ncols=n_steps,
                             figsize=(4*n_steps, 16),
                             sharex=False, sharey=False)

    # カラムタイトル
    for col in range(n_steps):
        axes[0, col].set_title(f"Step {col+1}", fontsize=14)

    # 各条件ごとに上段＝Post-edit、下段＝Final-state
    for idx, cond in enumerate(['A','B','C']):
        result = data[cond]
        edits = result['edits']
        finals = result.get('final_state_evaluations', [])

        for j, edit in enumerate(edits):
            triple = edit['triple']
            cands  = triple['candidates']
            post_p = edit['post_edit_probabilities']['probabilities']
            # Final-state の確率を JSON に合わせて読み替えてください
            final_p = finals[j]['final_state_probabilities']['probabilities']

            target_idx = cands.index(triple['object'])

            # — 上段：Post-edit —
            ax_top = axes[idx*2, j]
            bars = ax_top.bar(range(len(cands)), post_p, color='gray', edgecolor='black')
            # 強調
            bars[target_idx].set_edgecolor('black')
            bars[target_idx].set_linewidth(2)
            if j == 0:
                ax_top.set_ylabel(f"Cond {cond}\nPost-edit", fontsize=12)
            ax_top.set_xticks(range(len(cands)))
            ax_top.set_xticklabels(cands, rotation=45, ha='right', fontsize=8)
            ax_top.text(0.5, 0.9,
                        f"{triple['subject']}\n{triple['relation']}\n{triple['object']}",
                        transform=ax_top.transAxes,
                        fontsize=8, ha='center', va='top')

            # — 下段：Final-state —
            ax_bot = axes[idx*2+1, j]
            bars = ax_bot.bar(range(len(cands)), final_p, color='gray', edgecolor='black')
            bars[target_idx].set_edgecolor('black')
            bars[target_idx].set_linewidth(2)
            if j == 0:
                ax_bot.set_ylabel(f"Cond {cond}\nFinal-state", fontsize=12)
            ax_bot.set_xticks(range(len(cands)))
            ax_bot.set_xticklabels(cands, rotation=45, ha='right', fontsize=8)

    plt.suptitle("Knowledge Editing: Post-edit vs Final-state Probabilities", fontsize=16, y=0.92)
    plt.tight_layout(rect=[0,0,1,0.96])

    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot knowledge editing results (conditions A/B/C)")
    parser.add_argument("--fileA", required=True,
                        help="Condition A の結果 JSON パス")
    parser.add_argument("--fileB", required=True,
                        help="Condition B の結果 JSON パス")
    parser.add_argument("--fileC", required=True,
                        help="Condition C の結果 JSON パス")
    parser.add_argument("--out", default=None,
                        help="図を保存するパス（省略すると表示のみ）")
    args = parser.parse_args()

    paths = {'A': args.fileA, 'B': args.fileB, 'C': args.fileC}
    plot_results(paths, output_path=args.out)

if __name__ == "__main__":
    main()