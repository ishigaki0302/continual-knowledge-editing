#!/usr/bin/env python3
"""
Convert ZSRE data format to JSON format required by EasyEdit
"""

import json
import os

def convert_zsre_to_json(input_file, output_file, max_samples=1000):
    """Convert ZSRE tab-separated format to JSON"""
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:  # Limit samples for faster training
                break
                
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                relation = parts[0]
                question_template = parts[1]
                subject = parts[2]
                context = parts[3]
                answer = parts[4]
                
                # Create prompt by replacing XXX with subject
                prompt = question_template.replace('XXX', subject)
                
                # Create the required format
                data_point = {
                    "src": prompt,
                    "alt": answer,
                    "answers": [answer],  # Required by ZsreDataset
                    "rephrase": prompt,   # Use same prompt as rephrase for simplicity
                    "loc": f"nq question: {prompt}",
                    "loc_ans": answer
                }
                data.append(data_point)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} samples to {output_file}")
    return len(data)

def main():
    # Create training data
    train_count = convert_zsre_to_json(
        '../easyedit_base/data/zsre/relation_splits/train.0',
        '../easyedit_base/data/zsre/zsre_mend_train.json',
        max_samples=1000
    )
    
    # Create evaluation data
    eval_count = convert_zsre_to_json(
        '../easyedit_base/data/zsre/relation_splits/dev.0',
        '../easyedit_base/data/zsre/zsre_mend_eval.json',
        max_samples=200
    )
    
    print(f"Training samples: {train_count}")
    print(f"Evaluation samples: {eval_count}")

if __name__ == "__main__":
    main()