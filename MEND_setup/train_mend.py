#!/usr/bin/env python3
"""
Train MEND model for knowledge editing
"""

import os
import sys
import torch

# Add the easyedit_base directory to the Python path
sys.path.insert(0, '/app/EasyEdit/easyedit_base')

from easyeditor import MENDTrainingHparams
from easyeditor import EditTrainer
from easyeditor import ZsreDataset

def train_mend():
    """Train MEND model"""
    
    # Set up parameters
    # hparams = MENDTrainingHparams.from_hparams('../easyedit_base/hparams/TRAINING/MEND/gpt2-xl.yaml')
    hparams = MENDTrainingHparams.from_hparams('../easyedit_base/hparams/TRAINING/MEND/gpt-j-6B.yaml')
    
    # Fix model name to use HuggingFace model ID instead of local path
    # hparams.model_name = 'openai-community/gpt2-xl'
    # hparams.tokenizer_name = 'openai-community/gpt2-xl'
    hparams.model_name = 'EleutherAI/gpt-j-6b'
    hparams.tokenizer_name = 'EleutherAI/gpt-j-6b'
    
    # Override some parameters for faster training
    hparams.max_iters = 5000  # Reduce iterations for faster training
    hparams.model_save_pt = 1000  # Save more frequently
    hparams.log_interval = 100
    hparams.eval_log_interval = 500
    hparams.val_interval = 500
    hparams.early_stop_patience = 2000
    hparams.save = True  # Enable saving
    
    # Create output directory
    os.makedirs('../results/models/MEND', exist_ok=True)
    
    # Load training data (already converted from ZSRE format)
    print("Loading training dataset...")
    train_ds = ZsreDataset('../easyedit_base/data/zsre/zsre_mend_train.json', config=hparams)
    eval_ds = ZsreDataset('../easyedit_base/data/zsre/zsre_mend_eval.json', config=hparams)
    
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Evaluation dataset size: {len(eval_ds)}")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = EditTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    # Start training
    print("Starting MEND training...")
    trainer.run()
    
    print("Training completed!")
    print(f"Model saved to: {hparams.results_dir}/models/MEND/")

if __name__ == "__main__":
    train_mend()