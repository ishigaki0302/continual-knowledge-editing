#!/usr/bin/env python3
"""
Debug script to check model identity and parameter changes
"""

import json
import torch
from transformers import AutoTokenizer
from torch.nn.functional import softmax
from src.utils.easyedit_wrapper import EasyEditWrapper
import copy

def debug_model_editing():
    """Debug model editing to understand parameter changes"""
    
    print("=== Model Identity Debug ===")
    
    # Load test data
    with open('datasets/temp_ckndata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create test case
    subject = "Ryoma Ishigaki"
    relation = "Skills"
    test_object = "Python"
    
    relation_data = data['SharedRelations']['Skills']
    prompt = relation_data['prompt'].replace('[subject]', subject).replace(' [object].', "")
    candidates = relation_data['objects']
    
    print(f"Test case: {prompt} -> {test_object}")
    
    # Initialize wrapper
    wrapper = EasyEditWrapper(method="ROME", model_name="gpt2-xl")
    wrapper.initialize_editor()
    
    # Get original model
    original_model = wrapper.editor.model
    print(f"Original model ID: {id(original_model)}")
    
    # Get a specific parameter to track
    target_layer = "transformer.h.17.mlp.c_proj.weight"
    original_param = original_model.state_dict()[target_layer].clone()
    print(f"Original parameter norm: {original_param.norm().item():.6f}")
    
    # Perform edit
    print("\n=== Performing Edit ===")
    metrics, edited_model = wrapper.edit_model(
        prompts=prompt,
        ground_truth="unknown",
        target_new=test_object,
        subject=subject,
        edit_id="debug_edit"
    )
    
    print(f"Edited model ID: {id(edited_model)}")
    print(f"Same model instance? {original_model is edited_model}")
    
    # Check parameter changes
    edited_param = edited_model.state_dict()[target_layer]
    print(f"Edited parameter norm: {edited_param.norm().item():.6f}")
    
    param_diff = (edited_param - original_param).norm().item()
    print(f"Parameter difference norm: {param_diff:.6f}")
    
    if param_diff > 1e-6:
        print("✅ Parameters were changed during editing")
    else:
        print("❌ Parameters were NOT changed during editing")
    
    # Test with direct prompt that ROME was optimizing for
    test_prompt = prompt  # Use the same prompt as the edit
    tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== Testing Direct Prompt: '{test_prompt}' ===")
    
    # Test original model
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        orig_outputs = original_model(**inputs)
        orig_logits = orig_outputs.logits[0, -1, :]
    
    # Test edited model  
    with torch.no_grad():
        edit_outputs = edited_model(**inputs)
        edit_logits = edit_outputs.logits[0, -1, :]
    
    # Get logits for "Python" token
    python_token_id = tokenizer.encode(" Python", add_special_tokens=False)[0]
    
    orig_python_logit = orig_logits[python_token_id].item()
    edit_python_logit = edit_logits[python_token_id].item()
    
    print(f"Original 'Python' logit: {orig_python_logit:.6f}")
    print(f"Edited 'Python' logit: {edit_python_logit:.6f}")
    print(f"Logit change: {edit_python_logit - orig_python_logit:+.6f}")
    
    if abs(edit_python_logit - orig_python_logit) > 0.1:
        print("✅ Model output changed significantly for target token")
    else:
        print("❌ Model output did NOT change significantly for target token")
    
    # Check if models share parameters
    print(f"\n=== Parameter Sharing Check ===")
    orig_param_addr = original_model.state_dict()[target_layer].data_ptr()
    edit_param_addr = edited_model.state_dict()[target_layer].data_ptr()
    print(f"Original parameter memory address: {orig_param_addr}")
    print(f"Edited parameter memory address: {edit_param_addr}")
    print(f"Same memory location? {orig_param_addr == edit_param_addr}")

if __name__ == "__main__":
    debug_model_editing()