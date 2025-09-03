#!/usr/bin/env python3
"""
Simple test script for inference
"""

import sys
import os

# Set up the path
sys.path.append('.')

# Test if the script can run with basic arguments
if __name__ == '__main__':
    print("Testing inference script...")
    
    # Example command for testing
    test_command = [
        'inference.py',
        '--model-path', './model_save/summe/summe_0.pt',  # Update this path
        '--dataset-path', './datasets/eccv16_dataset_summe_google_pool5.h5',  # Update this path  
        '--video-name', 'video_1',
        '--save-plot', './test_output.png',
        '--device', 'cpu'  # Use CPU for testing
    ]
    
    print("Test command:")
    print("python " + " ".join(test_command))
    print("\nTo run the test, execute:")
    print(f"python {' '.join(test_command)}")
    
    # Basic file checks
    model_path = './model_save/summe/summe_0.pt'
    dataset_path = './datasets/eccv16_dataset_summe_google_pool5.h5'
    
    print(f"\nFile checks:")
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Dataset file exists: {os.path.exists(dataset_path)}")
    
    if not os.path.exists(model_path):
        print("❌ Model file not found! Please check the path.")
    if not os.path.exists(dataset_path):
        print("❌ Dataset file not found! Please check the path.")
        
    if os.path.exists(model_path) and os.path.exists(dataset_path):
        print("✅ All files found! You can run the inference.")
