#!/usr/bin/env python3
"""
FullTransNet Video Summarization Inference Script
Single script for Kaggle/Colab - loads trained model and shows video summarization results
"""

import os
import sys
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Import project modules
from helpers import data_helper, vsumm_helper
from model.transfomer_with_window import Transformer


def get_model(**kwargs):
    """Create and return the transformer model."""
    return Transformer(
        T=0,
        dim_in=kwargs.get('num_feature', 1024),
        heads=kwargs.get('num_head', 8),
        enlayers=kwargs.get('enlayers', 6),
        delayers=kwargs.get('delayers', 6),
        dim_mid=64,
        length=kwargs.get('length', 1536),
        window_size=kwargs.get('window_size', 16),
        attention_mode=kwargs.get('attention_mode', 'sliding_chunks'),
        stride=kwargs.get('stride', 1),
        dff=kwargs.get('dff', 2048)
    )


def load_video_data(dataset_path, video_name):
    """Load video data from HDF5 dataset."""
    print(f"Loading video data: {video_name} from {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as dataset:
        if video_name not in dataset:
            available_videos = list(dataset.keys())
            print(f"Available videos: {available_videos}")
            raise KeyError(f"Video '{video_name}' not found")
        
        video_file = dataset[video_name]
        
        # Load all necessary data
        seq = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = int(video_file['n_frames'][...])
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        
        # Optional fields
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)
    
    # Normalize ground truth scores
    gtscore -= gtscore.min()
    if gtscore.max() > 0:
        gtscore /= gtscore.max()
    
    print(f"✓ Video loaded: {seq.shape[0]} segments, {n_frames} frames")
    
    return {
        'seq': seq,
        'gtscore': gtscore, 
        'cps': cps,
        'n_frames': n_frames,
        'nfps': nfps,
        'picks': picks,
        'user_summary': user_summary
    }


def perform_inference(model, video_data, device='cuda'):
    """Perform inference on the video data."""
    print("Performing inference...")
    
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        seq_tensor = torch.as_tensor(video_data['seq'], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get keyshot summary from ground truth
        keyshot_summ = vsumm_helper.get_keyshot_summ(
            video_data['gtscore'], video_data['cps'], 
            video_data['n_frames'], video_data['nfps'], video_data['picks']
        )
        target = vsumm_helper.downsample_summ(keyshot_summ)
        target1 = seq_tensor.squeeze(0)[target]
        
        # Prepare global indices
        global_idxa = video_data['cps'][:, 0]
        global_idxb = video_data['cps'][:, 1]
        idx_mid = (global_idxa + global_idxb) // 2
        global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
        global_idx = np.concatenate((global_idx, idx_mid))
        
        # Forward pass
        out, _, _, _ = model(seq_tensor, target1, global_idx)
        
        # Process output to get prediction
        pred_summ1 = torch.zeros(len(target))
        a, b = out.shape
        
        for j in range(b):
            column = out[:, j]
            min_value = torch.min(column)
            max_value = torch.max(column)
            for i in range(a):
                if column[i] == max_value and max_value == torch.max(out[i, :]):
                    pred_summ1[j] = max_value
                    break
            else:
                pred_summ1[j] = min_value
        
        # Get keyshot summary from prediction
        keyshot_summ_pred = vsumm_helper.get_keyshot_summ(
            pred_summ1.cpu().numpy(), video_data['cps'], 
            video_data['n_frames'], video_data['nfps'], video_data['picks']
        )
    
    print("✓ Inference completed")
    
    return {
        'pred_scores': pred_summ1.cpu().numpy(),
        'keyshot_pred': keyshot_summ_pred,
        'keyshot_gt': keyshot_summ,
        'gt_scores': video_data['gtscore']
    }


def visualize_summary(video_name, video_data, results, save_path=None):
    """Create visualization of the video summary."""
    print("Creating visualization...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Video Summarization Results: {video_name}', fontsize=16, fontweight='bold')
    
    n_frames = video_data['n_frames']
    cps = video_data['cps']
    frame_axis = np.arange(n_frames)
    
    # 1. Ground Truth Importance Scores
    axes[0].plot(results['gt_scores'], 'b-', linewidth=2, label='Ground Truth Scores')
    axes[0].set_ylabel('Importance Score')
    axes[0].set_title('Ground Truth Importance Scores')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Add change points
    for cp in cps:
        axes[0].axvline(x=cp[0], color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(x=cp[1], color='red', linestyle='--', alpha=0.5)
    
    # 2. Predicted Scores (mapped to frame level)
    pred_frame_scores = np.zeros(len(results['gt_scores']))
    for i, score in enumerate(results['pred_scores']):
        if i < len(pred_frame_scores):
            pred_frame_scores[i] = score
    
    axes[1].plot(pred_frame_scores, 'r-', linewidth=2, label='Predicted Scores')
    axes[1].set_ylabel('Predicted Score')
    axes[1].set_title('Model Predicted Scores')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Ground Truth Summary
    axes[2].fill_between(frame_axis, results['keyshot_gt'], 
                        color='green', alpha=0.6, label='Ground Truth Summary')
    axes[2].set_ylabel('Selected')
    axes[2].set_title('Ground Truth Summary (Binary)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 4. Predicted Summary
    axes[3].fill_between(frame_axis, results['keyshot_pred'], 
                        color='orange', alpha=0.6, label='Predicted Summary')
    axes[3].set_ylabel('Selected')
    axes[3].set_title('Predicted Summary (Binary)')
    axes[3].set_xlabel('Frame Number')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Calculate metrics
    if results['keyshot_gt'] is not None and results['keyshot_pred'] is not None:
        f1 = vsumm_helper.f1_score(results['keyshot_pred'], results['keyshot_gt'])
        
        # Add text box with metrics
        total_frames = n_frames
        selected_frames = int(results['keyshot_pred'].sum())
        compression_ratio = (1 - selected_frames / total_frames) * 100
        
        textstr = f'F1-Score: {f1:.3f}\nSelected: {selected_frames}/{total_frames} frames\nCompression: {compression_ratio:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[3].text(0.02, 0.98, textstr, transform=axes[3].transAxes, 
                    verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    
    plt.show()
    return fig


def print_summary_stats(video_name, video_data, results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print(f"VIDEO SUMMARIZATION RESULTS: {video_name}")
    print("="*60)
    
    n_frames = video_data['n_frames']
    gt_frames = int(results['keyshot_gt'].sum())
    pred_frames = int(results['keyshot_pred'].sum())
    
    print(f"Total frames: {n_frames}")
    print(f"Ground truth summary: {gt_frames} frames ({gt_frames/n_frames*100:.1f}%)")
    print(f"Predicted summary: {pred_frames} frames ({pred_frames/n_frames*100:.1f}%)")
    
    if results['keyshot_gt'] is not None and results['keyshot_pred'] is not None:
        f1 = vsumm_helper.f1_score(results['keyshot_pred'], results['keyshot_gt'])
        overlap = np.sum(results['keyshot_pred'] & results['keyshot_gt'])
        precision = overlap / pred_frames if pred_frames > 0 else 0
        recall = overlap / gt_frames if gt_frames > 0 else 0
        
        print(f"\nMetrics:")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Overlap: {int(overlap)} frames")
    
    print("="*60)


def run_inference(model_path, dataset_path, video_name, save_plot=None, device='cuda'):
    """Main inference function - USE THIS IN KAGGLE/COLAB"""
    
    print("="*60)
    print("FULLTRANSNET VIDEO SUMMARIZATION INFERENCE")
    print("="*60)
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    try:
        # Load model
        print(f"\nLoading model from: {model_path}")
        model = get_model()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.eval().to(device)
        print("✓ Model loaded successfully")
        
        # Load video data
        print(f"\nLoading video data...")
        video_data = load_video_data(dataset_path, video_name)
        
        # Perform inference
        print(f"\nRunning inference...")
        results = perform_inference(model, video_data, device)
        
        # Print statistics
        print_summary_stats(video_name, video_data, results)
        
        # Create visualization
        print(f"\nCreating visualization...")
        fig = visualize_summary(video_name, video_data, results, save_plot)
        
        print("\n✓ Inference completed successfully!")
        
        return video_data, results, fig
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


# KAGGLE/COLAB USAGE EXAMPLES:

def example_summe():
    """Example for SumMe dataset"""
    return run_inference(
        model_path='./model_save/summe/summe_0.pt',
        dataset_path='./datasets/eccv16_dataset_summe_google_pool5.h5',
        video_name='video_1',
        save_plot='./results/summe_video_1.png'
    )

def example_tvsum():
    """Example for TVSum dataset"""
    return run_inference(
        model_path='./model_save/tvsum/tvsum_0.pt', 
        dataset_path='./datasets/eccv16_dataset_tvsum_google_pool5.h5',
        video_name='video_1',
        save_plot='./results/tvsum_video_1.png'
    )


if __name__ == '__main__':
    # Direct execution - run example
    print("Running FullTransNet Video Summarization Inference...")
    print("Make sure you have the model and dataset files in the correct paths!")
    
    # You can modify these paths as needed
    MODEL_PATH = './model_save/summe/summe_0.pt'
    DATASET_PATH = './datasets/eccv16_dataset_summe_google_pool5.h5'
    VIDEO_NAME = 'video_1'
    SAVE_PATH = './results/video_1_summary.png'
    
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Video: {VIDEO_NAME}")
    print(f"Save plot: {SAVE_PATH}")
    
    # Run the inference
    video_data, results, fig = run_inference(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        video_name=VIDEO_NAME,
        save_plot=SAVE_PATH
    )
    
    if video_data is not None:
        print("\n🎉 Inference completed successfully!")
        print("Check the results above and the saved PNG file.")
    else:
        print("\n❌ Inference failed. Check the error messages above.")
