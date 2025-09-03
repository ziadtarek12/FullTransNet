#!/usr/bin/env python3
"""
FullTransNet Video Summarization Inference Script
Single script for Kaggle/Colab - loads trained model and shows video summarization results
"""

import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import torch
import h5py
from pathlib import Path
import seaborn as sns

from helpers import init_helper, data_helper, vsumm_helper
from model.transfomer_with_window import Transformer


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_model(model_type, **kwargs):
    """Create and return the model."""
    if model_type == 'encoder-decoder':
        num_feature = kwargs['num_feature']
        num_head = kwargs['num_head']
        enlayer = kwargs['enlayers']
        delayer = kwargs['delayers']
        wid_size = kwargs['window_size']
        stride = kwargs['stride']
        length = kwargs['length']
        attention_mode = kwargs['attention_mode']
        dff = kwargs['dff']

        return Transformer(
            T=0,
            dim_in=num_feature,
            heads=num_head,
            enlayers=enlayer,
            delayers=delayer,
            dim_mid=64,
            length=length,
            window_size=wid_size,
            attention_mode=attention_mode,
            stride=stride,
            dff=dff
        )
    else:
        raise ValueError(f'Invalid model type {model_type}')


def load_video_data(dataset_path, video_name):
    """Load video data from H5 file."""
    logger = logging.getLogger(__name__)
    
    try:
        with h5py.File(dataset_path, 'r') as dataset:
            if video_name not in dataset:
                available_videos = list(dataset.keys())
                logger.error(f"Video '{video_name}' not found in dataset. Available videos: {available_videos}")
                return None
            
            video_file = dataset[video_name]
            
            # Load all necessary data
            seq = video_file['features'][...].astype(np.float32)
            gtscore = video_file['gtscore'][...].astype(np.float32)
            cps = video_file['change_points'][...].astype(np.int32)
            n_frames = video_file['n_frames'][...].astype(np.int32)
            nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
            picks = video_file['picks'][...].astype(np.int32)
            
            # Load user summary if available
            user_summary = None
            if 'user_summary' in video_file:
                user_summary = video_file['user_summary'][...].astype(np.float32)
            
            # Load ground truth summary if available
            gt_summary = None
            if 'gtsummary' in video_file:
                gt_summary = video_file['gtsummary'][...].astype(np.float32)
            
            # Normalize ground truth scores
            gtscore -= gtscore.min()
            if gtscore.max() > 0:
                gtscore /= gtscore.max()
            
            # Compute sequence differences
            seq_size = seq.shape
            seqdiff = []
            seq_temp = 0
            for j in range(seq_size[0]):
                if j == 0:
                    seq_current = seq[j]
                else:
                    seq_current = seq[j]
                    seq_temp = seq_current - seq_before
                    seqdiff.append(seq_temp)
                seq_before = seq_current
            seqdiff.append(seq_temp)
            seqdiff = np.array(seqdiff)
            
            logger.info(f"Loaded video '{video_name}' with {len(seq)} segments and {n_frames} frames")
            
            return {
                'seq': seq,
                'seqdiff': seqdiff,
                'gtscore': gtscore,
                'cps': cps,
                'n_frames': n_frames,
                'nfps': nfps,
                'picks': picks,
                'user_summary': user_summary,
                'gt_summary': gt_summary
            }
            
    except Exception as e:
        logger.error(f"Error loading video data: {e}")
        return None


def perform_inference(model, video_data, device):
    """Perform inference on the video data."""
    logger = logging.getLogger(__name__)
    print(f"Starting inference with {len(video_data['seq'])} segments...")
    
    seq = torch.as_tensor(video_data['seq'], dtype=torch.float32).unsqueeze(0).to(device)
    gtscore = video_data['gtscore']
    cps = video_data['cps']
    n_frames = video_data['n_frames']
    nfps = video_data['nfps']
    picks = video_data['picks']
    
    print(f"Video has {n_frames} frames and {len(cps)} segments")
    
    # Get keyshot summary from ground truth
    keyshot_summ = vsumm_helper.get_keyshot_summ(gtscore, cps, n_frames, nfps, picks)
    target = vsumm_helper.downsample_summ(keyshot_summ)
    target1 = seq.squeeze(0)[target]
    
    print(f"Ground truth summary has {np.sum(keyshot_summ)} selected frames")
    
    # Prepare global indices
    global_idxa = cps[:, 0]
    global_idxb = cps[:, 1]
    idx_mid = (global_idxa + global_idxb) // 2
    global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
    global_idx = np.concatenate((global_idx, idx_mid))
    
    model.eval()
    with torch.no_grad():
        out, _, _, _ = model(seq, target1, global_idx)
        
        # Process model output to get prediction
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
    
    # Convert prediction to keyshot summary
    keyshot_summ_pred = vsumm_helper.get_keyshot_summ(pred_summ1.cpu().numpy(), cps, n_frames, nfps, picks)
    
    logger.info("Inference completed successfully")
    
    return {
        'pred_scores': pred_summ1.cpu().numpy(),
        'keyshot_pred': keyshot_summ_pred,
        'keyshot_gt': keyshot_summ,
        'gt_scores': gtscore,
        'change_points': cps
    }


def create_visualization(video_name, video_data, inference_results, save_path=None):
    """Create a comprehensive visualization of the video summarization results."""
    logger = logging.getLogger(__name__)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Video Summarization Results: {video_name}', fontsize=16, fontweight='bold')
    
    n_frames = video_data['n_frames']
    cps = video_data['cps']
    
    # Create frame indices for visualization
    frame_indices = np.arange(n_frames)
    
    # 1. Ground Truth Importance Scores
    ax1 = axes[0]
    ax1.plot(frame_indices, inference_results['gt_scores'], 'b-', linewidth=1.5, alpha=0.7)
    ax1.fill_between(frame_indices, inference_results['gt_scores'], alpha=0.3, color='blue')
    ax1.set_title('Ground Truth Importance Scores', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Importance Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_frames)
    
    # Add change points
    for cp in cps:
        ax1.axvline(x=cp[0], color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(x=cp[1], color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 2. Ground Truth Summary
    ax2 = axes[1]
    gt_summary_plot = np.zeros(n_frames)
    for i, val in enumerate(inference_results['keyshot_gt']):
        gt_summary_plot[i] = val
    
    ax2.fill_between(frame_indices, gt_summary_plot, color='green', alpha=0.6, label='Ground Truth Summary')
    ax2.set_title('Ground Truth Summary (Binary)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selected')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_frames)
    
    # 3. Predicted Summary
    ax3 = axes[2]
    pred_summary_plot = np.zeros(n_frames)
    for i, val in enumerate(inference_results['keyshot_pred']):
        pred_summary_plot[i] = val
    
    ax3.fill_between(frame_indices, pred_summary_plot, color='orange', alpha=0.6, label='Predicted Summary')
    ax3.set_title('Predicted Summary (Binary)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Selected')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, n_frames)
    
    # 4. Comparison View
    ax4 = axes[3]
    
    # Create bars for ground truth and prediction
    bar_height = 0.4
    ax4.barh(0.5, n_frames, height=bar_height, color='lightgray', alpha=0.3, label='Full Video')
    
    # Ground truth segments
    gt_segments = []
    pred_segments = []
    
    # Find continuous segments for ground truth
    gt_start = None
    for i in range(len(inference_results['keyshot_gt'])):
        if inference_results['keyshot_gt'][i] > 0:
            if gt_start is None:
                gt_start = i
        else:
            if gt_start is not None:
                gt_segments.append((gt_start, i))
                gt_start = None
    if gt_start is not None:
        gt_segments.append((gt_start, len(inference_results['keyshot_gt'])))
    
    # Find continuous segments for prediction
    pred_start = None
    for i in range(len(inference_results['keyshot_pred'])):
        if inference_results['keyshot_pred'][i] > 0:
            if pred_start is None:
                pred_start = i
        else:
            if pred_start is not None:
                pred_segments.append((pred_start, i))
                pred_start = None
    if pred_start is not None:
        pred_segments.append((pred_start, len(inference_results['keyshot_pred'])))
    
    # Draw ground truth segments
    for start, end in gt_segments:
        ax4.barh(0.7, end - start, left=start, height=0.2, color='green', alpha=0.8)
    
    # Draw predicted segments
    for start, end in pred_segments:
        ax4.barh(0.3, end - start, left=start, height=0.2, color='orange', alpha=0.8)
    
    ax4.set_title('Summary Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Summary Type')
    ax4.set_yticks([0.3, 0.7])
    ax4.set_yticklabels(['Predicted', 'Ground Truth'])
    ax4.set_xlim(0, n_frames)
    ax4.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    if inference_results['keyshot_gt'] is not None and inference_results['keyshot_pred'] is not None:
        f1 = vsumm_helper.f1_score(inference_results['keyshot_pred'], inference_results['keyshot_gt'])
        
        # Add text box with metrics
        textstr = f'F1-Score: {f1:.3f}\nGT Summary: {np.sum(inference_results["keyshot_gt"]):.0f} frames\nPred Summary: {np.sum(inference_results["keyshot_pred"]):.0f} frames'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        logger.info(f"Visualization saved to {save_path}")
        print(f"✓ PNG file saved successfully: {save_path}")
        
        # Verify file exists
        if Path(save_path).exists():
            file_size = Path(save_path).stat().st_size / 1024  # Size in KB
            print(f"✓ File size: {file_size:.1f} KB")
        else:
            print(f"✗ Error: File was not saved to {save_path}")
    
    plt.show()
    
    return fig


def print_summary_statistics(video_name, video_data, inference_results):
    """Print detailed summary statistics."""
    logger = logging.getLogger(__name__)
    
    n_frames = video_data['n_frames']
    gt_summary = inference_results['keyshot_gt']
    pred_summary = inference_results['keyshot_pred']
    
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS FOR: {video_name}")
    print("="*60)
    
    print(f"Total video frames: {n_frames}")
    print(f"Total segments: {len(video_data['cps'])}")
    print(f"Feature dimension: {video_data['seq'].shape[1]}")
    
    if gt_summary is not None:
        gt_frames = np.sum(gt_summary)
        gt_percentage = (gt_frames / n_frames) * 100
        print(f"\nGround Truth Summary:")
        print(f"  Selected frames: {gt_frames:.0f} ({gt_percentage:.1f}%)")
    
    if pred_summary is not None:
        pred_frames = np.sum(pred_summary)
        pred_percentage = (pred_frames / n_frames) * 100
        print(f"\nPredicted Summary:")
        print(f"  Selected frames: {pred_frames:.0f} ({pred_percentage:.1f}%)")
    
    if gt_summary is not None and pred_summary is not None:
        f1 = vsumm_helper.f1_score(pred_summary, gt_summary)
        overlap = np.sum(pred_summary & gt_summary)
        precision = overlap / np.sum(pred_summary) if np.sum(pred_summary) > 0 else 0
        recall = overlap / np.sum(gt_summary) if np.sum(gt_summary) > 0 else 0
        
        print(f"\nEvaluation Metrics:")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Overlap: {overlap:.0f} frames")
    
    print("="*60)


def main():
    """Main inference function."""
    print("Starting FullTransNet Video Summarization Inference...")
    
    parser = argparse.ArgumentParser(description='Video Summarization Inference')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the H5 dataset file')
    parser.add_argument('--video-name', type=str, required=True,
                        help='Name of the video in the dataset (e.g., video_1)')
    
    # Model configuration (should match training configuration)
    parser.add_argument('--model', type=str, default='encoder-decoder')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--enlayers', type=int, default=6)
    parser.add_argument('--delayers', type=int, default=6)
    parser.add_argument('--length', type=int, default=1536)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--dff', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--attention-mode', type=str, default='sliding_chunks',
                        choices=('tvm', 'sliding_chunks', 'sliding_chunks_no_overlap'))
    
    # Output options
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save the visualization plot')
    parser.add_argument('--no-display', action='store_true',
                        help='Don\'t display the plot (useful for headless environments)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load video data
    logger.info(f"Loading video data: {args.video_name} from {args.dataset_path}")
    print(f"Loading video: {args.video_name}")
    video_data = load_video_data(args.dataset_path, args.video_name)
    if video_data is None:
        logger.error("Failed to load video data")
        print("ERROR: Failed to load video data")
        return
    print(f"✓ Video data loaded successfully")
    
    # Create model
    logger.info("Creating model...")
    print("Creating model...")
    model = get_model(args.model, **vars(args))
    model = model.to(device)
    print(f"✓ Model created and moved to {device}")
    
    # Load trained weights
    logger.info(f"Loading model weights from {args.model_path}")
    print(f"Loading model weights from: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
        print("✓ Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        print(f"ERROR: Failed to load model weights: {e}")
        return
    
    # Perform inference
    logger.info("Performing inference...")
    print("Performing inference...")
    inference_results = perform_inference(model, video_data, device)
    print("✓ Inference completed")
    
    # Print statistics
    print("Generating summary statistics...")
    print_summary_statistics(args.video_name, video_data, inference_results)
    
    # Create visualization
    if not args.no_display:
        logger.info("Creating visualization...")
        print("Creating visualization...")
        create_visualization(args.video_name, video_data, inference_results, args.save_plot)
    elif args.save_plot:
        logger.info("Creating and saving visualization...")
        print("Creating and saving visualization...")
        create_visualization(args.video_name, video_data, inference_results, args.save_plot)
    
    logger.info("Inference completed successfully!")
    print("🎉 All done! Inference completed successfully!")


if __name__ == '__main__':
    main()
