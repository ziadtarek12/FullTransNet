#!/usr/bin/env python3
"""
FullTransNet Video Summarization Inference Script with Video Creation
Single script for Kaggle/Colab - creates actual summarized video files
"""

import os
import sys
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import subprocess
import tempfile
import shutil

# Try to import cv2, fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Demo video creation will be limited.")

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
    
    # 2. Predicted Scores
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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    
    plt.show()
    return fig


def get_summary_frames(results):
    """Extract exact frame indices for video summary."""
    summary_binary = results['keyshot_pred']
    selected_frames = np.where(summary_binary == 1)[0]
    
    # Convert to frame ranges
    frame_ranges = []
    if len(selected_frames) > 0:
        start = selected_frames[0]
        
        for i in range(1, len(selected_frames)):
            if selected_frames[i] != selected_frames[i-1] + 1:
                frame_ranges.append((start, selected_frames[i-1]))
                start = selected_frames[i]
        
        frame_ranges.append((start, selected_frames[-1]))
    
    return selected_frames.tolist(), frame_ranges


def create_demo_video(frame_ranges, output_path, fps=30, width=640, height=480):
    """Create a demo video showing the selected segments (for when no input video is available)."""
    if not CV2_AVAILABLE:
        print("OpenCV not available. Creating text file with segment information instead.")
        create_segment_info_file(frame_ranges, output_path.replace('.mp4', '_segments.txt'), fps)
        return
    
    print(f"Creating demo video with {len(frame_ranges)} segments...")
    
    try:
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Failed to create video writer. Creating segment info file instead.")
            create_segment_info_file(frame_ranges, output_path.replace('.mp4', '_segments.txt'), fps)
            return
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for seg_idx, (start, end) in enumerate(frame_ranges):
            color = colors[seg_idx % len(colors)]
            duration_frames = max(1, int((end - start + 1) * fps / 30))  # Convert to output fps
            
            # Create a colored frame for this segment
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = color
            
            # Add text overlay (only if OpenCV text functions work)
            try:
                text = f"Segment {seg_idx + 1}"
                time_text = f"Frames {start}-{end}"
                cv2.putText(frame, text, (50, height//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, time_text, (50, height//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except:
                # Text overlay failed, continue without text
                pass
            
            # Write frames for this segment
            for _ in range(duration_frames):
                out.write(frame)
        
        out.release()
        # Remove the problematic cv2.destroyAllWindows() call for Kaggle compatibility
        print(f"✓ Demo video created: {output_path}")
        
    except Exception as e:
        print(f"Error creating demo video: {e}")
        print("Creating segment info file instead.")
        create_segment_info_file(frame_ranges, output_path.replace('.mp4', '_segments.txt'), fps)


def create_segment_info_file(frame_ranges, output_path, fps=30):
    """Create a text file with segment information when video creation is not possible."""
    with open(output_path, 'w') as f:
        f.write("VIDEO SUMMARY SEGMENTS\n")
        f.write("=" * 50 + "\n\n")
        
        total_duration = 0
        for i, (start, end) in enumerate(frame_ranges):
            start_time = start / fps
            end_time = end / fps
            duration = (end - start + 1) / fps
            total_duration += duration
            
            f.write(f"Segment {i+1}:\n")
            f.write(f"  Frames: {start} - {end}\n")
            f.write(f"  Time: {start_time:.2f}s - {end_time:.2f}s\n")
            f.write(f"  Duration: {duration:.2f}s\n\n")
        
        f.write(f"Total summary duration: {total_duration:.2f}s\n")
        f.write(f"Number of segments: {len(frame_ranges)}\n")
    
    print(f"✓ Segment information saved to: {output_path}")


def extract_video_with_ffmpeg(input_video, frame_ranges, output_path, fps=30):
    """Extract video segments using FFmpeg (if input video is available)."""
    if not os.path.exists(input_video):
        print(f"Input video {input_video} not found. Creating demo video instead.")
        create_demo_video(frame_ranges, output_path, fps)
        return
    
    print(f"Extracting video segments using FFmpeg...")
    
    # Create temporary directory for segments
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        
        # Extract each segment
        for i, (start, end) in enumerate(frame_ranges):
            start_time = start / fps
            duration = (end - start + 1) / fps
            segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            
            cmd = [
                'ffmpeg', '-i', input_video,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                segment_file, '-y'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                segment_files.append(segment_file)
                print(f"  ✓ Extracted segment {i+1}: {start_time:.2f}s - {start_time+duration:.2f}s")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to extract segment {i+1}: {e}")
        
        if segment_files:
            # Create concat file
            concat_file = os.path.join(temp_dir, "concat.txt")
            with open(concat_file, 'w') as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")
            
            # Concatenate segments
            cmd = [
                'ffmpeg', '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_path, '-y'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✓ Summary video created: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to create summary video: {e}")
                # Fallback to demo video
                create_demo_video(frame_ranges, output_path, fps)


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


def save_frame_data(video_name, video_data, results, selected_frames, frame_ranges, save_path, fps=30):
    """Save frame extraction data to JSON file."""
    
    # Calculate time ranges
    time_ranges = []
    for start, end in frame_ranges:
        time_ranges.append({
            'start_frame': int(start),
            'end_frame': int(end),
            'start_time': float(start / fps),
            'end_time': float(end / fps),
            'duration': float((end - start + 1) / fps)
        })
    
    # Create data structure with proper type conversion
    frame_data = {
        'video_info': {
            'video_name': video_name,
            'total_frames': int(video_data['n_frames']),
            'fps': int(fps),
            'total_duration': float(video_data['n_frames'] / fps)
        },
        'summary_info': {
            'selected_frames_count': len(selected_frames),
            'compression_ratio': float((1 - len(selected_frames) / video_data['n_frames']) * 100),
            'summary_duration': float(sum([(end - start + 1) / fps for start, end in frame_ranges])),
            'f1_score': float(vsumm_helper.f1_score(results['keyshot_pred'], results['keyshot_gt']))
        },
        'frame_extraction': {
            'selected_frames': [int(x) for x in selected_frames],
            'frame_ranges': [[int(start), int(end)] for start, end in frame_ranges],
            'time_ranges': time_ranges
        }
    }
    
    # Save to JSON file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(frame_data, f, indent=2)
    
    print(f"✓ Frame data saved to: {save_path}")
    return frame_data


def run_inference(model_path, dataset_path, video_name, save_plot=None, save_frames=None, 
                 save_video=None, input_video=None, fps=30, device='cuda'):
    """Main inference function with video creation."""
    
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
        
        # Extract frame information
        selected_frames, frame_ranges = get_summary_frames(results)
        
        print(f"\nExtracted {len(selected_frames)} frames in {len(frame_ranges)} segments")
        for i, (start, end) in enumerate(frame_ranges):
            duration = (end - start + 1) / fps
            print(f"  Segment {i+1}: frames {start}-{end} ({duration:.2f}s)")
        
        # Create visualization
        if save_plot:
            print(f"\nCreating visualization...")
            fig = visualize_summary(video_name, video_data, results, save_plot)
        else:
            fig = None
        
        # Save frame data
        frame_data = None
        if save_frames:
            print(f"\nSaving frame data...")
            frame_data = save_frame_data(video_name, video_data, results, selected_frames, frame_ranges, save_frames, fps)
        
        # Create summary video
        if save_video:
            print(f"\nCreating summary video...")
            if input_video and os.path.exists(input_video):
                extract_video_with_ffmpeg(input_video, frame_ranges, save_video, fps)
            else:
                create_demo_video(frame_ranges, save_video, fps)
        
        print("\n✓ Inference completed successfully!")
        
        return video_data, results, fig, selected_frames, frame_ranges, frame_data
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


# KAGGLE/COLAB USAGE EXAMPLES:

def example_summe():
    """Example for SumMe dataset with video creation"""
    return run_inference(
        model_path='./model_save/summe/summe_0.pt',
        dataset_path='./datasets/eccv16_dataset_summe_google_pool5.h5',
        video_name='video_1',
        save_plot='./results/summe_video_1.png',
        save_frames='./results/summe_video_1_frames.json',
        save_video='./results/summe_video_1_summary.mp4'
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FullTransNet Video Summarization with Video Creation')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to the HDF5 dataset file')
    parser.add_argument('--video-name', type=str, required=True,
                       help='Name of the video in the dataset')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save the visualization plot')
    parser.add_argument('--save-frames', type=str, default=None,
                       help='Path to save frame extraction data as JSON')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Path to save the summarized video')
    parser.add_argument('--input-video', type=str, default=None,
                       help='Path to input video file (if available)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for calculations')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("Running FullTransNet Video Summarization with Video Creation...")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Video: {args.video_name}")
    print(f"Save plot: {args.save_plot}")
    print(f"Save frames: {args.save_frames}")
    print(f"Save video: {args.save_video}")
    print(f"Input video: {args.input_video}")
    
    # Run the inference
    video_data, results, fig, selected_frames, frame_ranges, frame_data = run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        video_name=args.video_name,
        save_plot=args.save_plot,
        save_frames=args.save_frames,
        save_video=args.save_video,
        input_video=args.input_video,
        fps=args.fps,
        device=args.device
    )
    
    if video_data is not None:
        print("\n🎉 Video summarization completed successfully!")
        print("Generated files:")
        if args.save_plot:
            print(f"  📊 Visualization: {args.save_plot}")
        if args.save_frames:
            print(f"  📄 Frame data: {args.save_frames}")
        if args.save_video:
            print(f"  🎬 Summary video: {args.save_video}")
    else:
        print("\n❌ Video summarization failed.")
