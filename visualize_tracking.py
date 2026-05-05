#!/usr/bin/env python3
"""
Visualize MOTIP tracking results by overlaying bounding boxes and IDs on video frames.
"""

import cv2
import os
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

def generate_colors(num_colors):
    """Generate distinct colors for different track IDs."""
    np.random.seed(42)
    colors = {}
    for i in range(num_colors):
        colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
    return colors

def visualize_tracking(frames_dir, tracking_file, output_dir, max_frames=None, create_video=True, fps=15):
    """
    Overlay tracking results on video frames.

    Args:
        frames_dir: Directory containing extracted frames (img1/)
        tracking_file: Path to MOT format tracking results (.txt)
        output_dir: Directory to save visualized frames
        max_frames: Maximum number of frames to process (None = all)
        create_video: Whether to automatically create a video from frames
        fps: Framerate for the output video (default: 15)
    """

    # Read tracking results
    print(f"Reading tracking results from {tracking_file}...")
    tracks = {}
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])

            if frame_id not in tracks:
                tracks[frame_id] = []
            tracks[frame_id].append({
                'id': track_id,
                'bbox': (int(x), int(y), int(w), int(h))
            })

    # Get unique track IDs for color generation
    all_track_ids = set()
    for frame_tracks in tracks.values():
        for track in frame_tracks:
            all_track_ids.add(track['id'])

    colors = generate_colors(max(all_track_ids) + 1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if max_frames:
        frame_files = frame_files[:max_frames]

    print(f"Processing {len(frame_files)} frames...")

    # Process each frame
    for frame_file in tqdm(frame_files):
        frame_id = int(frame_file.split('.')[0])
        frame_path = os.path.join(frames_dir, frame_file)

        # Read frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}")
            continue

        # Draw tracking boxes for this frame
        if frame_id in tracks:
            for track in tracks[frame_id]:
                track_id = track['id']
                x, y, w, h = track['bbox']

                # Get color for this track
                color = colors.get(track_id, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                # Draw track ID
                label = f"ID:{track_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw background rectangle for text
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)

                # Draw text
                cv2.putText(frame, label, (x, y - 5), font, font_scale, (0, 0, 0), thickness)

        # Add frame number
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save visualized frame
        output_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_path, frame)

    print(f"✓ Visualization complete! Saved to {output_dir}")
    
    # Automatically create video if requested
    if create_video:
        print(f"\nCreating video at {fps} fps...")
        video_path = os.path.join(output_dir, 'tracking_video.mp4')
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite without asking
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', f'{output_dir}/*.jpg',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',  # Quality (lower = better, 23 is default)
            video_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Video created successfully: {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create video. You can create it manually with:")
            print(f"ffmpeg -framerate {fps} -pattern_type glob -i '{output_dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p {video_path}")
            print(f"Error: {e.stderr}")
        except FileNotFoundError:
            print(f"✗ ffmpeg not found. Please install ffmpeg to create videos automatically.")
            print(f"You can create the video manually with:")
            print(f"ffmpeg -framerate {fps} -pattern_type glob -i '{output_dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p {video_path}")
    else:
        print(f"\nTo create a video from frames:")
        print(f"ffmpeg -framerate {fps} -pattern_type glob -i '{output_dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p {output_dir}/tracking_video.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MOTIP tracking results")
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='Directory containing frames (e.g., datasets/OceanFish/test/BB36_mob_m1_full/img1)')
    parser.add_argument('--tracking-file', type=str, required=True,
                       help='Path to tracking results file (.txt)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save visualized frames')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip automatic video creation')
    parser.add_argument('--fps', type=int, default=15,
                       help='Framerate for output video (default: 15)')

    args = parser.parse_args()

    visualize_tracking(
        frames_dir=args.frames_dir,
        tracking_file=args.tracking_file,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        create_video=not args.no_video,
        fps=args.fps
    )
