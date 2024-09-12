#!/usr/bin/env python3
import argparse
import os

from imgcat import imgcat
from tqdm.auto import tqdm

from vid2frames import FrameDecoder
from projstate import project_dir_from_filename, ProjectState

def show_frames(decoder: FrameDecoder, num_frames: int = 10):
    """Show the top frames."""
    print(f"Showing {num_frames} most diverse sample frames using imgcat.  (iterm2 or similar required)")
    if num_frames > len(decoder):
        num_frames = len(decoder)
    for i in range(num_frames):
        fmd = decoder.fmd_by_rank(i)
        frame_num = fmd["frame_num"]
        print(f"\nPreview of #{i+1}/{num_frames} sample frame (frame #{frame_num})")
        imgcat(fmd["frame"])

def save_frames(decoder: FrameDecoder, num_frames: int, save_dir: str):
    """Save the top frames."""
    if num_frames > len(decoder):
        num_frames = len(decoder)
    progress = tqdm(range(num_frames), desc="Saving frames")
    for i in progress:
        fmd = decoder.fmd_by_rank(i)
        frame_num = fmd["frame_num"]
        fn = f"diverse-{i:03d}-frame-{frame_num}.jpeg"
        save_path = os.path.join(save_dir, fn)
        fmd["pil_img"].save(save_path)
    print(f"Saved {num_frames} frames to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--project-dir", type=str, default=None, help="Path to the project directory")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to analyze")
    parser.add_argument("--show-frames", type=int, default=10, help="Show the N most diverse sample frames")
    parser.add_argument("--save-frames", type=int, default=100, help="Save the N most diverse sample frames to the project directory")
    args = parser.parse_args()

    proj_dir = args.project_dir or project_dir_from_filename(args.video_path)
    project = ProjectState(project_dir=proj_dir, video_path=args.video_path)

    decoder = FrameDecoder(video_path=args.video_path, max_frames=args.max_frames)
    decoder.analyze()

    if args.show_frames > 0:
        show_frames(decoder, num_frames=args.show_frames)
    if args.save_frames > 0:
        save_frames(decoder, num_frames=args.save_frames, save_dir=project.subdir("sample_frames"))