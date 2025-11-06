#!/usr/bin/env python3
"""
This script peforms inference on a video and produces an annotated video of the results
"""
import argparse
import time
import os
import cv2
import sys
import warnings
import logging
import subprocess
import shutil
from io import StringIO
from pathlib import Path
from contextlib import contextmanager

from groundlight import Groundlight
from tqdm.auto import tqdm

from datetime import datetime

from projstate import ProjectState
from framegrab_web_server import FrameGrabWebServer
from drawing import draw_iqs

from threaded_video_writer import ThreadedVideoWriter

def optimize_video_for_web(input_path: Path, output_path: Path = None) -> Path:
    """
    Optimize MP4 video for web streaming by re-encoding to H.264 and moving moov atom to beginning.
    
    This ensures browser compatibility by:
    1. Re-encoding video to H.264 (browser-compatible codec)
    2. Moving moov atom to beginning (faststart for progressive playback)
    
    Requires ffmpeg to be installed. If ffmpeg is not available, returns the original path.
    
    Args:
        input_path: Path to input video file
        output_path: Optional output path. If None, creates optimized version in same directory.
        
    Returns:
        Path to optimized video (or original if optimization not possible)
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
    
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        print(f"Note: ffmpeg not found. Video will not be optimized for web streaming.")
        print(f"Video may not play in browsers if codec is not browser-compatible.")
        print(f"To enable optimization:")
        print(f"  1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print(f"  2. Install ffmpeg: brew install ffmpeg")
        print(f"  See INSTALL_FFMPEG.md for more options.")
        return input_path
    
    try:
        print(f"Optimizing video for web streaming (re-encoding to H.264)...")
        # Re-encode to H.264 for browser compatibility and optimize for web streaming
        # -c:v libx264: Use H.264 video codec (browser-compatible)
        # -preset fast: Good balance between speed and file size
        # -crf 23: High quality (lower = better quality, 18-28 is reasonable range)
        # -c:a copy: Copy audio stream without re-encoding (faster)
        # -movflags +faststart: Move moov atom to beginning for progressive playback
        result = subprocess.run(
            ['ffmpeg', '-i', str(input_path), 
             '-c:v', 'libx264',
             '-preset', 'fast',
             '-crf', '23',
             '-c:a', 'copy',
             '-movflags', '+faststart',
             str(output_path), '-y'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Video optimized for web (H.264 encoded): {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to optimize video: {e.stderr}")
        return input_path
    except Exception as e:
        print(f"Warning: Error optimizing video: {e}")
        return input_path

@contextmanager
def suppress_output():
    """Context manager to suppress stdout, stderr, warnings, and logging temporarily."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    # Temporarily increase logging level to suppress Groundlight SDK logs
    old_log_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL + 1)
    
    # Also suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.root.setLevel(old_log_level)

def infer_and_produce_video(project: ProjectState, 
                            detector_ids: list[str], 
                            frame_stride: int, 
                            web_preview_port: int,
                            human_review: str,
                            ) ->  None:
    gl = Groundlight()
    
    detectors = [gl.get_detector(detector_id) for detector_id in detector_ids]
    
    detector_names = [d.name for d in detectors]
    print(f'Using {len(detectors)} detectors: {detector_names}')
    
    # Create the video reader
    cap = cv2.VideoCapture(project.video_path)
    _, frame = cap.read()
    height, width, _ = frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # return to beginning
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # TODO this might not get the actual number of frames, is this a problem?
    
    # Create the video output directory if it doesn't already exist
    video_output_dir = os.path.join(project.project_dir, "video_output")
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Generate a filename for the output video
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H.%M.%S")
    filename = timestamp + ".mp4"
    filepath = os.path.join(video_output_dir, filename)
    
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Output FPS always matches input FPS (frame_stride only controls label update frequency)
    output_fps = input_fps
        
    print(f'Input video FPS: {input_fps} | Output video FPS: {output_fps} | Frame stride: {frame_stride} (labels updated every {frame_stride} frames)')
    
    # Create the video writer
    writer = ThreadedVideoWriter(filepath, output_fps, (width, height))
    
    message = f'Output video path: {filepath}'
    web_server = FrameGrabWebServer(f"Producing {filename}...", port=web_preview_port, message=message)
    
    try:
        # Process all frames, but only run inference every frame_stride frames
        # Labels persist on intermediate frames
        last_iqs = None  # Store last inference results to persist labels
        
        pbar = tqdm(range(total_frames), desc="Producing video")
        for frame_num in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                tqdm.write('Cannot read frame. Exiting...')
                break
            
            # Perform inference only every frame_stride frames
            if frame_num % frame_stride == 0:
                iqs = []
                for detector in detectors:
                    max_retries = 10
                    retries = 0
                    while True:
                        try:
                            # Suppress Groundlight SDK warnings to avoid breaking progress bar
                            with suppress_output():
                                iq = gl.submit_image_query(
                                    detector=detector,
                                    image=frame,
                                    human_review=human_review,
                                    wait=0.0,
                                )
                            break
                        except Exception as e:
                            retries += 1
                            if retries == max_retries:
                                raise RuntimeError(
                                    f'Repeatedly encountered an exception while submitting image queries to {detector.id}.'
                                )
                            tqdm.write(str(e))
                            time.sleep(1)
                            
                    iqs.append(iq)
                
                # Update last_iqs with new inference results
                last_iqs = iqs
            
            # Use last inference results to annotate frame (persists labels)
            if last_iqs is not None:
                # Annotate the frame (suppress warnings here too in case they come from result parsing)
                with suppress_output():
                    draw_iqs(detectors, last_iqs, frame)
            
            # Show in web browser (only for frames where inference happened)
            if frame_num % frame_stride == 0:
                web_server.show_image(frame)
            
            # Write all frames to output video
            writer.add_frame(frame)
    except KeyboardInterrupt:
        tqdm.write('User cancelled video production.')
    finally:
        cap.release()
        writer.stop()
        print(f'Finished producing video at {filepath}')
        
        # Optimize video for web streaming (if ffmpeg available)
        # Create temporary optimized file, then replace original
        temp_optimized = Path(filepath).parent / f"{Path(filepath).stem}_temp_opt{Path(filepath).suffix}"
        optimized_path = optimize_video_for_web(Path(filepath), temp_optimized)
        if optimized_path != Path(filepath) and optimized_path.exists():
            # Replace original with optimized version
            Path(filepath).unlink()  # Delete original
            optimized_path.rename(Path(filepath))  # Rename optimized to original name
            # Verify file is stable after rename
            import time
            time.sleep(0.1)
            if not Path(filepath).exists():
                raise RuntimeError("Optimized file disappeared after rename")
            print(f'Video optimized for web streaming')

if __name__ == "__main__":
    # Suppress Groundlight SDK warnings globally
    logging.getLogger('groundlight').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    parser.add_argument("--detector-ids", type=str, nargs="+", required=True, help="One or more detector IDs to use")
    parser.add_argument("--frame-stride", type=int, default=1,  help="Use every nth frame of the input video. Defaults to 1 (uses all frames).")
    parser.add_argument("--web-preview-port", type=int, default=5000,  help="The port used by the web preview.")
    parser.add_argument(
        "--human-review", 
        type=str, 
        default="NEVER", 
        choices=["NEVER", "ALWAYS", "DEFAULT"], 
        help="Specifies the cloud labeling behavior. Options are: 'NEVER' (never escalates to cloud labelers), 'ALWAYS' (always escalates), or 'DEFAULT' (only escalates ML answer is not confident)."
        )
    args = parser.parse_args()
    
    project = ProjectState.load(args.project_dir)
    
    infer_and_produce_video(project, args.detector_ids, args.frame_stride, args.web_preview_port, args.human_review)