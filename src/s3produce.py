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
from io import StringIO
from contextlib import contextmanager

from groundlight import Groundlight
from tqdm.auto import tqdm

from datetime import datetime

from projstate import ProjectState
from framegrab_web_server import FrameGrabWebServer
from drawing import draw_iqs

from threaded_video_writer import ThreadedVideoWriter

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
    if frame_stride == 1:
        output_fps = input_fps
    elif input_fps % frame_stride == 0:
        output_fps = int(input_fps // frame_stride)
    else:
        raise ValueError(f'Frame stride of {frame_stride} is invalid for input_fps={input_fps}. '
                        f'Must be a divisor of the input FPS.')
        
    print(f'Input video FPS: {input_fps} | Output video FPS: {output_fps} given frame stride of {frame_stride}')
    
    # Create the video writer
    writer = ThreadedVideoWriter(filepath, output_fps, (width, height))
    
    message = f'Output video path: {filepath}'
    web_server = FrameGrabWebServer(f"Producing {filename}...", port=web_preview_port, message=message)
    
    try:
        pbar = tqdm(range(0, total_frames, frame_stride), desc="Producing video")
        for frame_num in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                tqdm.write('Cannot read frame. Exiting...')
                break
            
            # Perform inference
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
                    
            # Annotate the frame (suppress warnings here too in case they come from result parsing)
            with suppress_output():
                draw_iqs(detectors, iqs, frame)
            
            # Show in web browser
            web_server.show_image(frame)
            
            # Write the frame
            writer.add_frame(frame)
    except KeyboardInterrupt:
        tqdm.write('User cancelled video production.')
    finally:
        cap.release()
        writer.stop()
        print(f'Finished producing video at {filepath}')

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