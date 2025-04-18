#!/usr/bin/env python3
"""
This script peforms inference on a video and produces an annotated video of the results
"""
import argparse
import time
import os
import cv2

from groundlight import Groundlight
from tqdm.auto import tqdm

from datetime import datetime

from projstate import ProjectState
from framegrab_web_server import FrameGrabWebServer
from drawing import draw_iqs

from threaded_video_writer import ThreadedVideoWriter

def infer_and_produce_video(project: ProjectState, detector_ids: list[str], frame_stride: int) ->  None:
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    writer = cv2.VideoWriter(filepath, fourcc, output_fps, (width, height))
    # writer = ThreadedVideoWriter(filepath=)
    
    web_server = FrameGrabWebServer("Video Producer")
    
    for frame_num in tqdm(range(0, total_frames, frame_stride), "Producing video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame. Exiting...')
            break
        
        # Perform inference
        iqs = []
        for detector in detectors:
            while True:
                try:
                    iq = gl.submit_image_query(
                        detector=detector,
                        image=frame,
                        human_review="NEVER",
                        wait=0.0,
                    )
                    break
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    
            iqs.append(iq)
                
        # Annotate the frame
        draw_iqs(detectors, iqs, frame)
        
        web_server.show_image(frame)
        
        # Write the frame
        writer.write(frame)
        
    cap.release()
    writer.release()
    
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    parser.add_argument("--detector-ids", type=str, nargs="+", required=True, help="One or more detector IDs to use")
    parser.add_argument("--frame-stride", type=int, default=1,  help="Use every n frames of the input video. Defaults to 1 (uses all frames).")
    args = parser.parse_args()
    
    project = ProjectState.load(args.project_dir)
    
    infer_and_produce_video(project, args.detector_ids, args.frame_stride)