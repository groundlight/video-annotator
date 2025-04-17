#!/usr/bin/env python3
"""
This script peforms inference on a video and produces an annotated video of the results
"""
import argparse
import time
from typing import Callable, Optional

from groundlight import Groundlight, ImageQuery, BinaryClassificationResult
from imgcat import imgcat
from tqdm.auto import tqdm

import os
from datetime import datetime

import cv2

from projstate import ProjectState
# from framemgr import FrameManager

from framegrab_web_server import FrameGrabWebServer

from drawing import draw_iqs

def get_iq_answer(iq: ImageQuery) -> str:
    """Get the answer from an ImageQuery, assuming it's a binary classification result.
    Returns UNSURE if the confidence is below the threshold.
    """
    threshold = iq.confidence_threshold
    if isinstance(iq.result, BinaryClassificationResult):
        result: BinaryClassificationResult = iq.result
        confidence = result.confidence
        if confidence is None:  # This means it's a human label
            return iq.result.label
        if confidence < threshold:
            return "UNSURE"
        return iq.result.label
    else:
        return "NONE"

def infer_and_produce_video(project: ProjectState, detector_ids: list[str]) ->  None:
    gl = Groundlight()
    
    detectors = [gl.get_detector(detector_id) for detector_id in detector_ids]
    
    for detector in detectors:
        print(detector)
    
    # Create the video reader
    cap = cv2.VideoCapture(project.video_path)
    _, frame = cap.read()
    height, width, _ = frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # return to beginning
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create the video output directory if it doesn't already exist
    video_output_dir = os.path.join(project.project_dir, "video_output")
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Generate a filename for the output video
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H.%M.%S")
    filename = timestamp + ".mp4"
    filepath = os.path.join(video_output_dir, filename)
    
    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    fps = cap.get(cv2.CAP_PROP_FPS) / 5
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    web_server = FrameGrabWebServer("Video Producer")
    
    for frame_num in tqdm(range(0, total_frames, 5), "Producing video"):
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
                
        # TODO annotate the frame
        draw_iqs(iqs, frame)
        
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
    args = parser.parse_args()
    
    project = ProjectState.load(args.project_dir)
    
    infer_and_produce_video(project, args.detector_ids)