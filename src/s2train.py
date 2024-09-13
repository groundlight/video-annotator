#!/usr/bin/env -S poetry run python
"""Submits frames to a Groundlight detector for training.
Waits for confident answers, which generally means human review.
All is done in diversity order, so the frames are spread out.
"""
import argparse

from groundlight import Groundlight
from imgcat import imgcat

from projstate import ProjectState
from framemgr import FrameManager

gl = Groundlight()

def build_detector(query: str, confidence: float):
    name = query[:20]  # would be nice if I didn't have to name the detector
    det = gl.get_or_create_detector(name=name, query=query, confidence_threshold=confidence)
    print(f"Detector {det} being used")
    return det


def submit_to_model(detector, fmd: dict, ask_async: bool = False):
    """Takes the frame-metadata dict and submits the frame to the model.
    """
    print(f"\n\n")
    imgcat(fmd["pil_img"])
    print(f"Submitting frame {fmd['frame_num']} to model.")
    iq_metadata = {
        "frame_num": fmd["frame_num"],
    }
    url = f"https://dashboard.groundlight.ai/reef/review/queue/detector/{detector.id}"
    print(f"Open the following URL in a browser to review the image:")
    print(f"    {url}")
    if ask_async:
        response = gl.ask_async(detector, fmd["pil_img"])
    else:
        response = gl.submit_image_query(
            detector, 
            fmd["pil_img"],  
            wait=120, 
            human_review="NEVER", 
            metadata=iq_metadata,
        )
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    parser.add_argument("--query", type=str, help="Query to define the model", required=True)
    parser.add_argument("--confidence", type=float, default=0.75, help="Confidence threshold for the model")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to submit to the model")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip")
    parser.add_argument("--ask-async", action="store_true", help="Don't wait for any responses to the image queries")
    args = parser.parse_args()

    project = ProjectState.load(args.project_dir)
    decoder = FrameManager.for_project(project)

    detector = build_detector(args.query, args.confidence)
    for i in range(args.skip_frames, args.skip_frames + args.num_frames):
        fmd = decoder.framedat_by_rank(i)
        submit_to_model(detector, fmd, ask_async=args.ask_async)