#!/usr/bin/env -S poetry run python
"""Submits frames to a Groundlight detector for training.
Waits for confident answers, which generally means human review.
All is done in diversity order, so the frames are spread out.
"""
import argparse
import time

from groundlight import Groundlight
from imgcat import imgcat

from projstate import ProjectState
from framemgr import FrameManager

gl = Groundlight()

def build_detector(query: str, confidence: float):
    name = query[:20]  # would be nice if I didn't have to name the detector
    det = gl.get_or_create_detector(name=name, query=query, confidence_threshold=confidence)
    print(f"Detector {det.id} being used")
    return det


def submit_to_model(detector, fmd: dict, ask_async: bool, wait: float, human_review: str) -> bool:
    """Takes the frame-metadata dict and submits the frame to the model.
    """
    print(f"\n\n")
    imgcat(fmd["pil_img"])
    print(f"Submitting frame {fmd['frame_num']} to model.")
    iq_metadata = {
        "frame_num": fmd["frame_num"],
    }
    url = f"https://dashboard.groundlight.ai/reef/review/queue/detector/{detector.id}"
    
    if human_review == "ALWAYS":
        message = f"Image submitted to cloud labeler. If you wish to review yourself, you can open this URL:\n\t{url}"
    elif human_review == "DEFAULT":
        message = f'Image submitted with default escalation behavior. Image will only escalate to cloud labeler if the ML result is not confident. If you wish to review yourself, you can open this URL:\n\t{url}'
    elif human_review == "NEVER":
        message = f"Open the following URL in a browser to review the image:\n\t{url}"
    else:
        raise ValueError(f'Unexpected value for human_review: {human_review}')
    
    print('-' * 50)
    print(message)
    
    if ask_async:
        response = gl.ask_async(detector, fmd["pil_img"], human_review=human_review)
    else:
        response = gl.submit_image_query(
            detector, 
            fmd["pil_img"],  
            wait=wait, 
            human_review=human_review, 
            metadata=iq_metadata,
        )
    print(response)

def submit_to_model_retry(detector, fmd: dict, ask_async: bool, wait: float, human_review: str) -> None:
    """Takes the frame-metadata dict and submits the frame to the model.
    """
    delay = 5
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            submit_to_model(detector, fmd, ask_async=ask_async, wait=wait, human_review=human_review)
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            import pdb; pdb.set_trace()
            print(f"Error submitting frame {fmd['frame_num']}: {e}.  Pausing for {delay} seconds.")
            time.sleep(delay)
            delay *= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Query to define the model. Either provide this or detector-id.")
    group.add_argument("--detector-id", type=str, help="ID of an existing detector. Either provide this or query.")
    
    parser.add_argument("--confidence", type=float, default=0.75, help="Confidence threshold for the model")
    parser.add_argument("--wait", type=float, default=120.0, help="The amount of time to wait for a confident answer.")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to submit to the model")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip")
    parser.add_argument("--ask-async", action="store_true", help="Don't wait for any responses to the image queries")
    parser.add_argument(
        "--human-review", 
        type=str, 
        default="NEVER", 
        choices=["NEVER", "ALWAYS", "DEFAULT"], 
        help="Specifies the cloud labeling behavior. Options are: 'NEVER' (never escalates to cloud labelers), 'ALWAYS' (always escalates), or 'DEFAULT' (only escalates ML answer is not confident)."
        )

    args = parser.parse_args()

    project = ProjectState.load(args.project_dir)
    decoder = FrameManager.for_project(project)
    
    if args.query is not None:
        detector = build_detector(args.query, args.confidence)
    else:
        detector = gl.get_detector(args.detector_id)
        
        if args.confidence is not None:
            gl.update_detector_confidence_threshold(detector, args.confidence)
            print(f"Updated {detector.id}'s confidence threshold to {args.confidence}")
            
    print(f'Using detector {detector}')
    
    for i in range(args.skip_frames, args.skip_frames + args.num_frames):
        fmd = decoder.framedat_by_rank(i)
        submit_to_model_retry(detector, fmd, ask_async=args.ask_async, wait=args.wait, human_review=args.human_review)