#!/usr/bin/env python3
"""
This script takes a video and a detector, and runs the detector on each frame.
It stores the results as metadata on each frame.
"""
import argparse

from groundlight import Groundlight, ImageQuery, BinaryClassificationResult
from tqdm.auto import tqdm

from projstate import ProjectState
from framemgr import FrameManager


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


def run_detector(decoder: FrameManager, *, 
    detector_id: str, 
    confidence_threshold: float | None, 
    verbose: bool = False, 
    max_frames: int | None = None,
):
    """Feed each frame through the detector, and record the results.
    """
    gl = Groundlight()
    detector = gl.get_detector(detector_id)
    answers = []
    N = min(len(decoder), max_frames) if max_frames else len(decoder)
    for frame_num in tqdm(range(N), desc="Inference"):
        framedat = decoder.framedat_by_num(frame_num)
        if "answer" in framedat:
            if verbose:
                print(f"Frame {frame_num} already answered: {framedat['answer']}")
            continue
        frame = framedat["pil_img"]
        iq = gl.ask_ml(detector, frame)
        answer = get_iq_answer(iq)
        if verbose:
            print(f"Frame {frame_num}: {iq.result}")
        md = {
            "answer": answer,
            "answer_confidence": iq.result.confidence,
            "iq_id": iq.id,
        }
        decoder.set_metadata(frame_num, **md)
        answers.append(answer)
    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    parser.add_argument("--detector-id", type=str, required=True, help="ID of the detector to use")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Confidence threshold for inference (overrides detector's value)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    project = ProjectState.load(args.project_dir)
    decoder = FrameManager.for_project(project)
    answers = run_detector(decoder, 
        detector_id=args.detector_id, 
        confidence_threshold=args.confidence_threshold, 
        verbose=args.verbose, 
        max_frames=args.max_frames
    )
    project.save()
