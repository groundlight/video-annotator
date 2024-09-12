#!/usr/bin/env python3
"""
This script takes a video and a detector, and runs the detector on each frame.
It stores the results as metadata on each frame.
"""
import argparse
from framemgr import FrameManager

from groundlight import Groundlight, ImageQuery, BinaryClassificationResult
from tqdm.auto import tqdm


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


def run_detector(decoder: FrameManager, detector_id: str, confidence_threshold: float | None, verbose: bool = False):
    """Feed each frame through the detector, and record the results.
    """
    gl = Groundlight()
    detector = gl.get_detector(detector_id)
    answers = []
    for frame_num in tqdm(range(len(decoder)), desc="Inference"):
        frame = decoder.get_frame(frame_num)
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
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to use")
    parser.add_argument("--detector-id", type=str, required=True, help="ID of the detector to use")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Confidence threshold for inference (overrides detector's value)")
    args = parser.parse_args()

    decoder = FrameManager(video_path=args.video_path, max_frames=args.max_frames)
    answers = run_detector(decoder, args.detector_id, args.confidence_threshold)
    decoder.save_metadata()
