#!/usr/bin/env python3
import argparse
from vid2frames import FrameDecoder

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


def run_detector(decoder: FrameDecoder, detector_id: str, confidence_threshold: float | None):
    """Feed each frame through the detector, and record the results.
    """
    gl = Groundlight()
    detector = gl.get_detector(detector_id)
    answers = []
    for frame_num in tqdm(range(len(decoder)), desc="Inference"):
        frame = decoder.get_frame(frame_num)
        iq = gl.ask_ml(detector, frame)
        answer = get_iq_answer(iq)
        print(f"Frame {frame_num}: {answer}")
        answers.append(answer)
    return answers


def build_video_from_answers(decoder: FrameDecoder, answers: list[str], output_path: str):
    """Build a video from the answers.
    """
    print(f"(NOT YET IMPLEMENTED) Building video from {len(answers)} answers")
    #TODO: Implement this.
    raise NotImplementedError("Not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to use")
    parser.add_argument("--output", type=str, default=None, help="Name of file to save the output video to")
    parser.add_argument("--detector-id", type=str, required=True, help="ID of the detector to use")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Confidence threshold for inference (overrides detector's value)")
    args = parser.parse_args()

    decoder = FrameDecoder(video_path=args.video_path, max_frames=args.max_frames)
    answers = run_detector(decoder, args.detector_id, args.confidence_threshold)
    if args.output:
        output_path = args.output
    else:
        output_path = f"{args.video_path.rsplit('.', 1)[0]}_output.mp4"
    build_video_from_answers(decoder, answers, output_path)
