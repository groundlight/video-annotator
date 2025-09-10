#!/usr/bin/env -S poetry run python
"""Submits frames to a Groundlight detector for training.
By default, waits for confident answers, which generally means human review.
All is done in diversity order, so the frames are spread out.
"""
import argparse
import time

from groundlight import Groundlight, ImageQuery, CountingResult, BinaryClassificationResult


def pprint_iq(iq: ImageQuery) -> None:
    """
    Pretty-print print the details of an image query.
    """
    print(f'"{iq.query}"')
    print(f'ID: {iq.id}')

    # Detector mode-specific attributes
    if isinstance(iq.result, CountingResult):
        print(f'Count: {iq.result.count}')
    elif isinstance(iq.result, BinaryClassificationResult):
        label = '-' if iq.result is None else iq.result.label.value
        print(f'Label: {label}')
    else:
        raise ValueError(
            f'Unsupported result type: {type(iq.result)}'
        )
        
    confidence = None if iq.result is None else iq.result.confidence
    confidence_str = '-' if confidence is None else f'{confidence * 100:.2f}%'
    print(f'Confidence: {confidence_str}')
    
    source = '-' if iq.result is None else iq.result.source
    print(f'Source: {source}')


def build_detector(query: str, confidence: float):
    name = query[:20]  # would be nice if I didn't have to name the detector
    det = gl.get_or_create_detector(name=name, query=query, confidence_threshold=confidence)
    print(f"Detector {det.id} being used")
    return det


def submit_to_model(detector, fmd: dict, ask_async: bool, wait: float, human_review: str) -> bool:
    """Takes the frame-metadata dict and submits the frame to the model.
    """
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
    print(f"Submitting frame {fmd['frame_num']} to model.")
    print(message)
    
    t1 = time.time()
    if ask_async:
        iq = gl.ask_async(detector, fmd["pil_img"], human_review=human_review, metadata=iq_metadata)
        print(f'Submitted {iq.id} asynchonously to Groundlight.')
    else:
        print('Submitting iq...')
        iq = gl.submit_image_query(detector, fmd["pil_img"], wait=0.0, human_review=human_review, metadata=iq_metadata)
        print(f'Finished submitting {iq.id}.')
        
        confidence = 0.0 if iq.result.confidence is None else iq.result.confidence
        confidence_threshold = detector.confidence_threshold
        if human_review == "ALWAYS":
            print('-' * 5 + "ML Result" + '-' * 5)
            pprint_iq(iq)
            print(f'human_review set to "ALWAYS". Waiting for human answer...')
            human_confidence = 1.0
            iq = gl.wait_for_confident_result(iq, confidence_threshold=human_confidence, timeout_sec=wait)
        elif human_review == "DEFAULT" and confidence < confidence_threshold:
            print('-' * 5 + "Preliminary ML Result" + '-' * 5)
            pprint_iq(iq)
            print(
                f'human_review is set to "DEFAULT" and the preliminary ML confidence ({confidence:.4f}) was below '
                f'the confidence threshold ({confidence_threshold:.4f}). Escalating {iq.id} to cloud labeler...'
                )
            iq = gl.wait_for_confident_result(iq, confidence_threshold=confidence_threshold, timeout_sec=wait)
            
        print('-' * 5 + "Final Result" + '-' * 5)
        pprint_iq(iq)
        
    t2 = time.time()
    elapsed_time = t2 - t1
    
    print(f'Result returned in {elapsed_time:.2f} seconds.')


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
            print(f"Error submitting frame {fmd['frame_num']}: {e}.  Pausing for {delay} seconds.")
            time.sleep(delay)
            delay *= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Path to the project directory")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Query to define the model for a new binary detector. Either provide this or detector-id")
    group.add_argument("--detector-id", type=str, help="ID of an existing detector. Either provide this or query")
    
    parser.add_argument("--confidence", type=float, default=0.75, help="Confidence threshold for the model")
    parser.add_argument("--wait", type=float, default=120.0, help="The amount of time to wait for a confident answer")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to submit to the model")
    parser.add_argument("--ask-async", action="store_true", help="Don't wait for any responses to the image queries")
    parser.add_argument(
        "--human-review", 
        type=str, 
        default="DEFAULT", 
        choices=["NEVER", "ALWAYS", "DEFAULT"], 
        help="Specifies the cloud labeling behavior. Options are: 'NEVER' (never escalates to cloud labelers), 'ALWAYS' (always escalates), or 'DEFAULT' (only escalates ML answer is not confident)"
        )

    args = parser.parse_args()
    
    # Connect to Groundlight client
    gl = Groundlight()
    
    # Deferring these expensive imports to make the CLI more responsive
    from projstate import ProjectState
    from framemgr import FrameManager
        
    project = ProjectState.load(args.project_dir)
    decoder = FrameManager.for_project(project)
    
    if args.query is not None:
        detector = build_detector(args.query, args.confidence)
    else:
        detector = gl.get_detector(args.detector_id)
        
        if args.confidence is not None:
            gl.update_detector_confidence_threshold(detector, args.confidence)
            print(f"Updated {detector.id}'s confidence threshold to {args.confidence}")
            
    detector_id = detector.id
    
    print('get_num_previously_submitted_frames starting...')
    num_previously_submitted_frames = project.get_num_previously_submitted_frames(detector_id)
    print('get_num_previously_submitted_frames finished.')
    
    num_frames = min(args.num_frames, len(decoder))
    print(f'Previously submitted {num_previously_submitted_frames} frames to detector {detector_id}. Submitting {num_frames} frames more...')
    
    i = 0
    num_submitted_frames = 0
    while True:
        # Get the next clustered frame by diversity rank
        try:
            frame_num = decoder.frame_num_by_rank(i)
            i += 1
        except IndexError:
            print(
                f'Reached the end of available clustered frames. Was only able to submit {num_submitted_frames} of the requested {num_frames} frames.'
                )
            break
        
        # Check if the frame has already been submitted
        if project.check_frame_submission(frame_num, detector_id):
            continue # Frame has already been submitted. Skipping...
            
        # Submit the frame and log the submission
        fmd = decoder.framedat_by_num(frame_num)
        try:
            submit_to_model_retry(detector, fmd, ask_async=args.ask_async, wait=args.wait, human_review=args.human_review)
        finally:
            project.log_frame_submission(frame_num, detector_id)
            num_submitted_frames += 1
        
        # Check if we have submitted the requested number of frames
        if num_submitted_frames == num_frames:
            print(f'Finshed submitting {num_frames} frames to {detector_id}.')
            break