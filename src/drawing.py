import cv2
import numpy as np
from groundlight import ImageQuery, CountingResult, BinaryClassificationResult

def draw_iqs(iqs: list[ImageQuery], frame: np.ndarray) -> None:
    for iq in iqs:
        if iq.result is None:
            continue
        elif isinstance(iq.result, CountingResult):
            if iq.rois is None:
                continue
            draw_bounding_boxes(iq.rois, frame)
        elif isinstance(iq.result, BinaryClassificationResult):
            raise NotImplementedError('Binary detectors not yet supported.')
        else:
            raise ValueError(
                f'Unsupported image query result type: {iq.result}'
            )
            
def draw_bounding_boxes(rois, frame) -> None:
    height, width = frame.shape[:2]

    for roi in rois:
        bbox = roi.geometry

        top_left = (int(bbox.left * width), int(bbox.top * height))
        bottom_right = (int(bbox.right * width), int(bbox.bottom * height))

        cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=2)

