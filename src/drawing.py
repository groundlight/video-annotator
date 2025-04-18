import cv2
import numpy as np
from groundlight import ImageQuery, Detector, ModeEnum

def generate_unique_bgr_colors(n: int) -> list:
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors

def draw_iqs(detectors: list[Detector], iqs: list[ImageQuery], frame: np.ndarray) -> None:
    counting_iqs = []
    for detector, counting_iq in zip(detectors, iqs):
        detector_mode = detector.mode
        if detector_mode == ModeEnum.COUNT:
            counting_iqs.append(counting_iq)
        else:
            raise NotImplementedError(
                f'Detector mode {detector_mode} is not yet supported.'
            )
            
    unique_bbox_colors = generate_unique_bgr_colors(len(counting_iqs))
    for n, counting_iq in enumerate(counting_iqs):
        color = unique_bbox_colors[n]
        draw_bounding_boxes(counting_iq.rois, frame, color)
            
def draw_bounding_boxes(rois, frame: np.ndarray, color: tuple) -> None:
    height, width = frame.shape[:2]

    rois = [] if rois is None else rois
    for roi in rois:
        bbox = roi.geometry

        top_left = (int(bbox.left * width), int(bbox.top * height))
        bottom_right = (int(bbox.right * width), int(bbox.bottom * height))

        cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=2)

