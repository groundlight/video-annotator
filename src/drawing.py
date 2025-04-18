import cv2
import numpy as np
from groundlight import ImageQuery, Detector, ModeEnum

def generate_unique_bgr_colors(n: int) -> list:
    """
    Generate n unique BGR colors, evenly spaced across the color wheel, starting with green.
    """
    colors = []
    for i in range(n):
        hue = int((60 + 180 * i / n) % 180)
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
    
    class_names = [d.mode_configuration["class_name"] for d in detectors]
    draw_class_labels(frame, class_names, unique_bbox_colors)
            
def draw_bounding_boxes(rois, frame: np.ndarray, color: tuple) -> None:
    height, width = frame.shape[:2]

    rois = [] if rois is None else rois
    for roi in rois:
        bbox = roi.geometry

        top_left = (int(bbox.left * width), int(bbox.top * height))
        bottom_right = (int(bbox.right * width), int(bbox.bottom * height))

        cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=2)

def draw_class_labels(frame: np.ndarray, class_names: list[str], colors: list[tuple]) -> None:
    """
    Draw the class labels on the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    margin = 4
    line_height = 0

    for i, (label, color) in enumerate(zip(class_names, colors)):
        label = label[:20] # restrict the length of the label
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        line_height = text_height + 2 * margin

        # Position from bottom left, stacking upward
        x = 10
        y = frame.shape[0] - 10 - i * line_height

        # Background rectangle
        top_left = (x - margin, y - text_height - margin)
        bottom_right = (x + text_width + margin, y + margin)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), thickness=-1)

        # Text
        cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)