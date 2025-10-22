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
    # Separate IQs by type
    bbox_iqs = []  # COUNT and BOUNDING_BOX modes
    banner_iqs = []  # BINARY and MULTI_CLASS modes
    
    for detector, iq in zip(detectors, iqs):
        detector_mode = detector.mode
        if detector_mode in (ModeEnum.COUNT, ModeEnum.BOUNDING_BOX):
            bbox_iqs.append((detector, iq))
        elif detector_mode in (ModeEnum.BINARY, ModeEnum.MULTI_CLASS):
            banner_iqs.append((detector, iq))
        else:
            raise NotImplementedError(
                f'Detector mode {detector_mode} is not yet supported.'
            )
    
    # Draw bounding boxes for COUNT and BOUNDING_BOX modes
    if bbox_iqs:
        unique_bbox_colors = generate_unique_bgr_colors(len(bbox_iqs))
        for n, (detector, iq) in enumerate(bbox_iqs):
            color = unique_bbox_colors[n]
            draw_bounding_boxes(iq.rois, frame, color)
        
        class_names = [d.mode_configuration.get("class_name", f"Detector {i+1}") for i, (d, _) in enumerate(bbox_iqs)]
        draw_class_labels(frame, class_names, unique_bbox_colors)
    
    # Draw banners for BINARY and MULTI_CLASS modes
    if banner_iqs:
        draw_banners(banner_iqs, frame)
            
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

def draw_banners(detector_iq_pairs: list[tuple[Detector, ImageQuery]], frame: np.ndarray) -> None:
    """
    Draw banners at the top of the frame showing query and result for binary/multiclass detectors.
    Each detector gets its own line.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 8
    frame_height, frame_width = frame.shape[:2]
    
    y_offset = margin
    
    for detector, iq in detector_iq_pairs:
        # Get the result text
        result_text = get_result_text(iq)
        
        # Format: "Query: Result"
        query_text = detector.query
        
        # Calculate available space for query (leaving room for result)
        result_width, _ = cv2.getTextSize(result_text, font, font_scale, thickness)
        available_width = frame_width - result_width[0] - 4 * margin
        
        # Clip query text to fit
        query_text = clip_text_to_width(query_text, font, font_scale, thickness, available_width)
        full_text = f"{query_text}: {result_text}"
        
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
        banner_height = text_height + 2 * margin
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_offset), (frame_width, y_offset + banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        text_x = margin
        text_y = y_offset + margin + text_height
        cv2.putText(frame, full_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        
        # Move to next line
        y_offset += banner_height

def get_result_text(iq: ImageQuery) -> str:
    """
    Extract the result text from an ImageQuery.
    """
    if iq.result is None:
        return "Pending"
    
    # Try to get the label (for binary and multiclass)
    if hasattr(iq.result, 'label'):
        if hasattr(iq.result.label, 'value'):
            return str(iq.result.label.value)
        return str(iq.result.label)
    
    # Fallback
    return "Unknown"

def clip_text_to_width(text: str, font, font_scale: float, thickness: int, max_width: int) -> str:
    """
    Clip text to fit within a maximum width, adding ellipsis if needed.
    """
    text_width, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    if text_width[0] <= max_width:
        return text
    
    # Binary search for the right length
    ellipsis = "..."
    ellipsis_width, _ = cv2.getTextSize(ellipsis, font, font_scale, thickness)
    available_width = max_width - ellipsis_width[0]
    
    if available_width <= 0:
        return ellipsis
    
    # Estimate characters that fit
    chars_per_pixel = len(text) / text_width[0]
    estimated_chars = int(available_width * chars_per_pixel)
    
    # Find the exact length
    for length in range(estimated_chars, 0, -1):
        test_text = text[:length] + ellipsis
        test_width, _ = cv2.getTextSize(test_text, font, font_scale, thickness)
        if test_width[0] <= max_width:
            return test_text
    
    return ellipsis