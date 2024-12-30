import os

import cv2

def format_time_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    out = ""
    if hours > 0:   
        out += f"{hours:02d}h"
    if minutes > 0:
        out += f"{minutes:02d}m"
    if seconds > 0:
        out += f"{seconds:02d}s"
    return out

def print_cvcap_metadata(video_path:str):
    print(f"Video path: {video_path}")
    cap = cv2.VideoCapture(video_path)
    print(f"CV2 object: {cap}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    file_size = os.path.getsize(video_path)
    file_format = os.path.splitext(video_path)[1]
    estimated_duration = frame_count / reported_fps
    print(f"Size (MB): {file_size / 1024 / 1024:.2f}")
    print(f"File format: {file_format}")
    print(f"Number of frames: {frame_count}")
    print(f"Resolution: {width}x{height}")
    if frame_count > 0:
        print(f"Bytes per frame: {file_size / frame_count:.1f}")
    print(f"Estimated duration: {format_time_hms(estimated_duration)}")
