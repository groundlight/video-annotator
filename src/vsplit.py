from dataclasses import dataclass
import os
import traceback

from PIL import Image
from tqdm.auto import tqdm
import cv2
import numpy as np

from lilutil import print_cvcap_metadata

@dataclass
class SplitRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    name: str


def bgr_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame[:, :, ::-1])

class VideoSplitter:
    """VideoSplitter is a class that processes a video file and can split it into regions.

    Example usage:
        vs = VideoSplitter('path_to_video.mp4')
        vs.print_metadata()
        vs.define_regions_4()
        vs.write_split_videos('quarters')
    """

    def __init__(self, video_file):
        self.video_file = video_file
        self.regions = []
        self._reload_video()

    def _reload_video(self):
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_file}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.file_size = os.path.getsize(self.video_file)
        self.file_format = os.path.splitext(self.video_file)[1]
        self.estimated_duration = self.frame_count / self.reported_fps

    def print_metadata(self):
        print_cvcap_metadata(self.video_file)

    def define_regions_4(self, names: list[str]=["upper_left", "upper_right", "lower_left", "lower_right"]):
        """Define 4 equally-sized regions for splitting.
        """
        self.regions = [
            SplitRegion(0, 0, self.width // 2, self.height // 2, names[0]),
            SplitRegion(self.width // 2, 0, self.width, self.height // 2, names[1]),
            SplitRegion(0, self.height // 2, self.width // 2, self.height, names[2]),
            SplitRegion(self.width // 2, self.height // 2, self.width, self.height, names[3])
        ]

    def _split_frame(self, frame: np.ndarray) -> list[np.ndarray]:
        out = []
        for region in self.regions:
            out.append(frame[region.y1:region.y2, region.x1:region.x2])
        return out

    def preview_regions(self) -> dict[str, Image.Image]:
        """Returns the first full frame, and then each of the splits.
        """
        ret, sample_frame = self.cap.read()
        out = {"full_frame": bgr_image(sample_frame)}
        if not ret:
            raise ValueError("Failed to read a frame from the video.")
        ndarray_crops = self._split_frame(sample_frame)
        for i, region in enumerate(self.regions):
            out[region.name] = bgr_image(ndarray_crops[i])
        return out

    def write_split_videos(self, codec: str = 'XVID'):
        """Writes the video to a series of files, one for each region.
        Codec options:
        - 'XVID' = MPEG-4 codec
        - 'MJPG' = Motion-JPEG codec
        - 'X264' = H.264 codec
        """
        self._reload_video()  # make sure we're at the start of the video
        num_regions = len(self.regions)
        if num_regions == 0:
            raise ValueError("No regions defined")

        # Create VideoWriters for each region
        filename_prefix = os.path.splitext(self.video_file)[0]
        out_filenames = [f"{filename_prefix}_{region.name}.avi" for region in self.regions]
        writers = []
        for i, out_filename in enumerate(out_filenames):
            size = (self.regions[i].x2 - self.regions[i].x1, self.regions[i].y2 - self.regions[i].y1)
            writer = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*codec), self.reported_fps, size)
            print(f"Writing to {out_filename} with size {size} and {self.reported_fps} fps using {codec} codec")
            writers.append(writer)

        try:
            # Process each frame
            for i in tqdm(range(self.frame_count)):
                if not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break

                for i, region in enumerate(self.regions):
                    cropped_frame = frame[region.y1:region.y2, region.x1:region.x2]
                    writers[i].write(cropped_frame)
            print(f"Finished writing all {self.frame_count} frames")
        finally:
            self.cap.release()
            for writer in writers:
                try:
                    writer.release()
                except Exception as e:
                    traceback.print_exc()

