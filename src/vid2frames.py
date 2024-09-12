#!/usr/bin/env python3
import argparse
import os
import json

from PIL import Image
from imgcat import imgcat
from tqdm.auto import tqdm
import cv2
import numpy as np

from qcluster import QCluster
from projstate import ProjectState




class FrameManager:
    """Analyzes all the frames in a video, recording metadata about them.
    """

    def __init__(self, video_path: str, max_frames: int = 0):
        """
        Args:
            :video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            if max_frames < self.total_frames:
                print(f"Limiting to {max_frames} frames.  Original frame count: {self.total_frames}")
                self.total_frames = max_frames
        print(f"Total frames: {self.total_frames}")
        self.qcluster = QCluster()
        self.state = ProjectState(video_path=video_path, frame_count=self.total_frames)

    @classmethod
    def for_project(cls, project: ProjectState):
        return cls(video_path=project.video_path, max_frames=project.frame_count)

    def __len__(self):
        return self.total_frames

    def analyze(self):
        """Analyzes the video frame by frame.  Calculates embeddings, 
        and clusters the frames for diversity.
        """
        print(f"Scanning video, embedding frames")
        progress = tqdm(range(self.total_frames), desc="Extracting frames")
        for frame_num in progress:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(frame)
            self.qcluster.add_image(frame, frame_num)
        print("Scan complete, clustering frames")
        order = self.qcluster.diversity_order()
        for i, frame_num in enumerate(order):
            self.state.update_frame_metadata(num=frame_num, diversity_rank=i)
        N = self.total_frames
        self.sorted_indices = sorted(range(N), key=lambda i: self.state.get_frame_metadata(i).md["diversity_rank"])
        print(f"Clustering complete.  Found {len(self.sorted_indices)} clusters")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame to make motion detection faster."""
        # check if it has too many pixels.  Max of 0.5MP
        if frame.shape[0] * frame.shape[1] > 500000:
            frame = cv2.resize(frame, (800, 600))
        return frame

    def framedat_by_rank(self, rank: int) -> dict:
        """Gets a bunch of data about a frame, from its rank.
        :return: fmd dict with frame, metadata, and frame number
        """
        frame_num = self.sorted_indices[rank]
        return self.framedat_by_num(frame_num)

    def framedat_by_num(self, frame_num: int) -> dict:
        """Gets a bunch of data about a frame, from its number.
        :return: fmd dict with frame, metadata, and frame number
        """
        fmd = self.state.get_frame_metadata(frame_num)
        out = {
            "pil_img": Image.fromarray(self.get_frame(frame_num)),
            "frame": self.get_frame(frame_num),
            "frame_num": frame_num,
        }
        out.update(fmd.md)
        return out

    def set_metadata(self, frame_num: int, **kwargs):
        """Set metadata for a frame."""
        self.state.update_frame_metadata(num=frame_num, **kwargs)

    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get the frame from the video given the frame number.
        Preprocesses before returning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, bgr_frame = self.cap.read()
        if not ret:
            raise ValueError(f"Error reading frame {frame_num}")
        # swap bgr to rgb
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.preprocess_frame(rgb_frame)

    def save_metadata(self):
        """Save the metadata to the file frame-info.json in the project directory."""
        fn = os.path.join(self.state.project_dir, "frame-info.json")
        big_json_obj = []
        for i in range(len(self.state.frame_metadata)):
            fmd = self.state.get_frame_metadata(i)
            big_json_obj.append(fmd.as_dict())
        with open(fn, "w") as f:
            json.dump(big_json_obj, f, indent=2)
        print(f"Saved frame metadata to {fn}")
