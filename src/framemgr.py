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
from projstate import ProjectState, FrameListMetadata


class FrameManager:
    """Analyzes all the frames in a video, recording metadata about them.
    """

    def __init__(self, video_path: str, max_frames: int = 0, frame_metadata: FrameListMetadata | None = None):
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
        if frame_metadata is None:
            self.metadata = FrameListMetadata()
        else:
            self.metadata = frame_metadata
        self.frame_diversity_order = None

    @classmethod
    def for_project(cls, project: ProjectState):
        args = {
            "video_path": project.video_path,
            "max_frames": len(project.frame_metadata),
            "frame_metadata": project.frame_metadata,
        }
        out = cls(**args)
        out._update_frame_diversity_order()
        return out
    def __len__(self):
        return self.total_frames

    def analyze(self):
        """Analyzes the video frame by frame.  Calculates embeddings, 
        and clusters the frames for diversity.
        """
        print(f"Scanning video, embedding frames")
        progress = tqdm(range(self.total_frames), desc="Embedding frames")
        for frame_num in progress:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(frame)
            self.qcluster.add_image(frame, frame_num)
        print("Scan complete, clustering frames")
        cluster_info = self.qcluster.analyze()
        for entry in cluster_info:
            self.metadata.update_frame_metadata(num=entry["id"], diversity_rank=entry["diversity_rank"], cluster=entry["cluster"])
        self._update_frame_diversity_order()
        print(f"Clustering complete.  Found {len(self.qcluster)} clusters")

    def _update_frame_diversity_order(self):
        N = self.total_frames
        def get_diversity_rank(i):  # just used by lambda below
            return self.metadata.get_frame_metadata(i)["diversity_rank"]
        self.frame_diversity_order = sorted(range(N), key=get_diversity_rank)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame to make motion detection faster."""
        # check if it has too many pixels.  Max of 0.5MP
        if frame.shape[0] * frame.shape[1] > 500000:
            frame = cv2.resize(frame, (800, 600))
        return frame

    def framedat_by_rank(self, rank: int) -> dict:
        """Gets a bunch of data about a frame, from its rank (a.k.a. diversity order).
        :return: framedat dict with all metadata, plus:
            - pil_img: PIL image of the frame
            - frame: numpy array of the frame
            - frame_num: the frame number
        """
        frame_num = self.frame_diversity_order[rank]
        return self.framedat_by_num(frame_num)

    def framedat_by_num(self, frame_num: int) -> dict:
        """Gets a bunch of data about a frame, from its number.
        :return: framedat dict with all metadata, plus:
            - pil_img: PIL image of the frame
            - frame: numpy array of the frame
            - frame_num: the frame number
        """
        fmd = self.metadata.get_frame_metadata(frame_num)
        out = {
            "pil_img": Image.fromarray(self.get_frame(frame_num)),
            "frame": self.get_frame(frame_num),
            "frame_num": frame_num,
        }
        out.update(fmd)
        return out

    def set_metadata(self, frame_num: int, **kwargs):
        """Set metadata for a frame."""
        self.metadata.update_frame_metadata(num=frame_num, **kwargs)

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

