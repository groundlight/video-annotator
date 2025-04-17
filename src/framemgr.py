#!/usr/bin/env python3
import argparse
import os
import json
from typing import Optional
from typing import Iterator

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

    def __init__(self, video_path: str, max_frames: int = 0, frame_metadata: Optional[FrameListMetadata] = None):
        """
        Args:
            :video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        print(video_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine the number of frames to use based on the total number of frames in the provided video
        # and the maximum frames requested by the user        
        if max_frames == 0:
            self.num_frames_to_use = self.total_frames
        else:
            self.num_frames_to_use = min(max_frames, self.total_frames)
            print(f'Using a cluster of {self.num_frames_to_use} frame of {self.total_frames} available frames.')
            
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
    
    def frame_indices_to_use(self) -> Iterator[int]:
        """
        Based on the length of the whole video and the number of frames that we actually want to use, generate
        the indices of the frames to use.
        """
        for i in range(self.num_frames_to_use):
            idx = int(round(i * (self.total_frames - 1) / (self.num_frames_to_use - 1)))
            yield idx

    def __len__(self):
        return self.num_frames_to_use

    def analyze(self):
        """Analyzes the video frame by frame.  Calculates embeddings, 
        and clusters the frames for diversity.
        """
        print(f"Scanning video and embedding frames...")
        
        # Evenly space the frames across the entire video
        progress = tqdm(self.frame_indices_to_use(), desc="Embedded frames")
        for frame_num in progress:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if not ret:
                print(f"Warning: didn't get frame {frame_num}")
                break
            frame = self.preprocess_frame(frame)
            self.qcluster.add_image(frame, frame_num)
            
        print("Scan complete, clustering frames...")
        cluster_info = self.qcluster.analyze()
        for entry in cluster_info:
            self.metadata.update_frame_metadata(num=entry["id"], diversity_rank=entry["diversity_rank"], cluster=entry["cluster"])
        self._update_frame_diversity_order()
        print(f"Clustering complete. Found {len(self.qcluster)} clusters")

    def _update_frame_diversity_order(self):
        def get_diversity_rank(i):  # just used by lambda below
            return self.metadata.get_frame_metadata(i)["diversity_rank"]
        self.frame_diversity_order = sorted(self.frame_indices_to_use(), key=get_diversity_rank)
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame to make motion detection faster."""
        # check if it has too many pixels.  For this we only need like 120k
        if frame.shape[0] * frame.shape[1] > 120000:
            frame = cv2.resize(frame, (400, 300))
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
        return rgb_frame
