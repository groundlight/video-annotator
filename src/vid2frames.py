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
from vidstate import VidState




class FrameDecoder:
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
        self.vidstate = VidState(video_path=video_path, frame_count=self.total_frames)

    def __len__(self):
        return self.total_frames

    def analyze(self):
        """Analyzes the video frame by frame.  Calculates embeddings, 
        and clusters the frames for diversity.
        """
        print(f"Scanning video, embedding frames")
        progress = tqdm(range(self.total_frames), desc="Extracting frames")
        for frame_num in progress:
            ret, bgr_frame = self.cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(bgr_frame)
            self.qcluster.add_image(frame, frame_num)
        print("Scan complete, clustering frames")
        order = self.qcluster.diversity_order()
        for i, frame_num in enumerate(order):
            self.vidstate.update_frame_metadata(num=frame_num, diversity_rank=i)
        N = self.total_frames
        self.sorted_indices = sorted(range(N), key=lambda i: self.vidstate.get_frame_metadata(i).md["diversity_rank"])
        print(f"Clustering complete.  Found {len(self.sorted_indices)} clusters")

    def preprocess_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame to make motion detection faster."""
        # check if it has too many pixels.  Max of 0.5MP
        if bgr_frame.shape[0] * bgr_frame.shape[1] > 500000:
            bgr_frame = cv2.resize(bgr_frame, (800, 600))
        # convert colors to RGB
        #return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return bgr_frame

    def fmd_by_rank(self, rank: int) -> dict:
        """Get the frame by rank.
        :return: fmd dict with frame, metadata, and frame number
        """
        frame_num = self.sorted_indices[rank]
        fmd = self.vidstate.get_frame_metadata(frame_num)
        out = {
            "pil_img": Image.fromarray(self.get_frame(frame_num)),
            "frame": self.get_frame(frame_num),
            "frame_num": frame_num,
        }
        out.update(fmd.md)
        return out


    def show_top_frames(self, num_frames: int = 10):
        """Show the top frames."""
        for i in range(num_frames):
            frame_num = self.sorted_indices[i]
            fmd = self.fmd_by_rank(i)
            print(f"Frame {fmd['frame_num']}: {fmd['diversity_rank']}")
            imgcat(fmd["frame"])

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
