import json
import os
from typing import List

class FrameMetadata(dict):
    """Metadata about a single frame.
    """

    def as_dict(self) -> dict:
        """Convert to a dictionary.
        (In case we want to add functionality here.)
        """
        return dict(self)


class FrameListMetadata():
    """Stores metadata about a list of frames.
    """
    frame_metadata: List[FrameMetadata] = []

    def update_frame_metadata(self, num: int, **kwargs):
        """Update the frame metadata for a given frame number.
        """
        fmd = self.get_frame_metadata(num)
        fmd.update(kwargs)

    def get_frame_metadata(self, num: int) -> FrameMetadata:
        """Get the frame metadata for a given frame number.
        """
        if len(self.frame_metadata) <= num:
            self._extend_frame_metadata(num)
        return self.frame_metadata[num]

    def _extend_frame_metadata(self, num: int):
        """Extend the frame metadata to the given frame number.
        """
        for i in range(len(self.frame_metadata), num + 1):
            new_fmd = FrameMetadata(frame_num=i)
            self.frame_metadata.append(new_fmd)

    def __len__(self):
        return len(self.frame_metadata)


class ProjectState():
    """Stores the full state of a video for the purposes of VQA.
    """
    video_path: str
    project_dir: str
    frame_count: int
    frame_metadata: FrameListMetadata

    def __init__(self, video_path: str, project_dir: str, frame_count: int = 0):
        self.video_path = video_path
        self.project_dir = project_dir
        self.frame_metadata = FrameListMetadata()
        self.frame_count = frame_count

    def subdir(self, name: str) -> str:
        """Get the path to a subdirectory of the project directory.
        """
        out_dir = os.path.join(self.project_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _as_dict(self) -> dict:
        """Convert to a dictionary.
        """
        out = {
            "video_path": self.video_path,
            "frame_count": self.frame_count,
        }
        return out

    def save(self):
        """Save the project state to the project directory.
        """
        fn = os.path.join(self.project_dir, "project-info.json")
        out = self._as_dict()
        with open(fn, "w") as f:
            json.dump(out, f, indent=2)
