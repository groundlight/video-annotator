import os
from typing import List

import pydantic

class FrameMetadata(pydantic.BaseModel):
    """Metadata for a single frame.
    """
    frame_num: int
    md: dict = {}


class ProjectState(pydantic.BaseModel):
    """Stores the full state of a video for the purposes of VQA.
    """
    video_path: str = ""
    project_dir: str = ""
    frame_count: int = 0
    frame_metadata: List[FrameMetadata] = []

    def update_frame_metadata(self, num: int, **kwargs):
        """Update the frame metadata for a given frame number.
        """
        fmd = self.get_frame_metadata(num)
        fmd.md.update(kwargs)

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

    def subdir(self, name: str) -> str:
        """Get the path to a subdirectory of the project directory.
        """
        out_dir = os.path.join(self.project_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

def project_dir_from_filename(filename: str) -> str:
    """Get the project directory from the filename.
    """
    base_filename = os.path.basename(filename)
    base_filename = os.path.splitext(base_filename)[0]  # Remove the .mp4
    out = os.path.join("./proj", base_filename)
    os.makedirs(out, exist_ok=True)
    return out