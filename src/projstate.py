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

    def save(self, out_dir: str):
        """Save the frame metadata to the project directory.
        """
        fn = os.path.join(out_dir, "frame-info.json")
        big_json_obj = []
        for i in range(len(self.frame_metadata)):
            fmd = self.frame_metadata[i]
            big_json_obj.append(fmd.as_dict())
        with open(fn, "w") as f:
            json.dump(big_json_obj, f, indent=2)
        print(f"Saved frame metadata to {fn}")

    @classmethod
    def load(cls, project_dir: str) -> "FrameListMetadata":
        """Load the frame metadata from the project directory.
        """
        fn = os.path.join(project_dir, "frame-info.json")
        with open(fn, "r") as f:
            big_json_obj = json.load(f)
        out = cls()
        for fmd in big_json_obj:
            out.frame_metadata.append(FrameMetadata(**fmd))
        return out


class ProjectState():
    """Stores the full state of a video for the purposes of VQA.
    """
    video_path: str
    project_dir: str
    frame_metadata: FrameListMetadata

    def __init__(self, video_path: str, project_dir: str):
        self.video_path = video_path
        self.project_dir = project_dir
        self.frame_metadata = FrameListMetadata()

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
        }
        return out

    def save(self):
        """Save the project state to the project directory.
        """
        self.frame_metadata.save(self.project_dir)
        fn = os.path.join(self.project_dir, "project-info.json")
        out = self._as_dict()
        with open(fn, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved project state to {fn}")
        print(f"Project dir: {self.project_dir}")

    @classmethod
    def load(cls, project_dir: str) -> "ProjectState":
        """Load the project state from the project directory.
        """
        fn = os.path.join(project_dir, "project-info.json")
        with open(fn, "r") as f:
            args = json.load(f)
        args["project_dir"] = project_dir
        out = cls(**args)
        out.frame_metadata = FrameListMetadata.load(project_dir)
        print(f"Loaded project state from {project_dir} with {len(out.frame_metadata)} frames")
        return out
