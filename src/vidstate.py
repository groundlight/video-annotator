from typing import List
import atexit
import shelve

import pydantic

class GlobalShelf:
    _instance = None

    @classmethod
    def get_instance(cls, filename, flag='c', protocol=None, writeback=False):
        if cls._instance is None:
            cls._instance = shelve.open(filename, flag, protocol, writeback)
            atexit.register(cls._instance.close)
        return cls._instance

global_shelf = None

def initialize_global_shelf(filename, flag='c', protocol=None, writeback=False):
    global global_shelf
    global_shelf = GlobalShelf.get_instance(filename, flag, protocol, writeback)


def get_shelf():
    if global_shelf is None:
        raise RuntimeError("Global shelf has not been initialized. Call initialize_global_shelf() first.")
    return global_shelf


class FrameMetadata(pydantic.BaseModel):
    """Metadata for a single frame.
    """
    frame_num: int
    md: dict = {}


class VidState(pydantic.BaseModel):
    """Stores the full state of a video for the purposes of VQA.
    """
    video_path: str
    frame_count: int
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
