import cv2
import numpy as np
from threading import Thread
from queue import Queue, Full, Empty
import time

class ThreadedVideoWriter:
    def __init__(self, filepath: str, fps: int, resolution: tuple) -> None:
        self.filepath = filepath
        self.resolution = resolution
        self.fps = fps

        self.queue = Queue(maxsize=10)
        self.writer = cv2.VideoWriter(
            filename=self.filepath,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=resolution,
        )
        self.run = False
        
        self.thread = Thread(target=self._run_loop, daemon=True)

        self._start()

    def add_frame(self, frame: np.ndarray) -> None:
        try:
            self.queue.put_nowait(frame)
        except Full:
            print("Video recorder queue full! Dropping frame.")

    def _start(self) -> None:
        self.run = True
        self.thread.start()

    def stop(self) -> None:
        self.run = False
        self.thread.join()
        self.writer.release()
        print('Video recording stopped.')

    def _run_loop(self) -> None:
        while self.run or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except Empty:
                continue  # No frame to write, loop again
            
            time.sleep(0.01)
            self.writer.write(frame)