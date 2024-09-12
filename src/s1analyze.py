#!/usr/bin/env python3
import argparse
from vid2frames import FrameDecoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to analyze")
    parser.add_argument("--show-top-frames", type=int, default=0, help="Show the top N most diverse frames")
    args = parser.parse_args()

    decoder = FrameDecoder(video_path=args.video_path, max_frames=args.max_frames)
    decoder.analyze()
    if args.show_top_frames > 0:
        decoder.show_top_frames(num_frames=args.show_top_frames)
    if args.output_dir is not None:
        #TODO: I'm not sure this is working.
        # It's certainly not used by anything downstream.
        decoder.vidstate.save(args.output_dir)