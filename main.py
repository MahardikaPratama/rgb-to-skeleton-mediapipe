#!/usr/bin/env python3
"""
main.py — Entry point for the RGB-to-Skeleton pipeline.

Run this file directly to process one or more videos.
See --help for all options.

Examples:
    python main.py --input data/raw/marah.mp4
    python main.py --input data/raw/
    python main.py --input data/raw/ --no-json --no-pickle
    python main.py --input data/raw/marah.mp4 --no-preview
"""

import os
import time
from datetime import timedelta
from src.core.pipeline import SkeletonPipeline
from src.core.cli import parse_args

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    pipeline = SkeletonPipeline(
        save_npy               = not args.no_npy,
        save_json              = not args.no_json,
        save_pickle            = not args.no_pickle,
        save_excel             = not args.no_excel,
        generate_preview       = not args.no_preview,
        generate_overlay       = not args.no_overlay,
        generate_skeleton_only = not args.no_skeleton_only,
        pickle_filename        = args.pickle_name,
    )

    input_path = args.input

    if os.path.isdir(input_path):
        pipeline.process_folder(input_path, label=args.label)
    elif os.path.isfile(input_path):
        pipeline.process_video(input_path, label=args.label)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Total execution time: {timedelta(seconds=int(elapsed))}")