"""Baseline: detect and track every person in a video."""

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/people_tracked.mp4"))
    parser.add_argument("--device", choices=("cpu", "auto", "0"), default="cpu")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if requested == "0":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 requested, but CUDA is unavailable.")
        return 0
    return "cpu"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    video = cv2.VideoCapture(str(args.input))
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    model = YOLO("yoloModel/yolo26n.pt")
    frame_count = 0
    started = time.perf_counter()

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.25,
                imgsz=640,
                device=device,
                verbose=False,
            )[0]
            writer.write(result.plot())
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    finally:
        video.release()
        writer.release()

    elapsed = time.perf_counter() - started
    speed = frame_count / elapsed if elapsed else 0.0
    print(f"Finished {frame_count} frames on {device} at {speed:.1f} FPS")
    print(f"Saved annotated video to {args.output}")


if __name__ == "__main__":
    main()
