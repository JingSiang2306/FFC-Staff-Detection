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
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path("trackers/custom_bytetrack.yaml"),
    )
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, choices=(640, 960, 1280), default=640)
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
    script_started = time.perf_counter()
    args = parse_args()
    device = resolve_device(args.device)
    video = cv2.VideoCapture(str(args.input))
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps else 0.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    model = YOLO("yoloModel/best_v1.1.pt")
    frame_count = 0
    processing_started = time.perf_counter()
    previous_frame_time = processing_started
    realtime_fps = 0.0

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                tracker=str(args.tracker),
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            annotated_frame = result.plot()

            # FPS Annotation
            current_frame_time = time.perf_counter()
            frame_elapsed = current_frame_time - previous_frame_time
            current_fps = 1.0 / frame_elapsed if frame_elapsed else 0.0
            realtime_fps = (
                current_fps
                if realtime_fps == 0.0
                else 0.9 * realtime_fps + 0.1 * current_fps
            )
            previous_frame_time = current_frame_time
            fps_text = f"FPS: {realtime_fps:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                fps_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated_frame)
            cv2.imshow("Annotated video", annotated_frame)
            cv2.waitKey(1)
            frame_count += 1
            print(f"\rprocessing {frame_count}/{total_frames} frames", end="", flush=True)
    finally:
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_elapsed = time.perf_counter() - processing_started
    total_elapsed = time.perf_counter() - script_started
    speed = frame_count / processing_elapsed if processing_elapsed else 0.0
    print("\r" + " " * 80, end="\r")
    print(f"processed {frame_count}/{total_frames} (Completed successfully)")
    print(f"Saved annotated video to {args.output}")
    print(f"Finished {frame_count} frames on {device} at {speed:.1f} FPS")
    print(f"Tracker: {args.tracker}")
    print(f"Tracking confidence floor: {args.conf}; image size: {args.imgsz}")
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Total processing time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
