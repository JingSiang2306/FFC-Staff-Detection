"""Run the CEPDOF OBB model as a person-detection-only video baseline."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLUE = (255, 0, 0)  # OpenCV uses BGR colour order.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/sample.mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/staff_detected_obb_baseline.mp4",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "yoloModel/best_CEPDOF_OBB_v1.0.pt",
    )
    parser.add_argument("--device", choices=("cpu", "auto", "0"), default="auto")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, choices=(640, 960, 1280), default=640)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Write the output without opening the live preview window",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if requested == "0":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 requested, but CUDA is unavailable.")
        return 0
    return "cpu"


def validate_args(args):
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"OBB model not found: {args.model}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output video paths must be different.")


def draw_obb(frame, corners, confidence):
    """Draw one blue rotated person box and its confidence score."""
    points = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [points], True, BLUE, 2, cv2.LINE_AA)

    label = f"person {confidence:.2f}"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    x = int(np.min(points[:, 0, 0]))
    y = int(np.min(points[:, 0, 1])) - 8
    x = max(0, min(x, frame.shape[1] - text_size[0] - 4))
    y = max(20, min(y, frame.shape[0] - 4))

    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        BLUE,
        2,
        cv2.LINE_AA,
    )


def draw_fps(frame, realtime_fps):
    fps_text = f"FPS: {realtime_fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        fps_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_person_count(frame, person_count):
    text = f"Persons: {person_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_width = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
    position = (frame.shape[1] - 20 - text_width, 35)
    cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    script_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    model = YOLO(str(args.model))
    if model.task != "obb":
        raise ValueError(f"Expected an OBB checkpoint, but model task is {model.task!r}.")

    video = cv2.VideoCapture(str(args.input))
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    source_fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / source_fps if source_fps else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Cannot create output video: {args.output}")

    frame_count = 0
    total_detections = 0
    stopped_by_user = False
    processing_started = time.perf_counter()
    previous_frame_time = processing_started
    realtime_fps = 0.0

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            result = model.predict(
                frame,
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            annotated_frame = frame.copy()
            obb = result.obb
            person_count = len(obb) if obb is not None else 0
            total_detections += person_count

            if obb is not None and person_count:
                corner_sets = obb.xyxyxyxy.cpu().numpy()
                confidences = obb.conf.cpu().tolist()
                for corners, confidence in zip(corner_sets, confidences):
                    draw_obb(annotated_frame, corners, confidence)

            current_frame_time = time.perf_counter()
            frame_elapsed = current_frame_time - previous_frame_time
            current_fps = 1.0 / frame_elapsed if frame_elapsed else 0.0
            realtime_fps = (
                current_fps
                if realtime_fps == 0.0
                else 0.9 * realtime_fps + 0.1 * current_fps
            )
            previous_frame_time = current_frame_time

            draw_fps(annotated_frame, realtime_fps)
            draw_person_count(annotated_frame, person_count)
            writer.write(annotated_frame)

            frame_count += 1
            print(f"\rProcessing {frame_count}/{total_frames} frames", end="", flush=True)

            if not args.no_display:
                cv2.imshow("CEPDOF OBB person-detection baseline", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break
    finally:
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_elapsed = time.perf_counter() - processing_started
    total_elapsed = time.perf_counter() - script_started
    processing_fps = frame_count / processing_elapsed if processing_elapsed else 0.0
    status = "Stopped by user" if stopped_by_user else "Completed successfully"
    average_people = total_detections / frame_count if frame_count else 0.0

    print("\r" + " " * 80, end="\r")
    print(f"Processed {frame_count}/{total_frames} frames ({status})")
    print(f"Saved annotated video to {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Confidence floor: {args.conf}; image size: {args.imgsz}")
    print(f"Average detected people per processed frame: {average_people:.2f}")
    print(f"Source video duration: {video_duration:.2f} seconds")
    print(f"Total processing time: {total_elapsed:.2f} seconds")
    print(f"Overall processing speed: {processing_fps:.1f} FPS")


if __name__ == "__main__":
    main()
