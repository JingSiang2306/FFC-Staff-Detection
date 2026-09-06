"""Detect people with OBB, verify staff tags, and confirm staff over time."""

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/sample.mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/staff_detected_obb.mp4",
    )
    parser.add_argument(
        "--person-model",
        type=Path,
        default=PROJECT_ROOT / "yoloModel/best_CEPDOF_OBB_v1.0.pt",
    )
    parser.add_argument(
        "--tag-model",
        type=Path,
        default=PROJECT_ROOT / "yoloModel/best_tag_v1.1.pt",
    )
    parser.add_argument("--device", choices=("cpu", "auto", "0"), default="auto")
    parser.add_argument(
        "--tracker",
        type=Path,
        default=PROJECT_ROOT / "trackers/custom_bytetrack.yaml",
    )
    parser.add_argument("--conf", type=float, default=0.10, help="Person confidence floor")
    parser.add_argument("--imgsz", type=int, choices=(640, 960, 1280), default=640)
    parser.add_argument("--tag-conf", type=float, default=0.30)
    parser.add_argument("--tag-imgsz", type=int, default=640)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.10,
        help="Padding around each OBB crop as a fraction of its width and height",
    )
    parser.add_argument("--vote-window", type=int, default=15)
    parser.add_argument("--vote-min", type=int, default=3)
    parser.add_argument("--staff-hold", type=int, default=45)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Write the output without opening the live OpenCV window",
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
    if not args.person_model.is_file():
        raise FileNotFoundError(f"Person OBB model not found: {args.person_model}")
    if not args.tag_model.is_file():
        raise FileNotFoundError(f"Tag model not found: {args.tag_model}")
    if args.tracker.is_absolute() and not args.tracker.is_file():
        raise FileNotFoundError(f"Tracker configuration not found: {args.tracker}")
    if not 0 <= args.conf <= 1 or not 0 <= args.tag_conf <= 1:
        raise ValueError("Confidence thresholds must be between 0 and 1.")
    if args.crop_padding < 0:
        raise ValueError("--crop-padding cannot be negative.")
    if args.tag_imgsz <= 0:
        raise ValueError("--tag-imgsz must be positive.")
    if args.vote_window <= 0 or args.vote_min <= 0 or args.staff_hold <= 0:
        raise ValueError("Voting values must be positive.")
    if args.vote_min > args.vote_window:
        raise ValueError("--vote-min cannot be greater than --vote-window.")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output video paths must be different.")


def order_obb_corners(corners):
    """Return cyclic OBB corners with a short edge first.

    The first edge becomes the horizontal crop axis and the adjacent long edge
    becomes the vertical crop axis. This produces a tight portrait-oriented crop
    without using the larger axis-aligned rectangle around a rotated person.
    """
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    cyclic = points[np.argsort(angles)]
    edge_lengths = np.linalg.norm(np.roll(cyclic, -1, axis=0) - cyclic, axis=1)
    short_edge_index = int(np.argmin(edge_lengths))
    return np.roll(cyclic, -short_edge_index, axis=0)


def crop_person_obb(frame, corners, padding):
    """Perspective-warp one OBB into a tight, portrait-oriented image crop."""
    p0, p1, p2, p3 = order_obb_corners(corners)
    horizontal = p1 - p0
    vertical = p3 - p0
    box_width = float(np.linalg.norm(horizontal))
    box_height = float(np.linalg.norm(vertical))
    if box_width < 2.0 or box_height < 2.0:
        return None

    horizontal_unit = horizontal / box_width
    vertical_unit = vertical / box_height
    horizontal_padding = box_width * padding
    vertical_padding = box_height * padding

    source = np.float32(
        [
            p0 - horizontal_unit * horizontal_padding - vertical_unit * vertical_padding,
            p1 + horizontal_unit * horizontal_padding - vertical_unit * vertical_padding,
            p2 + horizontal_unit * horizontal_padding + vertical_unit * vertical_padding,
            p3 - horizontal_unit * horizontal_padding + vertical_unit * vertical_padding,
        ]
    )
    crop_width = max(2, int(round(box_width + 2 * horizontal_padding)))
    crop_height = max(2, int(round(box_height + 2 * vertical_padding)))
    destination = np.float32(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        transform,
        (crop_width, crop_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def draw_staff_obb(frame, corners, track_id, tag_score):
    points = np.rint(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)
    colour = (0, 255, 0)
    label = f"STAFF ID {track_id} | Tag {tag_score:.2f}"
    cv2.polylines(frame, [points], True, colour, 2, cv2.LINE_AA)

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    x = int(np.min(points[:, 0, 0]))
    y = int(np.min(points[:, 0, 1])) - 8
    x = max(0, min(x, frame.shape[1] - text_size[0] - 4))
    y = max(20, min(y, frame.shape[0] - 4))
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
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


def draw_counts(frame, person_count, staff_count):
    lines = (f"Persons: {person_count}", f"Staff: {staff_count}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    right_margin = 20
    for index, text in enumerate(lines):
        text_width = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        position = (frame.shape[1] - right_margin - text_width, 35 + index * 32)
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main():
    script_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    person_model = YOLO(str(args.person_model))
    if person_model.task != "obb":
        raise ValueError(
            f"Expected an OBB person checkpoint, but model task is {person_model.task!r}."
        )
    tag_model = YOLO(str(args.tag_model))

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
        video.release()
        raise RuntimeError(f"Cannot create output video: {args.output}")

    tag_events = defaultdict(deque)
    active_staff_until = {}
    confirmed_staff_ids = set()
    tag_positive_events = 0
    frame_count = 0
    stopped_by_user = False
    warned_missing_ids = False
    processing_started = time.perf_counter()
    previous_frame_time = processing_started
    realtime_fps = 0.0

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            result = person_model.track(
                frame,
                persist=True,
                tracker=str(args.tracker),
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            annotated_frame = frame.copy()
            obb = result.obb
            person_count = len(obb) if obb is not None else 0

            tracked_people = []
            person_crops = []
            track_ids_tensor = getattr(obb, "id", None) if obb is not None else None
            if obb is not None and track_ids_tensor is not None:
                corner_sets = obb.xyxyxyxy.cpu().numpy()
                track_ids = track_ids_tensor.int().cpu().tolist()
                for corners, track_id in zip(corner_sets, track_ids):
                    crop = crop_person_obb(frame, corners, args.crop_padding)
                    if crop is not None and crop.size:
                        tracked_people.append((corners, track_id))
                        person_crops.append(crop)
            elif person_count and not warned_missing_ids:
                print(
                    "\nWarning: OBB detections were returned without track IDs; "
                    "staff voting cannot use those detections."
                )
                warned_missing_ids = True

            tag_scores = [0.0] * len(person_crops)
            if person_crops:
                tag_results = tag_model.predict(
                    person_crops,
                    classes=[0],
                    conf=args.tag_conf,
                    imgsz=args.tag_imgsz,
                    device=device,
                    verbose=False,
                )
                for index, tag_result in enumerate(tag_results):
                    if tag_result.boxes is not None and len(tag_result.boxes):
                        tag_scores[index] = float(tag_result.boxes.conf.max().item())

            vote_start = frame_count - args.vote_window + 1
            history_start = frame_count - max(args.vote_window, args.staff_hold) + 1
            active_staff_until = {
                track_id: active_until
                for track_id, active_until in active_staff_until.items()
                if active_until >= frame_count
            }
            staff_count = 0
            for (corners, track_id), tag_score in zip(tracked_people, tag_scores):
                history = tag_events[track_id]
                while history and history[0][0] < history_start:
                    history.popleft()
                if tag_score >= args.tag_conf:
                    history.append((frame_count, tag_score))
                    tag_positive_events += 1
                vote_count = sum(event_frame >= vote_start for event_frame, _ in history)
                if vote_count >= args.vote_min:
                    active_staff_until[track_id] = frame_count + args.staff_hold
                    confirmed_staff_ids.add(track_id)
                elif tag_score >= args.tag_conf and track_id in active_staff_until:
                    active_staff_until[track_id] = frame_count + args.staff_hold
                if track_id in active_staff_until:
                    recent_scores = [score for _, score in history]
                    display_score = max(recent_scores) if recent_scores else tag_score
                    draw_staff_obb(annotated_frame, corners, track_id, display_score)
                    staff_count += 1

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
            draw_counts(annotated_frame, person_count, staff_count)

            writer.write(annotated_frame)
            frame_count += 1
            print(f"\rprocessing {frame_count}/{total_frames} frames", end="", flush=True)
            if not args.no_display:
                cv2.imshow("Staff detection - OBB", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break
    finally:
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_elapsed = time.perf_counter() - processing_started
    total_elapsed = time.perf_counter() - script_started
    speed = frame_count / processing_elapsed if processing_elapsed else 0.0
    status = "Stopped by user" if stopped_by_user else "Completed successfully"
    print("\r" + " " * 80, end="\r")
    print(f"Processed {frame_count}/{total_frames} ({status})")
    print(f"Saved annotated video to {args.output}")
    print(f"Finished {frame_count} frames on {device} at {speed:.1f} FPS")
    print(f"Person OBB model: {args.person_model}")
    print(f"Tag model: {args.tag_model}")
    print(f"Tracker: {args.tracker}")
    print(f"Person confidence floor: {args.conf}; image size: {args.imgsz}")
    print(f"OBB crop padding: {args.crop_padding}")
    print(f"Tag confidence: {args.tag_conf}; tag image size: {args.tag_imgsz}")
    print(f"Temporal vote: {args.vote_min} positives within {args.vote_window} frames")
    print(f"Staff status hold: {args.staff_hold} frames after recent tag evidence")
    print(f"Tag-positive events: {tag_positive_events}")
    print(f"Confirmed staff track IDs: {sorted(confirmed_staff_ids)}")
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Total processing time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
