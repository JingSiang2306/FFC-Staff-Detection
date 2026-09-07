"""Detect people, verify staff tags, and confirm staff using temporal voting."""

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    """
    Read command-line options, using defaults for options not supplied.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/staff_detected.mp4"))
    parser.add_argument(
        "--csv-output", type=Path, default=None,
        help="Staff CSV path (default: video output path with a .csv extension)",
    )
    parser.add_argument("--person-model", type=Path, default=Path("yoloModel/best_v1.2.pt"))
    parser.add_argument("--tag-model", type=Path, default=Path("yoloModel/best_tag_v1.2.pt"))
    parser.add_argument("--device", choices=("cpu", "auto", "0"), default="auto")
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path("trackers/custom_bytetrack.yaml"),
    )
    parser.add_argument("--conf", type=float, default=0.10, help="Person confidence threshold")
    parser.add_argument("--imgsz", type=int, choices=(640, 960, 1280), default=640)
    parser.add_argument("--tag-conf", type=float, default=0.5, help="Tag confidence threshold")
    parser.add_argument("--tag-imgsz", type=int, default=640)
    parser.add_argument("--crop-padding", type=float, default=0.10)  # Add space on each side of a person crop.
    parser.add_argument("--vote-window", type=int, default=15)       # Look for evidence in this many recent frames.
    parser.add_argument("--vote-min", type=int, default=3)           # Require this many positive frames to confirm staff.
    parser.add_argument("--staff-hold", type=int, default=45)        # Keep staff status briefly without fresh evidence.
    parser.add_argument(
        "--person-show", action="store_true",
        help="Show thin red person boxes and small IDs below them (off by default)",
    )
    return parser.parse_args()


def resolve_device(requested):
    """
    Select the CPU or first CUDA GPU; reject an unavailable requested GPU.
    """
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if requested == "0":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 requested, but CUDA is unavailable.")
        return 0
    return "cpu"


def validate_args(args):
    """
    Stop early if model files are missing or option values are invalid.
    """
    if not args.person_model.is_file():
        raise FileNotFoundError(f"Person model not found: {args.person_model}")
    if not args.tag_model.is_file():
        raise FileNotFoundError(f"Tag model not found: {args.tag_model}")
    if not 0 <= args.conf <= 1 or not 0 <= args.tag_conf <= 1:
        raise ValueError("Confidence thresholds must be between 0 and 1.")
    if args.crop_padding < 0:
        raise ValueError("--crop-padding cannot be negative.")
    if args.vote_window <= 0 or args.vote_min <= 0 or args.staff_hold <= 0:
        raise ValueError("Voting values must be positive.")
    if args.vote_min > args.vote_window:
        raise ValueError("--vote-min cannot be greater than --vote-window.")
    
    csv_path = args.csv_output if args.csv_output is not None else args.output.with_suffix(".csv")
    protected_paths = (args.input, args.output, args.person_model, args.tag_model, args.tracker)
    if csv_path.resolve() in {path.resolve() for path in protected_paths}:
        raise ValueError("CSV output must not overwrite the video, models, or tracker configuration.")


def crop_person(frame, box, padding):
    """
    Extract a padded person crop without going beyond the image edges.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding                     # Scale horizontal padding with the person's width.
    pad_y = (y2 - y1) * padding                     # Scale vertical padding with the person's height.
    # Clip the expanded box to valid image coordinates.
    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(frame_width, int(x2 + pad_x))
    crop_y2 = min(frame_height, int(y2 + pad_y))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]  # Image slices use rows (y) before columns (x).


def draw_person_box(frame, box, track_id):
    """
    Draw a thin red person box with a small ID near its bottom-left corner.
    """
    x1, y1, x2, y2 = map(int, box)
    colour = (0, 0, 255)  # OpenCV stores colours in blue, green, red order.
    label = f"ID {track_id}" if track_id is not None else "ID ?"
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
    text_x = max(0, min(x1 + 2, frame.shape[1] - text_width - 2))
    # Keep the ID inside the image when the person's feet reach the bottom edge.
    text_y = y2 + 14 if y2 + 17 < frame.shape[0] else max(12, y2 - 6)
    # Draw a black outline first, then the coloured text for readability.
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, colour, 1, cv2.LINE_AA)


def draw_staff_box(frame, box, track_id, tag_score):
    """
    Draw the green staff box, track ID, and supplied tag confidence.
    The caller supplies the highest score in the retained history.
    """
    x1, y1, x2, y2 = map(int, box)
    colour = (0, 255, 0)
    label = f"STAFF ID {track_id} | Tag {tag_score:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    # Draw a black outline under the green label.
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
        2,
        cv2.LINE_AA,
    )


def staff_coordinates(box):
    """
    Return the person's bottom-centre position in original-frame pixels.
    Round once so the video label and CSV contain the same coordinates.
    """
    x1, _, x2, y2 = box
    return round((x1 + x2) / 2, 1), round(y2, 1)


def format_video_time(frame_index, fps):
    """
    Convert a zero-based frame index to video time in MM:SS:mmm format.
    Use the source frame rate, not the computer's processing speed.
    """
    total_ms = round(frame_index * 1000 / fps)
    minutes, remainder = divmod(total_ms, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}m {seconds:02d}s {milliseconds:03d}ms"


def draw_staff_coordinates(frame, box, x, y):
    """
    Stack X above Y beside the lower-right corner of the staff box.
    Move the label inside the image when it reaches an image edge.
    """
    lines = (f"X: {x:.1f}", f"Y: {y:.1f}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    sizes = [cv2.getTextSize(text, font, font_scale, 1) for text in lines]
    text_width = max(size[0][0] for size in sizes)
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    line_gap = text_height + baseline + 5
    frame_height, frame_width = frame.shape[:2]
    text_x = int(box[2]) + 6
    if text_x + text_width + 3 > frame_width:
        text_x = int(box[2]) - text_width - 6
    text_x = max(2, min(text_x, frame_width - text_width - 3))
    last_y = max(text_height + line_gap + 3, min(int(box[3]), frame_height - baseline - 3))
    for index, text in enumerate(lines):
        position = (text_x, last_y - line_gap + index * line_gap)
        # Draw a black outline beneath the green coordinates.
        # cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, position, font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)


def draw_fps(frame, realtime_fps):
    """
    Show the smoothed processing speed at the top-left corner.
    """
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
    """
    Show current-frame person and staff counts at the top-right corner.
    """
    lines = (f"Persons: {person_count}", f"Staff: {staff_count}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    right_margin = 20
    for index, text in enumerate(lines):
        # Measure each label so both lines align with the right margin.
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
    """
    Process the video once: track people, check their tags, and draw results.
    Keep staff evidence separate for each tracker ID.
    """
    script_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    csv_path = args.csv_output if args.csv_output is not None else args.output.with_suffix(".csv")
    device = resolve_device(args.device)
    video = cv2.VideoCapture(str(args.input))
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0                           # Use 25 FPS if the video reports no frame rate.
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps else 0.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save at the source size and frame rate, independent of processing speed.
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Cannot create output video: {args.output}")

    person_model = YOLO(str(args.person_model))
    tag_model = YOLO(str(args.tag_model))
    tag_events = defaultdict(deque)                                     # Store positive (frame number, score) pairs per ID.
    active_staff_until = {}                                             # Map active staff IDs to their expiry frame.
    confirmed_staff_ids = set()                                         # Keep all confirmed IDs for the final summary only.
    tag_positive_events = 0
    frame_count = 0
    processing_started = time.perf_counter()
    previous_frame_time = processing_started
    realtime_fps = 0.0
    csv_file = None
    csv_rows = 0

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["staff ID", "frame", "x", "y", "time"])
        while True:
            success, frame = video.read()
            if not success:
                break

            # Preserve tracker state between frames; class 0 is the person class.
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
            annotated_frame = frame.copy()                              # Keep the original image clean for tag detection.
            person_count = len(result.boxes) if result.boxes is not None else 0

            # Draw all red boxes first so the existing green staff boxes cover them.
            if args.person_show and result.boxes is not None:
                person_boxes = result.boxes.xyxy.cpu().tolist()
                person_ids = (
                    result.boxes.id.int().cpu().tolist()
                    if result.boxes.id is not None else [None] * len(person_boxes)
                )
                for person_box, person_id in zip(person_boxes, person_ids):
                    draw_person_box(annotated_frame, person_box, person_id)

            tracked_people = []
            person_crops = []
            # Only people with track IDs can build temporal staff evidence.
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    crop = crop_person(frame, box, args.crop_padding)
                    if crop is not None and crop.size:
                        # Keep these lists in the same order to match crops to IDs.
                        tracked_people.append((box, track_id))
                        person_crops.append(crop)

            tag_scores = [0.0] * len(person_crops)                      # A crop with no accepted tag keeps a zero score.
            if person_crops:
                # Check all person crops together with the separate tag model.
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
                        # Use the strongest tag prediction from this person's crop.
                        tag_scores[index] = float(tag_result.boxes.conf.max().item())

            vote_start = frame_count - args.vote_window + 1
            # Retain enough history for voting and the displayed confidence.
            history_start = frame_count - max(args.vote_window, args.staff_hold) + 1
            # Expire staff status even when an ID is missing from this frame.
            active_staff_until = {
                track_id: active_until
                for track_id, active_until in active_staff_until.items()
                if active_until >= frame_count
            }
            staff_count = 0
            frame_rows = []
            frame_time = format_video_time(frame_count, fps)
            for (box, track_id), tag_score in zip(tracked_people, tag_scores):
                history = tag_events[track_id]
                # Discard evidence older than the retained history.
                while history and history[0][0] < history_start:
                    history.popleft()
                if tag_score >= args.tag_conf:
                    # Record one positive vote for this person in this frame.
                    history.append((frame_count, tag_score))
                    tag_positive_events += 1
                vote_count = sum(event_frame >= vote_start for event_frame, _ in history)
                if vote_count >= args.vote_min:
                    # Confirm or refresh while enough votes remain in the window.
                    active_staff_until[track_id] = frame_count + args.staff_hold
                    confirmed_staff_ids.add(track_id)
                elif tag_score >= args.tag_conf and track_id in active_staff_until:
                    # One fresh positive can extend an already-active staff ID.
                    active_staff_until[track_id] = frame_count + args.staff_hold
                if track_id in active_staff_until:
                    recent_scores = [score for _, score in history]
                    # Show the retained peak score, not necessarily this frame's score.
                    display_score = max(recent_scores) if recent_scores else tag_score
                    draw_staff_box(annotated_frame, box, track_id, display_score)
                    x, y = staff_coordinates(box)
                    draw_staff_coordinates(annotated_frame, box, x, y)
                    # Export only displayed staff; frame numbering starts at 1.
                    frame_rows.append([track_id, frame_count + 1, x, y, frame_time])
                    staff_count += 1

            current_frame_time = time.perf_counter()
            frame_elapsed = current_frame_time - previous_frame_time
            current_fps = 1.0 / frame_elapsed if frame_elapsed else 0.0
            # Smooth sudden speed changes to make the FPS display easier to read.
            realtime_fps = (
                current_fps
                if realtime_fps == 0.0
                else 0.9 * realtime_fps + 0.1 * current_fps
            )
            previous_frame_time = current_frame_time
            draw_fps(annotated_frame, realtime_fps)
            draw_counts(annotated_frame, person_count, staff_count)

            writer.write(annotated_frame)                               # Save the same annotations shown in the preview.
            csv_writer.writerows(frame_rows)
            csv_rows += len(frame_rows)
            cv2.imshow("Staff detection", annotated_frame)
            frame_count += 1
            print(f"\rprocessing {frame_count}/{total_frames} frames", end="", flush=True)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Close the CSV on normal completion, early quit, or a processing error.
        if csv_file is not None:
            csv_file.close()
        # Release the video handles and window on completion, quit, or loop error.
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_elapsed = time.perf_counter() - processing_started       # Measure the processing stage after model loading.
    total_elapsed = time.perf_counter() - script_started                # Include setup and model loading in the total time.
    speed = frame_count / processing_elapsed if processing_elapsed else 0.0
    print("\r" + " " * 80, end="\r")
    print(f"processed {frame_count}/{total_frames} (Completed successfully)")
    print(f"Saved annotated video to {args.output}")
    print(f"Saved {csv_rows} staff occurrences to {csv_path}")
    print(f"Finished {frame_count} frames on {device} at {speed:.1f} FPS")
    print(f"Person model: {args.person_model}")
    print(f"Tag model: {args.tag_model}")
    print(f"Tracker: {args.tracker}")
    print(f"Person confidence floor: {args.conf}; image size: {args.imgsz}")
    print(f"Tag confidence: {args.tag_conf}; tag image size: {args.tag_imgsz}")
    print(f"Temporal vote: {args.vote_min} positives within {args.vote_window} frames")
    print(f"Staff status hold: {args.staff_hold} frames after recent tag evidence")
    print(f"Tag-positive events: {tag_positive_events}")
    print(f"Confirmed staff track IDs: {sorted(confirmed_staff_ids)}")
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Total processing time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
