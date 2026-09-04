"""Mine additional person crops from training-only video ranges for tag annotation."""

import argparse
import csv
import math
from pathlib import Path

def parse_range(value):
    try:
        start_text, end_text = value.split("-", 1)
        start, end = float(start_text), float(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use START-END seconds, for example 0-120") from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("A range must satisfy 0 <= START < END")
    return start, end


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample.mp4"))
    parser.add_argument("--model", type=Path, default=Path("yoloModel/best_v1.1.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/tag_expansion_candidates"))
    parser.add_argument("--range", dest="ranges", type=parse_range, action="append", required=True,
                        help="Training-only START-END seconds; repeat for multiple blocks")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested):
    import torch

    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return int(requested) if requested.isdigit() else requested


def inside_ranges(seconds, ranges):
    return any(start <= seconds < end for start, end in ranges)


def padded_box(box, frame_width, frame_height, padding):
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y)),
        min(frame_width, math.ceil(x2 + pad_x)), min(frame_height, math.ceil(y2 + pad_y)),
    )


def main():
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    if args.sample_fps <= 0 or args.padding < 0:
        raise ValueError("--sample-fps must be positive and --padding cannot be negative")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(f"Output folder is not empty: {args.output}")

    video = cv2.VideoCapture(str(args.input))
    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")
    source_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        raise RuntimeError("Video FPS is unavailable")
    duration = total_frames / source_fps
    for start, end in args.ranges:
        if end > duration + 1 / source_fps:
            raise ValueError(f"Range {start}-{end} exceeds video duration {duration:.2f}s")

    image_dir = args.output / "images" / "train"
    label_dir = args.output / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    device = resolve_device(args.device)
    frame_step = max(1, round(source_fps / args.sample_fps))
    manifest = []
    sampled_frames = 0

    try:
        for frame_number in range(total_frames):
            success, frame = video.read()
            if not success:
                break
            seconds = frame_number / source_fps
            if frame_number % frame_step or not inside_ranges(seconds, args.ranges):
                continue
            result = model.predict(frame, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                   device=device, verbose=False)[0]
            sampled_frames += 1
            height, width = frame.shape[:2]
            boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
            for person_index, (box, confidence) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = padded_box(box, width, height, args.padding)
                if x2 <= x1 or y2 <= y1:
                    continue
                stem = f"video_f{frame_number:07d}_t{seconds:010.3f}_person_{person_index:02d}"
                image_path = image_dir / f"{stem}.jpg"
                cv2.imwrite(str(image_path), frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (label_dir / f"{stem}.txt").write_text("", encoding="utf-8")
                manifest.append({
                    "crop_image": image_path.relative_to(args.output).as_posix(),
                    "source_video": args.input.as_posix(), "frame": frame_number,
                    "time_seconds": f"{seconds:.3f}", "person_index": person_index,
                    "person_confidence": f"{confidence:.4f}",
                    "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                })
            print(f"\rScanned frame {frame_number + 1}/{total_frames}; crops: {len(manifest)}", end="", flush=True)
    finally:
        video.release()

    with (args.output / "crop_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["crop_image", "source_video", "frame", "time_seconds", "person_index",
                  "person_confidence", "crop_x1", "crop_y1", "crop_x2", "crop_y2"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (args.output / "classes.txt").write_text("staff_tag\n", encoding="utf-8")
    ranges_text = ", ".join(f"{start:g}-{end:g}s" for start, end in args.ranges)
    print(f"\nSampled {sampled_frames} training-only frames and created {len(manifest)} crops")
    print(f"Ranges: {ranges_text}; effective sampling: about {source_fps / frame_step:.2f} FPS")
    print(f"Output: {args.output}")
    print("Review all crops and annotate only visible staff_tag boxes. Do not treat unreviewed empty labels as negatives.")


if __name__ == "__main__":
    main()
