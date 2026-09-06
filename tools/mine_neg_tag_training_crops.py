"""Mine staff-tag hard-negative candidates, then export only reviewed negatives.

Run from the project root:
    python tools/mine_neg_tag_training_crops.py
    python tools/mine_neg_tag_training_crops.py --export-reviewed

The first command creates clean crops, tag-box previews and review.csv.
Set decision=negative only when NO real staff tag is visible anywhere in a crop.
The second command copies those reviewed crops and creates empty YOLO tag labels.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import deque
from pathlib import Path


REVIEW_GUIDE = """These images are candidates for STAFF-TAG training, not confirmed negatives.

1. Inspect previews/ to see the model's tag boxes and confidence scores.
2. Inspect the corresponding clean images/ crop before deciding.
3. Edit only the decision column in review.csv:
   negative = No real visible staff tag anywhere in this crop.
   positive = A real tag is visible; keep it for separate tag annotation.
   skip     = Uncertain, too unclear, or an unwanted duplicate.
   blank    = Not reviewed yet.
4. Save review.csv and run the script again with --export-reviewed.

Only negative rows are exported to images/train and labels/train, with empty
YOLO label files for class 0: staff_tag. A hidden tag can be a negative example
for visible-tag detection, even if the person is staff. A neighbour's real tag
inside the crop makes it positive, not negative. Never train on the previews.

The exported folder contains only additional negatives. Merge them with your
existing tag TRAINING images/labels; do not train on this folder by itself.
Keep all training positives and avoid overwhelming them with extra negatives.
Validation and test images must not be added to training.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/sample.mp4"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("data/person_dataset_seed123/split_manifest.json"))
    parser.add_argument("--person-model", type=Path, default=Path("yoloModel/best_v1.2.pt"))
    parser.add_argument("--tag-model", type=Path, default=Path("yoloModel/best_tag_v1.1.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/tag_hard_negative_candidates"))
    parser.add_argument("--device", choices=("cpu", "auto", "0"), default="auto")
    parser.add_argument("--conf", type=float, default=0.10, help="Person detection floor")
    parser.add_argument("--tag-conf", type=float, default=0.30,
                        help="Mining floor; intentionally includes scores below deployment's 0.70")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tag-imgsz", type=int, default=640)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--min-gap", type=float, default=1.0,
                        help="Seconds between saves of strongly overlapping crops; 0 disables")
    parser.add_argument("--max-crops", type=int, default=500,
                        help="Stop after this many saved candidates")
    parser.add_argument("--export-reviewed", action="store_true",
                        help="Export rows marked negative in OUTPUT/review.csv; no inference")
    parser.add_argument("--reviewed-output", type=Path,
                        default=Path("data/tag_hard_negatives_reviewed"))
    return parser.parse_args()


def require_empty(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output is not empty: {path}. Choose a new output folder.")


def load_plan(path):
    """Use the frozen block assignments and retain the original boundary margins."""
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if plan.get("method") != "randomized_temporal_blocks":
        raise ValueError("Expected the temporal-block split_manifest.json from prepare_dataset.py.")
    blocks = {block["block_id"]: block for block in plan["blocks"]}
    if len(blocks) != len(plan["blocks"]):
        raise ValueError("Duplicate block IDs in split manifest.")
    assigned = set()
    for split in ("train", "val", "test"):
        ids = plan["splits"][split]["block_ids"]
        if len(set(ids)) != len(ids) or assigned.intersection(ids) or not set(ids) <= blocks.keys():
            raise ValueError("Split block IDs overlap, repeat, or reference missing blocks.")
        assigned.update(ids)
    if assigned != blocks.keys():
        raise ValueError("Some video blocks have no split assignment.")
    total = int(plan["video_info"]["decoded_frame_count"])
    cursor = 0
    for block in sorted(blocks.values(), key=lambda item: item["start_frame"]):
        if block["start_frame"] != cursor or block["end_frame"] < cursor:
            raise ValueError("Video blocks overlap or leave gaps in the manifest.")
        cursor = block["end_frame"] + 1
    if cursor != total:
        raise ValueError("Manifest blocks do not match its video frame count.")
    margin = plan["boundary_margin_frames"]
    if not isinstance(margin, int) or margin < 0:
        raise ValueError("Invalid boundary margin in split manifest.")
    training_frames = {}
    for block_id in plan["splits"]["train"]["block_ids"]:
        block = blocks[block_id]
        start, stop = block["start_frame"] + margin, block["end_frame"] + 1 - margin
        if start >= stop:
            raise ValueError("A training block is too short for its boundary margin.")
        training_frames.update((frame, block_id) for frame in range(start, stop))
    # Existing exported frames already have annotations; mine additional frames only.
    for split in ("train", "val", "test"):
        for frame in plan["splits"][split]["frame_indices"]:
            training_frames.pop(frame, None)
    if not training_frames:
        raise ValueError("No additional training frames are available.")
    return plan, training_frames


def padded_box(box, width, height, padding):
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    # Match detect_staff.py's crop geometry and preserve original-resolution pixels.
    return (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)))


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(area_a + area_b - overlap, 1)


def export_reviewed(args):
    review_path = args.output / "review.csv"
    with review_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        if not fields or not {"decision", "image"} <= set(fields):
            raise ValueError("review.csv must contain decision and image columns.")
        rows = list(reader)
    selected, seen = [], set()
    for row in rows:
        decision = (row.get("decision") or "").strip().lower()
        if decision not in ("", "negative", "positive", "skip"):
            raise ValueError(f"Unknown decision {decision!r}; use negative, positive, skip or blank.")
        if decision != "negative":
            continue
        source = (args.output / row["image"]).resolve()
        if source.parent != (args.output / "images").resolve() or not source.is_file():
            raise ValueError(f"Missing or invalid clean crop: {row['image']}")
        if source.name.casefold() in seen:
            raise ValueError(f"Duplicate negative row: {source.name}")
        seen.add(source.name.casefold())
        selected.append((row, source))
    if not selected:
        print("No rows marked negative. Review the candidates and edit the decision column first.")
        return
    require_empty(args.reviewed_output)
    images, labels = args.reviewed_output / "images/train", args.reviewed_output / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for row, source in selected:
        shutil.copy2(source, images / source.name)
        (labels / f"{source.stem}.txt").write_text("", encoding="utf-8")
    (args.reviewed_output / "classes.txt").write_text("staff_tag\n", encoding="utf-8")
    with (args.reviewed_output / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row, _ in selected)
    print(f"Exported {len(selected)} reviewed staff-tag negatives to {args.reviewed_output}")
    print("Merge these image/label pairs with tag TRAINING data. This is not a complete training dataset.")


def mine(args):
    for path in (args.input, args.person_model, args.tag_model, args.split_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    for value in (args.conf, args.tag_conf):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Confidence thresholds must be finite numbers in (0, 1].")
    if (not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or
            not math.isfinite(args.min_gap) or args.min_gap < 0 or
            not math.isfinite(args.crop_padding) or args.crop_padding < 0 or
            min(args.imgsz, args.tag_imgsz, args.max_crops) <= 0):
        raise ValueError("Invalid sampling, padding, image size or crop limit.")
    require_empty(args.output)
    plan, training_frames = load_plan(args.split_manifest)
    expected_name = Path(plan["video"].replace("\\", "/")).name
    if args.input.name != expected_name:
        raise ValueError(f"Use the original {expected_name} video matching the manifest, not an annotated output.")

    import cv2
    import torch
    from ultralytics import YOLO

    if args.device == "0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 requested, but CUDA is unavailable.")
    device = 0 if args.device == "0" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    video = cv2.VideoCapture(str(args.input))
    sampled = saved = 0
    completed = False
    config = None
    try:
        if not video.isOpened():
            raise RuntimeError(f"Cannot open video: {args.input}")
        fps = video.get(cv2.CAP_PROP_FPS)
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info = plan["video_info"]
        if (not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, info["fps"], rel_tol=1e-5) or
                total != info["decoded_frame_count"] or (width, height) != (info["width"], info["height"])):
            raise ValueError("Video FPS, dimensions or frame count differ from the split manifest.")
        if args.sample_fps > fps:
            raise ValueError("--sample-fps cannot exceed the source video's FPS.")
        stride = max(1, round(fps / args.sample_fps))
        if not any(frame % stride == 0 for frame in training_frames):
            raise ValueError("No new training frames fall on this sample grid; increase --sample-fps.")
        person_model, tag_model = YOLO(str(args.person_model)), YOLO(str(args.tag_model))
        if person_model.task != "detect" or tag_model.task != "detect":
            raise ValueError("Use ordinary YOLO detection weights for both models, not OBB/classification weights.")
        images, previews = args.output / "images", args.output / "previews"
        images.mkdir(parents=True)
        previews.mkdir(parents=True)
        (args.output / "REVIEW.txt").write_text(REVIEW_GUIDE, encoding="utf-8")
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
                      source_fps=fps, effective_sample_fps=fps / stride,
                      training_block_ids=plan["splits"]["train"]["block_ids"])
        fields = ["decision", "image", "preview", "frame", "time_seconds", "block_id", "person_index",
                  "person_confidence", "tag_confidence", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
                  "tag_boxes_xyxy_conf"]
        recent = deque()
        print(f"Mining staff-tag candidates from training blocks {config['training_block_ids']}")
        print(f"Sampling about {fps / stride:.2f} FPS; tag floor {args.tag_conf:.2f}; maximum {args.max_crops} crops")
        with (args.output / "review.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for frame_index in range(total):
                success, frame = video.read()
                if not success:
                    raise RuntimeError(f"Video decoding stopped early at frame {frame_index}/{total}.")
                if frame_index not in training_frames or frame_index % stride:
                    continue
                seconds = frame_index / fps
                sampled += 1
                result = person_model.predict(frame, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                              device=device, verbose=False)[0]
                people, crops = [], []
                if result.boxes is not None:
                    for index, (box, score) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),
                                                            result.boxes.conf.cpu().tolist())):
                        bounds = padded_box(box, width, height, args.crop_padding)
                        x1, y1, x2, y2 = bounds
                        if x2 <= x1 or y2 <= y1:
                            continue
                        people.append((index, score, bounds))
                        crops.append(frame[y1:y2, x1:x2])
                tag_results = tag_model.predict(crops, classes=[0], conf=args.tag_conf, imgsz=args.tag_imgsz,
                                                device=device, verbose=False) if crops else []
                if len(tag_results) != len(crops):
                    raise RuntimeError("Tag results do not match the person crop batch.")
                while recent and seconds - recent[0][0] >= args.min_gap:
                    recent.popleft()
                for (index, person_score, bounds), crop, tags in zip(people, crops, tag_results):
                    if tags.boxes is None or not len(tags.boxes):
                        continue
                    tag_boxes = [list(box) + [score] for box, score in zip(tags.boxes.xyxy.cpu().tolist(),
                                                                         tags.boxes.conf.cpu().tolist())
                                 if score >= args.tag_conf]
                    if not tag_boxes:
                        continue
                    # Suppress near repeats by location and time; no identity assumptions are made.
                    if args.min_gap and any(box_iou(bounds, previous) >= 0.70 for _, previous in recent):
                        continue
                    name = f"hardneg_f{frame_index:07d}_p{index:02d}.jpg"
                    preview = crop.copy()
                    for x1, y1, x2, y2, score in tag_boxes:
                        cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                        cv2.putText(preview, f"tag {score:.2f}", (max(0, int(x1)), max(12, int(y1) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                    for path, pixels in ((images / name, crop), (previews / name, preview)):
                        if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise RuntimeError(f"Could not save {path}")
                    writer.writerow(dict(decision="", image=f"images/{name}", preview=f"previews/{name}",
                                         frame=frame_index, time_seconds=f"{seconds:.3f}",
                                         block_id=training_frames[frame_index], person_index=index,
                                         person_confidence=f"{person_score:.4f}",
                                         tag_confidence=f"{max(box[4] for box in tag_boxes):.4f}",
                                         crop_x1=bounds[0], crop_y1=bounds[1], crop_x2=bounds[2], crop_y2=bounds[3],
                                         tag_boxes_xyxy_conf=json.dumps(tag_boxes)))
                    recent.append((seconds, bounds))
                    saved += 1
                    if saved >= args.max_crops:
                        break
                file.flush()
                print(f"\rFrame {frame_index + 1}/{total}; sampled {sampled}; candidates {saved}", end="", flush=True)
                if saved >= args.max_crops:
                    break
            else:
                completed = True
        print(f"\nSaved {saved} candidate crops to {args.output}")
        if not completed:
            print("Stopped at --max-crops; raise the limit and use a new output folder to scan further.")
        print("Review review.csv first, then use --export-reviewed. No negative labels have been assumed.")
    finally:
        video.release()
        if config is not None:
            config.update(sampled_frames=sampled, saved_candidates=saved, full_scan_completed=completed)
            (args.output / "mining_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if args.export_reviewed:
        export_reviewed(args)
    else:
        mine(args)


if __name__ == "__main__":
    main()
