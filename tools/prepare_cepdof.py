"""
Prepare selected CEPDOF sequences for YOLO detection and/or OBB training.
Place this file in FCC-Staff-Detection/tools and run it from the project root.
Images stay in CEPDOF; hard links are read-only aliases, not independent copies.
Only Python's standard library is needed unless --preview is enabled (OpenCV).
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil


# Edit these totals before preparing a NEW dataset. Zero skips the sequence.
IMAGE_COUNTS = {
    "Lunch1": 1000,
    "Lunch2": 0,
    "Lunch3": 0,
    "IRill": 0,
    "IRfilter": 0,
    "High_activity": 2000,
    "Edge_cases": 2000,
    "All_off": 0,
}

SPLITS = ("train", "val", "test")
RATIOS = (0.6, 0.2, 0.2)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def allocate(total):
    """Round split sizes while keeping their sum equal to the requested total."""
    raw = [total * ratio for ratio in RATIOS]
    sizes = [math.floor(value) for value in raw]
    order = sorted(range(3), key=lambda i: (-(raw[i] - sizes[i]), i))
    for i in order[:total - sum(sizes)]:
        sizes[i] += 1
    return dict(zip(SPLITS, sizes))


def natural_key(record):
    # Sort frame 2 before frame 10, even if filenames are not zero-padded.
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", record["file_name"])]


def corners(bbox):
    """Convert CEPDOF clockwise degrees into four ordered pixel corners."""
    if len(bbox) != 5 or not all(math.isfinite(float(v)) for v in bbox):
        raise ValueError(f"Invalid five-value CEPDOF box: {bbox}")
    cx, cy, width, height, degrees = map(float, bbox)
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive box size: {bbox}")
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + dx*c - dy*s, cy + dx*s + dy*c)
            for dx, dy in [(-width/2, -height/2), (width/2, -height/2),
                           (width/2, height/2), (-width/2, height/2)]]


def label_lines(boxes, width, height, formats):
    """Build normalized labels; fail on edge cases instead of dropping people."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    rows = {fmt: [] for fmt in formats}
    for box in boxes:
        pts = corners(box)
        if any(x < -1e-5 or y < -1e-5 or x > width+1e-5 or y > height+1e-5
               for x, y in pts):
            raise ValueError(f"Box extends outside image; review annotation: {box}")
        # Clamp only negligible floating-point roundoff, not actual geometry.
        pts = [(min(width, max(0, x)), min(height, max(0, y))) for x, y in pts]
        if "obb" in rows:
            values = [v for x, y in pts for v in (x/width, y/height)]
            rows["obb"].append("0 " + " ".join(f"{v:.8f}" for v in values))
        if "detect" in rows:
            xs, ys = zip(*pts)
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            values = ((x1+x2)/(2*width), (y1+y2)/(2*height),
                      (x2-x1)/width, (y2-y1)/height)
            rows["detect"].append("0 " + " ".join(f"{v:.8f}" for v in values))
    return {fmt: "\n".join(lines) + ("\n" if lines else "")
            for fmt, lines in rows.items()}


def split_images(images, count, seed, sequence, gap):
    """Sample within three disjoint chronological sections, separated by gaps."""
    images = sorted(images, key=natural_key)
    usable = len(images) - 2*gap
    if usable < count:
        raise ValueError(f"{sequence}: requested {count}, but only {max(0, usable)} "
                         f"remain after two gaps of {gap}. Lower count or --gap.")
    targets = allocate(count)
    if min(targets.values()) == 0:
        raise ValueError(f"{sequence}: select at least 5 images for three splits")
    # Separate per-sequence randomness keeps other sequences stable when disabled.
    rng = random.Random(f"{seed}:{sequence}:sections")
    order = list(SPLITS)
    rng.shuffle(order)
    spare = allocate(usable-count)
    position, selected, sections = 0, [], []
    for index, split in enumerate(order):
        size = targets[split] + spare[split]
        pool = images[position:position+size]
        picks = random.Random(f"{seed}:{sequence}:{split}").sample(pool, targets[split])
        selected.extend((split, item) for item in sorted(picks, key=natural_key))
        sections.append({"split": split, "first": pool[0]["file_name"],
                         "last": pool[-1]["file_name"], "available": size,
                         "selected": targets[split]})
        position += size + (gap if index < 2 else 0)
    return selected, sections


def find_image(root, sequence, filename):
    """Support images directly in the sequence or in an images subfolder."""
    relative = Path(filename.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image filename in JSON: {filename}")
    candidates = [root/sequence/relative, root/relative,
                  root/"images"/sequence/relative, root/sequence/"images"/relative]
    matches = {p.resolve() for p in candidates if p.is_file()}
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {sequence}/{filename}; found "
                         f"{len(matches)}. Checked: " + ", ".join(map(str, candidates)))
    return matches.pop()


def prepare_plan(args, counts, formats):
    records, sources, sections = [], {}, {}
    for sequence, count in counts.items():
        if count == 0:
            continue
        annotation_path = args.source/"annotations"/f"{sequence}.json"
        raw = annotation_path.read_bytes()
        data = json.loads(raw)
        images = data["images"]
        ids = [im["id"] for im in images]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate image IDs in {annotation_path}")
        names = [im["file_name"].casefold() for im in images]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate image filenames in {annotation_path}")
        people = {c["id"] for c in data["categories"] if c["name"].lower() == "person"}
        if not people:
            raise ValueError(f"No person category in {annotation_path}")
        by_image = defaultdict(list)
        known = set(ids)
        for ann in data["annotations"]:
            if ann["image_id"] not in known:
                raise ValueError(f"Annotation refers to unknown image: {ann['image_id']}")
            if ann["category_id"] not in people or ann.get("iscrowd", 0):
                raise ValueError(f"Unsupported category/crowd annotation: {ann}")
            by_image[ann["image_id"]].append(ann["bbox"])
        selected, sections[sequence] = split_images(images, count, args.seed, sequence, args.gap)
        sources[sequence] = {"json": str(annotation_path),
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "available": len(images), "selected": count}
        for split, im in selected:
            path = find_image(args.source, sequence, im["file_name"])
            boxes = by_image[im["id"]]
            try:
                labels = label_lines(boxes, im["width"], im["height"], formats)
            except ValueError as exc:
                raise ValueError(f"{sequence}/{im['file_name']}: {exc}") from exc
            # Sequence and ID prevent filename collisions between source folders.
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(im["id"]))
            name = f"{sequence}__{safe_id}{path.suffix.lower()}"
            records.append({"sequence": sequence, "image_id": im["id"],
                            "source": str(path), "split": split, "name": name,
                            "width": im["width"], "height": im["height"],
                            "boxes": boxes, "labels": labels})
        print(f"{sequence}: {len(images)} available, {count} selected {allocate(count)}")
    names = [Path(r["name"]).stem.casefold() for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Output filename collision after sanitizing image IDs")
    return records, sources, sections


def save_preview(record, destination):
    import cv2
    import numpy as np
    image = cv2.imread(record["source"])
    if image is None:
        raise ValueError(f"Cannot decode {record['source']}")
    if image.shape[:2] != (record["height"], record["width"]):
        raise ValueError(f"Image dimensions differ from JSON: {record['source']}")
    for box in record["boxes"]:
        pts = np.rint(corners(box)).astype(np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        cv2.rectangle(image, tuple(lo), tuple(hi), (255, 160, 0), 1)
    cv2.putText(image, "Green: OBB | Blue: ordinary box", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write preview: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT.parent/"CEPDOF")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"data"/"cepdof_yolo")
    parser.add_argument("--format", choices=("both", "detect", "obb"), default="both")
    parser.add_argument("--count", action="append", default=[], metavar="SEQUENCE=N",
                        help="Override a count; repeat for multiple sequences")
    parser.add_argument("--only", nargs="+", choices=list(IMAGE_COUNTS),
                        help="Disable all sequences except these")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gap", type=int, default=10,
                        help="Unselected frames between chronological sections (default: 10)")
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="Save up to N random previews per sequence per split")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without writing")
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    counts = IMAGE_COUNTS.copy()
    for override in args.count:
        name, separator, value = override.partition("=")
        if not separator or name not in counts:
            parser.error(f"Use --count SEQUENCE=N; unknown override {override!r}")
        try:
            counts[name] = int(value)
        except ValueError:
            parser.error(f"Count must be an integer: {override}")
    if args.only:
        counts = {name: n if name in args.only else 0 for name, n in counts.items()}
    if any(type(n) is not int or n < 0 for n in counts.values()):
        parser.error("IMAGE_COUNTS values must be non-negative integers")
    if not any(counts.values()) or args.gap < 0 or args.preview < 0:
        parser.error("Select at least one sequence and use non-negative gap/preview values")
    if args.output == args.source or args.source in args.output.parents:
        parser.error("Output must be outside the original CEPDOF folder")
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output to preserve its split.")
    formats = ("detect", "obb") if args.format == "both" else (args.format,)
    records, sources, sections = prepare_plan(args, counts, formats)
    totals = dict(Counter(r["split"] for r in records))
    print(f"Total: {len(records)} images; {totals}; formats: {formats}")
    if args.dry_run:
        print("Dry run passed. Image existence and selected annotations checked; no files written.")
        return
    if args.preview:
        import cv2  # Check the optional dependency before creating output.
    args.output.mkdir(parents=True)
    # This marker distinguishes interrupted preparation from a complete dataset.
    marker = args.output/"PREPARATION_INCOMPLETE.txt"
    marker.write_text("Preparation is incomplete. Do not train from this folder.\n")
    for fmt in formats:
        for split in SPLITS:
            for kind in ("images", "labels"):
                (args.output/fmt/kind/split).mkdir(parents=True)
    for index, record in enumerate(records, 1):
        source = Path(record["source"])
        for fmt in formats:
            destination = args.output/fmt/"images"/record["split"]/record["name"]
            if args.image_mode == "hardlink":
                try:
                    os.link(source, destination)
                except OSError as exc:
                    raise OSError("Hard link failed. Use a NEW output with --image-mode copy "
                                  "if the source and output are on different volumes.") from exc
            else:
                shutil.copy2(source, destination)
            label = args.output/fmt/"labels"/record["split"]/(destination.stem+".txt")
            label.write_text(record["labels"][fmt], encoding="utf-8")
        if index % 100 == 0 or index == len(records):
            print(f"Preparing {index}/{len(records)}")
    if args.preview:
        groups = defaultdict(list)
        for record in records:
            groups[(record["sequence"], record["split"])].append(record)
        for (sequence, split), group in groups.items():
            picks = random.Random(f"{args.seed}:{sequence}:{split}:preview").sample(
                group, min(args.preview, len(group)))
            for record in picks:
                save_preview(record, args.output/"previews"/sequence/split/record["name"])
    for fmt in formats:
        root = args.output/fmt
        yaml = (f"path: {json.dumps(root.as_posix())}\ntrain: images/train\n"
                "val: images/val\ntest: images/test\nnames:\n  0: person\n")
        (root/"data.yaml").write_text(yaml, encoding="utf-8")
    manifest = {"version": 1, "seed": args.seed, "image_counts": counts,
                "split_ratios": dict(zip(SPLITS, RATIOS)), "gap_frames": args.gap,
                "method": "Three chronological sections; seeded order; random sampling within each",
                "image_mode": args.image_mode, "formats": formats, "sources": sources,
                "sections": sections, "totals": totals,
                "images": [{k: v for k, v in r.items() if k != "labels"} for r in records]}
    (args.output/"split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    marker.unlink()
    print(f"Done: {args.output}")
    print("Keep linked images read-only. Freeze the manifest before training.")
    for fmt in formats:
        print(f"{fmt} YAML: {args.output/fmt/'data.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        raise SystemExit(f"ERROR: {error}") from error
