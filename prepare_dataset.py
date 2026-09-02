"""Extract a reproducible 60/20/20 temporal-block dataset from a video."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

# Set the dataset split sizes.
RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}

# Read the command-line options provided by the user.
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/sample.mp4"))
    parser.add_argument("--output", type=Path, default=Path("data/person_dataset"))
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--frames-per-block", type=int, default=8)
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


# Read video properties and count frames that actually decode.
def inspect_video(path: Path) -> dict:

    # Open the video.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    # Store the video properties.
    info = {
        "reported_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    decoded = 0
    # Count the frames that can read successfully.
    while cap.read()[0]:
        decoded += 1
    cap.release()

    if decoded == 0:
        raise RuntimeError("The video contains no decodable frames.")
    info["decoded_frame_count"] = decoded
    info["duration_seconds"] = decoded / info["fps"] if info["fps"] > 0 else None
    return info


# Create blocks, assign whole blocks, and sample frames within each block.
def make_plan(
    frame_count: int, block_count: int, samples: int, margin: int, seed: int
) -> tuple[list[tuple[int, int]], dict[str, list[int]], dict[str, list[int]]]:
    
    if block_count <= 0 or block_count > frame_count or block_count % 5:
        raise ValueError("--blocks must be divisible by 5 and no larger than the video.")
    if samples <= 0 or margin < 0:
        raise ValueError("--frames-per-block must be positive and --margin non-negative.")
    
    # Divide the video into blocks with nearly equal sizes.
    base, extra = divmod(frame_count, block_count)
    blocks, start = [], 0
    for block_id in range(block_count):
        # Give earlier blocks one extra frame when needed.
        end = start + base + (block_id < extra)
        blocks.append((start, end))  # End is exclusive.
        start = end

    # Use the seed to make the random split repeatable.
    rng = random.Random(seed)
    shuffled = list(range(block_count))
    rng.shuffle(shuffled)
    # Divide the shuffled block list into a 60/20/20 split.
    train_end = block_count * 3 // 5
    val_end = train_end + block_count // 5
    split_blocks = {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }

    # Store the sampled frame numbers for each dataset split.
    split_frames = {name: [] for name in RATIOS}
    for split, block_ids in split_blocks.items():
        for block_id in block_ids:
            start, end = blocks[block_id]
            usable_start, usable_end = start + margin, end - margin
            usable = usable_end - usable_start
            if usable < samples:
                raise ValueError("A block is too short for the samples and margin.")

            # One random frame per section prevents samples clustering together.
            for section in range(samples):
                low = usable_start + usable * section // samples
                high = usable_start + usable * (section + 1) // samples
                split_frames[split].append(rng.randrange(low, high))

    return blocks, split_blocks, split_frames


# Create the dataset folders, images, and configuration files.
def main() -> None:
    args = get_args()
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100.")
    # Stop before overwriting an existing dataset or annotations.
    if args.output.is_file() or (args.output.exists() and any(args.output.iterdir())):
        raise FileExistsError(
            f"Output is not empty: {args.output}\n"
            "Choose a new --output folder to protect existing annotations."
        )

    video_info = inspect_video(args.video)
    frame_count = video_info["decoded_frame_count"]
    blocks, split_blocks, split_frames = make_plan(
        frame_count, args.blocks, args.frames_per_block, args.margin, args.seed
    )

    # Create matching image and label folders for every split.
    for split in RATIOS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Map each selected frame number to its dataset split.
    frame_lookup = {
        frame: split for split, frames in split_frames.items() for frame in frames
    }
    cap = cv2.VideoCapture(str(args.video))
    saved_count = 0
    # Always release the video, even if saving an image fails.
    try:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            # Save the frame only when it was selected in the plan.
            split = frame_lookup.get(frame_index)
            if split:
                path = args.output / "images" / split / f"frame_{frame_index:06d}.jpg"
                saved = cv2.imwrite(
                    str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                )
                if not saved:
                    raise RuntimeError(f"Failed to save: {path}")
                saved_count += 1
            frame_index += 1
    finally:
        cap.release()

    if saved_count != len(frame_lookup):
        raise RuntimeError(f"Expected {len(frame_lookup)} images; saved {saved_count}.")

    # Record the split details so the dataset can be reproduced.
    manifest = {
        "video": args.video.as_posix(),
        "video_info": video_info,
        "method": "randomized_temporal_blocks",
        "ratios": RATIOS,
        "seed": args.seed,
        "block_count": args.blocks,
        "frames_per_block": args.frames_per_block,
        "boundary_margin_frames": args.margin,
        "blocks": [
            {"block_id": i, "start_frame": start, "end_frame": end - 1}
            for i, (start, end) in enumerate(blocks)
        ],
        "splits": {
            split: {
                "block_ids": split_blocks[split],
                "frame_indices": sorted(split_frames[split]),
            }
            for split in RATIOS
        },
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Quote the absolute dataset path safely for the YAML file.
    dataset_path = json.dumps(args.output.resolve().as_posix())
    (args.output / "data.yaml").write_text(
        f"path: {dataset_path}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )

    print(f"Video frames: {frame_count}")
    for split in RATIOS:
        print(
            f"{split:>5}: {len(split_blocks[split])} blocks, "
            f"{len(split_frames[split])} images"
        )
    print(f"Dataset created at: {args.output}")
    print("Review the images and annotate every visible person before training.")


if __name__ == "__main__":
    main()
