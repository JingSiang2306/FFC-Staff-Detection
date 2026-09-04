"""Create a reproducible tag dataset with a balanced training split."""

import argparse
import csv
import random
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/tag_dataset_v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/tag_dataset_balanced_v2")
    )
    parser.add_argument("--data-yaml", type=Path, default=Path("data/tag_dataset_balanced_v2/tag_data.yaml"))
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def validate_label(label_path):
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 values at {label_path}:{line_number}")

        class_id = int(float(parts[0]))
        coordinates = list(map(float, parts[1:]))
        if class_id != 0:
            raise ValueError(f"Expected class 0 at {label_path}:{line_number}")
        if not all(0 <= value <= 1 for value in coordinates[:2]):
            raise ValueError(f"Invalid box centre at {label_path}:{line_number}")
        if not all(0 < value <= 1 for value in coordinates[2:]):
            raise ValueError(f"Invalid box size at {label_path}:{line_number}")
    return bool(lines)


def load_split(source_root, split):
    image_dir = source_root / "images" / split
    label_dir = source_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing images or labels for split: {split}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len({path.stem for path in image_paths}) != len(image_paths):
        raise ValueError(f"Duplicate image stems found in split: {split}")

    items = []
    image_stems = {path.stem for path in image_paths}
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")
        items.append((image_path, label_path, validate_label(label_path)))

    extra_labels = sorted(
        path for path in label_dir.glob("*.txt") if path.stem not in image_stems
    )
    if extra_labels:
        raise ValueError(f"Label has no matching image: {extra_labels[0]}")
    return items


def check_destinations(output_root, data_yaml):
    if output_root.exists() and any(output_root.rglob("*")):
        raise FileExistsError(f"Output folder is not empty: {output_root}")
    if data_yaml.exists():
        raise FileExistsError(f"Data YAML already exists: {data_yaml}")


def copy_items(items, output_root, split, selection_name, manifest_rows):
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image_path, label_path, is_positive in sorted(items):
        shutil.copy2(image_path, output_images / image_path.name)
        shutil.copy2(label_path, output_labels / label_path.name)
        manifest_rows.append(
            {
                "split": split,
                "image": image_path.name,
                "label": "positive" if is_positive else "negative",
                "selection": selection_name(is_positive),
            }
        )


def main():
    args = parse_args()
    if args.negative_ratio < 0:
        raise ValueError("--negative-ratio must be zero or greater.")

    check_destinations(args.output, args.data_yaml)
    rng = random.Random(args.seed)
    splits = {split: load_split(args.source, split) for split in SPLITS}

    train_positives = [item for item in splits["train"] if item[2]]
    train_negatives = [item for item in splits["train"] if not item[2]]
    if not train_positives:
        raise ValueError("The training split contains no positive tag labels.")

    negative_count = min(
        len(train_negatives), len(train_positives) * args.negative_ratio
    )
    sampled_negatives = rng.sample(train_negatives, negative_count)
    selected_train = train_positives + sampled_negatives
    manifest_rows = []

    copy_items(
        selected_train,
        args.output,
        "train",
        lambda positive: "all_positive" if positive else "sampled_negative",
        manifest_rows,
    )
    for split in ("val", "test"):
        copy_items(
            splits[split],
            args.output,
            split,
            lambda positive: "evaluation_positive" if positive else "evaluation_negative",
            manifest_rows,
        )

    manifest_path = args.output / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("split", "image", "label", "selection")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (args.output / "classes.txt").write_text("staff_tag\n", encoding="utf-8")
    args.data_yaml.write_text(
        f"path: {args.output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: staff_tag\n",
        encoding="utf-8",
    )

    print(
        f"train: {len(train_positives)} positive + "
        f"{negative_count} negative = {len(selected_train)}"
    )
    for split in ("val", "test"):
        positives = sum(item[2] for item in splits[split])
        print(
            f"{split}: {positives} positive + "
            f"{len(splits[split]) - positives} negative = {len(splits[split])}"
        )
    print(f"Created balanced dataset in {args.output}")
    print(f"Created training configuration: {args.data_yaml}")


if __name__ == "__main__":
    main()
