"""Create padded person crops for staff-tag annotation."""

import argparse
import csv
import math
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MANIFEST_FIELDS = (
    "split",
    "crop_image",
    "source_image",
    "person_index",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/person_dataset_seed123"))
    parser.add_argument("--output", type=Path, default=Path("data/tag_dataset"))
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--person-class", type=int, default=0)
    return parser.parse_args()


def to_crop_box(label, image_width, image_height, padding):
    _, x_center, y_center, box_width, box_height = label
    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height

    x_padding = box_width * padding
    y_padding = box_height * padding
    x1 = max(0, math.floor(x_center - box_width / 2 - x_padding))
    y1 = max(0, math.floor(y_center - box_height / 2 - y_padding))
    x2 = min(image_width, math.ceil(x_center + box_width / 2 + x_padding))
    y2 = min(image_height, math.ceil(y_center + box_height / 2 + y_padding))
    return x1, y1, x2, y2


def read_person_labels(label_path, person_class):
    if not label_path.exists():
        return []

    labels = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid label at {label_path}:{line_number}")

        class_id = int(float(parts[0]))
        if class_id == person_class:
            labels.append((class_id, *map(float, parts[1:5])))
    return labels


def check_output(output_root):
    if output_root.exists() and any(output_root.rglob("*")):
        raise FileExistsError(
            f"Output folder is not empty: {output_root}. "
            "Choose another --output folder to avoid mixing dataset versions."
        )


def main():
    args = parse_args()
    if args.padding < 0:
        raise ValueError("--padding must be zero or greater.")

    check_output(args.output)
    manifest_rows = []
    split_counts = {}

    for split in SPLITS:
        source_images = args.source / "images" / split
        source_labels = args.source / "labels" / split
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Missing images or labels for split: {split}")

        output_images = args.output / "images" / split
        output_labels = args.output / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path
            for path in source_images.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        crop_count = 0

        for image_path in image_paths:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
                width, height = image.size
                label_path = source_labels / f"{image_path.stem}.txt"
                person_labels = read_person_labels(label_path, args.person_class)

                for person_index, label in enumerate(person_labels):
                    x1, y1, x2, y2 = to_crop_box(label, width, height, args.padding)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop_name = f"{image_path.stem}_person_{person_index:02d}.jpg"
                    crop_path = output_images / crop_name
                    image.crop((x1, y1, x2, y2)).save(crop_path, quality=95)

                    # Empty labels represent negative samples until a visible tag is annotated.
                    (output_labels / f"{Path(crop_name).stem}.txt").write_text("")
                    manifest_rows.append(
                        {
                            "split": split,
                            "crop_image": crop_path.relative_to(args.output).as_posix(),
                            "source_image": image_path.as_posix(),
                            "person_index": person_index,
                            "crop_x1": x1,
                            "crop_y1": y1,
                            "crop_x2": x2,
                            "crop_y2": y2,
                        }
                    )
                    crop_count += 1

        split_counts[split] = crop_count

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "crop_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    (args.output / "classes.txt").write_text("staff_tag\n", encoding="utf-8")
    print(f"Created {sum(split_counts.values())} person crops in {args.output}")
    for split in SPLITS:
        print(f"{split}: {split_counts[split]}")
    print("Next: annotate visible staff tags; leave true negative labels empty.")


if __name__ == "__main__":
    main()
