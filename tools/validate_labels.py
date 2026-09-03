"""Validate YOLO detection labels and optionally create annotated previews."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPSILON = 1e-6


@dataclass
class Issue:
    level: str
    path: Path
    message: str
    line: int | None = None

    def format(self, dataset: Path) -> str:
        try:
            shown_path = self.path.relative_to(dataset)
        except ValueError:
            shown_path = self.path
        location = f"{shown_path}:{self.line}" if self.line else str(shown_path)
        return f"[{self.level}] {location} - {self.message}"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/person_dataset"))
    parser.add_argument("--class-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--preview-dir", type=Path, default=Path("outputs/label_previews")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-issues", type=int, default=100)
    return parser.parse_args()


def index_files(paths: list[Path], kind: str, issues: list[Issue]) -> dict[str, Path]:
    """Index files by stem and report duplicate names."""
    indexed: dict[str, Path] = {}
    for path in paths:
        if path.stem in indexed:
            issues.append(
                Issue("ERROR", path, f"Duplicate {kind} stem '{path.stem}'.")
            )
        else:
            indexed[path.stem] = path
    return indexed


def read_label(
    path: Path, allowed_ids: set[int], issues: list[Issue]
) -> list[tuple[int, float, float, float, float]]:
    """Parse one YOLO label file and return valid boxes."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        issues.append(Issue("ERROR", path, f"Cannot read label file: {error}"))
        return []

    non_empty_lines = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    if not non_empty_lines:
        issues.append(Issue("WARNING", path, "Label file is empty."))
        return []

    boxes = []
    for line_number, line in non_empty_lines:
        values = line.split()
        if len(values) != 5:
            issues.append(
                Issue("ERROR", path, f"Expected 5 values, found {len(values)}.", line_number)
            )
            continue

        try:
            class_id = int(values[0])
        except ValueError:
            issues.append(Issue("ERROR", path, "Class ID must be an integer.", line_number))
            continue

        try:
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError:
            issues.append(
                Issue("ERROR", path, "Coordinates must be numbers.", line_number)
            )
            continue

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            issues.append(
                Issue("ERROR", path, "Coordinates must be finite numbers.", line_number)
            )
            continue
        if class_id not in allowed_ids:
            issues.append(
                Issue(
                    "ERROR",
                    path,
                    f"Class ID {class_id} is not allowed; expected {sorted(allowed_ids)}.",
                    line_number,
                )
            )

        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            issues.append(
                Issue("ERROR", path, "Box centre must be between 0 and 1.", line_number)
            )
            continue
        if not (0 < width <= 1 and 0 < height <= 1):
            issues.append(
                Issue("ERROR", path, "Box width and height must be above 0 and at most 1.", line_number)
            )
            continue

        x_min, x_max = x_center - width / 2, x_center + width / 2
        y_min, y_max = y_center - height / 2, y_center + height / 2
        if x_min < -EPSILON or y_min < -EPSILON or x_max > 1 + EPSILON or y_max > 1 + EPSILON:
            issues.append(
                Issue("ERROR", path, "Box extends outside the image boundary.", line_number)
            )
            continue

        if class_id in allowed_ids:
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def validate_dataset(
    dataset: Path, allowed_ids: set[int], allow_missing: bool
) -> tuple[dict, list[Issue], dict[Path, list[tuple[int, float, float, float, float]]]]:
    issues: list[Issue] = []
    boxes_by_image = {}
    stats = {"images": 0, "labels": 0, "boxes": 0, "missing": 0, "unmatched": 0}

    for split in SPLITS:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir():
            issues.append(Issue("ERROR", image_dir, "Image directory is missing."))
            continue
        if not label_dir.is_dir():
            issues.append(Issue("ERROR", label_dir, "Label directory is missing."))
            continue

        image_paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_paths = sorted(path for path in label_dir.iterdir() if path.suffix.lower() == ".txt")
        images = index_files(image_paths, "image", issues)
        labels = index_files(label_paths, "label", issues)
        stats["images"] += len(image_paths)
        stats["labels"] += len(label_paths)

        for stem, image_path in images.items():
            label_path = labels.get(stem)
            if label_path is None:
                level = "WARNING" if allow_missing else "ERROR"
                issues.append(Issue(level, image_path, "Matching label file is missing."))
                stats["missing"] += 1
                boxes_by_image[image_path] = []
                continue

            boxes = read_label(label_path, allowed_ids, issues)
            boxes_by_image[image_path] = boxes
            stats["boxes"] += len(boxes)

        for stem, label_path in labels.items():
            if stem not in images:
                issues.append(Issue("ERROR", label_path, "Matching image file is missing."))
                stats["unmatched"] += 1

    return stats, issues, boxes_by_image


def create_previews(
    boxes_by_image: dict[Path, list[tuple[int, float, float, float, float]]],
    output_dir: Path,
    count: int,
    seed: int,
) -> list[str]:
    if count <= 0:
        raise ValueError("--preview-count must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Preview folder is not empty: {output_dir}\n"
            "Choose another --preview-dir to avoid mixing preview runs."
        )

    items = sorted(boxes_by_image.items(), key=lambda item: str(item[0]))
    selected = random.Random(seed).sample(items, min(count, len(items)))
    warnings = []

    for image_path, boxes in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            warnings.append(f"Could not read preview image: {image_path}")
            continue

        image_height, image_width = image.shape[:2]
        for class_id, x_center, y_center, width, height in boxes:
            x1 = max(0, round((x_center - width / 2) * image_width))
            y1 = max(0, round((y_center - height / 2) * image_height))
            x2 = min(image_width - 1, round((x_center + width / 2) * image_width))
            y2 = min(image_height - 1, round((y_center + height / 2) * image_height))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"person ({class_id})",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        split = image_path.parent.name
        preview_path = output_dir / split / image_path.name
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_path), image):
            warnings.append(f"Could not save preview image: {preview_path}")

    return warnings


def main() -> int:
    args = get_args()
    if args.max_issues <= 0:
        raise ValueError("--max-issues must be positive.")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {args.dataset}")

    stats, issues, boxes_by_image = validate_dataset(
        args.dataset, set(args.class_ids), args.allow_missing_labels
    )
    errors = sum(issue.level == "ERROR" for issue in issues)
    warnings = sum(issue.level == "WARNING" for issue in issues)

    print(f"Dataset: {args.dataset}")
    print(f"Images: {stats['images']}")
    print(f"Label files: {stats['labels']}")
    print(f"Valid boxes: {stats['boxes']}")
    print(f"Missing labels: {stats['missing']}")
    print(f"Unmatched labels: {stats['unmatched']}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    print("Note: Preview inspection is still required to find people that were not boxed.")

    if issues:
        print("\nIssues:")
        for issue in issues[: args.max_issues]:
            print(issue.format(args.dataset))
        hidden = len(issues) - args.max_issues
        if hidden > 0:
            print(f"... {hidden} additional issues not shown")

    if args.preview:
        preview_warnings = create_previews(
            boxes_by_image, args.preview_dir, args.preview_count, args.seed
        )
        print(f"\nPreview images: {args.preview_dir}")
        for warning in preview_warnings:
            print(f"[WARNING] {warning}")

    if errors:
        print("\nValidation failed. Fix the errors before training.")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
