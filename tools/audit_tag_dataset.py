"""Audit YOLO staff-tag labels and create size reports and contact sheets."""

import argparse
import csv
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tag_dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tag_audit"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--thumb-size", type=int, default=240)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--sheet-rows", type=int, default=5)
    return parser.parse_args()


def find_image(images_dir, stem):
    matches = [p for p in images_dir.iterdir() if p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for label stem '{stem}', found {len(matches)}")
    return matches[0]


def read_boxes(label_path):
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 values")
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        if class_id != 0 or any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid class or coordinate at {label_path}:{line_number}")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Non-positive box size at {label_path}:{line_number}")
        boxes.append(values)
    return boxes


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_contact_sheets(items, output_dir, split, thumb_size, columns, rows):
    per_sheet = columns * rows
    font = ImageFont.load_default()
    for page, start in enumerate(range(0, len(items), per_sheet), 1):
        canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 34)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, item in enumerate(items[start : start + per_sheet]):
            image = Image.open(item["image_path"]).convert("RGB")
            image_draw = ImageDraw.Draw(image)
            width, height = image.size
            for box in item["boxes"]:
                xc, yc, bw, bh = box
                x1 = int((xc - bw / 2) * width)
                y1 = int((yc - bh / 2) * height)
                x2 = int((xc + bw / 2) * width)
                y2 = int((yc + bh / 2) * height)
                image_draw.rectangle((x1, y1, x2, y2), outline="lime", width=max(2, width // 200))
            image.thumbnail((thumb_size - 8, thumb_size - 8))
            cell = index % per_sheet
            col, row = cell % columns, cell // columns
            x = col * thumb_size + (thumb_size - image.width) // 2
            y = row * (thumb_size + 34) + (thumb_size - image.height) // 2
            canvas.paste(image, (x, y))
            label = f"{item['stem']}  {item['tag_width']:.0f}x{item['tag_height']:.0f}px"
            draw.text((col * thumb_size + 4, row * (thumb_size + 34) + thumb_size + 5), label[:38], fill="black", font=font)
        canvas.save(output_dir / f"{split}_positives_{page:02d}.jpg", quality=92)


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_lines = [f"Training image size: {args.imgsz}", ""]
    for split in SPLITS:
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images or labels for split: {split}")

        positive_items = []
        for label_path in sorted(label_dir.glob("*.txt")):
            boxes = read_boxes(label_path)
            if not boxes:
                continue
            image_path = find_image(image_dir, label_path.stem)
            with Image.open(image_path) as image:
                width, height = image.size
            scale = min(args.imgsz / width, args.imgsz / height)
            for box_index, (xc, yc, bw, bh) in enumerate(boxes):
                tag_width = bw * width
                tag_height = bh * height
                rows.append({
                    "split": split, "image": image_path.name, "box_index": box_index,
                    "crop_width": width, "crop_height": height,
                    "tag_width_px": round(tag_width, 3), "tag_height_px": round(tag_height, 3),
                    "tag_area_percent": round(100 * bw * bh, 4),
                    "resized_width_px": round(tag_width * scale, 3),
                    "resized_height_px": round(tag_height * scale, 3),
                })
                positive_items.append({
                    "stem": image_path.stem, "image_path": image_path, "boxes": boxes,
                    "tag_width": tag_width, "tag_height": tag_height,
                })

        widths = [row["resized_width_px"] for row in rows if row["split"] == split]
        heights = [row["resized_height_px"] for row in rows if row["split"] == split]
        summary_lines.extend([
            f"{split}: {len(widths)} tag boxes",
            f"  resized width  p10/median/p90: {percentile(widths, .1):.1f} / {statistics.median(widths) if widths else 0:.1f} / {percentile(widths, .9):.1f} px",
            f"  resized height p10/median/p90: {percentile(heights, .1):.1f} / {statistics.median(heights) if heights else 0:.1f} / {percentile(heights, .9):.1f} px",
        ])
        build_contact_sheets(positive_items, args.output, split, args.thumb_size, args.columns, args.sheet_rows)

    report_path = args.output / "tag_box_sizes.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"Reports and contact sheets: {args.output}")


if __name__ == "__main__":
    main()
