"""
Train YOLO OBB on CEPDOF's rotated person boxes.

Place this file in FCC-Staff-Detection/tools. Relative paths are resolved from
the project root, so the script can be run from any terminal directory.
"""

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def project_path(value):
    """Resolve a command-line path relative to the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def polygon_area(points):
    """Calculate the area of a four-corner OBB polygon."""
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2


def check_dataset(data_path):
    """Check that train and validation folders contain valid YOLO OBB labels."""
    import yaml

    if not data_path.is_file():
        raise ValueError(f"Dataset YAML does not exist: {data_path}")

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if names not in ({0: "person"}, {"0": "person"}, ["person"]):
        raise ValueError("Dataset must contain exactly one class: 0: person")

    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    if any((path / "PREPARATION_INCOMPLETE.txt").exists() for path in (root, root.parent)):
        raise ValueError("CEPDOF preparation is incomplete")

    for split in ("train", "val"):
        if split not in data:
            raise ValueError(f"Missing '{split}' entry in {data_path}")

        images_dir = root / data[split]
        labels_dir = root / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise ValueError(
                f"Expected images/{split} and labels/{split} beneath {root}"
            )

        images = sorted(
            path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"No images found in {images_dir}")

        box_count = 0
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Missing label for {image_path.name}: {label_path}")

            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                parts = line.split()
                if len(parts) != 9 or parts[0] != "0":
                    raise ValueError(
                        f"Expected '0 x1 y1 x2 y2 x3 y3 x4 y4' in "
                        f"{label_path}, line {line_number}"
                    )

                coordinates = [float(value) for value in parts[1:]]
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                    raise ValueError(
                        f"OBB coordinates must be finite and normalized in {label_path}, "
                        f"line {line_number}"
                    )

                points = list(zip(coordinates[0::2], coordinates[1::2]))
                if len(set(points)) != 4 or polygon_area(points) <= 1e-10:
                    raise ValueError(
                        f"Degenerate OBB in {label_path}, line {line_number}"
                    )
                box_count += 1

        if box_count == 0:
            raise ValueError(f"No person annotations found in {split}")
        print(f"{split}: {len(images)} images, {box_count} rotated person boxes")

    # The test split remains untouched until the final model configuration is chosen.


def find_metric_column(fieldnames, metric):
    """Support small Ultralytics naming differences between releases."""
    if metric == "mAP50-95":
        matches = [
            name for name in fieldnames
            if name.startswith("metrics/") and "mAP50-95" in name
        ]
    else:
        matches = [
            name for name in fieldnames
            if name.startswith("metrics/")
            and "mAP50" in name
            and "mAP50-95" not in name
        ]
    if len(matches) != 1:
        raise ValueError(f"Could not identify one {metric} column in results.csv: {matches}")
    return matches[0]


def plot_accuracy_vs_epoch(save_dir):
    """Plot validation OBB mAP and add clear reporting columns to results.csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path = save_dir / "results.csv"
    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]

    if not rows:
        raise ValueError(f"No epoch metrics found in {results_path}")

    map_column = find_metric_column(list(rows[0]), "mAP50-95")
    map50_column = find_metric_column(list(rows[0]), "mAP50")
    epochs = [int(float(row["epoch"])) for row in rows]
    maps = [float(row[map_column]) for row in rows]
    maps50 = [float(row[map50_column]) for row in rows]
    if not all(math.isfinite(value) for value in maps + maps50):
        raise ValueError("Non-finite validation mAP detected; inspect the training log")

    best_index = max(range(len(maps)), key=maps.__getitem__)
    for index, row in enumerate(rows):
        row["accuracy_mAP50-95"] = maps[index]
        row["is_final_epoch"] = int(index == len(rows) - 1)

    temporary_path = results_path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(results_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, maps, label="Validation OBB mAP50-95", linewidth=2)
    axis.plot(epochs, maps50, label="Validation OBB mAP50", alpha=0.7)
    axis.scatter(
        [epochs[best_index]],
        [maps[best_index]],
        color="green",
        zorder=3,
        label=f"Highest epoch mAP50-95: {maps[best_index]:.4f}",
    )
    axis.scatter(
        [epochs[-1]],
        [maps[-1]],
        color="orange",
        zorder=4,
        label=f"Final epoch mAP50-95: {maps[-1]:.4f}",
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Validation OBB mAP (0-1)",
        ylim=(0, 1),
        title="CEPDOF rotated person detection",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_dir / "accuracy_vs_epoch.png", dpi=160)
    plt.close(figure)

    print(f"Final epoch validation OBB mAP50-95: {maps[-1]:.4f}")
    print(
        f"Highest logged epoch OBB mAP50-95: {maps[best_index]:.4f} "
        f"at epoch {epochs[best_index]}"
    )
    print("The deployment checkpoint remains weights/best.pt.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cepdof_yolo/obb/data.yaml")
    parser.add_argument("--model", default=None, help="Optional local OBB .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch", type=int, default=4, help="Use -1 for automatic GPU batch sizing"
    )
    parser.add_argument("--device", default=None, help="0 for CUDA or cpu; default is auto")
    parser.add_argument("--workers", type=int, default=0, help="Windows-friendly default")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run one complete epoch")
    args = parser.parse_args()

    if (
        min(args.imgsz, args.epochs) <= 0
        or args.batch == 0
        or args.batch < -1
        or args.workers < 0
        or args.freeze < 0
        or args.patience < 0
        or not math.isfinite(args.lr0)
        or args.lr0 <= 0
    ):
        parser.error("Invalid training parameter")

    data_path = project_path(args.data)
    check_dataset(data_path)

    if args.model:
        weights = project_path(args.model)
        if not weights.is_file() or weights.suffix.lower() != ".pt":
            parser.error("--model must point to an existing OBB .pt checkpoint")
    else:
        local_weights = PROJECT_ROOT / "yoloModel" / "yolo26n-obb.pt"
        weights = local_weights if local_weights.is_file() else "yolo26n-obb.pt"

    name = args.name or f"yolo26n_cepdof_obb_{args.imgsz}_v1"
    if Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
        parser.error("--name must be a folder name, not a path")
    if args.smoke_test:
        name += "_smoke"

    project = PROJECT_ROOT / "outputs" / "training"
    if (project / name).exists():
        parser.error("Run folder already exists. Choose another --name to preserve it")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "obb":
        parser.error("Use an OBB checkpoint such as yolo26n-obb.pt")

    training_epochs = 1 if args.smoke_test else args.epochs
    print(f"Starting weights: {weights}")
    print(f"Dataset: {data_path}")
    model.train(
        data=str(data_path),
        epochs=training_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.lr0,
        patience=args.patience,
        freeze=args.freeze,
        seed=args.seed,
        deterministic=True,
        amp=not args.no_amp,
        single_cls=False,
        val=True,
        plots=True,
        save=True,
        cache=False,
        resume=False,
        project=str(project),
        name=name,
        exist_ok=False,
        # A rotation around the fisheye centre is realistic for overhead people.
        degrees=180.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.5,
        close_mosaic=0 if args.smoke_test else min(10, training_epochs),
        mixup=0.0,
        copy_paste=0.0,
    )

    save_dir = Path(model.trainer.save_dir)
    plot_accuracy_vs_epoch(save_dir)
    print(f"Results: {save_dir}")
    print(f"Checkpoint: {save_dir / 'weights' / 'best.pt'}")
    if args.smoke_test:
        print("Smoke test only. Start the full run again from yolo26n-obb.pt.")


if __name__ == "__main__":
    main()
