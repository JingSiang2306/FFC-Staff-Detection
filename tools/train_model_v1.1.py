import argparse
from pathlib import Path

import csv
import math

import albumentations as A
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "person_dataset_seed123" / "data.yaml"


def parse_args():
    """Read training options from the command line."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26n for overhead person detection.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--model", default="yoloModel/yolo26n.pt", help="Pretrained model weights")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 selects automatically")
    parser.add_argument("--workers", type=int, default=4, help="Data-loading worker processes")
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch to verify the setup")
    return parser.parse_args()


def resolve_device(requested):
    """Convert the device option into an Ultralytics device value."""
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return int(requested) if requested.isdigit() else requested


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
    """Plot validation mAP and add clear reporting columns to results.csv."""
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
    args = parse_args()
    data_path = args.data.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    device = resolve_device(args.device)
    epochs = 1 if args.smoke_test else args.epochs
    run_name = "yolo26n_fisheye_v1.2_smoke" if args.smoke_test else "yolo26n_fisheye_v1.2-plot"

    print(f"Mode: {'smoke test' if args.smoke_test else 'full training'}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {epochs}")
    print("The test split will not be evaluated during training.\n")

    model = YOLO(args.model)

    # Apply the requested hue, brightness, and grayscale changes.
    custom_transforms = [
        A.HueSaturationValue(
            hue_shift_limit=(-20, 20),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=0.0,
            p=1.0,
        ),
        A.ToGray(p=0.25),
    ]

    model.train(
        data=str(data_path),
        device=device,
        imgsz=args.imgsz,
        epochs=epochs,
        batch=args.batch,
        workers=args.workers,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        seed=123,
        deterministic=True,
        amp=True,
        single_cls=False,
        degrees=45.0,
        flipud=0.5,
        fliplr=0.5,
        augmentations=custom_transforms,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        project=str(PROJECT_ROOT / "outputs" / "training"),
        name=run_name,
        exist_ok=False,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    plot_accuracy_vs_epoch(save_dir)
    print(f"\nCompleted successfully. Results: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
