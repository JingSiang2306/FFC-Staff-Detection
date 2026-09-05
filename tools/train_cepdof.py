"""
Train ordinary YOLO on CEPDOF's converted, axis-aligned person boxes.
Place in FCC-Staff-Detection/tools. Relative arguments use the project root.
Example: python tools/train_cepdof.py --smoke-test --device 0
"""

import argparse
import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()


def check_dataset(data_path):
    """Catch OBB labels and incomplete preparation before starting training."""
    import yaml
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if names not in ({0: "person"}, {"0": "person"}, ["person"]):
        raise ValueError("Dataset must contain exactly one class: 0: person")
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        raise ValueError("Use an absolute path in data.yaml, as prepare_cepdof.py generates")
    if any((p/"PREPARATION_INCOMPLETE.txt").exists() for p in (root, root.parent)):
        raise ValueError("Dataset preparation is incomplete")
    for split in ("train", "val"):
        images = root / data[split]
        labels = root/"labels"/split
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"Expected prepared images/{split} and labels/{split}: {root}")
        files = [p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not files:
            raise ValueError(f"No images in {images}")
        boxes = 0
        for image in files:
            label = labels/(image.stem+".txt")
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Expected ordinary five-column person labels: {label}")
                x, y, w, h = map(float, parts[1:])
                if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (x,y,w,h))
                        or w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6
                        or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
                    raise ValueError(f"Invalid normalized box in {label}: {line}")
                boxes += 1
        if not boxes:
            raise ValueError(f"No person annotations in {split}")
        print(f"{split}: {len(files)} images, {boxes} person boxes")
    # The test split is reserved for evaluation after model selection.


def plot_accuracy_vs_epoch(save_dir):
    """Plot validation mAP and add explicit reporting columns to results.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = save_dir/"results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"No epoch metrics in {path}")
    epochs = [int(float(row["epoch"])) for row in rows]
    maps = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    map50 = [float(row["metrics/mAP50(B)"]) for row in rows]
    if not all(math.isfinite(v) for v in maps+map50):
        raise ValueError("Non-finite validation mAP; inspect the training log")
    best = max(range(len(maps)), key=maps.__getitem__)
    for i, row in enumerate(rows):
        row["accuracy_mAP50-95"] = maps[i]
        row["is_final_epoch"] = int(i == len(rows)-1)
    # Replace atomically after reading the complete original table.
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, maps, label="Validation mAP50-95", linewidth=2)
    ax.plot(epochs, map50, label="Validation mAP50", alpha=0.7)
    ax.scatter([epochs[best]], [maps[best]], color="green", zorder=3,
               label=f"Highest epoch mAP50-95: {maps[best]:.4f}")
    ax.scatter([epochs[-1]], [maps[-1]], color="orange", zorder=4,
               label=f"Final epoch mAP50-95: {maps[-1]:.4f}")
    ax.set(xlabel="Epoch", ylabel="Validation mAP (0–1)", ylim=(0, 1),
           title="CEPDOF ordinary person detection")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir/"accuracy_vs_epoch.png", dpi=160)
    plt.close(fig)
    print(f"Final epoch validation mAP50-95: {maps[-1]:.4f}")
    print(f"Highest logged epoch mAP50-95: {maps[best]:.4f} at epoch {epochs[best]}")
    print("These are epoch-log metrics; best.pt is selected by Ultralytics fitness.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cepdof_yolo/detect/data.yaml")
    parser.add_argument("--model", default=None, help="Optional local .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4, help="Use -1 for GPU automatic batch sizing")
    parser.add_argument("--device", default=None, help="0 for CUDA, cpu for CPU; default auto")
    parser.add_argument("--workers", type=int, default=0, help="Windows-friendly default")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="One full epoch, separate run")
    args = parser.parse_args()
    if (min(args.imgsz, args.epochs) <= 0 or args.batch == 0 or args.batch < -1
            or args.workers < 0 or args.freeze < 0 or args.patience < 0
            or not math.isfinite(args.lr0) or args.lr0 <= 0):
        parser.error("Invalid training parameter")
    data_path = project_path(args.data)
    check_dataset(data_path)
    if args.model:
        weights = project_path(args.model)
        if not weights.is_file() or weights.suffix.lower() != ".pt":
            parser.error("--model must point to an existing .pt checkpoint")
    else:
        local = PROJECT_ROOT/"yoloModel"/"yolo26n.pt"
        weights = local if local.is_file() else "yolo26n.pt"
    name = args.name or "yolo26n_cepdof_detect_v1.0"
    if Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
        parser.error("--name must be a folder name, not a path")
    if args.smoke_test:
        name += "_smoke"
    project = PROJECT_ROOT/"outputs"/"training"
    if (project/name).exists():
        parser.error("Run folder exists. Choose another --name to preserve the earlier run")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.task != "detect":
        parser.error("Use an ordinary detection checkpoint, not an OBB/segmentation model")
    print(f"Starting weights: {weights}\nDataset: {data_path}")
    model.train(
        data=str(data_path), epochs=1 if args.smoke_test else args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
        optimizer="AdamW", lr0=args.lr0, patience=args.patience,
        freeze=args.freeze, seed=args.seed, deterministic=True, amp=not args.no_amp,
        single_cls=False, val=True, plots=True, save=True, cache=False, resume=False,
        project=str(project), name=name, exist_ok=False,
        # Preserve converted box tightness: arbitrary rotations enlarge AABBs again.
        degrees=0.0, shear=0.0, perspective=0.0,
        flipud=0.5, fliplr=0.5, translate=0.1, scale=0.3,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        mosaic=0.5, close_mosaic=0 if args.smoke_test else min(10, args.epochs),
        mixup=0.0, copy_paste=0.0,
    )
    save_dir = Path(model.trainer.save_dir)
    plot_accuracy_vs_epoch(save_dir)
    print(f"Results: {save_dir}\nCheckpoint: {save_dir/'weights'/'best.pt'}")
    if args.smoke_test:
        print("Smoke test only. Start the full run again from the original weights.")


if __name__ == "__main__":
    main()
