import argparse
from pathlib import Path

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


def main():
    args = parse_args()
    data_path = args.data.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    device = resolve_device(args.device)
    epochs = 1 if args.smoke_test else args.epochs
    run_name = "yolo26n_custom_aug_smoke" if args.smoke_test else "yolo26n_custom_aug_v1"

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
    print(f"\nCompleted successfully. Results: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
