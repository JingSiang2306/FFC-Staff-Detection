import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "tag_dataset_balanced" / "tag_data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "yoloModel" / "yolo26n.pt"


def parse_args():
    """Read training options from the command line."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26n to detect staff tags.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to tag_data.yaml")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pretrained YOLO weights")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 selects automatically")
    parser.add_argument("--workers", type=int, default=4, help="Data-loading worker processes")
    parser.add_argument("--name", default="yolo26n_staff_tag_v1.0", help="Unique output run name")
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
    run_name = f"{args.name}_smoke" if args.smoke_test else args.name

    print(f"Mode: {'smoke test' if args.smoke_test else 'full training'}")
    print(f"Data: {data_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {epochs}")
    print("The test split will remain unseen during training.\n")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        device=device,
        imgsz=args.imgsz,
        epochs=epochs,
        batch=args.batch,
        workers=args.workers,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        seed=123,
        deterministic=True,
        amp=True,
        single_cls=False,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.0,
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
