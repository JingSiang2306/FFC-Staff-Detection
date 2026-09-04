import argparse
import csv
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "person_dataset_seed123" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "yoloModel" / "best_v1.1.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "outputs" / "test_evaluation"


def parse_args():
    """Read test-evaluation options from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model on the test split.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to best.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--workers", type=int, default=4, help="Data-loading worker processes")
    parser.add_argument("--name", default="yolo26n_fisheye_v1.1_test", help="Output run name")
    return parser.parse_args()


def resolve_device(requested):
    """Convert the device option into an Ultralytics device value."""
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return int(requested) if requested.isdigit() else requested


def read_dataset_names(data_path):
    """Read and normalize class names from data.yaml."""
    with data_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not data.get("test"):
        raise ValueError("data.yaml does not define a test split.")

    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(class_id): str(name) for class_id, name in names.items()}
    else:
        raise ValueError("data.yaml does not contain valid class names.")

    return names


def write_metrics_csv(output_path, model_path, imgsz, metrics):
    """Save the main test metrics in one CSV row."""
    row = {
        "split": "test",
        "model": str(model_path),
        "image_size": imgsz,
        "precision": f"{metrics.box.mp:.6f}",
        "recall": f"{metrics.box.mr:.6f}",
        "mAP50": f"{metrics.box.map50:.6f}",
        "mAP50-95": f"{metrics.box.map:.6f}",
        "fitness": f"{metrics.fitness:.6f}",
        "inference_ms_per_image": f"{metrics.speed.get('inference', 0.0):.3f}",
    }

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    project_path = DEFAULT_PROJECT.resolve()
    run_path = project_path / args.name

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")
    if run_path.exists():
        raise FileExistsError(f"Output already exists: {run_path}\nUse a different --name to keep results separate.")

    dataset_names = read_dataset_names(data_path)
    device = resolve_device(args.device)
    model = YOLO(str(model_path))

    if model.names != dataset_names:
        print(f"Using dataset class names for reporting: {dataset_names}")
        model.model.names = dataset_names

    print("Evaluating the frozen test split. Do not tune the model using this result.")
    metrics = model.val(
        data=str(data_path),
        split="test",
        device=device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        single_cls=False,
        plots=True,
        project=str(project_path),
        name=args.name,
        exist_ok=True,
    )

    write_metrics_csv(run_path / "test_metrics.csv", model_path, args.imgsz, metrics)

    print("\nTest evaluation completed successfully")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Results:    {run_path}")


if __name__ == "__main__":
    main()
