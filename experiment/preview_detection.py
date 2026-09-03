"""Preview YOLO bounding boxes on selected images."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


OUTPUT_DIR = Path(__file__).resolve().parent / "annotated"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", default="yoloModel/yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def next_output_path():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    number = 1
    while (OUTPUT_DIR / f"Annotated_{number}.jpg").exists():
        number += 1
    return OUTPUT_DIR / f"Annotated_{number}.jpg"


def main():
    args = get_args()
    model = YOLO(args.model)

    for image_path in args.images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        result = model.predict(
            image,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        annotated = result.plot()

        cv2.imshow(f"Detection Preview - {image_path.name}", annotated)
        print("Press any key in the preview window to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        answer = input("Save this annotated image? (Y/N): ").strip().lower()
        if answer == "y":
            output_path = next_output_path()
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
