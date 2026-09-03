"""Sample a fisheye video and convert the frames to an equirectangular strip."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiment" / "convertedIMG"


def parse_args() -> argparse.Namespace:
    """Read command-line settings for the conversion experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from an overhead fisheye video and unwrap the visible "
            "hemisphere into a 360-degree equirectangular strip."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Number of converted frames to save per second (default: 1.0).",
    )
    parser.add_argument(
        "--fisheye-fov",
        type=float,
        default=180.0,
        help="Full field of view across the fisheye diameter (default: 180).",
    )
    parser.add_argument(
        "--lens-model",
        choices=("equidistant", "equisolid", "stereographic", "orthographic"),
        default="equidistant",
        help="Approximate fisheye projection model (default: equidistant).",
    )
    parser.add_argument(
        "--center-x",
        type=float,
        default=None,
        help="Fisheye-circle centre x coordinate; default is the image centre.",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        default=None,
        help="Fisheye-circle centre y coordinate; default is the image centre.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Usable fisheye radius in pixels; default fits inside the image.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Output height; default is approximately the fisheye radius.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Output width; default keeps equal horizontal/vertical angular scale.",
    )
    parser.add_argument(
        "--azimuth-offset",
        type=float,
        default=0.0,
        help="Rotate the panorama seam clockwise in degrees (default: 0).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace images from an earlier experiment in the output folder.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Stop early when a setting cannot produce a valid conversion."""
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than 0.")
    if not 0 < args.fisheye_fov < 360:
        raise ValueError("--fisheye-fov must be between 0 and 360 degrees.")
    if args.lens_model == "orthographic" and args.fisheye_fov > 180:
        raise ValueError("The orthographic model requires --fisheye-fov <= 180.")
    if args.radius is not None and args.radius <= 0:
        raise ValueError("--radius must be greater than 0.")
    if args.output_height is not None and args.output_height < 2:
        raise ValueError("--output-height must be at least 2 pixels.")
    if args.output_width is not None and args.output_width < 2:
        raise ValueError("--output-width must be at least 2 pixels.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")


def prepare_output_folder(output_dir: Path, overwrite: bool) -> None:
    """Create a clean destination without touching unrelated files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_items = list(output_dir.iterdir())

    if existing_items and not overwrite:
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}\n"
            "Use --overwrite to replace a previous conversion."
        )

    if overwrite:
        unknown_items = [
            item
            for item in existing_items
            if not (
                item.is_file()
                and (
                    (item.name.startswith("frame_") and item.suffix.lower() == ".jpg")
                    or item.name == "conversion_manifest.json"
                )
            )
        ]
        if unknown_items:
            names = ", ".join(item.name for item in unknown_items)
            raise RuntimeError(
                "The output folder contains unrelated items and was not changed: "
                f"{names}"
            )

        for item in existing_items:
            item.unlink()


def projection_radius(
    polar_angle: np.ndarray,
    maximum_angle: float,
    radius: float,
    lens_model: str,
) -> np.ndarray:
    """Convert an incident angle into a radius for a fisheye lens model."""
    if lens_model == "equidistant":
        normalized_radius = polar_angle / maximum_angle
    elif lens_model == "equisolid":
        normalized_radius = np.sin(polar_angle / 2) / math.sin(maximum_angle / 2)
    elif lens_model == "stereographic":
        normalized_radius = np.tan(polar_angle / 2) / math.tan(maximum_angle / 2)
    else:
        normalized_radius = np.sin(polar_angle) / math.sin(maximum_angle)

    return normalized_radius * radius


def build_remap(
    frame_width: int,
    frame_height: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Build the pixel lookup shared by every frame in the video."""
    center_x = args.center_x if args.center_x is not None else (frame_width - 1) / 2
    center_y = args.center_y if args.center_y is not None else (frame_height - 1) / 2

    default_radius = min(
        center_x,
        center_y,
        frame_width - 1 - center_x,
        frame_height - 1 - center_y,
    )
    radius = args.radius if args.radius is not None else default_radius

    if radius <= 0:
        raise ValueError("The calculated fisheye radius is not valid.")

    maximum_angle = math.radians(args.fisheye_fov / 2)
    output_height = args.output_height or max(2, int(round(radius)))
    output_width = args.output_width or max(
        2,
        int(round(output_height * (2 * math.pi / maximum_angle))),
    )

    # The top is the horizon/outer edge; the bottom is the downward centre.
    vertical_fraction = np.linspace(1.0, 0.0, output_height, dtype=np.float32)
    polar_angle = vertical_fraction[:, None] * maximum_angle

    # Do not duplicate the 0/360-degree seam in the final column.
    horizontal_fraction = np.arange(output_width, dtype=np.float32) / output_width
    azimuth = (
        horizontal_fraction[None, :] * (2 * math.pi)
        + math.radians(args.azimuth_offset)
    )

    source_radius = projection_radius(
        polar_angle,
        maximum_angle,
        radius,
        args.lens_model,
    )
    map_x = center_x + source_radius * np.sin(azimuth)
    map_y = center_y - source_radius * np.cos(azimuth)

    projection_info: dict[str, float | int | str] = {
        "input_width": frame_width,
        "input_height": frame_height,
        "output_width": output_width,
        "output_height": output_height,
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "fisheye_fov_degrees": args.fisheye_fov,
        "vertical_coverage_degrees": args.fisheye_fov / 2,
        "lens_model": args.lens_model,
        "azimuth_offset_degrees": args.azimuth_offset,
    }
    return map_x.astype(np.float32), map_y.astype(np.float32), projection_info


def estimate_sample_count(
    reported_frames: int,
    source_fps: float,
    sample_fps: float,
) -> int | None:
    """Estimate the number of outputs for the progress display."""
    if reported_frames <= 0:
        return None

    sample_number = 0
    while round(sample_number * source_fps / sample_fps) < reported_frames:
        sample_number += 1
    return sample_number


def convert_video(args: argparse.Namespace) -> dict[str, object]:
    """Decode the video, sample frames, convert them, and save a manifest."""
    input_path = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video was not found: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError("The video does not report a valid frame rate.")
    if args.sample_fps > source_fps:
        capture.release()
        raise ValueError(
            f"--sample-fps ({args.sample_fps}) cannot exceed the video FPS "
            f"({source_fps:.3f})."
        )

    prepare_output_folder(output_dir, args.overwrite)
    expected_samples = estimate_sample_count(
        reported_frames,
        source_fps,
        args.sample_fps,
    )

    frame_index = 0
    sample_number = 0
    map_x: np.ndarray | None = None
    map_y: np.ndarray | None = None
    projection_info: dict[str, float | int | str] | None = None
    saved_frames: list[dict[str, float | int | str]] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            target_frame = round(sample_number * source_fps / args.sample_fps)
            if frame_index >= target_frame:
                if map_x is None or map_y is None:
                    frame_height, frame_width = frame.shape[:2]
                    map_x, map_y, projection_info = build_remap(
                        frame_width,
                        frame_height,
                        args,
                    )
                elif frame.shape[:2] != (
                    projection_info["input_height"],
                    projection_info["input_width"],
                ):
                    raise RuntimeError("The video frame size changed during decoding.")

                converted = cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                timestamp = frame_index / source_fps
                filename = f"frame_{frame_index:06d}_t{timestamp:07.3f}s.jpg"
                output_path = output_dir / filename
                written = cv2.imwrite(
                    str(output_path),
                    converted,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not written:
                    raise RuntimeError(f"Failed to save image: {output_path}")

                saved_frames.append(
                    {
                        "sample_number": sample_number,
                        "source_frame": frame_index,
                        "timestamp_seconds": round(timestamp, 6),
                        "filename": filename,
                    }
                )
                sample_number += 1

                total_text = str(expected_samples) if expected_samples else "?"
                print(
                    f"\rConverted {sample_number}/{total_text}",
                    end="",
                    flush=True,
                )

            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError("The video contains no decodable frames.")
    if not saved_frames or projection_info is None:
        raise RuntimeError("No frames were selected for conversion.")

    manifest: dict[str, object] = {
        "input_video": str(input_path),
        "output_directory": str(output_dir),
        "source_fps": source_fps,
        "requested_sample_fps": args.sample_fps,
        "reported_frame_count": reported_frames,
        "decoded_frame_count": frame_index,
        "converted_frame_count": len(saved_frames),
        "jpeg_quality": args.jpeg_quality,
        "projection": projection_info,
        "frames": saved_frames,
    }
    manifest_path = output_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\rConverted {len(saved_frames)}/{len(saved_frames)} (Completed successfully)")
    print(f"Images: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    """Run the standalone preprocessing experiment."""
    args = parse_args()
    try:
        validate_args(args)
        convert_video(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
