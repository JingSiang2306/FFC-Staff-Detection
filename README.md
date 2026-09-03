# FCC Staff Detection

CPU-first proof of concept for the FootfallCam AI Evaluation Test. The final goal is to report every frame containing a staff member and, as a bonus, the staff member's coordinates.

## Current milestone

`detect_staff.py` currently performs the baseline stage only:

1. reads the input video;
2. detects the COCO `person` class with YOLO26n;
3. maintains person IDs with ByteTrack;
4. writes an annotated MP4.

Staff-tag verification and CSV output will be added after this baseline is visually checked.

Dataset preparation and annotation validation are provided as reusable scripts under the Git-tracked `tools/` folder.

## Project layout

```text
FCC-Staff-Detection/
├── .gitignore
├── detect_staff.py
├── requirements.txt
├── README.md
├── tools/                  # reusable utilities; tracked by Git
│   ├── prepare_dataset.py  # temporal train/val/test extraction
│   └── validate_labels.py  # YOLO label checks and previews
├── data/
│   ├── sample.mp4          # local only; ignored by Git
│   └── person_dataset/     # images and labels; ignored by Git
├── outputs/                # generated results; ignored by Git
└── notes/
    └── FCC AI Evaluation - Engineering Notes.md
```

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the supplied video at `data/sample.mp4`.

## Run

CPU, the safe default:

```powershell
python detect_staff.py
```

Choose CUDA automatically when available:

```powershell
python detect_staff.py --device auto
```

Require the first NVIDIA GPU:

```powershell
python detect_staff.py --device 0
```

The annotated result is written to `outputs/people_tracked.mp4`.

## Dataset tools

Create the temporal-block dataset:

```powershell
python tools/prepare_dataset.py --seed 123
```

Validate the completed YOLO labels:

```powershell
python tools/validate_labels.py --dataset data/person_dataset
```

Validate the labels and generate 20 preview images:

```powershell
python tools/validate_labels.py --dataset data/person_dataset --preview --preview-count 20
```

The `tools/` folder is committed to GitHub. Dataset images, labels, videos, model weights, and generated outputs remain local unless the repository policy is changed deliberately.

## Important limitation

This milestone tracks all detected people. It does not yet identify the staff member. Separating these stages makes detection and tracking errors visible before staff-tag logic is added.
