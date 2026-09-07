# FFC Staff Detection

The goal is to identify frames containing staff and, as a bonus, provide staff coordinates. The current main script produces a live preview, annotated video and per-frame staff coordinate CSV.

## Current milestone

`detect_staff.py` now:

1. reads the input video and detects people with the fine-tuned person model;
2. tracks those people using the selected tracker configuration;
3. extracts padded crops from the original frame and checks them with a separate `staff_tag` detector;
4. confirms staff using recent tag evidence for each tracker ID, with temporary status retention;
5. displays staff boxes, IDs, tag scores, bottom-centre coordinates, processing FPS and current-frame counts, and writes an annotated MP4 plus a staff-occurrence CSV.

### Demo

<video src="outputs/demo/1_staff_detect-Normal-Compressed.mp4" width="100%" controls>
  Demo videos located at outputs/demo.
</video>

### Temporal voting: actual current behaviour

- A new tracker ID needs at least 5 tag-positive frames within the latest 15 processed video frames. These need not be consecutive.
- Whenever 5 votes still lie in that window, the expiry is set to the current frame plus 75. Even without a new positive, votes remaining in the window can refresh this expiry.
- An already-active staff ID can also refresh its expiry with one new tag-positive frame. Status expires after the stored expiry frame; it is not permanent for the whole video.
- A new ID starts without the previous ID's evidence. There is no cross-ID identity stitching. If an ID transfers to another person, its retained staff status can transfer too until it expires.
- The displayed `Tag` value is the **highest accepted score in the retained history**, not necessarily the current frame's score. A high displayed value therefore does not prove the tag is currently visible.
- A staff box is drawn only when the person is currently returned with a usable tracked crop. Holding staff status does not reconstruct a missing person box.
- `Persons` and `Staff` are current-frame counts. The final console list contains all IDs ever confirmed, not a count of unique real staff members.

## Setup

Use the existing working virtual environment when reproducing the accepted result. For a fresh environment, use a Python version supported by the chosen PyTorch/Ultralytics packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the video at `data/sample.mp4`, provide both trained weights in `yoloModel/`, and retain `trackers/custom_bytetrack.yaml`. A CUDA-capable GPU also needs a compatible CUDA-enabled PyTorch installation; otherwise `auto` uses the CPU.

## Run

Run from the repository root. Relative model, tracker, input and output paths are resolved from the working directory.

Choose CUDA automatically when available (the current default):

```powershell
python detect_staff.py
```

Force CPU execution:

```powershell
python detect_staff.py --device cpu
```

Require the first NVIDIA GPU:

```powershell
python detect_staff.py --device 0
```

Show all returned person boxes as thin red boxes with small IDs below them for diagnosis:

```powershell
python detect_staff.py --person-show
```

Show staff's trajectory for latest 60 frames

```powershell
python detect_staff.py --trajectory-show
```

Staff retain the red person ID below the box, while the thicker green staff box is drawn over the red box. Without this flag, only staff boxes are drawn. Press `q` in the preview to stop early; the saved video then contains only the processed portion.

The default input path is `data/sample.mp4`. Use `--input` for new input video.

```powershell
python detect_staff.py --input data/sample-2.mp4
```

The annotated result is written to `outputs/staff_detected.mp4`. Use a different `--output` filename to preserve previous results.

### Staff coordinates and CSV

Each displayed staff box has green `X` and `Y` labels near its lower-right corner, with X above Y. The text moves inward at image edges. Coordinates use the person's bottom centre: `x = (x1 + x2) / 2`, `y = y2`, in original-frame pixels, rounded to one decimal place. The origin is the image's top-left corner; x increases rightwards and y downwards. These are image positions, not calibrated floor/world coordinates.

By default, the CSV shares the video's output name: `outputs/staff_detected.csv`. Override its location with:

```powershell
python detect_staff.py --csv-output outputs/staff_frames.csv
```

The columns, in order, are `staff ID,frame,x,y,time`. There is one row per displayed staff ID per frame, including frames where staff status is held without a fresh tag detection. Multiple staff in one frame produce multiple rows; frames with no displayed staff produce none. If no staff are confirmed, the CSV contains only its header.

Frame numbering starts at **1**; frame 1 has time `00m 00s 000ms`. Time uses minutes, seconds and milliseconds, calculated as `(frame - 1) / source_fps`. At 25 FPS, frame 26 is `00m 01s 000ms`. This represents video time, not processing time. Explicit units prevent Excel from misinterpreting the values. Timing assumes a constant frame rate.

The CSV closes on completion, `q`, or a processing error, preserving rows already written. Existing CSV/video output files are overwritten when their paths are reused. The staff ID is the tracker ID, so ID fragmentation or transfer remains possible. Export is one-pass and does not backfill frames before staff confirmation.

### Defaults in the frozen script

| Option | Default |
|---|---|
| `--person-model` | `yoloModel/best_v1.2.pt` |
| `--tag-model` | `yoloModel/best_tag_v1.2.pt` |
| `--tracker` | `trackers/custom_bytetrack.yaml` |
| `--device` | `auto` |
| `--conf`, `--imgsz` | `0.10`, `640` |
| `--tag-conf`, `--tag-imgsz` | `0.5`, `640` |
| `--crop-padding` | `0.10` on each side |
| `--vote-window`, `--vote-min` | `15`, `5` |
| `--staff-hold` | `75` frames |
| `--person-show` | Off |
| `--trajectory-show` | Off |
| `--csv-output` | Same path as `--output`, with a `.csv` extension |

List all options:

```powershell
python detect_staff.py --help
```

## Dataset tools

Create the temporal-block person dataset:

```powershell
python tools/prepare_dataset.py --seed 123
```

The documented split uses train/validation/test proportions of 60/20/20 with temporal blocks and boundary separation. Block assignment may be randomized; this is not permission to randomly redistribute neighbouring crops. Preserve the actual split manifest and use training-only source blocks for expansion and mining.

Validate the completed person labels, optionally with previews:

```powershell
python tools/validate_labels.py --dataset data/person_dataset
python tools/validate_labels.py --dataset data/person_dataset --preview --preview-count 20
```

Tag-model workflow already completed during development:

1. `prepare_tag_dataset.py`: generate padded crops from person labels while preserving source splits.
2. Manually label visible tags as class `0: staff_tag`; review empty labels rather than assuming all unlabelled crops are negatives.
3. `balance_tag_dataset.py`: retain all training positives and reproducibly sample training negatives (4:1, seed 123); keep full validation/test sets unchanged.
4. `train_tag_model.py`: train the separate tag detector, keeping the person-model experiment independent.
5. `mine_neg_tag_training_crops.py`: review difficult tag predictions from training data and add only confirmed negatives to a later training version. Do not mine validation/test examples into training.

### Recorded tag dataset checkpoints

| Dataset checkpoint | Training | Validation | Test |
|---|---|---|---|
| Original annotated crops | 26 positive + 1,776 negative | 7 positive + 565 negative | 5 positive + 606 negative |
| First balanced set | 26 positive + 104 negative | Unchanged | Unchanged |
| Positive expansion, balanced v1.1 | 43 positive + 172 negative | Unchanged | Unchanged |
| Negative expansion, balanced v1.2 | 43 positive + 189 negative | Unchanged | Unchanged |

## Limitations

The accepted result is a proof of concept. It is not guaranteed to detect every staff occurrence or generalize to an unknown video. 

Occlusion, changing appearance, tiny people near image edges and hidden tags can cause missed detections or new IDs. Better person detection, better tag detection and better tracking address different parts of this problem. Temporal holding reduces short gaps but can retain incorrect status after an ID transfer; stricter tag thresholds can reduce false positives while increasing missed staff periods.

The current pipeline exports bottom-centre coordinates and staff-occurrence rows, but remains one-pass: it cannot annotate or export earlier frames retrospectively after later staff confirmation. Missing detections and unconfirmed IDs also produce no staff rows. Consequently, CSV export alone does not guarantee that every true staff occurrence is captured. Two-pass recovery remains deferred due to current detection and tracking performance is not reliable enough, with the same person frequently receiving different tracking IDs.
