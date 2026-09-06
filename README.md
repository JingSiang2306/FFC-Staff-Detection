# FCC Staff Detection

The goal is to identify frames containing staff and, as a bonus, provide staff coordinates. The current main script produces a live preview, annotated video and per-frame staff coordinate CSV.

## Current milestone

`detect_staff.py` now:

1. reads the input video and detects people with the fine-tuned person model;
2. tracks those people using the selected tracker configuration;
3. extracts padded crops from the original frame and checks them with a separate `staff_tag` detector;
4. confirms staff using recent tag evidence for each tracker ID, with temporary status retention;
5. displays staff boxes, IDs, tag scores, bottom-centre coordinates, processing FPS and current-frame counts, and writes an annotated MP4 plus a staff-occurrence CSV.

### Development closure and experiment history

This closes implementation work for the accepted milestone, not every possible accuracy limitation. The history below consolidates available project conversations and the latest uploaded main script; it is not a verified inventory of every chat or a remote issue-tracker closure.

| Workstream | Outcome at the freeze |
|---|---|
| Person detection | Pretrained YOLO baseline, overhead-person fine-tuning and model revisions completed. The main script now selects `best_v1.2.pt`. Earlier `item` class-name metadata was corrected to `person`. |
| Runtime and tracking | CPU/GPU selection, progress, timing and video output implemented. ByteTrack threshold/buffer tuning and BoT-SORT ReID comparisons were performed. The current default remains the custom ByteTrack YAML. |
| Staff-tag model | Person-crop preparation, tag annotation, reproducible balancing, positive expansion and separate training implemented. The latest main script selects `best_tag_v1.2.pt`; final v1.2 numerical metrics are not established by the available evidence. |
| Hard-negative mining | `tools/mine_neg_tag_training_crops.py` created to find difficult candidates for the **tag detector**. Human review is required before treating a candidate as a negative. |
| Temporal logic and display | Permanent staff assignment replaced with expiring per-ID status. Staff confidence, current-frame counts and optional red person boxes/IDs implemented. The earlier trailing `video` NameError is absent from the latest script. |
| Fisheye and CEPDOF experiments | Fisheye preprocessing, CEPDOF preparation/training, and ordinary-box versus oriented-box (OBB) experiments explored separately. OBB person-only and staff-pipeline scripts were developed as experiments; the main script remains ordinary-box based. |
| Remaining accuracy issues | Occasional false positives, missed people, hidden tags, fragmented IDs and identity transfers are accepted limitations for this milestone, not claimed to be solved. |
| Frame/coordinate export | Added after the initial freeze: per-displayed-staff bottom-centre coordinates in both preview/video and CSV. |
| Two-pass output | Still deferred. Earlier frames are not retrospectively recovered after later staff confirmation. |

### Temporal voting: actual current behaviour

- A new tracker ID needs at least 3 tag-positive frames within the latest 15 processed video frames. These need not be consecutive.
- Whenever 3 votes still lie in that window, the expiry is set to the current frame plus 45. Even without a new positive, votes remaining in the window can refresh this expiry.
- An already-active staff ID can also refresh its expiry with one new tag-positive frame. Status expires after the stored expiry frame; it is not permanent for the whole video.
- A new ID starts without the previous ID's evidence. There is no cross-ID identity stitching. If an ID transfers to another person, its retained staff status can transfer too until it expires.
- The displayed `Tag` value is the **highest accepted score in the retained history**, not necessarily the current frame's score. A high displayed value therefore does not prove the tag is currently visible.
- A staff box is drawn only when the person is currently returned with a usable tracked crop. Holding staff status does not reconstruct a missing person box.
- `Persons` and `Staff` are current-frame counts. The final console list contains all IDs ever confirmed, not a count of unique real staff members.

## Project layout

Documented paths are listed below. This is not a filesystem scan of the Windows repository; experiment locations and additional utility filenames should be checked locally before submission.

| Path | Purpose |
|---|---|
| `detect_staff.py` | Main ordinary-box person/tag inference pipeline |
| `README.md`, `requirements.txt` | Project guide and environment dependencies |
| `trackers/custom_bytetrack.yaml` | Default tracker configuration; retain the version used for the accepted run |
| `yoloModel/best_v1.2.pt` | Current default person weights |
| `yoloModel/best_tag_v1.2.pt` | Current default tag weights |
| `tools/` | Dataset preparation, label validation, balancing, mining and training utilities |
| `data/sample.mp4` | Supplied demonstration video |
| `data/person_dataset/` | Person images and YOLO labels |
| `data/tag_dataset*/` | Original and versioned tag datasets, manifests and dataset YAML files |
| `data/cepdof_yolo/` | Prepared CEPDOF experimental data, including the OBB branch |
| `experiment/` | Separate experimental scripts/results; not the main inference entry point |
| `outputs/` | Annotated videos, training runs and diagnostics |
| `notes/FCC AI Evaluation - Engineering Notes.md` | Detailed design and experiment history |

The external CEPDOF source was kept beside this repository at `FCC/CEPDOF`, while the repository was at `FCC/FCC-Staff-Detection`.

## Setup

Use the existing working virtual environment when reproducing the accepted result. For a fresh environment, use a Python version supported by the chosen PyTorch/Ultralytics packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the video at `data/sample.mp4`, provide both trained weights in `yoloModel/`, and retain `trackers/custom_bytetrack.yaml`. Model weights and the complete environment are not bundled with this README. A CUDA-capable GPU also needs a compatible CUDA-enabled PyTorch installation; otherwise `auto` uses the CPU.

Before final submission, retain the working dependency versions, tracker YAML, exact run command and selected weights so the result can be reproduced on another computer.

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

Show all returned person boxes as thin red boxes with small IDs below them:

```powershell
python detect_staff.py --person-show
```

Staff retain the red person ID below the box, while the thicker green staff box is drawn over the red box. Without this flag, only staff boxes are drawn. Press `q` in the preview to stop early; the saved video then contains only the processed portion.

The annotated result is written to `outputs/staff_detected.mp4`. Use a different `--output` filename to preserve previous results.

### Staff coordinates and CSV

Each displayed staff box has green `X` and `Y` labels near its lower-right corner, with X above Y. The text moves inward at image edges. Coordinates use the person's bottom centre: `x = (x1 + x2) / 2`, `y = y2`, in original-frame pixels, rounded to one decimal place. The origin is the image's top-left corner; x increases rightwards and y downwards. These are image positions, not calibrated floor/world coordinates.

By default, the CSV shares the video's output name: `outputs/staff_detected.csv`. Override its location with:

```powershell
python detect_staff.py --csv-output outputs/staff_frames.csv
```

The columns, in order, are `staff ID,frame,x,y,time`. There is one row per displayed staff ID per frame, including frames where staff status is held without a fresh tag detection. Multiple staff in one frame produce multiple rows; frames with no displayed staff produce none. If no staff are confirmed, the CSV contains only its header.

Frame numbering starts at **1**; frame 1 has time `00:00:000`. Time is `MM:SS:mmm`, with three millisecond digits, computed as `(frame - 1) / source_fps`. At 25 FPS, frame 26 is `00:01:000`. It is video time, not processing time. This follows the source FPS timeline (appropriate for the supplied constant-frame-rate video), not per-frame presentation timestamps from a variable-frame-rate source. Minutes may exceed 59.

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
| `--vote-window`, `--vote-min` | `15`, `3` |
| `--staff-hold` | `45` frames |
| `--person-show` | Off |
| `--csv-output` | Same path as `--output`, with a `.csv` extension |

These are the uploaded script's defaults, not proof of the command-line overrides used in a particular video. Earlier `--tag-conf 0.7` trials are historical experiments, not the current default. Tracker association thresholds and buffer duration are configured separately in the YAML.

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
4. Audit tag visibility/size, mine training-only expansion candidates, annotate new positives and create a separate dataset version.
5. `train_tag_model.py`: train the separate tag detector, keeping the person-model experiment independent.
6. `mine_neg_tag_training_crops.py`: review difficult tag predictions from training data and add only confirmed negatives to a later training version. Do not mine validation/test examples into training.

### Recorded tag dataset checkpoints

| Dataset checkpoint | Training | Validation | Test |
|---|---|---|---|
| Original annotated crops | 26 positive + 1,776 negative | 7 positive + 565 negative | 5 positive + 606 negative |
| First balanced set | 26 positive + 104 negative | Unchanged | Unchanged |
| Positive expansion, balanced v2 | 43 positive + 172 negative | Unchanged | Unchanged |

The expansion added 17 labelled positives. These are historical counts, not a verified count of the later hard-negative/v1.2 dataset. Keep the final dataset's selection manifest for the report.

The initial tag baseline was ineffective. The subsequent supplied v1.1 plots showed mAP50 of 0.496 and a confusion matrix with 4 true positives, 2 false positives and 3 false negatives at its evaluation settings. These are small-validation-set results, **not** final v1.2 metrics or unknown-video accuracy. Validation has only seven positive images and test only five; report raw counts alongside percentages.

Separate CEPDOF ordinary-box and OBB preparation/training experiments were also explored. The ordinary-box run was reported to reach a best mAP50-95 of 0.5235 at epoch 83, after which keeping the best checkpoint was recommended. This is an experimental person-detector result, not an end-to-end staff-detection score; verify against its saved logs before citing it in the report.

### Dataset path troubleshooting

Moving or renaming a dataset folder does not update paths inside `tag_data.yaml`. The earlier `tag_dataset_balanced_v1.2` training error still pointed internally at `tag_dataset_balanced_v2/images/val`.

Ensure the YAML `path` points to the actual dataset root on the current computer and that `train`, `val`, and `test` resolve to existing image folders. An absolute root using forward slashes avoids ambiguity between machines; update it when moving the project. Changing the global Ultralytics download directory is not the fix for a stale dataset root.

The `tools/` folder is intended to be tracked by Git. Dataset images, labels, videos, model weights and generated outputs remain local unless the repository policy is changed deliberately. Do not regenerate or overwrite accepted datasets merely to write the report.

## Important limitation

The accepted result is a proof of concept. It is not guaranteed to detect every staff occurrence or generalize to an unknown interview video. Training on one scene can still learn scene-specific cues even when using a tag detector.

Occlusion, changing appearance, tiny people near image edges and hidden tags can cause missed detections or new IDs. Better person detection, better tag detection and better tracking address different parts of this problem. Temporal holding reduces short gaps but can retain incorrect status after an ID transfer; stricter tag thresholds can reduce false positives while increasing missed staff periods.

The current pipeline exports bottom-centre coordinates and staff-occurrence rows, but remains one-pass: it cannot annotate or export earlier frames retrospectively after later staff confirmation. Missing detections and unconfirmed IDs also produce no staff rows. Consequently, CSV export alone does not guarantee that every true staff occurrence is captured. Two-pass recovery remains deferred.

For report writing, use the frozen main script and accepted output as the implemented system. Separate experimental branches and proposed features from completed work. Preserve the final command, selected weights, tracker configuration, dataset manifests and actual evaluation logs. Final v1.2 metrics, final dataset counts and the exact local repository inventory still need those local records; do not substitute the subjective 80/100 assessment.

For a future Codex session, provide this README, the engineering notes and relevant experiment logs in the repository, then ask it to read them before working. Treat the latest code and recorded run configuration as authoritative when older chat recommendations differ. Development remains frozen unless explicitly reopened.
