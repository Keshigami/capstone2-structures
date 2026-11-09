---
title: Historical structures classifier
group: capstone
role: Computer Vision
port: 8102
status: working
one_liner: Upload a photo of a church or monument detail and a transfer-learned CNN names which of ten architectural elements it is, with a confidence.
---
# Historical structures classifier

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (75 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8102/.

Two-minute walkthrough:
1. Capstone 2, part 1: ten classes of heritage structure; pick a gallery sample
2. Altar: one click classifies it
3. Top-3 bars: the class and its confidence
4. Apse
5. Bell tower
6. Model numbers: test accuracy, best validation accuracy, epochs
7. Train versus validation curves; the dashed line is where augmentation switches on
8. Per-class test accuracy
<!-- /demo:auto -->

## Brief
A government tourism agency wants to monitor centuries-old buildings and flag the ones that need maintenance.
Step one is knowing what a photo shows: an altar, an apse, a bell tower, a column, the inside or outside of a dome,
a flying buttress, a gargoyle, stained glass or a vault. The task (Caltech CTME capstone 2, part 1) is a TensorFlow
model that predicts that category from an image, built with transfer learning because ten thousand photos is not
enough to train a CNN from scratch. "Solved" is a model that gets most held-out test photos right and a way for a
non-engineer to try it.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | pinned, reproducible, TensorFlow wheels available for Apple silicon |
| Model or core logic | TensorFlow 2.21 / Keras 3, MobileNetV2 ImageNet backbone (frozen) + Dense(256)/Dropout(0.5)/softmax head | the notebook's head on a backbone that trains on a laptop CPU in seconds instead of VGG16's hours |
| Interface | FastAPI + one HTML page (Jinja2, no build step) | upload or click a sample, top-3 with bars, live metrics; shares `common/ui.css` |
| Data | 10,235 training JPEGs in 10 class folders, 1,473 test images in a separate folder (154 MB, read in place) | the course dataset; corrupt files detected and excluded |
| Tests | pytest + FastAPI TestClient, 9 tests, offline, 3 s | health contract, metrics, predictions for three classes, bad input, gallery page |

## Libraries used
`tensorflow`/`keras` (backbone, head, training, saving `.keras`), `pillow` (decode, corrupt-image check, resize),
`numpy` (image arrays, per-class accuracy, confusion matrix), `matplotlib` (accuracy and loss curves),
`fastapi` + `uvicorn` + `python-multipart` + `jinja2` (the service and page), `pytest` + `httpx` (tests),
`ruff` (lint), `pandas` (installed for parity with the other capstones; not needed by this one).

## How it was built
1. Read the two course notebooks and the problem statement. The notebook plots 8 images per class with OpenCV,
   removes truncated images with PIL (`Image.open` + `load`, `os.remove` on failure), loads VGG16 without its top,
   freezes it, adds Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(10, softmax), compiles with Adam and
   categorical cross-entropy, stops early on `val_accuracy`, trains at 224px without augmentation and then with
   it (rotation 20°, shifts 0.2, shear 0.2, zoom 0.2, flip), and plots the curves. It used the test folder as the
   validation set.
2. Data scan (`src/structures/data.py`): same readability check, but the source folders are never written to;
   unreadable files are listed and skipped (3 in the test set, 0 in training) and counts per class are logged
   and saved to `models/dataset_scan.json`. Images are resized once into a cached array under `data/capstone2/`.
3. Split: 15% of the training folder held out per class for validation; the test folder is used only once, at
   the end. This is the main change from the notebook, which validated on the test set.
4. Model (`src/structures/model.py`): the notebook's head, optimiser, loss and callbacks kept; backbone switched
   to MobileNetV2 (EfficientNetB0 selectable) with the input scaling inside the model. Added the problem
   statement's "stop at a chosen validation accuracy" callback next to EarlyStopping. Augmentation is the
   notebook's ImageDataGenerator settings as Keras layers, wrapped around the model only while training.
5. Training (`src/train.py`): `--quick` (150 images per class, 160px, 2 plain + 2 augmented epochs) and `--full`
   (everything, 224px, up to 10 + 10 with early stopping). Both phases are run as in the notebook; the weights
   with the best validation accuracy across phases are saved. Outputs: model, classes, metrics (train/val/test,
   per class, confusion matrix, epoch history, wall time), curves, and one test image per class for the gallery.
6. Service (`app.py`): `/`, `/predict` (file upload or `sample=<class>`), `/health`, `/metrics`, `/curves.png`.
7. Tests and `justfile`: `setup`, `train`, `train-full`, `run`, `up`, `down`, `test`, `demo`.

## Results
Quick mode on this machine (M4 Max, CPU), `models/metrics-quick.json`:

| Number | Value | Meaning |
|---|---|---|
| Test accuracy | 89.3% | 1,473 never-seen images from the separate test folder; chance is 10% |
| Best validation accuracy | 93.6% | 220 held-out training images, baseline phase, epoch 2 (the saved weights) |
| Train accuracy at that epoch | 89.6% | below validation because dropout is on while training |
| Per class (test) | flying_buttress 98.7, altar 97.1, dome(outer) 96.4, gargoyle 92.0, stained_glass 92.0, dome(inner) 90.7, bell_tower 90.0, vault 89.0, column 79.0, apse 52.6 | apse (57 test images) is confused with bell_tower and column |
| Cost | 1,280 training images, 4 epochs, 13 s training, 18 s wall time | 150 images per class is enough for a frozen backbone to reach ~90% |

The notebook reached 0.90 validation accuracy with VGG16 on all images (validating on the test folder). A good
number for this dataset with a fine-tuned backbone is mid-90s; the quick model is within a few points of the
notebook on 15% of the data.

Full mode (`just train-full`, the model now deployed), `models/metrics-full.json`:

| Number | Value | Meaning |
|---|---|---|
| Test accuracy | **93.3%** | same 1,473 test images; the notebook reported 0.90 with VGG16 |
| Best validation accuracy | 95.6% | 1,535 held-out training images, baseline phase, epoch 4 (the saved weights) |
| Per class (test) | dome(outer) 98.8, gargoyle 97.1, stained_glass 96.9, altar 95.7, bell_tower 95.3, column 95.2, vault 92.1, dome(inner) 90.7, flying_buttress 78.2, apse 61.4 | apse is still the weak class; flying buttresses are often photographed with the whole facade |
| Cost | 8,700 training images, 14 epochs (early stopping in both phases), 12.2 min on CPU | augmentation peaked at 93.6% validation, below the plain phase, so the plain weights were kept |

## Demo details
- Run: `just setup && just up`, open `http://localhost:8102/` (`just train` rebuilds a quick model in ~25 s; `just train-full` the full one in ~12 min).
- Two-minute walkthrough: (1) click `gargoyle` in the sample gallery; the result card shows the image, the predicted
  class, its confidence and the top-3 bars, with a pill confirming the test label. (2) Click `apse`, then scroll to
  the per-class table: apse is the weak class (61.4% full, 52.6% quick), the gallery images are the model's most confident correct
  test images, so the table is the honest number. (3) Drop any photo of a cathedral window, a dome or a gargoyle
  from your own files onto the page; it classifies on drop. (4) Look at the curves: the dashed line is where augmentation switches on; validation accuracy
  drops after it in both modes, so the script kept the plain-phase weights.
- Video clip: `demo/clips/capstone2-structures.webm` (recorded by `just demo-video` at the root; coordinator owns it).
- Sample inputs: `static/samples/<class>.jpg`, one per class, taken from the test set. Expected result: the
  matching class at >99% for all ten (they were selected as confident correct predictions).
- Script hooks: `#input` (file input), `#run` (Classify button), `#result` (result container); gallery buttons
  carry `data-sample="<class>"` and classify on click without an upload.

## Limits and next steps
- `just train` (quick, 25 s) rebuilds a 15%-of-data demo model at 160px and overwrites the deployed full model;
  `just train-full` (12 min) restores the 93.3% one. Both metrics files are kept side by side.
- Augmentation hurts in a short run on a frozen backbone with a Flatten head; the script keeps the better phase.
  A production version would fine-tune the top blocks at a low learning rate and use global average pooling.
- Apse and column are weak; they need more data or a backbone fine-tune, and a confusion-matrix view in the UI.
- Validation here is split from training data, so the numbers are not directly comparable to the notebook's 0.90,
  which validated on the test set.
- CPU inference (~50 ms per image) and a single process: fine for a demo, not for a queue of maintenance photos.
