# How it was built

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

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | pinned, reproducible, TensorFlow wheels available for Apple silicon |
| Model or core logic | TensorFlow 2.21 / Keras 3, MobileNetV2 ImageNet backbone (frozen) + Dense(256)/Dropout(0.5)/softmax head | the notebook's head on a backbone that trains on a laptop CPU in seconds instead of VGG16's hours |
| Interface | FastAPI + one HTML page (Jinja2, no build step) | upload or click a sample, top-3 with bars, live metrics; shares `common/ui.css` |
| Data | 10,235 training JPEGs in 10 class folders, 1,473 test images in a separate folder (154 MB, read in place) | the course dataset; corrupt files detected and excluded |
| Tests | pytest + FastAPI TestClient, 9 tests, offline, 3 s | health contract, metrics, predictions for three classes, bad input, gallery page |

## Libraries
`tensorflow`/`keras` (backbone, head, training, saving `.keras`), `pillow` (decode, corrupt-image check, resize),
`numpy` (image arrays, per-class accuracy, confusion matrix), `matplotlib` (accuracy and loss curves),
`fastapi` + `uvicorn` + `python-multipart` + `jinja2` (the service and page), `pytest` + `httpx` (tests),
`ruff` (lint), `pandas` (installed for parity with the other capstones; not needed by this one).
