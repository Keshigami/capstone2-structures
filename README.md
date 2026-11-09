<!-- public-top -->
# Historical structures classifier

Upload a photo of a church or monument detail and a transfer-learned CNN names which of ten architectural elements it is, with a confidence. Caltech CTME PGC AIML, capstone.

**Watch it first:** [the 90-second film](demo/film.mp4) (what it does, how it is built, the proof, the answer) or the [60-second cut](demo/film-60s.mp4). The plain walkthrough with captions is [demo/demo.mp4](demo/demo.mp4).

## Try it in three commands

```bash
git clone https://github.com/Keshigami/capstone2-structures.git && cd capstone2-structures
just setup        # creates the venv with uv (Python 3.11)
just up           # http://127.0.0.1:8102/  (just down to stop)
```

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (`brew install uv just` on a Mac). The trained model is not in the repo (the trained model is 193 MB, over GitHub's file limit); `just train` rebuilds it from the Caltech `Capstone 2` dataset, which you point at with `DATA_ROOT`. No API key is needed anywhere in this project. `just test` runs the lint and the offline tests.

Step by step: [docs/Getting-Started.md](docs/Getting-Started.md), then [how it was built](docs/How-It-Was-Built.md), [results](docs/Results.md), [limits](docs/Limits-and-Next-Steps.md) and [the film](docs/The-Film.md). The same pages are on the [wiki](https://github.com/Keshigami/capstone2-structures/wiki).
<!-- /public-top -->
# capstone2-structures

## Demo walkthrough
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

Transfer-learning image classifier for ten kinds of historical architecture (altar, apse, bell tower, column,
inner dome, outer dome, flying buttress, gargoyle, stained glass, vault). Caltech CTME PGC AIML capstone 2,
part 1, rebuilt as a runnable service: a training script with a quick and a full mode, a FastAPI app on
port 8102 with an upload page and a click-to-classify sample gallery, and offline tests.

## What it does

- `src/train.py` scans the dataset, drops unreadable images the way the notebook did (PIL open + load, counts
  logged and written to `models/dataset_scan.json`), trains a frozen ImageNet backbone with the notebook's head,
  evaluates on the separate test folder and writes `models/structures.keras`, `classes.json`, `metrics.json`,
  `curves.png`, plus one test image per class into `static/samples/` for the gallery.
- `src/predict.py <image>` prints the top-3 classes with probabilities.
- `app.py` serves `GET /` (upload or click a sample, top-3 with bars, the model's numbers and curves),
  `POST /predict` (multipart `file` or form field `sample=<class>`), `GET /health`, `GET /metrics`, `GET /curves.png`.

## Data

Read in place from `~/Caltech CTME/PGC AIML - Capstone/Datasets/Capstone 2/Part 1/dataset_hist_structures 2/
dataset_hist_structures/` (override with `STRUCTURES_DATA=<that folder>` or `DATA_ROOT=<the Datasets folder>`).
Nothing is written there; resized-image caches and scan results go to `../data/capstone2/`.

| Split | Folder | Images | Notes |
|---|---|---|---|
| train + val | `Stuctures_Dataset/<class>/` | 10,235 | 407 (flying_buttress) to 1,919 (column) per class, 0 corrupt |
| test | `Dataset_test/Dataset_test_original_1478/<class>/` | 1,473 | 1,477 files; 3 unreadable JPEGs dropped (2 gargoyle, 1 stained_glass), 1 `.bat` skipped |

The notebook used the test folder as its validation set. Here 15% of the training folder is held out
(stratified, seed 42) for validation and early stopping; the test folder is only touched for the final numbers.

## Model

The notebook: VGG16 (ImageNet, frozen) -> Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(10, softmax),
Adam, categorical cross-entropy, EarlyStopping on `val_accuracy` (patience 3), 224px, first without augmentation
then with (rotation 20°, shifts 0.2, shear 0.2, zoom 0.2, horizontal flip). Its reported validation accuracy was
0.90 (VGG16, 10 epochs, Colab GPU).

Here the head, optimiser, loss, callbacks, two-phase schedule and augmentation settings are the same; the
backbone is MobileNetV2 (`--backbone efficientnetb0` also works) so a full run fits on a laptop CPU. A custom
`StopAtValAccuracy` callback (problem-statement task 6, threshold `--stop-at 0.97`) runs alongside EarlyStopping.
Preprocessing (`Rescaling` to [-1, 1]) is inside the saved model, so it takes raw RGB pixels. The weights with
the best validation accuracy across both phases are the ones saved, and `metrics.json` says which phase and
epoch that was.

| Mode | Train images | Size | Epochs | Target |
|---|---|---|---|---|
| `--quick` (default) | 150 per class, 1,280 train + 220 val | 160px | 2 plain + 2 augmented | under a minute |
| `--full` | all 10,235 (8,700 train + 1,535 val) | 224px | up to 10 + 10, early stopping | reproduce the notebook |

## Numbers achieved (this machine, M4 Max, CPU only)

Quick mode, from `models/metrics-quick.json`:

| Metric | Value |
|---|---|
| Test accuracy (1,473 images, untouched test folder) | **89.3%** |
| Best validation accuracy (220 held-out training images) | 93.6% (baseline phase, epoch 2) |
| Train accuracy at that epoch | 89.6% |
| Epochs | 4 (2 plain + 2 augmented), 12.8 s training, 18 s wall time with warm caches (25 s cold) |

Per-class test accuracy: flying_buttress 98.7, altar 97.1, dome(outer) 96.4, gargoyle 92.0, stained_glass 92.0,
dome(inner) 90.7, bell_tower 90.0, vault 89.0, column 79.0, apse 52.6. Apse (57 test images) is mostly confused
with bell_tower and column; from the outside an apse is just a round wall with a roof.

In quick mode the augmented phase lowers validation accuracy (0.936 -> 0.891 -> 0.800): two epochs of strong
augmentation on top of a frozen backbone with a spatial Flatten head is not enough to converge. The script keeps
the baseline weights in that case, and `curves.png` shows the dip. The original notebook saw the same thing at
its scale (0.904 without augmentation, 0.897 with).

Full mode, from `models/metrics-full.json` (this is the model currently in `models/structures.keras`):

| Metric | Value |
|---|---|
| Test accuracy (1,473 images) | **93.3%** (loss 0.282) |
| Best validation accuracy (1,535 held-out training images) | 95.6% (baseline phase, epoch 4; these weights were kept) |
| Train accuracy at that epoch | 93.1% |
| Epochs | 14 (7 plain + 7 augmented; EarlyStopping with patience 3 ended both phases), 12.2 min wall time on CPU |

Per-class test accuracy: dome(outer) 98.8, gargoyle 97.1, stained_glass 96.9, altar 95.7, bell_tower 95.3,
column 95.2, vault 92.1, dome(inner) 90.7, flying_buttress 78.2, apse 61.4. Validation accuracy per epoch:
0.921 0.934 0.954 0.956 0.954 0.946 0.951 | 0.907 0.917 0.917 0.936 0.928 0.923 0.932. The augmented phase
again peaked below the baseline (0.936 vs 0.956), in line with the notebook's own result (0.897 vs 0.904).
Against the notebook's VGG16 at 0.90: 93.3% on the test folder with a backbone 6x smaller, on a laptop CPU.
The 202 MB model file is mostly the 16M-parameter Flatten->Dense(256) layer plus its Adam state.

## How to run

```sh
just setup        # uv sync, copy the shared stylesheet, train a quick model if none exists
just train        # quick model (default); just train-full for everything at 224px
just test         # ruff + pytest (TestClient, offline; needs a trained model)
just run          # foreground server on http://127.0.0.1:8102
just up / down    # background server, pid in .server.pid, log in .server.log
just demo         # CLI prediction on three samples, then /health, /predict (upload and sample), /metrics
uv run python src/predict.py static/samples/gargoyle.jpg --top 3
curl -s -F file=@static/samples/altar.jpg localhost:8102/predict
curl -s -F sample=vault localhost:8102/predict
```

Requirements: `uv`, `just`, Python 3.11 (pinned in `.python-version`), the dataset folder above. The first
training run downloads MobileNetV2 ImageNet weights (9 MB) into `~/.keras`; everything after that is offline.

## Two-minute walkthrough for an interviewer

1. `just up`, open http://localhost:8102/. The header shows the backbone, input size and mode; the right-hand
   card is empty until you act.
2. Click `gargoyle` in the sample gallery. The result card shows the image, the predicted class with its
   confidence, and the top-3 bars; the pill says whether it matched the test label.
3. Click `apse`: still right, but scroll to the per-class table and note apse at 61.4% (full) or 52.6% (quick) on
   the test set, the one weak class. The gallery images were chosen as the model's most confident correct test images, so the
   gallery is a best case; the table is the honest number.
4. Drag in any photo of a church window or a gargoyle from your own files; the page classifies on drop.
5. `curl -s localhost:8102/metrics | jq .epochs` and `models/curves.png`: the dashed line is where augmentation
   switches on; validation drops after it, so the plain-phase weights were kept.
6. `just train` (25 s) shows the whole pipeline end to end on a subset; `just train-full` (12 min) rebuilds the 93.3% model.

## Tests

`tests/test_app.py` runs against the FastAPI app with the trained model through `TestClient`: `/health`
contract, `/metrics` sanity, `/predict` for three classes by upload (known class, three rows, probabilities sum to
1 over ten classes), `/predict` by sample name, a mostly-correct check on four gallery samples, bad input returns
400/404, and the page lists ten samples with the `#input`/`#run`/`#result` ids. Offline, about 3 s.

## Limits

- Quick mode trains on 15% of the data at 160px and is what `just train` rebuilds in 25 s; the full-mode numbers
  above are the ones to quote, and the full model is the one deployed until `just train` overwrites it.
- Apse and column are the weak classes; a fine-tuning phase (unfreeze the top of the backbone, low learning
  rate) and `GlobalAveragePooling2D` instead of `Flatten` would be the first two things to try.
- The notebook's validation set was the test folder; here validation is split from training data, so the
  numbers are not directly comparable to the notebook's 0.90.
- CPU only; no `tensorflow-metal`. Inference is ~50 ms per image, fine for a demo, not for batch throughput.
- One model file (~40 MB, git-ignored); `just train` recreates it in under a minute.

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Dataset: the Caltech CTME capstone data is not in this repo. Point `DATA_ROOT` at the `Datasets` folder (`Capstone 1`, `Capstone 2`) before `just train`.
Not included (rebuilt by `just train`): models/structures.keras (192 MB, rebuild with `just train`)
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
