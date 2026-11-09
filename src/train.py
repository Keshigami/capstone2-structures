"""Train the classifier.

  uv run python src/train.py --quick   # <=150 images/class, 160px, 2 baseline + 2 augmented epochs (minutes on CPU)
  uv run python src/train.py --full    # all 10,235 images, 224px, up to 10 + 10 epochs with early stopping

Writes models/structures.keras, classes.json, metrics.json, curves.png, dataset_scan.json and one sample
image per class into static/samples/ for the web gallery.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from structures import (  # noqa: E402
    CLASSES_PATH,
    CURVES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    SAMPLES_DIR,
    SCAN_PATH,
    slug,
)
from structures import data as D  # noqa: E402

log = logging.getLogger("structures.train")

MODES = {
    "quick": {"per_class": 150, "image_size": 160, "epochs_baseline": 2, "epochs_aug": 2, "batch_size": 32},
    "full": {"per_class": None, "image_size": 224, "epochs_baseline": 10, "epochs_aug": 10, "batch_size": 64},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="small subset, 160px, 4 epochs (default)")
    g.add_argument("--full", action="store_true", help="all images, 224px, early stopping")
    ap.add_argument("--backbone", default="mobilenetv2", choices=("mobilenetv2", "efficientnetb0"))
    ap.add_argument("--per-class", type=int, help="override max training images per class")
    ap.add_argument("--image-size", type=int, help="override input size")
    ap.add_argument("--epochs-baseline", type=int, help="override epochs without augmentation")
    ap.add_argument("--epochs-aug", type=int, help="override epochs with augmentation")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--val-fraction", type=float, default=0.15, help="held-out share of the training set")
    ap.add_argument("--stop-at", type=float, default=0.97, help="stop when val_accuracy reaches this")
    ap.add_argument("--patience", type=int, default=3, help="EarlyStopping patience on val_accuracy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true", help="rescan the folders and reload images")
    args = ap.parse_args(argv)
    args.mode = "full" if args.full else "quick"
    for k, v in MODES[args.mode].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    t0 = time.time()
    log.info("mode=%s backbone=%s image_size=%d per_class=%s epochs=%d+%d batch=%d",
             args.mode, args.backbone, args.image_size, args.per_class, args.epochs_baseline,
             args.epochs_aug, args.batch_size)

    # 1. Scan both folders, drop corrupt images, log counts per class.
    train_scan = D.scan_directory(D.train_dir(), use_cache=not args.no_cache)
    test_scan = D.scan_directory(D.test_dir(), use_cache=not args.no_cache)
    classes = train_scan.classes
    if test_scan.classes != classes:
        raise SystemExit(f"class folders differ between train {classes} and test {test_scan.classes}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_PATH.write_text(json.dumps({"train": train_scan.summary(), "test": test_scan.summary()}, indent=2))

    # 2. Subsample (quick mode) and split train/val stratified. The test folder is only used at the end.
    selected = D.limit_per_class(train_scan.files, args.per_class, args.seed)
    train_files, val_files = D.stratified_split(selected, args.val_fraction, args.seed)
    train_paths, y_train = D.flatten(train_files, classes)
    val_paths, y_val = D.flatten(val_files, classes)
    test_paths, y_test = D.flatten(test_scan.files, classes)
    log.info("images: train=%d val=%d test=%d", len(train_paths), len(val_paths), len(test_paths))

    # 3. Load and resize images (cached as .npy under data/capstone2/).
    s = args.image_size
    x_train = D.load_arrays(train_paths, s, f"train-{args.mode}", use_cache=not args.no_cache)
    x_val = D.load_arrays(val_paths, s, f"val-{args.mode}", use_cache=not args.no_cache)
    x_test = D.load_arrays(test_paths, s, "test", use_cache=not args.no_cache)

    import keras
    import tensorflow as tf

    from structures.model import build_model, callbacks, with_augmentation

    keras.utils.set_random_seed(args.seed)
    n = len(classes)
    y_train_oh = keras.utils.to_categorical(y_train, n)
    y_val_oh = keras.utils.to_categorical(y_val, n)

    def dataset(x, y, shuffle: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train = dataset(x_train, y_train_oh, shuffle=True)
    ds_val = dataset(x_val, y_val_oh, shuffle=False)

    # 4. Frozen backbone + the notebook's head.
    model = build_model(n, s, backbone=args.backbone)
    model.summary(print_fn=log.info)

    # 5. Phase 1: no augmentation. Phase 2: continue with augmentation (as in the notebook).
    history: dict[str, list[float]] = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    phases: list[dict] = []
    best: tuple[float, list, str] = (-1.0, [], "")  # (val_accuracy, weights, phase) across both phases
    t_train = time.time()
    for phase, epochs, trainee in (
        ("baseline", args.epochs_baseline, model),
        ("augmented", args.epochs_aug, with_augmentation(model, s)),
    ):
        if epochs <= 0:
            continue
        log.info("phase %s: up to %d epochs", phase, epochs)
        cbs = callbacks(args.stop_at, args.patience)
        h = trainee.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=cbs, verbose=2, shuffle=False)
        for k in history:
            history[k] += [float(v) for v in h.history[k]]
        phase_best = max(h.history["val_accuracy"])  # EarlyStopping restored these weights
        phases.append({"phase": phase, "epochs": len(h.history["loss"]), "best_val_accuracy": float(phase_best),
                       "stopped_at_threshold": cbs[0].triggered_epoch is not None})
        if phase_best > best[0]:
            best = (float(phase_best), model.get_weights(), phase)
    # Keep the best validation checkpoint across phases (augmentation can end below the baseline on short runs).
    model.set_weights(best[1])
    log.info("kept weights from phase %s (val_accuracy %.4f)", best[2], best[0])
    train_seconds = time.time() - t_train

    # 6. Evaluate on the untouched test folder.
    probs = model.predict(x_test.astype(np.float32), batch_size=args.batch_size, verbose=0)
    pred = probs.argmax(axis=1)
    test_acc = float((pred == y_test).mean())
    y_test_oh = keras.utils.to_categorical(y_test, n)
    test_loss = float(keras.losses.categorical_crossentropy(y_test_oh, probs).numpy().mean())
    per_class = {c: float((pred[y_test == i] == i).mean()) for i, c in enumerate(classes)}
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(y_test, pred, strict=True):
        confusion[t, p] += 1
    log.info("test accuracy %.4f (loss %.4f)", test_acc, test_loss)
    for c, a in per_class.items():
        log.info("  %-16s %.3f", c, a)

    # 7. Save artifacts. train_accuracy is reported at the epoch whose weights were kept.
    best_epoch = history["val_accuracy"].index(best[0])
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps({"classes": classes, "image_size": s, "backbone": args.backbone}, indent=2))
    metrics = {
        "mode": args.mode,
        "backbone": args.backbone,
        "image_size": s,
        "batch_size": args.batch_size,
        "classes": classes,
        "images": {
            "train": len(train_paths), "val": len(val_paths), "test": len(test_paths),
            "per_class_train": {c: len(v) for c, v in train_files.items()},
            "per_class_test": test_scan.counts(),
            "available_train": train_scan.total(),
            "corrupt_removed": {"train": train_scan.corrupt_total(), "test": test_scan.corrupt_total()},
        },
        "epochs": {"baseline": next((p["epochs"] for p in phases if p["phase"] == "baseline"), 0),
                   "augmented": next((p["epochs"] for p in phases if p["phase"] == "augmented"), 0),
                   "total": len(history["loss"]), "phases": phases},
        "train_accuracy": history["accuracy"][best_epoch],
        "train_accuracy_final": history["accuracy"][-1],
        "val_accuracy": best[0],
        "best_epoch": best_epoch + 1,
        "val_accuracy_final": history["val_accuracy"][-1],
        "selected_phase": best[2],
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "per_class_test_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
        "history": history,
        "wall_time_seconds": round(time.time() - t0, 1),
        "train_seconds": round(train_seconds, 1),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    plot_curves(history, phases, CURVES_PATH)
    # Keep a per-mode copy so quick and full numbers can both be quoted.
    shutil.copyfile(METRICS_PATH, MODELS_DIR / f"metrics-{args.mode}.json")
    shutil.copyfile(CURVES_PATH, MODELS_DIR / f"curves-{args.mode}.png")
    copy_samples(test_scan, classes, probs, y_test, test_paths)
    log.info("saved %s, %s, %s, %s in %.0fs", MODEL_PATH.name, CLASSES_PATH.name, METRICS_PATH.name,
             CURVES_PATH.name, metrics["wall_time_seconds"])
    return 0


def plot_curves(history: dict[str, list[float]], phases: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0c1018")
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.set_facecolor("#121826")
        ax.plot(epochs, history[key], marker="o", color="#4fd1c5", label="train")
        ax.plot(epochs, history["val_" + key], marker="o", color="#f2b84b", label="validation")
        if phases and phases[0]["phase"] == "baseline" and len(phases) > 1:
            ax.axvline(phases[0]["epochs"] + 0.5, color="#7c869d", linestyle="--", linewidth=1)
            ax.text(phases[0]["epochs"] + 0.6, ax.get_ylim()[0], "augmentation on", color="#aab3c8", fontsize=8,
                    va="bottom")
        ax.set_title(title, color="#e8ecf6")
        ax.set_xlabel("epoch", color="#aab3c8")
        ax.tick_params(colors="#aab3c8")
        for spine in ax.spines.values():
            spine.set_color("#3a4457")
        ax.legend(facecolor="#182033", edgecolor="#3a4457", labelcolor="#e8ecf6")
        ax.grid(color="#2a3346", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def copy_samples(test_scan: D.ScanResult, classes: list[str], probs: np.ndarray, y_test: np.ndarray,
                 test_paths: list[str]) -> None:
    """One gallery image per class, taken from the test set: the model's most confident correct one
    among the first 20 files of the class (falls back to the first file if none is correct)."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES_DIR.glob("*.jpg"):
        old.unlink()
    index = {p: i for i, p in enumerate(test_paths)}
    chosen: dict[str, str] = {}
    for ci, c in enumerate(classes):
        candidates = test_scan.files[c][:20]
        best, best_p = candidates[0], -1.0
        for p in candidates:
            i = index[p]
            if probs[i].argmax() == ci and probs[i, ci] > best_p:
                best, best_p = p, float(probs[i, ci])
        chosen[c] = best
        target = SAMPLES_DIR / f"{slug(c)}.jpg"
        if Path(best).suffix.lower() in (".jpg", ".jpeg"):
            shutil.copyfile(best, target)
        else:
            from PIL import Image

            with Image.open(best) as im:
                im.convert("RGB").save(target, "JPEG", quality=90)
    (SAMPLES_DIR / "samples.json").write_text(json.dumps(
        {slug(c): {"class": c, "file": f"{slug(c)}.jpg", "source": Path(p).name} for c, p in chosen.items()},
        indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
