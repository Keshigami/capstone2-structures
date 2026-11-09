"""Dataset scan (with corrupt-image removal), sampling, split and array loading.

The source folders under ~/Caltech CTME are read in place and never written to. The notebook deleted
unreadable files with os.remove; here they are detected the same way (PIL open + load) and excluded.
Derived files (array caches) go under <repo>/data/capstone2/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from structures import PROJECT_DIR

log = logging.getLogger("structures.data")

DERIVED_DIR = PROJECT_DIR.parent / "data" / "capstone2"
DEFAULT_DATASETS = Path.home() / "Caltech CTME" / "PGC AIML - Capstone" / "Datasets"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def dataset_root() -> Path:
    """Folder that contains Stuctures_Dataset/ and Dataset_test/.

    Override with STRUCTURES_DATA (that folder) or DATA_ROOT (the Caltech Datasets folder).
    """
    if os.environ.get("STRUCTURES_DATA"):
        return Path(os.environ["STRUCTURES_DATA"]).expanduser()
    root = Path(os.environ.get("DATA_ROOT", DEFAULT_DATASETS)).expanduser()
    return root / "Capstone 2" / "Part 1" / "dataset_hist_structures 2" / "dataset_hist_structures"


def train_dir() -> Path:
    return dataset_root() / "Stuctures_Dataset"


def test_dir() -> Path:
    return dataset_root() / "Dataset_test" / "Dataset_test_original_1478"


def is_readable(path: Path) -> bool:
    """Same check as the notebook's identify_and_remove_truncated_images: open and fully decode."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001 - PIL raises OSError, SyntaxError, ValueError... for broken files
        return False


@dataclass
class ScanResult:
    root: str
    files: dict[str, list[str]] = field(default_factory=dict)  # class -> readable image paths
    corrupt: dict[str, list[str]] = field(default_factory=dict)  # class -> unreadable image files
    skipped: dict[str, list[str]] = field(default_factory=dict)  # class -> non-image files

    @property
    def classes(self) -> list[str]:
        return sorted(self.files)

    def counts(self) -> dict[str, int]:
        return {c: len(self.files[c]) for c in self.classes}

    def total(self) -> int:
        return sum(self.counts().values())

    def corrupt_total(self) -> int:
        return sum(len(v) for v in self.corrupt.values())

    def summary(self) -> dict:
        return {
            "root": self.root,
            "images": self.total(),
            "per_class": self.counts(),
            "corrupt_removed": self.corrupt_total(),
            "corrupt_files": {c: v for c, v in self.corrupt.items() if v},
            "non_image_files_skipped": {c: v for c, v in self.skipped.items() if v},
        }


def scan_directory(root: Path, workers: int = 16, use_cache: bool = True) -> ScanResult:
    """List <root>/<class>/* images, drop unreadable ones, log counts per class."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {root} (set STRUCTURES_DATA or DATA_ROOT)")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    candidates: dict[str, list[Path]] = {}
    skipped: dict[str, list[str]] = {}
    for c in classes:
        entries = sorted(p for p in (root / c).iterdir() if p.is_file() and not p.name.startswith("."))
        candidates[c] = [p for p in entries if p.suffix.lower() in IMAGE_SUFFIXES]
        skipped[c] = [p.name for p in entries if p.suffix.lower() not in IMAGE_SUFFIXES]

    cache = _scan_cache_path(root, candidates)
    if use_cache and cache.exists():
        data = json.loads(cache.read_text())
        result = ScanResult(root=str(root), files=data["files"], corrupt=data["corrupt"], skipped=data["skipped"])
        log.info("scan cache hit %s: %d images, %d corrupt", root.name, result.total(), result.corrupt_total())
        return result

    result = ScanResult(root=str(root), skipped=skipped)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c in classes:
            flags = list(pool.map(is_readable, candidates[c]))
            result.files[c] = [str(p) for p, ok in zip(candidates[c], flags, strict=True) if ok]
            result.corrupt[c] = [p.name for p, ok in zip(candidates[c], flags, strict=True) if not ok]
            log.info(
                "%-16s %5d ok  %2d corrupt  %2d skipped (%s)",
                c, len(result.files[c]), len(result.corrupt[c]), len(skipped[c]), root.name,
            )
    log.info("%s: %d readable images in %d classes, %d corrupt removed",
             root.name, result.total(), len(classes), result.corrupt_total())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"files": result.files, "corrupt": result.corrupt, "skipped": result.skipped}))
    return result


def _scan_cache_path(root: Path, candidates: dict[str, list[Path]]) -> Path:
    h = hashlib.sha1()
    for c in sorted(candidates):
        for p in candidates[c]:
            st = p.stat()
            h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return DERIVED_DIR / "scan" / f"{root.name}-{h.hexdigest()[:12]}.json"


def limit_per_class(files: dict[str, list[str]], per_class: int | None, seed: int) -> dict[str, list[str]]:
    if per_class is None:
        return {c: list(v) for c, v in files.items()}
    rng = random.Random(seed)
    return {c: sorted(rng.sample(v, min(per_class, len(v)))) for c, v in files.items()}


def stratified_split(
    files: dict[str, list[str]], val_fraction: float, seed: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rng = random.Random(seed)
    train: dict[str, list[str]] = {}
    val: dict[str, list[str]] = {}
    for c, v in files.items():
        v = list(v)
        rng.shuffle(v)
        n_val = max(1, int(round(len(v) * val_fraction)))
        val[c], train[c] = v[:n_val], v[n_val:]
    return train, val


def flatten(files: dict[str, list[str]], classes: list[str]) -> tuple[list[str], np.ndarray]:
    paths: list[str] = []
    labels: list[int] = []
    for i, c in enumerate(classes):
        for p in files.get(c, []):
            paths.append(p)
            labels.append(i)
    return paths, np.asarray(labels, dtype=np.int64)


def load_image(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_arrays(paths: list[str], size: int, name: str, workers: int = 16, use_cache: bool = True) -> np.ndarray:
    """Decode and resize all images to a uint8 array [N, size, size, 3]; cached as .npy under data/."""
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12]
    cache = DERIVED_DIR / "arrays" / f"{name}-{size}px-{key}.npy"
    if use_cache and cache.exists():
        arr = np.load(cache)
        log.info("array cache hit %s: %s", name, arr.shape)
        return arr
    arr = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, img in enumerate(pool.map(lambda p: load_image(p, size), paths)):
            arr[i] = img
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    log.info("loaded %s: %s (%.0f MB)", name, arr.shape, arr.nbytes / 1e6)
    return arr
