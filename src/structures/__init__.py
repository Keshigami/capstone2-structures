"""Historical-structure image classifier (Caltech CTME capstone 2, part 1)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
SAMPLES_DIR = PROJECT_DIR / "static" / "samples"
MODEL_PATH = MODELS_DIR / "structures.keras"
CLASSES_PATH = MODELS_DIR / "classes.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
CURVES_PATH = MODELS_DIR / "curves.png"
SCAN_PATH = MODELS_DIR / "dataset_scan.json"


def slug(class_name: str) -> str:
    """File-safe name for a class: 'dome(inner)' -> 'dome_inner'."""
    return class_name.replace("(", "_").replace(")", "").replace(" ", "_").lower()
