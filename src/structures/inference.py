"""Load the saved model once and classify PIL images."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

from structures import CLASSES_PATH, MODEL_PATH


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH, classes_path: Path = CLASSES_PATH):
        import keras  # local import keeps `structures` importable without TensorFlow loaded

        if not Path(model_path).exists() or not Path(classes_path).exists():
            raise FileNotFoundError(f"no trained model at {model_path}; run `just train` first")
        self.model = keras.saving.load_model(model_path, compile=False)
        self.classes: list[str] = json.loads(Path(classes_path).read_text())["classes"]
        self.image_size = int(self.model.input_shape[1])

    def preprocess(self, image: Image.Image) -> np.ndarray:
        im = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)[None]

    def predict(self, image: Image.Image, top: int = 3) -> dict:
        probs = self.model.predict(self.preprocess(image), verbose=0)[0].astype(float)
        order = np.argsort(-probs)
        return {
            "predicted": self.classes[int(order[0])],
            "top": [{"label": self.classes[int(i)], "probability": float(probs[i])} for i in order[:top]],
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs, strict=True)},
            "image_size": self.image_size,
        }

    def predict_bytes(self, data: bytes, top: int = 3) -> dict:
        with Image.open(BytesIO(data)) as im:
            im.load()
            return self.predict(im, top=top)

    def predict_path(self, path: str | Path, top: int = 3) -> dict:
        with Image.open(path) as im:
            return self.predict(im, top=top)
