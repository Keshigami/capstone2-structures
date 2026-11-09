"""Transfer-learning model: frozen ImageNet backbone + the notebook's head, augmentation and callbacks.

The notebook used VGG16 -> Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(10, softmax), Adam,
categorical cross-entropy, EarlyStopping on val_accuracy. The head and training setup are kept; the
backbone defaults to MobileNetV2 so a full run fits on a laptop CPU in minutes instead of hours.
"""

from __future__ import annotations

import keras
from keras import layers

BACKBONES = ("mobilenetv2", "efficientnetb0")


def build_model(
    num_classes: int,
    image_size: int,
    backbone: str = "mobilenetv2",
    dense_units: int = 256,
    dropout: float = 0.5,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Model that takes raw RGB uint8/float images in [0, 255] and returns class probabilities."""
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    shape = (image_size, image_size, 3)
    inputs = keras.Input(shape=shape, name="image")
    if backbone == "mobilenetv2":
        # MobileNetV2 expects inputs in [-1, 1]; keep the preprocessing inside the saved model.
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")(inputs)
        base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=shape)
    else:
        x = inputs  # EfficientNetB0 includes its own preprocessing and takes [0, 255]
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=shape)
    base.trainable = False  # freeze all convolutional layers (also keeps BatchNorm in inference mode)
    x = base(x, training=False)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    model = keras.Model(inputs, outputs, name=f"structures_{backbone}")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def augmentation(image_size: int) -> keras.Sequential:
    """The notebook's ImageDataGenerator settings as Keras layers (active only while training)."""
    augs: list[layers.Layer] = [
        layers.RandomFlip("horizontal"),  # horizontal_flip=True
        layers.RandomRotation(20 / 360),  # rotation_range=20 degrees
        layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),  # width/height_shift_range=0.2
        layers.RandomZoom(0.2, fill_mode="nearest"),  # zoom_range=0.2
    ]
    if hasattr(layers, "RandomShear"):
        augs.append(layers.RandomShear(0.2, fill_mode="nearest"))  # shear_range=0.2
    return keras.Sequential([keras.Input(shape=(image_size, image_size, 3)), *augs], name="augmentation")


def with_augmentation(model: keras.Model, image_size: int) -> keras.Model:
    """Training-time wrapper: augmentation -> model. Weights are shared, so `model` is what gets saved."""
    wrapped = keras.Sequential([keras.Input(shape=(image_size, image_size, 3)), augmentation(image_size), model])
    wrapped.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return wrapped


class StopAtValAccuracy(keras.callbacks.Callback):
    """Problem statement task 6: stop once validation accuracy reaches a chosen number."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.triggered_epoch: int | None = None

    def on_epoch_end(self, epoch, logs=None):
        acc = (logs or {}).get("val_accuracy")
        if acc is not None and acc >= self.threshold:
            print(f"\nval_accuracy {acc:.4f} >= {self.threshold:.2f}: stopping training")
            self.triggered_epoch = epoch + 1
            self.model.stop_training = True


def callbacks(threshold: float, patience: int = 3) -> list[keras.callbacks.Callback]:
    return [
        StopAtValAccuracy(threshold),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
    ]
