"""HTTP tests against the FastAPI app with the quick model. Offline; no network."""

import math
import re

import pytest

THREE_CLASSES = [("altar", "altar"), ("gargoyle", "gargoyle"), ("stained_glass", "stained_glass")]


def test_health(client, classes):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "capstone2-structures"
    assert body["model_loaded"] is True
    assert body["classes"] == classes and len(classes) == 10


def test_metrics(client, classes):
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["classes"] == classes
    for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
        assert 0.0 <= m[key] <= 1.0
    assert set(m["per_class_test_accuracy"]) == set(classes)
    assert m["epochs"]["total"] == len(m["history"]["accuracy"]) >= 1
    assert m["images"]["test"] > 1000 and m["mode"] in ("quick", "full")


@pytest.mark.parametrize(("sample", "label"), THREE_CLASSES)
def test_predict_upload_returns_known_class(client, classes, samples_dir, sample, label):
    path = samples_dir / f"{sample}.jpg"
    with path.open("rb") as fh:
        r = client.post("/predict", files={"file": (path.name, fh, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted"] in classes
    assert len(body["top"]) == 3
    assert all(row["label"] in classes for row in body["top"])
    assert body["top"] == sorted(body["top"], key=lambda row: -row["probability"])
    assert math.isclose(sum(body["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert set(body["probabilities"]) == set(classes)
    assert body["filename"] == path.name


def test_predict_sample_by_name(client, classes):
    r = client.post("/predict", data={"sample": "dome(inner)"})
    assert r.status_code == 200, r.text
    assert r.json()["predicted"] in classes


def test_gallery_samples_classify_correctly_mostly(client):
    """The gallery images were chosen as confidently-correct test images, so most must still be right."""
    correct = 0
    for slug, label in [("altar", "altar"), ("vault", "vault"), ("dome_outer", "dome(outer)"), ("column", "column")]:
        body = client.post("/predict", data={"sample": slug}).json()
        correct += body["predicted"] == label
    assert correct >= 3


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/predict")
    assert r.status_code == 400
    r = client.post("/predict", data={"sample": "nope"})
    assert r.status_code == 404


def test_gallery_page_lists_ten_samples(client, classes):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert len(re.findall(r'data-sample="', html)) == 10
    for c in classes:
        assert f'data-label="{c}"' in html
    for anchor in ('id="input"', 'id="run"', 'id="result"', 'header class="top"', "/static/ui.css"):
        assert anchor in html
    assert client.get("/static/ui.css").status_code == 200
    assert client.get("/static/samples/gargoyle.jpg").status_code == 200
