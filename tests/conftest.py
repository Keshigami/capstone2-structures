import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def client():
    """App with the trained model loaded. Fails (not skips) when `just train` has not been run."""
    model = ROOT / "models" / "structures.keras"
    assert model.exists(), f"{model} missing: run `just train` first"
    from app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def classes() -> list[str]:
    return json.loads((ROOT / "models" / "classes.json").read_text())["classes"]


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return ROOT / "static" / "samples"
