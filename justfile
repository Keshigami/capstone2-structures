# capstone2-structures: transfer-learning classifier for historical structure images, port 8102.
set shell := ["bash", "-cu"]
port := "8102"
pidfile := ".server.pid"

# Install deps, pull the shared stylesheet, train a quick model if there is none.
setup:
    uv sync
    cp common/ui.css static/ui.css
    [ -f models/structures.keras ] || just train

# Quick model: <=150 images/class, 160px, 4 epochs. About two minutes on an M4 Max.
train:
    uv run python src/train.py --quick

# Full model: all 10,235 images, 224px, early stopping. Tens of minutes on CPU.
train-full:
    uv run python src/train.py --full

# Serve in the foreground.
run:
    uv run uvicorn app:app --host 127.0.0.1 --port {{port}}

# Serve in the background (pid in .server.pid, log in .server.log) and wait for /health.
up:
    #!/usr/bin/env bash
    set -euo pipefail
    just down >/dev/null 2>&1 || true
    uv sync -q
    nohup .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port {{port}} > .server.log 2>&1 &
    echo $! > {{pidfile}}
    for i in $(seq 1 90); do curl -sf localhost:{{port}}/health >/dev/null 2>&1 && break; sleep 1; done
    curl -s localhost:{{port}}/health || { echo "server did not come up; see .server.log"; exit 1; }
    echo

down:
    #!/usr/bin/env bash
    if [ -f {{pidfile}} ]; then kill "$(cat {{pidfile}})" 2>/dev/null || true; rm -f {{pidfile}}; fi
    lsof -ti tcp:{{port}} 2>/dev/null | xargs kill 2>/dev/null || true
    echo "stopped"

test:
    uv run ruff check
    uv run pytest

# Scripted end-to-end call: CLI prediction, then the HTTP API on a running server (starts one if needed).
demo:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "== CLI: top-3 for three samples"
    uv run python src/predict.py static/samples/gargoyle.jpg static/samples/stained_glass.jpg static/samples/dome_inner.jpg
    started=0
    if ! curl -sf localhost:{{port}}/health >/dev/null 2>&1; then just up >/dev/null; started=1; fi
    echo "== GET /health"; curl -s localhost:{{port}}/health; echo
    echo "== POST /predict (file upload: altar.jpg)"; curl -s -F file=@static/samples/altar.jpg localhost:{{port}}/predict; echo
    echo "== POST /predict (gallery sample: vault)"; curl -s -F sample=vault localhost:{{port}}/predict; echo
    echo "== GET /metrics (summary)"
    curl -s localhost:{{port}}/metrics | uv run python -c "import json,sys; m=json.load(sys.stdin); print({k: m[k] for k in ('mode','backbone','image_size','test_accuracy','val_accuracy','train_accuracy','wall_time_seconds')})"
    [ "$started" = 1 ] && just down >/dev/null || true
