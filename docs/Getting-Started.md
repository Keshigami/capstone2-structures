# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv.
- No API key is needed anywhere in this project.

## Three commands
```bash
git clone https://github.com/Keshigami/capstone2-structures.git && cd capstone2-structures
just setup
just up           # http://127.0.0.1:8102/  (just down to stop)
```
The trained model is not in the repo (the trained model is 193 MB, over GitHub's file limit); `just train` rebuilds it from the Caltech `Capstone 2` dataset, which you point at with `DATA_ROOT`.

## Commands
`just setup`, `just train` where a model is trained, `just up` / `just down`, `just test`, `just demo`.

## What you see
- Run: `just setup && just up`, open `http://localhost:8102/` (`just train` rebuilds a quick model in ~25 s; `just train-full` the full one in ~12 min).
- Two-minute walkthrough: (1) click `gargoyle` in the sample gallery; the result card shows the image, the predicted
  class, its confidence and the top-3 bars, with a pill confirming the test label. (2) Click `apse`, then scroll to
  the per-class table: apse is the weak class (61.4% full, 52.6% quick), the gallery images are the model's most confident correct
  test images, so the table is the honest number. (3) Drop any photo of a cathedral window, a dome or a gargoyle
  from your own files onto the page; it classifies on drop. (4) Look at the curves: the dashed line is where augmentation switches on; validation accuracy
  drops after it in both modes, so the script kept the plain-phase weights.
- Video clip: `demo/clips/capstone2-structures.webm` (recorded by `just demo-video` at the root; coordinator owns it).
- Sample inputs: `static/samples/<class>.jpg`, one per class, taken from the test set. Expected result: the
  matching class at >99% for all ten (they were selected as confident correct predictions).
- Script hooks: `#input` (file input), `#run` (Classify button), `#result` (result container); gallery buttons
  carry `data-sample="<class>"` and classify on click without an upload.

## Try this
Try it: click apse, then read the table.
