# Limits and next steps

- `just train` (quick, 25 s) rebuilds a 15%-of-data demo model at 160px and overwrites the deployed full model;
  `just train-full` (12 min) restores the 93.3% one. Both metrics files are kept side by side.
- Augmentation hurts in a short run on a frozen backbone with a Flatten head; the script keeps the better phase.
  A production version would fine-tune the top blocks at a low learning rate and use global average pooling.
- Apse and column are weak; they need more data or a backbone fine-tune, and a confusion-matrix view in the UI.
- Validation here is split from training data, so the numbers are not directly comparable to the notebook's 0.90,
  which validated on the test set.
- CPU inference (~50 ms per image) and a single process: fine for a demo, not for a queue of maintenance photos.
