# Results

Quick mode on this machine (M4 Max, CPU), `models/metrics-quick.json`:

| Number | Value | Meaning |
|---|---|---|
| Test accuracy | 89.3% | 1,473 never-seen images from the separate test folder; chance is 10% |
| Best validation accuracy | 93.6% | 220 held-out training images, baseline phase, epoch 2 (the saved weights) |
| Train accuracy at that epoch | 89.6% | below validation because dropout is on while training |
| Per class (test) | flying_buttress 98.7, altar 97.1, dome(outer) 96.4, gargoyle 92.0, stained_glass 92.0, dome(inner) 90.7, bell_tower 90.0, vault 89.0, column 79.0, apse 52.6 | apse (57 test images) is confused with bell_tower and column |
| Cost | 1,280 training images, 4 epochs, 13 s training, 18 s wall time | 150 images per class is enough for a frozen backbone to reach ~90% |

The notebook reached 0.90 validation accuracy with VGG16 on all images (validating on the test folder). A good
number for this dataset with a fine-tuned backbone is mid-90s; the quick model is within a few points of the
notebook on 15% of the data.

Full mode (`just train-full`, the model now deployed), `models/metrics-full.json`:

| Number | Value | Meaning |
|---|---|---|
| Test accuracy | **93.3%** | same 1,473 test images; the notebook reported 0.90 with VGG16 |
| Best validation accuracy | 95.6% | 1,535 held-out training images, baseline phase, epoch 4 (the saved weights) |
| Per class (test) | dome(outer) 98.8, gargoyle 97.1, stained_glass 96.9, altar 95.7, bell_tower 95.3, column 95.2, vault 92.1, dome(inner) 90.7, flying_buttress 78.2, apse 61.4 | apse is still the weak class; flying buttresses are often photographed with the whole facade |
| Cost | 8,700 training images, 14 epochs (early stopping in both phases), 12.2 min on CPU | augmentation peaked at 93.6% validation, below the plain phase, so the plain weights were kept |
