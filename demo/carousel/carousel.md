# Historical structures classifier · carousel

Spec: film/specs/capstone2-structures.json · 8 slides, 1080x1350.

## 1/8 Hook

- Altar. One hundred percent.

## 2/8 The question

- Column has 1,631 photos. Apse has 437.
- Can it learn the rare ones?
- column 1,631

## 3/8 What it does

- Historical structures classifier
- Capstone 2 · Heritage tourism agency
- Ten classes, one photo, 93% right
- max@baluyot ~ $ curl -s -F file=@static/samples/altar.jpg localhost:8102/predict

## 4/8 How it is built

- 1. Ten class folders — pillow · numpy
- 2. MobileNetV2, frozen — tensorflow · keras
- 3. Plain, then augmented — keras
- 4. POST /predict — FastAPI
- 5. Gallery page — HTML + ui.css
- Keep the best phase: augmentation lost here

## 5/8 The result

- 4 models on the same holdout
- Chance 10.0%
- Notebook VGG16 (validated on test) 90.0%
- MobileNetV2 quick, 15% of data 89.3%
- MobileNetV2 full 93.3% (winner)

## 6/8 The proof

- Altar: one click, top-3 bars
- screen: capstone2-structures/demo/poster.png

## 7/8 The answer and the limit

- 93.3%
- test accuracy · 1,473 photos never seen in training
- Apse: 61%. It looks like a tower from outside.
- Notebook 90.0 → 93.3 here. And it ships.

## 8/8 Try it

- Try it: click apse, then read the table.
- max@baluyot ~ $ git clone https://github.com/Keshigami/capstone2-structures.git
- max@baluyot ~ $ just setup
- max@baluyot ~ $ just up
- github.com/Keshigami/capstone2-structures
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
