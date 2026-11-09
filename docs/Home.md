# Historical structures classifier

Upload a photo of a church or monument detail and a transfer-learned CNN names which of ten architectural elements it is, with a confidence. Caltech CTME PGC AIML, capstone, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/capstone2-structures/blob/main/demo/film.mp4).

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the brief to a running service, step by step.
- [Results](Results.md): the numbers and what they mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show.

## The brief
A government tourism agency wants to monitor centuries-old buildings and flag the ones that need maintenance.
Step one is knowing what a photo shows: an altar, an apse, a bell tower, a column, the inside or outside of a dome,
a flying buttress, a gargoyle, stained glass or a vault. The task (Caltech CTME capstone 2, part 1) is a TensorFlow
model that predicts that category from an image, built with transfer learning because ten thousand photos is not
enough to train a CNN from scratch. "Solved" is a model that gets most held-out test photos right and a way for a
non-engineer to try it.

## The one-line answer
93.3% · test accuracy · 1,473 photos never seen in training

## One honest limit
Apse: 61%. It looks like a tower from outside.
