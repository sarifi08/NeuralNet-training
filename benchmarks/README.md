# Iteration log

This folder is the record of *what you tried, when, and how it scored*. It is half of your grade.

Every time you run `03_benchmark.py --tag <tag>`, three things land here:

| File | What it is |
|------|------------|
| `<tag>.json`            | Numbers: completion rate, median lap, crashes, full run tracks. |
| `<tag>_paths.png`       | All benchmark runs overlaid on the (x, z) plane. |
| `<tag>_progress.png`    | Bar chart of checkpoints reached per run. |
| `<tag>_overlay.png`     | (Optional) Where you drove during data collection vs. where the NN drove. Pass `--data data_<tag>.npz` to produce it. |

After a few iterations you also run:

```
python 04_compare.py v1 v2 v3 ...
```

which writes `_history.png` — completion + crashes across iterations, side by side.

---

## Tag naming

Pick names that say *what changed*, not just `v2`, `v3`. Examples:

```
v1-bc                     baseline behavioral cloning, 12→64→32→2 MLP
v2-recovery               added 200 recovery samples
v3-deepnet                12→128→64→32→2
v4-dagger                 corrected with 600 DAgger samples after watching v3 fail
v5-cnn-hybrid             added the 32×32 grid, hybrid CNN+MLP
v6-augment                added action smoothing + heading-error noise
```

Same tags should appear in your **git commit messages** ("v3-deepnet: deeper net, 8% better completion"). The instructor reads both.

---

## Required commits per iteration

For each iteration:

1. **Commit the data file** (`data_<tag>.npz`) **only if you collected new data**. Otherwise reuse the previous file.
2. **Commit the weights** (`nav_<tag>.npz`).
3. **Commit `benchmarks/<tag>.json` and the PNGs.**
4. **Write a one-paragraph commit message** — what you changed, what you predicted would happen, what actually happened.

A good commit message:

> v4-dagger: hand-corrected 600 frames after watching v3 stall against the
> rocky cluster on seed 42. Predicted: completion 60%→80%. Actual: 60%→80%
> on seed 42, 20%→60% on seed 7 (the bot was overfit to seed 42 terrain).

A bad commit message:

> updated stuff
