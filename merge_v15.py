"""Build data_v15.npz — focused DAgger on CP2→CP3 and CP5→CP6 trouble spots.

Combines:
  - data_v14.npz        — all previous merged data (already filtered, 32 719 frames)
  - data_v15_s42.npz    — NEW: focused laps on seed 42 (filter here)
  - data_v15_s7.npz     — NEW: focused laps on seed 7  (filter here)
  - data_v15_s99.npz    — NEW: focused laps on seed 99 (filter here)

New data targets the two persistent failure modes seen in v14 benchmarks:
  - CP2→CP3: bot was taking a bad line and getting stuck ~40 s per run
  - CP5→CP6: bot charged the wall at full throttle, escape couldn't free it

Filter: throttle > 0.3 AND speed > 1.0 — removes reverse/recovery frames that
would teach the network to brake and back up during normal driving.

Run:
    python merge_v15.py
"""
import sys
from pathlib import Path
import numpy as np

NEW_FILES = [
    "data_v15_s42.npz",
    "data_v15_s7.npz",
    "data_v15_s99.npz",
]


def filter_clean(states, actions):
    mask = (actions[:, 0] > 0.3) & (states[:, 0] > 1.0)
    return states[mask], actions[mask]


def stats(states, actions, name):
    if len(states) == 0:
        print(f"  {name}: n=    0  (empty after filter)")
        return
    corr = float(np.corrcoef(states[:, 1], actions[:, 1])[0, 1])
    print(f"  {name}: n={len(states):5d}  corr={corr:+.3f}  "
          f"steer_std={actions[:,1].std():.3f}  thr_mean={actions[:,0].mean():+.3f}")


def main():
    missing = [f for f in ["data_v14.npz"] + NEW_FILES if not Path(f).exists()]
    if missing:
        print("Missing files — collect them first:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    base = np.load("data_v14.npz")
    base_s, base_a = base["states"], base["actions"]

    d42 = np.load("data_v15_s42.npz")
    d7  = np.load("data_v15_s7.npz")
    d99 = np.load("data_v15_s99.npz")

    d42_s, d42_a = filter_clean(d42["states"], d42["actions"])
    d7_s,  d7_a  = filter_clean(d7["states"],  d7["actions"])
    d99_s, d99_a = filter_clean(d99["states"], d99["actions"])

    print("Components:")
    stats(base_s, base_a, "v14 base (all prev)  ")
    stats(d42_s,  d42_a,  "v15 s42 CP2→3 CP5→6  ")
    stats(d7_s,   d7_a,   "v15 s7  CP2→3 CP5→6  ")
    stats(d99_s,  d99_a,  "v15 s99 CP2→3 CP5→6  ")

    states  = np.vstack([base_s, d42_s, d7_s, d99_s])
    actions = np.vstack([base_a, d42_a, d7_a, d99_a])

    print(f"\nMerged data_v15.npz:")
    stats(states, actions, "all                  ")

    np.savez("data_v15.npz", states=states, actions=actions)
    print(f"\nSaved data_v15.npz  ({len(states)} frames)")


if __name__ == "__main__":
    main()
