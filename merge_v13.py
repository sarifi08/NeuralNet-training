"""Build data_v13.npz for the targeted DAgger iteration.

Combines:
  - data_v11a.npz       — clean auto cruise (already filtered)
  - data_v12_s42.npz    — earlier human laps on seed 42 (filter here)
  - data_v12_s7.npz     — earlier human laps on seed 7  (filter here)
  - data_v13_s42.npz    — NEW: targeted DAgger on seed 42 (filter here)
  - data_v13_s7.npz     — NEW: targeted DAgger on seed 7  (filter here)
  - data_v13_s99.npz    — NEW: targeted DAgger on seed 99 (filter here)

The v13 files focus on v12's failure modes:
  - Going AROUND walls when the cp is on the other side of one (vs.
    driving at the wall and then trying to course-correct).
  - Hitting cps with a sharper terminal angle so we don't graze past
    at 5m (the close_pass values v12 produced).
  - Slower, more deliberate driving in low-friction zones so the
    network learns to throttle-down rather than overshoot in mud / ice.

Filter is identical to merge_v12.py: throttle > 0.3 AND speed > 1.0.
That removes the human's reverse-out-of-a-crash frames and the
near-zero-speed wedges, neither of which generalize to good policies.

Skipping data_v2.npz (corr -0.179, dilutes signal) and data_v8_s99.npz
(corr +0.101, wrong sign entirely) — same exclusions as merge_v12.py.

Run this once all three data_v13_s*.npz files exist:
    python merge_v13.py
"""
import sys
from pathlib import Path
import numpy as np

REQUIRED = [
    "data_v11a.npz",
    "data_v12_s42.npz",
    "data_v12_s7.npz",
    "data_v13_s42.npz",
    "data_v13_s7.npz",
    "data_v13_s99.npz",
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
    missing = [f for f in REQUIRED if not Path(f).exists()]
    if missing:
        print("Missing data files — collect them first:")
        for f in missing:
            print(f"  {f}")
        print("\nSee `python 01_collect_laps.py --help` for the collector.")
        sys.exit(1)

    a    = np.load("data_v11a.npz")          # already filtered
    h42  = np.load("data_v12_s42.npz")
    h7   = np.load("data_v12_s7.npz")
    d42  = np.load("data_v13_s42.npz")
    d7   = np.load("data_v13_s7.npz")
    d99  = np.load("data_v13_s99.npz")

    h42_s, h42_a = filter_clean(h42["states"], h42["actions"])
    h7_s,  h7_a  = filter_clean(h7["states"],  h7["actions"])
    d42_s, d42_a = filter_clean(d42["states"], d42["actions"])
    d7_s,  d7_a  = filter_clean(d7["states"],  d7["actions"])
    d99_s, d99_a = filter_clean(d99["states"], d99["actions"])

    print("Components after filter:")
    stats(a["states"],  a["actions"],  "auto v11a (s42+s7)   ")
    stats(h42_s,        h42_a,         "human v12 s42        ")
    stats(h7_s,         h7_a,          "human v12 s7         ")
    stats(d42_s,        d42_a,         "DAgger v13 s42 (NEW) ")
    stats(d7_s,         d7_a,          "DAgger v13 s7  (NEW) ")
    stats(d99_s,        d99_a,         "DAgger v13 s99 (NEW) ")

    states  = np.vstack([a["states"],  h42_s, h7_s, d42_s, d7_s, d99_s])
    actions = np.vstack([a["actions"], h42_a, h7_a, d42_a, d7_a, d99_a])

    print(f"\nMerged data_v13.npz:")
    stats(states, actions, "all                  ")

    np.savez("data_v13.npz", states=states, actions=actions)
    print(f"\nSaved data_v13.npz ({len(states)} frames)")


if __name__ == "__main__":
    main()
