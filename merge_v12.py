"""Build data_v12.npz from auto-cruise + human-laps.

Components (all filtered: throttle > 0.3 AND speed > 1.0):
  data_v11a.npz       — auto-cruise (already filtered, 5,857 frames, corr -0.809)
  data_v12_s42.npz    — human laps on seed 42 (4,811 frames raw)
  data_v12_s7.npz     — human laps on seed 7  (3,614 frames raw)

Why this works (where v10 didn't):
  - v10 mixed escape frames into training → contradictory signals
  - v11a had clean signal but only covered cps 1-5
  - v12 = v11a's clean cruise + human's FULL 12-cp coverage (2-3 laps each)

Skip data_v2.npz — its correlation is -0.179 (very noisy), would dilute.
"""
import numpy as np

a   = np.load("data_v11a.npz")     # auto, already filtered
h42 = np.load("data_v12_s42.npz")  # human seed 42 — filter here
h7  = np.load("data_v12_s7.npz")   # human seed 7  — filter here


def filter_clean(states, actions):
    mask = (actions[:, 0] > 0.3) & (states[:, 0] > 1.0)
    return states[mask], actions[mask]


def stats(states, actions, name):
    corr = np.corrcoef(states[:, 1], actions[:, 1])[0, 1]
    print(f"  {name}: n={len(states):5d}  corr={corr:+.3f}  "
          f"steer_std={actions[:,1].std():.3f}  thr_mean={actions[:,0].mean():+.3f}")


h42_s, h42_a = filter_clean(h42["states"], h42["actions"])
h7_s,  h7_a  = filter_clean(h7["states"],  h7["actions"])

print("Components:")
stats(a["states"], a["actions"],  "auto v11a (s42+s7)")
stats(h42_s,       h42_a,         "human s42 filt   ")
stats(h7_s,        h7_a,          "human s7  filt   ")

states  = np.vstack([a["states"],  h42_s, h7_s])
actions = np.vstack([a["actions"], h42_a, h7_a])

print(f"\nMerged data_v12.npz:")
stats(states, actions, "all              ")

np.savez("data_v12.npz", states=states, actions=actions)
print(f"\nSaved data_v12.npz")
