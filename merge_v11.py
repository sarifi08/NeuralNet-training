"""Three candidate datasets — train all three and benchmark to find best.

Diagnostic from data_v10.npz showed:
  data_v2 (human, 12293 frames): corr -0.179  ← weak signal, dominates by volume
  data_v8_s42 (auto, filtered):   corr -0.682
  data_v8_s7  (auto, filtered):   corr -0.852

Three mixes:
  v11a — auto only (clean signal, low volume)
  v11b — filtered human + filtered auto (clean signal, balanced volume)
  v11c — auto upsampled 3× + human (auto dominates, recovery from human)
"""
import numpy as np

d_v2 = np.load("data_v2.npz")
d_42 = np.load("data_v8_s42.npz")
d_s7 = np.load("data_v8_s7.npz")


def clean_filter(states, actions):
    """Keep only frames where bot was actually driving forward."""
    mask = (actions[:, 0] > 0.3) & (states[:, 0] > 1.0)
    return states[mask], actions[mask]


def stats(states, actions, name):
    corr = np.corrcoef(states[:, 1], actions[:, 1])[0, 1]
    print(f"  {name}: n={len(states):5d}  corr={corr:+.3f}  "
          f"steer_std={actions[:,1].std():.3f}  thr_mean={actions[:,0].mean():+.3f}")


# Components
v2s_clean,  v2a_clean  = clean_filter(d_v2["states"], d_v2["actions"])
s42s_clean, s42a_clean = clean_filter(d_42["states"], d_42["actions"])
ss7s_clean, ss7a_clean = clean_filter(d_s7["states"], d_s7["actions"])

print("Filtered components:")
stats(v2s_clean,  v2a_clean,  "v2_clean (human)")
stats(s42s_clean, s42a_clean, "s42_clean (auto)")
stats(ss7s_clean, ss7a_clean, "s7_clean  (auto)")
print()

# v11a — auto only
states_a  = np.vstack([s42s_clean,  ss7s_clean])
actions_a = np.vstack([s42a_clean,  ss7a_clean])
np.savez("data_v11a.npz", states=states_a, actions=actions_a)
print("v11a (auto only):")
stats(states_a, actions_a, "data_v11a")

# v11b — filtered human + filtered auto
states_b  = np.vstack([v2s_clean,  s42s_clean,  ss7s_clean])
actions_b = np.vstack([v2a_clean,  s42a_clean,  ss7a_clean])
np.savez("data_v11b.npz", states=states_b, actions=actions_b)
print("v11b (filtered human + auto):")
stats(states_b, actions_b, "data_v11b")

# v11c — auto upsampled 3× + ALL human (raw)
states_c  = np.vstack([np.tile(states_a, (3, 1)),  d_v2["states"]])
actions_c = np.vstack([np.tile(actions_a, (3, 1)), d_v2["actions"]])
np.savez("data_v11c.npz", states=states_c, actions=actions_c)
print("v11c (auto×3 + raw human):")
stats(states_c, actions_c, "data_v11c")

print("\nWrote data_v11a.npz, data_v11b.npz, data_v11c.npz")
