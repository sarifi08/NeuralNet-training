# Deep Analysis — `LearnML_in3D` (Drive2Win Final Project)

*Generated from a full read of the codebase, 12 benchmark logs, 14 dataset versions, 10 trained model checkpoints, and the entire 37-commit git history.*

---

## 1. What this project actually is

The course assignment is to train a neural network that drives a 3D bot through a 12-checkpoint time-trial course on a remote simulator (`https://ml.ferit.tech`). The bot will eventually race against 19 classmates' bots in a 5-round tournament (3 obstacle-free rounds + 2 obstacle rounds), and the grade is split 50/50 between **process** (the git history + the `benchmarks/` folder) and **final tournament performance**.

The starter scaffold ships with:

- A `game_client.GameClient` that talks to the server via REST (sensors) and WebSocket (control commands).
- `drive2win/` — a Python package with `nn.py` (NumPy MLP + Adam + reference backprop), `normalize.py` (input/output scaling), `eval.py` (run-policy + scoring), `benchmark.py` (the canonical, do-not-edit evaluator), and `viz.py` (every plotting function).
- Four numbered scripts — `01_collect.py` (drive + record), `02_train.py` (where you fill in `my_backward()` and train), `03_benchmark.py` (run the canonical scorer and write `benchmarks/<tag>.json` + PNGs), and `04_compare.py` (cross-iteration table + history plot).
- The strict rule: **one commit per iteration**, each commit must explain what changed and why.

The 12-feature input vector is `[speed, heading_error, checkpoint_distance, ray_0..ray_7, ground_friction]`, normalized to roughly `[-1, 1]` by `drive2win.normalize`. The 2-element output is `(throttle, steering)`, both squashed by `tanh`. The default architecture is `12 → 64 → 32 → 2` with `ReLU/ReLU/tanh`, MSE loss, Adam at `lr=1e-3`, batch 64, 300 epochs, 90/10 train/val split.

---

## 2. The headline numbers (every benchmark on disk)

| tag | runs/seed | duration | seed 42 | seed 7 | seed 99 | notes |
|---|---|---|---|---|---|---|
| **v1** | 1 | 60 s | 0/1, max_cp=0, 5 crashes | – | – | First baseline. |
| **v2** | 1 | 60 s | 0/1, max_cp=1, 0 crashes | – | – | Recovery data added. |
| **v3** | 1 | 60 s | 0/1, max_cp=0, 1 crash | – | – | Deeper net (12→128→64→2). Reverted. |
| **v4** | 1 | 60 s | 0/1, max_cp=0, 9 crashes | – | – | Stuck-recovery override added — backfired. |
| **v10** | 3 | 60 s | 0/3, max_cp=0 | 0/3, max_cp=0 | 0/3, max_cp=1 | Naive merge of human + auto data. |
| **v11a** | 3 | 60 s | 0/3, max_cp=2 | 0/3, max_cp=0 | 0/3, max_cp=2 | Auto-only, filtered. **Best so far at 1.3 cp/run.** |
| **v11c** | 3 | 60 s | 0/3, max_cp=1 | 0/3, max_cp=0 | 0/3, max_cp=1 | Auto×3 + raw human. Worse than v11a. |
| **v12** | 3 | 60 s | 0/3, max_cp=1 | 0/3, max_cp=0 | 0/3, max_cp=1 | Clean auto + human full-laps (11,720 frames, corr −0.631). |
| **v12_fix** | 3 | 60 s | 0/3, max_cp=2 | 0/3, max_cp=1 | 0/3, max_cp=1 | Same model, runtime fix (wall-in-front stuck detection). |
| **v12_runtime** | 3 | 60 s | 0/3, max_cp=1 | 0/3, max_cp=0 | 0/3, max_cp=1 | Same model, full 3-point-turn escape + boost. |
| **v13** | 1 | 180 s | 0/1, max_cp=2 | 0/1, max_cp=0 | 0/1, max_cp=1 | DAgger expansion to 18,301 frames — regressed vs v12. |
| **v14-recovery** | 1 | 180 s | 0/1, max_cp=1 | 0/1, max_cp=1 | 0/1, max_cp=0 | Latest recovery-focused attempt. |

**No model has ever completed a full 12-checkpoint lap.** The current ceiling is 2/12 checkpoints. The honest summary the user wrote in the v13 commit body — *"Going from 0 to 0 is a null result, not a discovery"* — captures the state of play at the head of the branch.

---

## 3. Codebase map — what each file does

```
LearnML_in3D/
├── 01_collect.py          ← original 5-phase data collector (smooth / turns /
│                            obstacles / bad terrain / recovery), polls
│                            positions on a side thread for the path overlay
├── 01_collect_laps.py     ← single-purpose lap collector added in commit
│                            7d3a8c1 ("minimal lap-only collector for Phase 4")
├── 02_train.py            ← contains my_backward() (filled in correctly),
│                            gradient-checks against numerical_gradient,
│                            trains 300 epochs Adam, saves nav_<tag>.npz
├── 03_benchmark.py        ← canonical run-and-log entrypoint
├── 04_compare.py          ← reads all benchmarks/<tag>.json + writes _history.png
│
├── auto_collect.py        ← rule-based "expert" controller that drives
│                            autonomously while recording, used to generate
│                            data_v8_*.npz, data_v11a, etc.
├── run_agent.py           ← runtime that opens the browser, polls REST sensors
│                            on a background thread, runs the policy at 20 Hz,
│                            handles 3-point-turn escapes, distance-aware
│                            steering boost, post-escape bias, and per-frame
│                            JSONL diagnostic logging
├── diagnose_inputs.py     ← compares the live /sensors → 12-vector
│                            mapping vs. the recording-mode → 12-vector
│                            mapping (sanity check that train and infer
│                            see the same numbers)
├── debug_rest.py          ← standalone REST sensor probe
├── debug_ws.py            ← standalone WebSocket message probe
│
├── merge_v11.py           ← v11a (auto only) / v11b (filt human + auto) /
│                            v11c (auto×3 + raw human) candidate datasets
├── merge_v12.py           ← v11a (clean auto) + filtered human full-laps
├── merge_v13.py           ← v12 components + 3 new targeted DAgger files
│
├── game_client.py         ← starter SDK (REST + WS); not modified
│
├── drive2win/
│   ├── nn.py              ← MLP forward/backward, Adam, init, save/load,
│   │                        gradient check helpers
│   ├── normalize.py       ← SPD_MAX=20, DIST_MAX=100, RAY_MAX=50, the
│   │                        sensors_to_input() and normalize_states()
│   │                        single source of truth
│   ├── smooth.py          ← EMA action smoother (alpha=0.7 default), exposes
│   │                        last_raw on the wrapper for diagnostics
│   ├── eval.py            ← run_policy + score_runs (used by benchmark)
│   ├── benchmark.py       ← the canonical evaluator (DO NOT EDIT)
│   └── viz.py             ← every plot the project needs
│
├── data_v*.npz            ← 24 dataset files spanning v1–v14
├── nav_v*.npz             ← 10 trained model checkpoints
├── fig_actions_v*.png     ← per-iteration throttle/steering histograms
├── fig_heading_v*.png     ← per-iteration heading-error vs steering scatter
├── fig_loss_v*.png        ← per-iteration train/val loss curves
├── checkpoints_seed*.json ← world-space (x,y,z) of each checkpoint as the
│                            auto-collector hit it
└── benchmarks/
    ├── README.md
    ├── v*.json            ← 12 benchmark logs (every commit)
    └── v*_frames.jsonl    ← per-frame diagnostic logs for v12 and v13 runs
                              (gitignored — too large)
```

---

## 4. The work, broken down by domain

### 4.1 The neural network (`drive2win/nn.py` and `02_train.py`)

The architecture stayed at the default `12 → 64 → 32 → 2` with `ReLU / ReLU / tanh` for the entire project — except for one short-lived experiment in v3 (`12 → 128 → 64 → 2`, `nav_v3.npz` confirms the wider shapes), which was reverted in commit `f34c757` after it was found to overfit to a right-turn bias and underperform v2.

The user's `my_backward()` in `02_train.py` is a textbook chain-rule implementation:

1. Output gradient `dy = 2(y − target) / (n · n_outputs)` for MSE.
2. tanh derivative `dz3 = dy · (1 − y²)`.
3. Hidden 2 ReLU mask: `dz2 = (dz3 @ W3ᵀ) · (z2 > 0)`.
4. Hidden 1 ReLU mask: `dz1 = (dz2 @ W2ᵀ) · (z1 > 0)`.

The script gradient-checks every parameter against `numerical_gradient` to a tolerance of `1e-4` before any training starts, deliberately upcasting weights to `float64` only for the check (the commit `7809086` specifically calls out: *"float64 precision used in the check to avoid float32 cancellation noise"*). All 6 parameters pass at max relative error `< 1e-9`. Training itself stays in `float32` for speed.

The training loop is a clean Adam mini-batch SGD with random val-split, best-val checkpointing, and a 25-epoch print cadence. The reference `backward()` in `nn.py` is identical to `my_backward()` — used by all later iterations once the gradient check is green.

### 4.2 The data collection pipeline

Three separate collectors were written across iterations:

- **`01_collect.py`** — the original 5-phase scripted collector (smooth laps → tight turns → obstacle clusters → bad terrain → low-speed recovery → wall recovery), which also threads a position poller alongside the recording so the `_overlay.png` figure later compares the human's training drive against the network's test drive on the same x-z plane.
- **`01_collect_laps.py`** — added in commit `7d3a8c1` once the project realised that the 5-phase collector spends two minutes on "deliberately drive into walls" and "slow-speed reverse" frames that are exactly what the merge filter then strips out (*"those frames are exactly what we filter out at merge time… collecting them just wastes the user's session"*). This stripped-down version drives a single instruction: complete the 12-checkpoint loop for N minutes, with print-time correlation/std/throttle stats.
- **`auto_collect.py`** — a hand-engineered rule-based driver that records itself driving. The motivation, in the commit (`2ae7630`): *"Human driving produces ~−0.3 heading_error→steering correlation due to visual driving noise. Rule-based controller gives ~−0.8+ correlation with perfect sign convention."* The user just leaves the browser open while the script drives and records.

The auto-controller went through five rewrites, all preserved in the git log:

1. *Original* — heading-pursuit + bilateral wall-push. Failed: pushes from both walls cancelled in corridors and the bot stopped (`9710bb1`).
2. *Follow-The-Gap (FGM)* — ray-angle scoring `0.55·clearance + 0.45·cos(angle − target)`. Failed: the chosen angle flipped between discrete ray bins each frame, the wheels never settled, the bot couldn't accelerate (`58715b8`).
3. *Smooth heading-pursuit + decisive single-side override* — kept. Continuous `steer = -he/1.2`, only fires a strong unilateral correction when the front ray drops under 7 m, then chooses left or right by `(front_left + 0.4·left)` vs `(front_right + 0.4·right)`.
4. *Distance-aware steering and brake-on-overshoot* — `steer_scale(d)` returns `0.45 / 0.75 / 1.2` for `d < 6 / d < 12 / else`, and throttle drops to `0.25` when `distance < 10` AND `|he| > 1.0` so the bot tightens its turn radius near the checkpoint instead of orbiting past at high speed (`b9cea29`).
5. *Two-phase escape + post-escape bias* — 1.5 s reverse with hard turn (Ackermann reverse: `steer = +1` swings the front LEFT, so to rotate the bot left during reverse you set `steer = +1`), then 1.0 s forward in the same rotation, then a 3-second decaying steering bias toward the side the escape committed to so heading-pursuit doesn't immediately swing the bot back into the wall it just left.

A *frustration-detour* state was tried (`0d53b39`) — drive sideways for 7 s after two consecutive failed escapes, ignoring `heading_error`. It destroyed the correlation (+0.15 from −0.6) and was reverted in `fe68b1a` in favour of force-flipping escape direction on consecutive failures.

Two further refinements survived:

- **Stuck detection requires a wall in front** (`3611cc1`, `bf45ff5`): the original `if speed < 0.3: stuck_streak += 1` falsely triggered reverses on mud / ice / sand where speed naturally drops to 0.1–0.3 even at full forward throttle. The fix gates incrementing on `min(rays[0], rays[1], rays[7]) < 5.0` — a real obstacle in front, not just slow ground.
- **Proportional friction softening** (`e55aaa5`): when `friction < 0.6`, scale BOTH throttle and steering by `soft = max(0.5, 0.4 + friction)` so hard locks on ice don't spin the bot. Old code only reduced throttle below `friction < 0.4` and left steering untouched.

### 4.3 Dataset versions

24 `data_*.npz` files exist on disk. The interesting ones, in order:

| file | frames | source | notes |
|---|---|---|---|
| `data_v1.npz` | 6,036 | human, 5-phase | First baseline, includes positions array. |
| `data_v2.npz` | 12,293 | human | Larger but corr only −0.179 ("very noisy"). |
| `data_v4.npz` | 18,550 | human | Combined with `data_v4_dagger.npz` — 6,257 hand-corrected frames. |
| `data_v5_auto.npz` | 7,158 | auto (early) | First rule-based collection. |
| `data_v8_s{42,7,99}.npz` | 4,810 each | auto (refined) | Per-seed; corr improved to −0.5 to −0.7. |
| `data_v10.npz` | 21,913 | naive merge | corr fell to −0.386 — escape frames diluted the signal. |
| `data_v11a.npz` | 5,857 | auto, filtered | corr **−0.809**, val_loss 0.021 (9× better than v10), but only covers cps 1–5. |
| `data_v11b.npz` | 14,154 | filt human + auto | Intermediate. |
| `data_v11c.npz` | 29,864 | auto×3 + raw human | Worse than v11a (corr −0.540). |
| `data_v12_s{42,7}.npz` | 4,811 / 3,614 | human full-laps | Recorded with `01_collect_laps.py` to fix v11a's coverage gap. |
| `data_v12.npz` | 11,720 | merge | corr **−0.631**, full 12-cp coverage. |
| `data_v13_s{42,7,99}.npz` | ~3,610 each | targeted DAgger | New: drive AROUND walls when the cp is on the other side, sharper terminal angles, deliberate slow-downs in low friction. |
| `data_v13.npz` | 18,301 | merge | val_mse 0.0875 (lowest ever), but model regressed in benchmark. |
| `data_v14-recovery.npz` | 8,561 | human, recovery-focused | Most recent collection. |

The merge filter (`merge_v11.py`, `merge_v12.py`, `merge_v13.py`) is a one-liner that has shaped most of the project's improvements:

```python
mask = (actions[:, 0] > 0.3) & (states[:, 0] > 1.0)
```

It keeps only frames where the bot was actually driving forward (throttle > 0.3 AND speed > 1 m/s), removing the auto-controller's own escape sequences (which use `throttle = -0.8` with hard steer) and the human's reverse-out-of-a-crash frames. Auto data without this filter contains roughly 30 % escape frames whose heading→steering relationship is anti-correlated with normal driving, and mixing them in averages contradictions and produces mediocre output everywhere.

The honest, written-down realization in the v13 commit body is that this same filter has now become a problem too: it kept `thr_mean = 1.000` across all three new DAgger components, meaning every surviving frame is full-throttle and the *"slow down in mud"* frames the user deliberately drove are being filtered out before training. Loosening the filter is identified as *"a real lever (not yet pulled)"* for the next attempt.

### 4.4 The runtime (`run_agent.py`)

This is the most heavily-iterated file in the repo. It does six things that the canonical `drive2win/benchmark.py` does not:

1. **Auto-opens the browser** for each session — the simulator's physics engine runs in the browser, so without a tab open the server sends no sensor data and the bot doesn't move. The discovery and fix for this is in commits `6759450` (replace WS state polling with REST polling) and `54885b7` (the v1-bc commit that flagged the original `steps=0` symptom).
2. **Decouples sensor polling from control** (commit `0847aaa`): a background thread continuously polls `GET /sensors` (~4 Hz over the transatlantic link), and the main control loop reads the latest cached reading and sends WS control commands at the full target 20 Hz. Without this, the loop is bottlenecked on REST latency.
3. **Flattens the nested REST sensor response** (`fd7b0dd`) — `heading_error` lives inside `navigation{}` and `ground_friction` inside `ground{}` over REST, while `sensors_to_input()` expects flat keys. The `flatten_sensors()` helper reshapes them.
4. **Three-point-turn escape** (`9b50d6d`): when `stuck_streak >= 30` AND a wall is in front, pick `escape_dir` from which forward arc is more open (`rays[1]+[2]+[3]` vs `rays[5]+[6]+[7]`), reverse 1.5 s with `steer = -escape_dir` (Ackermann), forward 1.1 s with `steer = +0.7·escape_dir`. Net rotation ≈ 60–90°. After the escape, a 3-second decaying bias keeps the bot from immediately swinging back into the same wall.
5. **Distance-aware steering boost** (`9b50d6d`): `× 1.5` when `distance < 8 m`, `× 1.2` when `< 14 m`, otherwise no boost. Fixes the network's tendency to produce gentle steering (small `heading_error → small output`) when the cp is close, which would graze past at 4–5 m.
6. **Two later runtime bug fixes** that are detective work in their own right:
   - `799bb9b` *(stop infinite escape loop in wall-pocket wedges)* — the v12 seed-42 run wasted 22 seconds in a wall pocket because `stuck_streak` was being incremented during the escape's own reverse and forward phases. The moment esc_fwd ended, the trigger fired again. The fix gates `stuck_streak += 1` behind `not in_escape`. Doesn't free the bot from the wedge but stops the runtime from making it worse.
   - `5a0d8f0` *(flip escape direction on consecutive failed escapes)* — same v12 seed-42 run: 10 escape triggers in 22 s, every one with the same `dir = +1` because the rays didn't change between attempts. The fix tracks `last_escape_succeeded` (did `sp ≥ 0.3` during the escape?) and forces `-last_escape_dir` if the geometric pick repeats. Tagged in the live status with `(FLIP, prev escape failed)`.

The runtime also added a **per-frame JSONL diagnostic log** (`479abcc`, the `--log-frames` flag): every control frame appends `t, step, sp, he, d, fr, rays, cp, nn_t/nn_s (raw), cmd_t/cmd_s (final), phase, stuck, esc_rev/esc_fwd/post_esc` and an `event` tag on cp / escape / crash. This separation of *raw network output* from *commanded action after smoothing + boost + post-escape bias* is the diagnostic tool that surfaced the v13 wrong-sign-at-critical-moment finding.

The companion `820a31d` fix (`cp_hit close_pass distance label`) is small but representative of the level of care: cp-pass events were logging `d_at_hit ≈ 45 m` because by the time `checkpoint_index` increments, `nav.distance` has already updated to point at the *next* checkpoint. The fix keeps a 1-frame `prev_distance` and reports that, renamed to `close_pass` to make the meaning unambiguous.

### 4.5 The visualisations and diagnostics

`drive2win/viz.py` is used unchanged. The key plots — produced for every iteration — are:

- `fig_loss_<tag>.png` — train + val MSE across epochs. Used to check for overfitting (val rises while train falls).
- `fig_actions_<tag>.png` — throttle and steering histograms. Used to check for symmetry (did the demonstrator only ever turn right?).
- `fig_heading_<tag>.png` — heading-error vs steering scatter. Should slope downward; if not, the network can't learn to navigate from this data.
- `benchmarks/<tag>_paths.png` — all benchmark runs overlaid on the (x, z) plane.
- `benchmarks/<tag>_progress.png` — bar chart of checkpoints reached per run.
- `benchmarks/<tag>_overlay.png` — your training drive in gray vs. the NN's test drive in blue. The single most-revealing plot.

A separate `diagnose_inputs.py` exists to compare the live `/sensors → 12-vector` mapping against the recording-mode `sample.state → 12-vector` mapping side by side. Because the `flatten_sensors()` bug in iteration 1 silently fed the network the wrong inputs at inference time, this script is the safety net that confirms train and infer see identical numbers.

### 4.6 The EMA action smoother

`drive2win/smooth.py` (commit `12717a1`) wraps any policy in an exponential moving average:

```
smoothed = α · current + (1 − α) · prev
```

with `α = 0.7` as the default — keeps 70 % of the current prediction and 30 % of the previous output, eliminating the high-frequency steering oscillations that come from per-frame independent BC predictions without adding any learnable parameters. It exposes `policy.last_raw` so the runtime can read the underlying network's pre-smoothing output for diagnostics without paying for a second forward pass.

---

## 5. The git history as a narrative

The 37 commits, in order, tell a coherent story. The summary:

1. **Scaffold + baseline (`7809086` → `54885b7`)**: project structure, EMA smoother, gradient check passing, `v1-bc` trained, `val_mse = 0.195`, `0/1` checkpoints in benchmark.
2. **Get the runtime working at all (`6759450`, `fd7b0dd`, `0847aaa`)**: discover that REST polling is required (the WS doesn't push state), flatten the nested response, decouple control from REST latency. Without this triplet `steps = 0` for every benchmark run.
3. **Architecture experiment (`b131168`, `f34c757`)**: try `12→128→64→2`, find it overfits to a right-turn bias and underperforms v2, revert.
4. **Stuck-recovery override + fallout (`aff3a59`)**: add a runtime reverse override when stuck. Triggers spuriously on slow ground → 9 crashes on v4. Marks the start of the runtime-vs-network blame game that runs through the rest of the project.
5. **Rule-based auto-collector (`2ae7630` → `9710bb1` → `58715b8` → `b9cea29`)**: build a controller that drives itself well enough to record clean data. Each rewrite preserves what worked and discards what didn't, with the failure modes explained in the commit body.
6. **Friction softening + wall-in-front stuck detection (`3611cc1`, `e55aaa5`)**: realize that mud / ice / sand were sabotaging the controller because speed-only stuck detection counted any slow surface as "stuck" and triggered a reverse off the mud, into something behind, back onto the mud, forever.
7. **Mix experiments (`78d64c5`)**: discover the corr −0.179 / −0.506 / −0.690 problem with `data_v10`, write `merge_v11.py` to filter escape frames, produce v11a / v11b / v11c.
8. **Phase-7 close (`3718503`)**: the project's de-facto status report. Names v11a as the best model so far, identifies *data coverage* as the bottleneck (auto controller never finished a lap, so cps 6+ are absent from training data), and decides to pivot to human full-lap collection.
9. **Lap-only collector + 12-cp fix (`7d3a8c1`, `ce4d378`, `30f03f2`)**: write `01_collect_laps.py`, fix the `TARGET_CHECKPOINTS = 8 → 12` constant in two places, fix the on-screen instruction string.
10. **v12 (`296b805`)**: clean auto + human laps merged, corr −0.631, full 12-cp coverage. `val_loss = 0.057`. Best-trained model in the project.
11. **Runtime cleanup for v12 (`bf45ff5`, `9b50d6d`)**: port the wall-in-front fix into the runtime, add 3-point-turn escape, distance boost, post-escape bias.
12. **Multi-seed benchmark (`1b4e9d7`)**: v12 / v12_fix / v12_runtime — same model, three runtime variants, all three seeds. Top run: 2/12 cps on seed 42. Honest commit body acknowledges *"data_v12 has correlation −0.631 and full-lap coverage but the trained MLP cannot replicate the user's 35-cp/4min driving on the same seeds."*
13. **Diagnostic plumbing (`479abcc`, `820a31d`)**: add `--log-frames` per-frame JSONL logging and live status print so the next debugging round isn't blind.
14. **Wall-pocket fixes (`799bb9b`, `5a0d8f0`)**: stop the runtime from making wedges worse — gate stuck_streak during escape, force-flip on consecutive failed escapes.
15. **v13 DAgger expansion (`5876102`, `d06ac40`)**: add three new targeted DAgger files, train v13. `val_mse = 0.0875` (lowest yet), but benchmark regresses to 0.00 cp/run from v12's 0.22. The commit body is the most thorough postmortem in the log: identifies three hypotheses (the throttle filter wiped out slow-driving frames, seed 99 widened the distribution beyond model capacity, the network outputs the wrong sign at the critical moment), refuses to retrain blind, and proposes the next move (a hybrid runtime that blends rule-based heading pursuit with the network).

---

## 6. What's actually been figured out, and what's left

### Solved
- The `my_backward()` chain rule, gradient-checked end-to-end against numerical gradients, max relative error `< 1e-9`.
- The browser-tab-required physics, REST-vs-WS message format, and decoupled-control-Hz performance issues.
- A rule-based auto-collector that produces high-correlation (−0.8) data at scale.
- The escape-frame contamination problem, and the merge-time filter that removes it.
- A 3-point-turn escape with consecutive-failure flipping and post-escape bias that keeps the runtime out of trivial infinite loops.
- Friction-aware throttle/steer softening so ice and mud no longer make the controller spin out.
- A diagnostic JSONL log + live status that separates the network's raw output from every runtime override.

### Identified but not yet fixed
- **The trained MLP cannot replicate the user's own driving on the same seed.** v12 has corr −0.631 over 11,720 frames, the user can run ~35 checkpoints in 4 minutes on seed 7, and the trained network plateaus at 1–2 checkpoints. This is the unresolved core of the project.
- **The throttle > 0.3 filter is now over-aggressive** — it strips out the deliberate slow-driving frames the user records to teach the network mud/ice behavior, leaving a corpus where every surviving frame is `thr ≈ 1.0`.
- **Wrong-sign steering at the critical moment** — the v13 frame log showed `nn_s = +0.40` at `he = +1.33` on seed 42 t=11.2 s (cp clearly on the LEFT, network steered RIGHT).
- **Lap completion is an open problem.** Best ever: 2/12 checkpoints. A full lap remains uncrossed.

### Proposed next moves (already named in the commits)
- **Hybrid runtime**: blend rule-based heading pursuit with the network — addresses the wrong-sign-at-critical-moment failure directly without needing more data.
- **Loosen the throttle filter** so slow-driving frames survive into training.
- **Architecture / training-schedule changes that haven't been tried yet**: leaky ReLU, action-delta prediction, ensemble of two seeds, the CNN-on-grid32 path, a sklearn `Pipeline` wrapper.

---

## 7. What grading will see

The commit history is dense, narrative, and disciplined — every iteration has a tag, a hypothesis, a result, and (often) a frame-log excerpt as evidence. The benchmarks folder has 12 `.json` logs plus the 4 frame-log JSONL files. The figures (`fig_actions_v*`, `fig_heading_v*`, `fig_loss_v*`, plus the per-tag `_paths.png` / `_progress.png` / `_overlay.png` triplets) are present for every iteration through v14-recovery. The `INSTRUCTOR_TODO.md` and original README in the upstream repo are untouched; everything in this folder is the user's added work.

The single weakness against the rubric is the final-performance number: no completed lap means the *50 % final-tournament* component is currently worth few points. Everything else — process, hypothesis-driven iteration, honest negative-result logging, evidence in the commit bodies, instrumentation that proves the next iteration won't be blind — is in unusually good shape for a course project at this stage.
