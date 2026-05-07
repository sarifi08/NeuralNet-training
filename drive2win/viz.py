"""Visualization tools — see your data, your training, and your driver in motion.

Every function here writes a PNG to disk and prints the path. Use them in your
training scripts, in your iteration notes, or directly from the REPL.

The point of this module is that you should never be guessing what your model
is doing. If a number on the benchmark looks weird, *plot it*.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ── Path overlays ────────────────────────────────────────────────────────
def plot_path(track: Sequence[dict], out: str = "path.png",
              title: str = "Path", color: str = "tab:blue") -> str:
    """Plot a single (x, z) path from an eval `track` list.

    Each entry of `track` is {"t": float, "position": {"x": ..., "z": ...}, ...}.
    """
    xs = [p["position"].get("x", 0.0) for p in track if p.get("position")]
    zs = [p["position"].get("z", 0.0) for p in track if p.get("position")]
    plt.figure(figsize=(7, 7))
    plt.plot(xs, zs, "-o", ms=3, lw=1, color=color)
    if xs:
        plt.plot(xs[0], zs[0], "o", ms=10, color="green", label="start")
        plt.plot(xs[-1], zs[-1], "X", ms=10, color="red", label="end")
    plt.gca().set_aspect("equal")
    plt.title(title); plt.xlabel("x"); plt.ylabel("z")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


def plot_path_overlay(training_xz: np.ndarray | None,
                      test_track: Sequence[dict],
                      out: str = "path_overlay.png",
                      title: str = "Training drive (gray) vs NN test (blue)") -> str:
    """Overlay where YOU drove (training data) and where the NN drove (test).

    `training_xz` is an (N, 2) array of (x, z) collected during data collection.
    Pass None if you didn't capture positions while driving — only the test
    path will be drawn.
    """
    plt.figure(figsize=(8, 8))
    if training_xz is not None and len(training_xz):
        plt.plot(training_xz[:, 0], training_xz[:, 1],
                 "-", lw=0.6, alpha=0.4, color="gray", label="training drive")

    xs = [p["position"].get("x", 0.0) for p in test_track if p.get("position")]
    zs = [p["position"].get("z", 0.0) for p in test_track if p.get("position")]
    plt.plot(xs, zs, "-o", ms=3, lw=1.4, color="tab:blue", label="NN test")
    if xs:
        plt.plot(xs[0], zs[0], "o", ms=10, color="green")
        plt.plot(xs[-1], zs[-1], "X", ms=10, color="red")

    plt.gca().set_aspect("equal")
    plt.title(title); plt.xlabel("x"); plt.ylabel("z")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


def plot_multi_run_paths(runs: Iterable[dict], out: str = "paths_multi.png",
                         title: str = "All benchmark runs") -> str:
    """One figure, all run tracks layered. Helpful for variance across seeds."""
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(runs):
        track = r.get("track") or []
        xs = [p["position"].get("x", 0.0) for p in track if p.get("position")]
        zs = [p["position"].get("z", 0.0) for p in track if p.get("position")]
        cps = r.get("checkpoints_passed", "?")
        plt.plot(xs, zs, "-", lw=1.2, color=cmap(i % 10),
                 label=f"run{i+1}  cp={cps}")
    plt.gca().set_aspect("equal")
    plt.title(title); plt.xlabel("x"); plt.ylabel("z")
    plt.grid(True, alpha=0.3); plt.legend(loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


# ── Dataset diagnostics ──────────────────────────────────────────────────
def plot_action_histograms(actions: np.ndarray, out: str = "actions.png") -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].hist(actions[:, 0], bins=30, color="steelblue", edgecolor="black")
    ax[0].set(title="Throttle distribution", xlabel="throttle", ylabel="count")
    ax[1].hist(actions[:, 1], bins=30, color="coral", edgecolor="black")
    ax[1].set(title="Steering distribution", xlabel="steering")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


def plot_heading_vs_steering(states_raw: np.ndarray, actions: np.ndarray,
                             out: str = "heading_vs_steering.png") -> str:
    """heading_error (column 1 in the 12-vec) versus steering. Should slope down."""
    plt.figure(figsize=(7, 5))
    plt.scatter(states_raw[:, 1], actions[:, 1], s=4, alpha=0.3)
    plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
    plt.xlabel("heading_error (rad)"); plt.ylabel("steering")
    plt.title("Did you steer toward the target?  (downward slope = yes)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


# ── Training-time diagnostics ────────────────────────────────────────────
def plot_loss_curves(train_losses, val_losses, out: str = "loss.png") -> str:
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train MSE")
    plt.plot(val_losses, label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Training curve")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


# ── Run-time diagnostics ─────────────────────────────────────────────────
def plot_speed_profile(track: Sequence[dict], out: str = "speed.png") -> str:
    ts = [p["t"] for p in track]
    sp = [p.get("speed", 0.0) for p in track]
    plt.figure(figsize=(9, 3.5))
    plt.plot(ts, sp, "-o", ms=3)
    plt.axhline(0.3, color="red", lw=0.5, ls="--", label="stuck threshold")
    plt.xlabel("time (s)"); plt.ylabel("speed")
    plt.title("Speed over the run"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


def plot_checkpoint_progress(runs: Iterable[dict], out: str = "checkpoint_progress.png") -> str:
    """Bar chart of checkpoints reached per run — fast eyeball of consistency."""
    runs = list(runs)
    plt.figure(figsize=(7, 3.5))
    plt.bar(range(1, len(runs) + 1),
            [r.get("checkpoints_passed", 0) for r in runs],
            color="steelblue", edgecolor="black")
    plt.xlabel("run"); plt.ylabel("checkpoints"); plt.title("Checkpoints per run")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out


# ── Iteration history ────────────────────────────────────────────────────
def plot_iteration_history(history: list[dict], out: str = "iteration_history.png") -> str:
    """Plot benchmark numbers across iterations.

    `history` is a list of dicts, each like:
        {"label": "v1-bc",       "completion": 0.0, "median_lap": float("inf"), "crashes": 1.8}
        {"label": "v2-recovery", "completion": 0.4, "median_lap": 58.0,         "crashes": 1.0}
    """
    labels = [h["label"] for h in history]
    comp   = [h.get("completion", 0.0) for h in history]
    crash  = [h.get("crashes", 0.0)    for h in history]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(labels, comp, color="seagreen", edgecolor="black")
    ax[0].set(title="Completion rate", ylim=(0, 1)); ax[0].grid(axis="y", alpha=0.3)
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(labels, crash, color="firebrick", edgecolor="black")
    ax[1].set(title="Mean crashes / run"); ax[1].grid(axis="y", alpha=0.3)
    ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"  saved {out}")
    return out
