"""Step 4 — Compare iterations against each other.

Run:  python 04_compare.py v1 v2 v3 v4

It reads benchmarks/<tag>.json for each tag you list, prints a table, and
saves benchmarks/_history.png — completion rate and crash rate side by side.

Use this every few iterations so you can SEE whether you're going forward.
The instructor's "process" grade rewards visible improvement curves.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from drive2win import viz


def _aggregate(log: dict) -> dict:
    """Average summary across all seeds in a single log."""
    seeds = log.get("seeds", [])
    if not seeds:
        return {"completion": 0.0, "median_lap": float("inf"), "crashes": 0.0,
                "max_cp": 0, "n_seeds": 0}
    comp = sum(s["summary"]["completion_rate"] for s in seeds) / len(seeds)
    med  = [s["summary"]["median_lap_time"] for s in seeds
            if s["summary"]["median_lap_time"] != float("inf")]
    median_lap = (sum(med) / len(med)) if med else float("inf")
    crashes = sum(s["summary"]["mean_crashes"] for s in seeds) / len(seeds)
    max_cp = max(s["summary"]["max_checkpoints"] for s in seeds)
    return {"completion": comp, "median_lap": median_lap,
            "crashes": crashes, "max_cp": max_cp, "n_seeds": len(seeds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="Iteration tags to compare.")
    ap.add_argument("--dir", default="benchmarks")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    history = []
    print(f"{'tag':<20s}  {'compl':>7s}  {'med_lap':>9s}  {'crashes':>8s}  {'cp':>5s}  seeds")
    for tag in args.tags:
        p = out_dir / f"{tag}.json"
        if not p.exists():
            print(f"  ! missing {p}, skipping")
            continue
        log = json.loads(p.read_text())
        a = _aggregate(log)
        history.append({"label": tag, **a})
        ml = "—" if a["median_lap"] == float("inf") else f"{a['median_lap']:.1f}s"
        print(f"{tag:<20s}  {a['completion']:>6.0%}   "
              f"{ml:>9s}  {a['crashes']:>7.2f}   {a['max_cp']:>4d}   {a['n_seeds']}")

    if history:
        viz.plot_iteration_history(history, out=str(out_dir / "_history.png"))


if __name__ == "__main__":
    main()
