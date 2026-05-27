"""Tournament bot — plugs nav_v14.npz into RoomBot for the live competition.

Usage:
    # Smoke test your own room first:
    python my_bot.py --room sarifi --name sarifi

    # Tournament day (professor announces the room name):
    python my_bot.py --room <room_name> --name sarifi
"""
import argparse
import numpy as np
from game_client import RoomBot
from drive2win.agent import make_policy

WEIGHTS = "nav_v14.npz"
SERVER  = "https://ml.ferit.tech"

# Load once at startup — all wrappers (escape, checkpoint precision,
# near-miss recovery) are included via drive2win/agent.py make_policy.
_policy = make_policy(WEIGHTS)


def _obs_to_state(obs: dict) -> dict:
    """Convert RoomBot obs → GameClient state format so our wrappers work."""
    nav = obs.get("navigation") or {}
    return {
        "sensors": {
            "speed":               obs.get("speed", 0.0),
            "heading_error":       nav.get("heading_error", 0.0),
            "checkpoint_distance": nav.get("distance", 999.0),
            "rays":                obs.get("rays") or [50.0] * 8,
            "ground_friction":     obs.get("ground_friction", 1.0),
            "navigation": {
                "checkpoints_completed": obs.get("checkpoints_passed", 0),
            },
        }
    }


def controller(obs: dict):
    """Called at 20 Hz by RoomBot. Returns (throttle, steering)."""
    if obs.get("race_phase") != "racing":
        return 0.0, 0.0
    state = _obs_to_state(obs)
    return _policy(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="sarifi",
                    help="Room name (use your name for solo tests; "
                         "professor announces the tournament room)")
    ap.add_argument("--name", default="sarifi",
                    help="Bot display name shown on the leaderboard")
    args = ap.parse_args()

    print(f"Loading weights: {WEIGHTS}")
    print(f"Joining room   : {args.room}  as  {args.name}")
    print("Waiting for round_start…\n")

    bot = RoomBot(SERVER, room=args.room, name=args.name)
    standings = bot.run(controller, hz=20.0)

    print("\n=== Final standings ===")
    for r in standings:
        print(f"  #{r.get('rank')}  {r.get('name')}  "
              f"checkpoints={r.get('total_checkpoints')}")


if __name__ == "__main__":
    main()
