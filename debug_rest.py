"""Debug REST sensor + REST control loop.

Run this, open the browser URL when prompted, wait for the bot to appear,
then the script will poll sensors every 0.5s for 15s and print the structure.

    python debug_rest.py
"""
import json, time, webbrowser, requests

SERVER = "https://ml.ferit.tech"
http = requests.Session()

resp = http.post(f"{SERVER}/api/session", json={
    "mode": "time_trial", "player_name": "debug_rest", "config": {"seed": 42}
})
resp.raise_for_status()
data = resp.json()
session_id = data["session_id"]
browser_url = data.get("browser_url", f"{SERVER}/?session={session_id}")

print(f"Session: {session_id}")
print(f"Opening browser: {browser_url}")
webbrowser.open(browser_url)

print("\nWaiting 5s for browser to load, then polling sensors…")
time.sleep(5)

for i in range(10):
    try:
        r = http.get(f"{SERVER}/api/session/{session_id}/sensors", timeout=3)
        print(f"\n[poll {i+1}] status={r.status_code}")
        if r.status_code == 200:
            s = r.json()
            print("  keys:", list(s.keys()))
            print("  FULL:", json.dumps(s, indent=2)[:600])
            break
        else:
            print("  body:", r.text[:200])
    except Exception as e:
        print(f"  error: {e}")
    time.sleep(0.5)

# Also try sending a REST control command
print("\nTrying REST control command (throttle=0.5, steering=0)…")
try:
    r = http.post(f"{SERVER}/api/session/{session_id}/control",
                  json={"throttle": 0.5, "steering": 0.0})
    print(f"  status={r.status_code}  body={r.text[:200]}")
except Exception as e:
    print(f"  error: {e}")

http.delete(f"{SERVER}/api/session/{session_id}")
print("\ndone.")
