"""Debug script — prints every raw WS message for 30 seconds.

Run this, open the browser URL when prompted, and let it capture messages.
This shows us the exact message format the server uses.

    python debug_ws.py
"""
import json, time, threading, webbrowser
import websocket, requests

SERVER = "https://ml.ferit.tech"
http = requests.Session()

# Create session
resp = http.post(f"{SERVER}/api/session", json={
    "mode": "time_trial",
    "player_name": "debug",
    "config": {"seed": 42},
})
resp.raise_for_status()
data = resp.json()
session_id = data["session_id"]
browser_url = data.get("browser_url", f"{SERVER}/?session={session_id}")

print(f"Session: {session_id}")
print(f"Opening browser: {browser_url}")
webbrowser.open(browser_url)

ws_url = f"wss://{SERVER.split('//')[1]}/ws?session_id={session_id}"
messages = []

def on_message(ws, msg):
    parsed = json.loads(msg)
    messages.append(parsed)
    print(f"[{time.strftime('%H:%M:%S')}] type={parsed.get('type')}  "
          f"keys={list(parsed.keys())}")
    # Print first non-ping message in full
    if parsed.get("type") not in ("pong", "ping") and len(messages) <= 3:
        print("  FULL:", json.dumps(parsed, indent=2)[:800])

def on_open(ws):
    print("WS connected — waiting for messages (30s)…")

def on_error(ws, err):
    print("WS error:", err)

def on_close(ws, code, msg):
    print("WS closed")

ws = websocket.WebSocketApp(ws_url,
    on_message=on_message, on_open=on_open,
    on_error=on_error, on_close=on_close)

t = threading.Thread(target=ws.run_forever, daemon=True)
t.start()
time.sleep(30)
ws.close()

print(f"\nTotal messages received: {len(messages)}")
if messages:
    types = {}
    for m in messages:
        k = m.get("type", "UNKNOWN")
        types[k] = types.get(k, 0) + 1
    print("Message type counts:", types)
    print("\nFirst message keys:", list(messages[0].keys()))

# Cleanup
http.delete(f"{SERVER}/api/session/{session_id}")
