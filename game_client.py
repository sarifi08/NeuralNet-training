"""
ML Simulation Game Client — Python SDK

Provides a simple interface for students to interact with the 3D simulation
without worrying about HTTP requests or WebSocket management.

Usage:
    from game_client import GameClient

    client = GameClient("https://ml.ferit.tech", api_key="mlsim_abc123...")
    session = client.create_session(mode="target_practice")
    data = client.fire_projectile(angle=45, force=70)
    sensors = client.get_sensors()
    client.send_control(throttle=0.8, steering=-0.3)
"""

import json
import math
import time
import threading
import numpy as np
import requests
import websocket


# Ground-friction lookup keyed by terrain ID (mirror of TERRAIN_TYPES in
# shared/types.ts). Same values, same order: grass, dirt, sand, mud, ice,
# rock, pavement.
TERRAIN_FRICTION = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.4, 5: 1.2, 6: 1.1}

# Ray angles in degrees, ported verbatim from src/agents/SensorSystem.ts:
# RAYCAST_ANGLES. 0 = forward, increasing CCW relative to heading.
RAYCAST_ANGLES_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
RAYCAST_MAX_RANGE = 50.0


def _parse_world_map_payload(raw: dict) -> dict:
    """Convert a WorldMapSnapshot JSON payload into the numpy-backed dict
    used by _compute_grid_local() and _compute_rays_8().
    """
    gs = int(raw["grid_size"])
    return {
        "resolution": float(raw["resolution"]),
        "world_size": float(raw["world_size"]),
        "grid_size": gs,
        "x_min": float(raw["x_min"]),
        "z_min": float(raw["z_min"]),
        "terrain_ids": np.asarray(raw["terrain_ids"], dtype=np.float32).reshape(gs, gs),
        "elevations": np.asarray(raw["elevations"], dtype=np.float32).reshape(gs, gs),
        "obstacles": np.asarray(raw["obstacles"], dtype=np.float32).reshape(gs, gs),
        "checkpoints": list(raw.get("checkpoints", []) or []),
        "world_version": int(raw.get("world_version", 0)),
    }


def _compute_grid_local(cache: dict, px: float, pz: float, heading: float, cps_completed: int) -> np.ndarray:
    """Build the 32x32x4 heading-aligned CNN grid from a cached snapshot."""
    GRID = 32
    CELL = 2.0
    HALF = (GRID - 1) / 2.0
    cols = np.arange(GRID, dtype=np.float32)
    rows = np.arange(GRID, dtype=np.float32)
    cc, rr = np.meshgrid(cols, rows)
    local_x = (cc - HALF) * CELL
    local_z = (rr - HALF) * CELL
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    world_x = px + local_x * cos_h - local_z * sin_h
    world_z = pz + local_x * sin_h + local_z * cos_h

    res = cache["resolution"]
    gs = cache["grid_size"]
    ix = np.floor((world_x - cache["x_min"]) / res).astype(np.int32)
    iz = np.floor((world_z - cache["z_min"]) / res).astype(np.int32)
    ix_c = np.clip(ix, 0, gs - 1)
    iz_c = np.clip(iz, 0, gs - 1)

    terrain = cache["terrain_ids"][iz_c, ix_c]
    elev = cache["elevations"][iz_c, ix_c]
    obs = cache["obstacles"][iz_c, ix_c]

    oob = (np.abs(world_x) > 100) | (np.abs(world_z) > 100)
    obs = np.where(oob, 1.0, obs).astype(np.float32)

    ch0 = (terrain / 6.0).astype(np.float32)
    MIN_H = -4.0
    MAX_H = 4.0
    ch1 = np.clip((elev - MIN_H) / (MAX_H - MIN_H), 0.0, 1.0).astype(np.float32)
    ch2 = obs

    checkpoints = cache["checkpoints"]
    if checkpoints:
        target = checkpoints[cps_completed % len(checkpoints)]["position"]
        tx = float(target.get("x", 0.0))
        tz = float(target.get("z", 0.0))
        dx = world_x - tx
        dz = world_z - tz
        dist = np.sqrt(dx * dx + dz * dz)
        ch3 = np.clip(1.0 - dist / 200.0, 0.0, 1.0).astype(np.float32)
    else:
        ch3 = np.zeros_like(ch0, dtype=np.float32)

    return np.stack([ch0, ch1, ch2, ch3], axis=0)


def _compute_rays_8(cache: dict, px: float, pz: float, heading: float,
                    max_dist: float = RAYCAST_MAX_RANGE) -> list:
    """8-direction obstacle raycast against the cached obstacle grid."""
    res = float(cache["resolution"])
    gs = int(cache["grid_size"])
    x_min = float(cache["x_min"])
    z_min = float(cache["z_min"])
    obstacles = cache["obstacles"]

    step = 0.5
    n_steps = int(max_dist / step) + 1

    distances = []
    for angle_deg in RAYCAST_ANGLES_DEG:
        angle_rad = (angle_deg * math.pi) / 180.0 + heading
        dir_x = -math.sin(angle_rad)
        dir_z = -math.cos(angle_rad)

        hit_dist = max_dist
        for k in range(1, n_steps + 1):
            d = k * step
            wx = px + dir_x * d
            wz = pz + dir_z * d
            if abs(wx) > 100 or abs(wz) > 100:
                hit_dist = d
                break
            ix = int(math.floor((wx - x_min) / res))
            iz = int(math.floor((wz - z_min) / res))
            if ix < 0 or ix >= gs or iz < 0 or iz >= gs:
                hit_dist = d
                break
            if obstacles[iz, ix] >= 0.5:
                hit_dist = d
                break
        distances.append(float(min(hit_dist, max_dist)))

    return distances


class GameClient:
    """Client for the ML simulation game server."""

    def __init__(self, server_url: str = "https://ml.ferit.tech", api_key: str = "None"):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.session_id = None
        self._ws = None
        self._ws_thread = None
        self._latest_state = None
        self._state_lock = threading.Lock()
        self._callbacks = {}
        self._world_map = None

        self._http = requests.Session()
        if api_key:
            self._http.headers["X-API-Key"] = api_key

    # ── Session Management ────────────────────────────────────────────────────

    def create_session(self, mode: str = "free_play", player_name: str = "student",
                       config: dict = None) -> dict:
        payload = {"mode": mode, "player_name": player_name, "config": config or {}}
        resp = self._http.post(f"{self.server_url}/api/session", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        browser_url = data.get("browser_url", f"http://localhost:5173/?session={self.session_id}")
        print(f"Session created: {self.session_id} (mode: {mode})")
        print(f"Open this URL in your browser: {browser_url}")
        return data

    def get_state(self) -> dict:
        self._check_session()
        resp = self._http.get(f"{self.server_url}/api/session/{self.session_id}/state")
        resp.raise_for_status()
        return resp.json()

    def configure(self, **kwargs) -> dict:
        self._check_session()
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/configure", json=kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def delete_session(self):
        if self.session_id:
            self._http.delete(f"{self.server_url}/api/session/{self.session_id}")
            self.session_id = None
        self.disconnect_ws()

    # ── Sensors ───────────────────────────────────────────────────────────────

    def get_sensors(self) -> dict:
        self._check_session()
        resp = self._http.get(f"{self.server_url}/api/session/{self.session_id}/sensors")
        resp.raise_for_status()
        return resp.json()

    def get_ground_grid(self) -> dict:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/grid"
        )
        resp.raise_for_status()
        return resp.json()

    def get_grid_observation(self) -> np.ndarray:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/grid32"
        )
        resp.raise_for_status()
        data = resp.json()
        grid = np.array(data["grid"]["data"])
        return grid.transpose(2, 0, 1)

    def cache_world_map(self, force: bool = False) -> dict:
        if self._world_map is not None and not force:
            return self._world_map
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/world_map"
        )
        resp.raise_for_status()
        cache = _parse_world_map_payload(resp.json())
        self._world_map = cache
        return cache

    def get_grid_local(self) -> np.ndarray:
        if self._world_map is None:
            self.cache_world_map()
        cache = self._world_map

        with self._state_lock:
            state = self._latest_state
        if state is None:
            raise RuntimeError("no live state yet — call connect_ws() and wait one tick")

        pos = state.get("position") or {}
        px = float(pos.get("x", 0.0))
        pz = float(pos.get("z", 0.0))
        heading = float(state.get("heading", 0.0))

        sensors = state.get("sensors") or {}
        nav = sensors.get("navigation") or {}
        cps_completed = int(nav.get("checkpoints_completed", 0))

        return _compute_grid_local(cache, px, pz, heading, cps_completed)

    def get_sensor_history(self, count: int = 100) -> list:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/history",
            params={"count": count},
        )
        resp.raise_for_status()
        return resp.json()["history"]

    # ── Projectiles ───────────────────────────────────────────────────────────

    def fire_projectile(self, angle: float = 45, force: float = 50,
                        yaw_offset: float = 0) -> dict:
        self._check_session()
        payload = {"angle": angle, "force": force, "yaw_offset": yaw_offset}
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/fire",
            json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def fire_batch(self, projectiles: list) -> list:
        self._check_session()
        payload = {"projectiles": projectiles}
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/fire/batch",
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def get_targets(self) -> list:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/targets"
        )
        resp.raise_for_status()
        return resp.json()["targets"]

    def get_wind(self) -> dict:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/wind"
        )
        resp.raise_for_status()
        return resp.json()

    # ── Agent Control ─────────────────────────────────────────────────────────

    def send_control(self, throttle: float = 0, steering: float = 0) -> dict:
        self._check_session()
        payload = {"throttle": throttle, "steering": steering}
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/control", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    # ── WebSocket (Real-time Control) ─────────────────────────────────────────

    def connect_ws(self, on_state=None):
        self._check_session()
        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        ws_url = f"{ws_scheme}://{self.server_url.split('//')[1]}/ws?session_id={self.session_id}"
        if self.api_key:
            ws_url += f"&api_key={self.api_key}"

        if on_state:
            self._callbacks["state"] = on_state

        def on_message(ws, message):
            data = json.loads(message)
            if data.get("type") == "state":
                with self._state_lock:
                    self._latest_state = data
                if "state" in self._callbacks:
                    self._callbacks["state"](data)
            elif data.get("type") == "pong":
                pass
            else:
                event_type = data.get("type")
                if event_type in self._callbacks:
                    self._callbacks[event_type](data)

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status, close_msg):
            print("WebSocket disconnected")

        def on_open(ws):
            print(f"WebSocket connected (session: {self.session_id})")

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        time.sleep(0.5)

    def send_control_ws(self, throttle: float = 0, steering: float = 0):
        if not self._ws:
            raise RuntimeError("WebSocket not connected. Call connect_ws() first.")
        msg = json.dumps({
            "type": "control",
            "session_id": self.session_id,
            "throttle": float(np.clip(throttle, -1, 1)),
            "steering": float(np.clip(steering, -1, 1)),
        })
        self._ws.send(msg)

    def get_latest_state(self) -> dict:
        with self._state_lock:
            return self._latest_state

    def on_event(self, event_type: str, callback):
        self._callbacks[event_type] = callback

    def disconnect_ws(self):
        if self._ws:
            self._ws.close()
            self._ws = None

    # ── Recording (Behavioral Cloning) ────────────────────────────────────────

    def start_recording(self, sample_rate: int = 20, include_grid: bool = False) -> dict:
        self._check_session()
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/recording/start",
            json={"sample_rate": sample_rate, "include_grid": include_grid},
        )
        resp.raise_for_status()
        return resp.json()

    def stop_recording(self) -> dict:
        self._check_session()
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/recording/stop"
        )
        resp.raise_for_status()
        return resp.json()

    def get_recording(self) -> dict:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/recording"
        )
        resp.raise_for_status()
        return resp.json()

    def get_recording_as_arrays(self) -> tuple:
        recording = self.get_recording()
        states = []
        actions = []
        for sample in recording["samples"]:
            s = sample["state"]
            state_vec = [
                s["speed"],
                s["heading_error"],
                s["checkpoint_distance"],
                *s["rays"],
                s["ground_friction"],
            ]
            action_vec = [sample["action"]["throttle"], sample["action"]["steering"]]
            states.append(state_vec)
            actions.append(action_vec)
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

    def get_recording_positions(self) -> np.ndarray:
        recording = self.get_recording()
        positions = []
        for sample in recording["samples"]:
            s = sample["state"]
            positions.append([s.get("position_x", 0.0), s.get("position_z", 0.0)])
        return np.array(positions, dtype=np.float32)

    def get_recording_with_grid(self) -> tuple:
        recording = self.get_recording()
        states = []
        actions = []
        grids = []
        for sample in recording["samples"]:
            s = sample["state"]
            if "grid32" not in s:
                raise RuntimeError(
                    "recording has no grid32 samples — start with "
                    "start_recording(include_grid=True)"
                )
            state_vec = [
                s["speed"],
                s["heading_error"],
                s["checkpoint_distance"],
                *s["rays"],
                s["ground_friction"],
            ]
            action_vec = [sample["action"]["throttle"], sample["action"]["steering"]]
            states.append(state_vec)
            actions.append(action_vec)
            grids.append(s["grid32"])
        grid_stack = np.array(grids, dtype=np.float32).reshape(-1, 32, 32, 4)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            grid_stack,
        )

    # ── Map & Exploration ─────────────────────────────────────────────────────

    def get_explored_map(self) -> dict:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/map/explored"
        )
        resp.raise_for_status()
        return resp.json()

    def get_terrain_ground_truth(self) -> list:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/map/terrain"
        )
        resp.raise_for_status()
        return resp.json()["samples"]

    # ── Anomaly System ────────────────────────────────────────────────────────

    def configure_anomalies(self, enabled: bool = True, malfunction_rate: float = 0.1,
                            terrain_anomaly_rate: float = 0.05, malfunction_types: list = None,
                            duration_range: tuple = (10, 50)) -> dict:
        self._check_session()
        payload = {
            "enabled": enabled,
            "agent_malfunction_rate": malfunction_rate,
            "terrain_anomaly_rate": terrain_anomaly_rate,
            "malfunction_types": malfunction_types or ["steering_invert", "throttle_scale", "random_jitter"],
            "malfunction_duration_range": list(duration_range),
        }
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/anomalies/configure",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def get_anomaly_labels(self) -> list:
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/anomalies/labels"
        )
        resp.raise_for_status()
        return resp.json()["labels"]

    # ── Competition ───────────────────────────────────────────────────────────

    def join_competition(self, student_id: str, agent_name: str) -> dict:
        payload = {"student_id": student_id, "agent_name": agent_name}
        resp = self._http.post(f"{self.server_url}/api/competition/join", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"Joined competition room: {data['room_id']} ({data['players']} players)")
        return data

    def get_competition_state(self) -> dict:
        resp = self._http.get(f"{self.server_url}/api/competition/state")
        resp.raise_for_status()
        return resp.json()

    def get_leaderboard(self) -> dict:
        resp = self._http.get(f"{self.server_url}/api/competition/leaderboard")
        resp.raise_for_status()
        return resp.json()

    def connect_competition_ws(self, student_id: str, agent_name: str, on_state=None):
        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        ws_url = (
            f"{ws_scheme}://{self.server_url.split('//')[1]}/ws/competition"
            f"?student_id={student_id}&agent_name={agent_name}"
        )
        if self.api_key:
            ws_url += f"&api_key={self.api_key}"

        if on_state:
            self._callbacks["competition_state"] = on_state

        def on_message(ws, message):
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "state":
                with self._state_lock:
                    self._latest_state = data
                if "state" in self._callbacks:
                    self._callbacks["state"](data)
            elif msg_type == "competition_state":
                if "competition_state" in self._callbacks:
                    self._callbacks["competition_state"](data)
            elif msg_type in self._callbacks:
                self._callbacks[msg_type](data)

        def on_error(ws, error):
            print(f"Competition WebSocket error: {error}")

        def on_close(ws, close_status, close_msg):
            print("Competition WebSocket disconnected")

        def on_open(ws):
            print(f"Competition WebSocket connected ({student_id}: {agent_name})")

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        time.sleep(0.5)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def collect_sensor_data(self, n_samples: int, interval: float = 0.2) -> list:
        self._check_session()
        readings = []
        for i in range(n_samples):
            reading = self.get_sensors()
            readings.append(reading)
            if i < n_samples - 1:
                time.sleep(interval)
            if (i + 1) % 50 == 0:
                print(f"Collected {i + 1}/{n_samples} samples")
        print(f"Collection complete: {len(readings)} samples")
        return readings

    def run_control_loop(self, policy_fn, duration: float = 60, hz: float = 20):
        self._check_session()
        if not self._ws:
            self.connect_ws()
            time.sleep(0.5)

        interval = 1.0 / hz
        start_time = time.time()
        steps = 0

        print(f"Running control loop at {hz}Hz for {duration}s...")
        try:
            while time.time() - start_time < duration:
                state = self.get_latest_state()
                if state is not None:
                    throttle, steering = policy_fn(state)
                    self.send_control_ws(throttle, steering)
                    steps += 1
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Control loop interrupted")

        elapsed = time.time() - start_time
        print(f"Control loop finished: {steps} steps in {elapsed:.1f}s ({steps/elapsed:.1f} Hz)")

    def _check_session(self):
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect_ws()
        self.delete_session()

    def __repr__(self):
        status = f"session={self.session_id}" if self.session_id else "no session"
        ws = "ws=connected" if self._ws else "ws=disconnected"
        return f"GameClient({self.server_url}, {status}, {ws})"


# ─────────────────────────────────────────────────────────────────────────────
#  Tournament client — RoomBot
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (heading) from a quaternion using THREE.js YXZ Euler order."""
    sy = 2.0 * (qw * qy - qx * qz)
    cy = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(sy, cy)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class RoomBot:
    """Tournament client. Plug a controller in, call run(), drive until the
    tournament ends.

    Example:
        from game_client import RoomBot

        def my_controller(obs):
            return 0.7, obs["navigation"]["heading_error"] * 0.4

        bot = RoomBot("https://ml.ferit.tech", room="demo", name="Alice")
        standings = bot.run(my_controller, hz=20.0)
        print(standings)
    """

    def __init__(self, server_url: str = "https://ml.ferit.tech", room: str = "main",
                 name: str = "bot", api_key: str = None):
        self.server_url = server_url.rstrip("/")
        self.room = room
        self.name = name
        self.api_key = api_key

        self._ws = None
        self._ws_thread = None
        self._stop = threading.Event()
        self._connected = threading.Event()

        self._lock = threading.Lock()
        self._bot_key = None
        self._phase = "lobby"
        self._round_index = 0
        self._latest_bots = []
        self._latest_state_t = 0
        self._world_map = None
        self._world_map_round = -1
        self._standings = []
        self._tournament_done = False
        self._last_pos_for_speed = None
        self._last_speed = 0.0

        self._http = requests.Session()
        if api_key:
            self._http.headers["X-API-Key"] = api_key

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, controller, hz: float = 20.0) -> list:
        """Connect, ready up, drive at `hz` until tournament_end."""
        if self._ws is None:
            self._connect()
        self._connected.wait(timeout=5.0)

        period = 1.0 / hz
        dt_warn_threshold = 0.100
        avg_dt = period
        next_tick = time.time()
        try:
            while not self._stop.is_set():
                with self._lock:
                    done = self._tournament_done
                if done:
                    break
                t0 = time.time()
                obs = self._build_obs()
                if obs is not None:
                    try:
                        out = controller(obs)
                        throttle, steering = float(out[0]), float(out[1])
                    except Exception as e:
                        print(f"[RoomBot:{self.name}] controller error: {e}")
                        throttle, steering = 0.0, 0.0
                    if self._phase == "racing":
                        self._send_control(throttle, steering)
                dt = time.time() - t0
                avg_dt = 0.9 * avg_dt + 0.1 * dt
                if avg_dt > dt_warn_threshold:
                    print(f"[RoomBot:{self.name}] WARNING: controller avg dt {avg_dt*1000:.0f}ms "
                          f"exceeds budget — frames will drop")
                    avg_dt = period
                next_tick += period
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.time()
        except KeyboardInterrupt:
            print(f"[RoomBot:{self.name}] interrupted")
        finally:
            self.disconnect()
        return list(self._standings)

    def disconnect(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── WS lifecycle ──────────────────────────────────────────────────────────

    def _connect(self) -> None:
        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        host = self.server_url.split("//", 1)[1]
        ws_url = f"{ws_scheme}://{host}/ws/room/bot?room={self.room}&name={self.name}"
        if self.api_key:
            ws_url += f"&api_key={self.api_key}"

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def _on_open(self, ws):
        print(f"[RoomBot:{self.name}] connected to room '{self.room}' — signaling ready")
        try:
            ws.send(json.dumps({"type": "ready", "ready": True}))
        except Exception:
            pass

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        if t == "bot_assigned":
            with self._lock:
                self._bot_key = msg.get("bot_key")
                rs = msg.get("room_state") or {}
                self._phase = rs.get("phase", "lobby")
                self._round_index = int(rs.get("round_index", 0))
            print(f"[RoomBot:{self.name}] bot_key={self._bot_key}")
            self._connected.set()
        elif t == "round_start":
            ridx = int(msg.get("round_index", 0))
            with self._lock:
                self._phase = "racing"
                self._round_index = ridx
                self._last_pos_for_speed = None
                self._last_speed = 0.0
            print(f"[RoomBot:{self.name}] round_start idx={ridx} seed={msg.get('seed')} "
                  f"obstacles={msg.get('obstacles')}")
            self._fetch_world_map(ridx)
        elif t == "state_update":
            bots = msg.get("bots") or []
            with self._lock:
                self._latest_bots = bots
                self._latest_state_t = int(msg.get("t", 0))
        elif t == "round_end":
            with self._lock:
                self._phase = "round_end"
            print(f"[RoomBot:{self.name}] round_end idx={msg.get('round_index')}")
        elif t == "tournament_end":
            standings = msg.get("standings") or []
            with self._lock:
                self._phase = "finished"
                self._standings = standings
                self._tournament_done = True
            print(f"[RoomBot:{self.name}] tournament_end")
            for r in standings:
                print(f"  #{r.get('rank')} {r.get('name')} cps={r.get('total_checkpoints')}")
        elif t == "error":
            code = msg.get("code")
            print(f"[RoomBot:{self.name}] error: {code} {msg.get('message')}")
            if code in ("auth_failed", "unauthorized", "forbidden"):
                self._stop.set()

    def _on_error(self, ws, err):
        print(f"[RoomBot:{self.name}] ws error: {err}")

    def _on_close(self, ws, code, reason):
        print(f"[RoomBot:{self.name}] disconnected ({code} {reason})")
        self._connected.set()
        self._stop.set()

    # ── observation building ──────────────────────────────────────────────────

    def _fetch_world_map(self, round_index: int) -> None:
        url = f"{self.server_url}/api/room/{self.room}/world_map"
        try:
            for attempt in range(5):
                resp = self._http.get(url, timeout=3.0)
                if resp.status_code == 200:
                    cache = _parse_world_map_payload(resp.json())
                    with self._lock:
                        self._world_map = cache
                        self._world_map_round = round_index
                    print(f"[RoomBot:{self.name}] cached world_map for round {round_index}")
                    return
                if resp.status_code in (404, 504):
                    time.sleep(0.5)
                    continue
                resp.raise_for_status()
            print(f"[RoomBot:{self.name}] world_map fetch failed after retries")
        except Exception as e:
            print(f"[RoomBot:{self.name}] world_map fetch error: {e}")

    def _self_state(self) -> dict:
        with self._lock:
            bot_key = self._bot_key
            bots = list(self._latest_bots)
        if not bot_key:
            return None
        for b in bots:
            if b.get("bot_key") == bot_key:
                return b
        return None

    def _other_bots(self) -> list:
        with self._lock:
            bot_key = self._bot_key
            bots = list(self._latest_bots)
        return [b for b in bots if b.get("bot_key") != bot_key]

    def _build_obs(self) -> dict:
        self_state = self._self_state()
        if self_state is None:
            return None
        pos = self_state.get("position") or {}
        rot = self_state.get("rotation") or {}
        px = float(pos.get("x", 0.0))
        py = float(pos.get("y", 0.0))
        pz = float(pos.get("z", 0.0))
        heading = _quat_to_yaw(
            float(rot.get("x", 0.0)),
            float(rot.get("y", 0.0)),
            float(rot.get("z", 0.0)),
            float(rot.get("w", 1.0)),
        )

        now = time.time()
        with self._lock:
            last = self._last_pos_for_speed
        if last is None:
            speed = 0.0
        else:
            lx, lz, lt = last
            dt = max(now - lt, 1e-3)
            speed = math.sqrt((px - lx) ** 2 + (pz - lz) ** 2) / dt
            speed = 0.6 * speed + 0.4 * self._last_speed
        with self._lock:
            self._last_pos_for_speed = (px, pz, now)
            self._last_speed = speed

        cps_completed = int(self_state.get("checkpoints", 0))

        with self._lock:
            cache = self._world_map
            round_idx = self._round_index
            phase = self._phase

        if cache is not None:
            rays = _compute_rays_8(cache, px, pz, heading)
            ground_friction = self._lookup_ground_friction(cache, px, pz)
            grid32 = _compute_grid_local(cache, px, pz, heading, cps_completed)
            navigation = self._compute_navigation(cache, px, pz, heading, cps_completed)
        else:
            rays = [RAYCAST_MAX_RANGE] * 8
            ground_friction = 1.0
            grid32 = np.zeros((4, 32, 32), dtype=np.float32)
            navigation = {"distance": 0.0, "heading_error": 0.0, "checkpoint_index": 0}

        return {
            "position": {"x": px, "y": py, "z": pz},
            "heading": heading,
            "speed": float(speed),
            "rays": rays,
            "ground_friction": float(ground_friction),
            "grid32": grid32,
            "navigation": navigation,
            "checkpoints_passed": cps_completed,
            "round_index": round_idx,
            "race_phase": phase,
            "other_bots": self._other_bots(),
        }

    def _lookup_ground_friction(self, cache: dict, px: float, pz: float) -> float:
        res = float(cache["resolution"])
        gs = int(cache["grid_size"])
        ix = int(math.floor((px - float(cache["x_min"])) / res))
        iz = int(math.floor((pz - float(cache["z_min"])) / res))
        if ix < 0 or ix >= gs or iz < 0 or iz >= gs:
            return 1.0
        tid = int(cache["terrain_ids"][iz, ix])
        return TERRAIN_FRICTION.get(tid, 1.0)

    def _compute_navigation(self, cache: dict, px: float, pz: float,
                            heading: float, cps_completed: int) -> dict:
        checkpoints = cache.get("checkpoints") or []
        if not checkpoints:
            return {"distance": 0.0, "heading_error": 0.0, "checkpoint_index": 0}
        idx = cps_completed % len(checkpoints)
        target = checkpoints[idx].get("position") or {}
        tx = float(target.get("x", 0.0))
        tz = float(target.get("z", 0.0))
        dx = tx - px
        dz = tz - pz
        distance = math.sqrt(dx * dx + dz * dz)
        target_angle = math.atan2(-dx, -dz)
        heading_error = _wrap_pi(target_angle - heading)
        return {
            "distance": float(distance),
            "heading_error": float(heading_error),
            "checkpoint_index": int(idx),
        }

    def _send_control(self, throttle: float, steering: float) -> None:
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({
                "type": "control",
                "throttle": float(np.clip(throttle, -1.0, 1.0)),
                "steering": float(np.clip(steering, -1.0, 1.0)),
            }))
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self) -> str:
        return f"RoomBot({self.server_url}, room={self.room}, name={self.name})"
