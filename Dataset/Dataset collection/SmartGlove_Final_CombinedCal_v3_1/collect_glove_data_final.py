"""
SMART GLOVE - DATA COLLECTOR  V3  (runs on the laptop)

Works with the ESP32 firmware SMART_GLOVE_V3. The glove runs on a battery and
connects to this program over the laptop's Wi-Fi hotspot (TCP port 5000).

Start it in PowerShell from the folder that contains this file:
    python collect_glove_data.py

Only the Python standard library is used (Python 3.8 or newer).
Everything is saved in the folder SmartGlove_Dataset next to this file.

Participant flow: quick sensor check -> ONE 5 s combined neutral calibration
(flex baseline + accel baseline + gyro bias + orientation reference) -> 3 s
fist reference -> 10 balanced rounds -> final 5 s neutral check.
There is no separate gyro-bias pose, wiggle test, dry run, or 30-minute test.
"""

import csv
import math
import os
import select
import shutil
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# 1. SETTINGS
# ============================================================================

HOST = "0.0.0.0"
PORT = 5000

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "SmartGlove_Dataset"
REAL_DIR = DATASET_DIR / "raw"        # real participants  P01, P02, ...
PILOT_DIR = DATASET_DIR / "pilot"     # dry runs           X01, X02, ...

GESTURES = [
    ("G01", "Fist"),
    ("G02", "Excellent"),
    ("G03", "Stop"),
    ("G04", "Thumbs Up"),
    ("G05", "This Way"),
    ("G06", "Wait"),
    ("G07", "Hey You!"),
    ("G08", "Victory"),
    ("G09", "Call Me"),
    ("G10", "Attention"),
]
N_ROUNDS = 5                          # every gesture once per round -> 5 repetitions each
COLLECTOR_VERSION = "3.1.0"

# Balanced (Williams) order for 10 gestures. Participant p uses rows p, p+1, ... p+9.
# Within every participant: each gesture appears once per round, once in every
# position, and every gesture-to-gesture transition happens exactly once.
WILLIAMS_BASE = [0, 1, 9, 2, 8, 3, 7, 4, 6, 5]

TRIAL_SAMPLES, CAL_SAMPLES, FIST_SAMPLES, CHECK_SAMPLES, DRIFT_SAMPLES = 50, 125, 75, 50, 125

# Quality-check limits (tune them with the dry runs; every value is logged).
STILL_MAX_SPREAD_DEG = 2.0    # calibration/checkpoint: max rotation away from the mean
STILL_MAX_DPS = 20.0          # calibration/checkpoint: peak rotation speed (200 Hz, from the glove)
TRIAL_WARN_DPS = 30.0         # trial: warn "hand may have moved" above this peak speed
TRIAL_WARN_SPREAD_DEG = 3.0   # trial: warn above this rotation during the 2 s hold
FLEX_WARN_RANGE = 200         # trial: warn if a finger reading changes more than this in 2 s
FLEX_LOW, FLEX_HIGH = 5, 4090 # readings at or beyond these are treated as saturated
INTERVAL_MS, INTERVAL_TOL_MS = 40, 3
LATE_MAX_US = 3000            # a sample started more than 3 ms late = timing fault
MAX_AUTO_REDO = 3             # after this many failed tries: real data must redo/pause; pilot may override
WIGGLE_SECONDS = float(os.environ.get("GLOVE_WIGGLE_SECONDS", "20"))
WIGGLE_MIN_FLEX_RANGE = 100   # each finger should change at least this much while wiggling

TIMEOUTS = {"PING": 3, "STATUS": 3, "CAL": 12, "FIST": 9, "CHECK": 7,
            "DRIFT": 12, "TRIAL": 7, "TEST": 3}

CSV_HEADER = [
    "timestamp_ms", "trial_id",
    "flex_thumb", "flex_index", "flex_middle", "flex_ring", "flex_little",
    "qw", "qx", "qy", "qz",
    "gx", "gy", "gz",
    "ax", "ay", "az",
    "roll", "pitch", "yaw",
]
INT_COLUMNS = {0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16}

TRIAL_LOG_HEADER = [
    "time", "session", "round", "position", "gesture_id", "gesture_name", "trial_id", "attempt",
    "result", "reason", "max_dps", "spread_deg", "flex_range_max", "flex_saturated",
    "interval_min_ms", "interval_max_ms", "late_max_us", "imu_missed", "temp_c", "rail_mv",
    "epoch", "boot", "ref_id", "data_file",
]
SESSION_LOG_HEADER = ["time", "session", "event", "details"]
REFERENCE_HEADER = ["time", "session", "ref_id", "attempt", "qw", "qx", "qy", "qz",
                    "spread_deg", "max_dps", "still_ok", "override", "boot", "epoch"]
CAL_SUMMARY_HEADER = [
    "time", "session", "ref_id",
    "flex_thumb_mean", "flex_index_mean", "flex_middle_mean", "flex_ring_mean", "flex_little_mean",
    "gyro_bias_x_dps", "gyro_bias_y_dps", "gyro_bias_z_dps", "gyro_bias_std_max_dps", "gyro_bias_dev_max_dps",
    "accel_x_mean_g", "accel_y_mean_g", "accel_z_mean_g",
    "qw", "qx", "qy", "qz", "spread_deg", "max_dps", "temp_c", "boot", "epoch"
]
PARTICIPANT_HEADER = ["participant_id", "kind", "created", "hand_length_mm", "hand_breadth_mm",
                      "age_band", "notes", "order_start_row"]
DEFINITION_HEADER = ["gesture_id", "gesture_name", "description", "arm_posture"]
DEFAULT_POSTURE = "Elbow on the table, forearm as in the neutral pose"
PLACEHOLDER = "(fill in: short description of the hand shape)"


class LinkError(Exception):
    """The connection to the glove was lost or timed out."""


class ProtocolError(Exception):
    """The glove sent something we did not expect."""


class PauseRequested(Exception):
    """The operator pressed Q / Ctrl+C."""


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def line(char="=", n=70):
    print(char * n)


def ask(prompt, allowed=None):
    """input() that turns Ctrl+C / end-of-input into a pause request."""
    while True:
        try:
            answer = input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            raise PauseRequested()
        if allowed is None or answer in allowed:
            return answer
        print("  Please type one of: " + ", ".join(a if a else "ENTER" for a in allowed))


# ============================================================================
# 2. FILES
# Normal acquisition is append-only and flushed to disk. Crash recovery may
# reconstruct a damaged gesture CSV, but only after saving the complete old
# file plus removed rows in quarantine/.
# ============================================================================

def _retry_on_lock(action, path):
    while True:
        try:
            return action()
        except PermissionError:
            print()
            print(f"  Cannot write {path.name}. Is it open in Excel or another program?")
            print("  Close it, then press ENTER to try again. (Nothing has been lost.)")
            try:
                input("  > ")
            except (KeyboardInterrupt, EOFError):
                print()


def append_rows(path, header, rows):
    """Append rows to a CSV (header if the file is new), then force them to disk."""
    def action():
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
    _retry_on_lock(action, path)


def write_new_file(path, header, rows):
    """Write a complete file in one step (temporary file, then rename)."""
    tmp = path.with_name(path.name + ".tmp")

    def action():
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    _retry_on_lock(action, path)


def read_csv_dicts(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_event(pdir, session, event, details=""):
    append_rows(pdir / "session_log.csv", SESSION_LOG_HEADER, [[now_iso(), session, event, details]])


def ensure_dataset():
    for d in (REAL_DIR, PILOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    gmap = DATASET_DIR / "gesture_map.csv"
    if not gmap.exists():
        write_new_file(gmap, ["gesture_id", "gesture_name"], [list(g) for g in GESTURES])
    gdef = DATASET_DIR / "gesture_definitions.csv"
    if not gdef.exists():
        write_new_file(gdef, DEFINITION_HEADER,
                       [[gid, name, PLACEHOLDER, DEFAULT_POSTURE] for gid, name in GESTURES])


def load_definitions():
    defs = {gid: {"description": "", "arm_posture": DEFAULT_POSTURE} for gid, _ in GESTURES}
    for row in read_csv_dicts(DATASET_DIR / "gesture_definitions.csv"):
        gid = (row.get("gesture_id") or "").strip()
        if gid in defs:
            desc = (row.get("description") or "").strip()
            defs[gid]["description"] = "" if desc == PLACEHOLDER else desc
            defs[gid]["arm_posture"] = (row.get("arm_posture") or "").strip() or DEFAULT_POSTURE
    return defs


# ============================================================================
# 3. LINK TO THE GLOVE  (every command has a sequence number)
# ============================================================================

def parse_fields(parts):
    out = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class Reply:
    def __init__(self, kind, code="", fields=None, rows=None, ref=None, stats=None, header=None):
        self.kind = kind            # PONG / STATUS / TEST / ERR / BLOCK
        self.code = code            # error code, block type, or OK
        self.fields = fields or {}
        self.rows = rows or []
        self.ref = ref
        self.stats = stats or {}
        self.header = header or {}


class GloveLink:
    def __init__(self, port=PORT):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # stops a second copy of this program from silently sharing the port
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server.bind((HOST, port))
        except OSError as e:
            print(f"Cannot open TCP port {port}: {e}")
            print("Is another copy of this program already running? Close it and try again.")
            sys.exit(1)
        self.server.listen(4)
        self.conn = None
        self.buf = b""
        self.seq = 0
        self.hello = {}
        self.peer = ""

    # --- connection handling ---
    def close_conn(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None
        self.buf = b""

    def pending(self):
        """True if the glove has opened a new connection that we have not accepted yet."""
        try:
            r, _, _ = select.select([self.server], [], [], 0)
            return bool(r)
        except OSError:
            return False

    def accept(self, timeout=None):
        """Wait for the glove. Returns the HELLO fields, or None on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        warned_old = False
        while True:
            wait = 1.0 if deadline is None else max(0.0, min(1.0, deadline - time.monotonic()))
            r, _, _ = select.select([self.server], [], [], wait)
            if not r:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue
            conn, addr = self.server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.close_conn()
            self.conn, self.peer = conn, addr[0]
            try:
                hello = self.read_line(5.0)
            except (LinkError, ProtocolError):
                self.close_conn()
                continue
            parts = hello.split(",")
            if len(parts) < 2 or parts[0] != "HELLO":
                self.close_conn()
                continue
            if parts[1] != "SMART_GLOVE_V3":
                if not warned_old:
                    print(f"\n  The glove at {addr[0]} runs old firmware ({parts[1]}).")
                    print("  Upload SmartGlove_V3.ino to the ESP32, then power it on again.")
                    warned_old = True
                self.close_conn()
                continue
            self.hello = parse_fields(parts[2:])
            self.hello["name"] = parts[1]
            return self.hello

    # --- low-level I/O ---
    def read_line(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.conn is None:
                raise LinkError("no answer from the glove (timeout)")
            try:
                r, _, _ = select.select([self.conn], [], [], remaining)
                if not r:
                    raise LinkError("no answer from the glove (timeout)")
                chunk = self.conn.recv(8192)
            except OSError as e:
                raise LinkError(f"connection error: {e}")
            if not chunk:
                raise LinkError("the glove closed the connection")
            self.buf += chunk
            if len(self.buf) > 2_000_000:
                raise ProtocolError("too much unexpected data")
        raw, _, self.buf = self.buf.partition(b"\n")
        return raw.decode("ascii", "replace").strip()

    def send(self, text):
        if self.conn is None:
            raise LinkError("not connected")
        try:
            self.conn.sendall((text + "\n").encode("ascii"))
        except OSError as e:
            raise LinkError(f"connection error: {e}")

    # --- request / reply ---
    def request(self, cmd, arg=""):
        self.seq += 1
        seq = self.seq
        self.send(f"{seq} {cmd}" + (f" {arg}" if arg else ""))
        deadline = time.monotonic() + TIMEOUTS.get(cmd, 5)
        while True:
            text = self.read_line(max(0.05, deadline - time.monotonic()))
            if not text:
                continue
            parts = text.split(",")
            tag = parts[0]
            if tag == "BEGIN" and len(parts) >= 3:
                lseq = self._int(parts[2])
                if lseq != seq:
                    self._skip_block(parts[1], parts[2], deadline)   # late answer to an old command
                    continue
                return self._read_block(parts, seq, deadline)
            if tag in ("PONG", "STATUS", "TEST", "ERR") and len(parts) >= 2:
                if self._int(parts[1]) != seq:
                    continue                                          # stale line from an old command
                code = parts[2] if len(parts) > 2 and "=" not in parts[2] else ""
                return Reply(tag, code, parse_fields(parts[2:]))
            # anything else (HELLO, stray rows) is ignored

    @staticmethod
    def _int(text):
        try:
            return int(text)
        except ValueError:
            return -1

    def _skip_block(self, kind, seq_text, deadline):
        end = f"END,{kind},{seq_text}"
        while self.read_line(max(0.05, deadline - time.monotonic())) != end:
            pass

    def _read_block(self, parts, seq, deadline):
        kind = parts[1]
        header = parse_fields(parts[3:])
        try:
            n = int(header.get("n", "-1"))
        except ValueError:
            raise ProtocolError("bad block header: " + ",".join(parts))
        rows = []
        for _ in range(n):
            text = self.read_line(max(0.05, deadline - time.monotonic()))
            row = text.split(",")
            if len(row) != len(CSV_HEADER):
                raise ProtocolError("malformed data row: " + text[:80])
            try:
                for i, v in enumerate(row):
                    int(v) if i in INT_COLUMNS else float(v)
            except ValueError:
                raise ProtocolError("non-numeric data row: " + text[:80])
            rows.append(row)
        ref, stats = None, {}
        while True:
            text = self.read_line(max(0.05, deadline - time.monotonic()))
            p = text.split(",")
            if p[0] == "REF" and len(p) == 6 and self._int(p[1]) == seq:
                try:
                    ref = tuple(float(v) for v in p[2:6])
                except ValueError:
                    raise ProtocolError("bad reference line")
            elif p[0] == "STATS" and len(p) >= 2 and self._int(p[1]) == seq:
                stats = parse_fields(p[2:])
            elif text == f"END,{kind},{seq}":
                return Reply("BLOCK", kind, rows=rows, ref=ref, stats=stats, header=header)
            else:
                raise ProtocolError("unexpected line inside a block: " + text[:80])


# ============================================================================
# 4. QUALITY CHECKS  (orientation maths uses the same conventions as the firmware)
# ============================================================================

def quats(rows):
    return [(float(r[7]), float(r[8]), float(r[9]), float(r[10])) for r in rows]


def normalize(q):
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / n for c in q)


def mean_quat(qs):
    a = normalize(qs[0])
    s = [0.0, 0.0, 0.0, 0.0]
    for q in qs:
        q = normalize(q)
        sign = -1.0 if sum(x * y for x, y in zip(a, q)) < 0 else 1.0
        for i in range(4):
            s[i] += sign * q[i]
    return normalize(tuple(s))


def angle_deg(q1, q2):
    d = abs(sum(x * y for x, y in zip(normalize(q1), normalize(q2))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def spread_deg(rows):
    qs = quats(rows)
    m = mean_quat(qs)
    return max(angle_deg(q, m) for q in qs)


def fnum(d, key, default=float("nan")):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def block_metrics(block):
    rows = block.rows
    flex = [[int(r[c]) for r in rows] for c in range(2, 7)]
    ranges = [max(f) - min(f) for f in flex]
    t = [int(r[0]) for r in rows]
    iv = [b - a for a, b in zip(t, t[1:])] or [INTERVAL_MS]
    return {
        "max_dps": fnum(block.stats, "max_dps"),
        "spread_deg": spread_deg(rows),
        "flex_range_max": max(ranges),
        "flex_ranges": ranges,
        "saturated": any(v <= FLEX_LOW or v >= FLEX_HIGH for f in flex for v in f),
        "interval_min": min(iv),
        "interval_max": max(iv),
        "first_ts": t[0],
        "late_max_us": fnum(block.stats, "late_max_us", 0.0),
        "imu_missed": fnum(block.stats, "imu_missed", 0.0),
        "temp_c": fnum(block.stats, "temp_c"),
        "rail_mv": block.stats.get("rail_mv", ""),
    }


def technical_problems(block, m, expected_n, trial_id=None, session=None):
    """Faults that make a block unusable (never a judgement about the gesture itself)."""
    problems = []
    if len(block.rows) != expected_n:
        problems.append(f"expected {expected_n} samples, got {len(block.rows)}")
    if trial_id is not None and any(int(r[1]) != trial_id for r in block.rows):
        problems.append("trial number mismatch")
    if m["first_ts"] > INTERVAL_TOL_MS or m["interval_min"] < INTERVAL_MS - INTERVAL_TOL_MS \
            or m["interval_max"] > INTERVAL_MS + INTERVAL_TOL_MS:
        problems.append(f"sample timing off ({m['interval_min']}-{m['interval_max']} ms)")
    if m["late_max_us"] > LATE_MAX_US:
        problems.append(f"a sample started {m['late_max_us'] / 1000:.1f} ms late")
    if m["imu_missed"] > 0:
        problems.append(f"IMU missed {int(m['imu_missed'])} sample(s) during capture")
    if session is not None:
        h = block.header
        if h.get("ref") != session.ref_id:
            problems.append("recorded against a different calibration")
        if h.get("boot") != session.boot or str(h.get("epoch")) != str(session.epoch):
            problems.append("glove restarted or orientation reset during recording")
    return problems


def still_ok(m):
    return m["spread_deg"] <= STILL_MAX_SPREAD_DEG and (math.isnan(m["max_dps"]) or m["max_dps"] <= STILL_MAX_DPS)


def describe_still(m):
    return (f"rotation during hold {m['spread_deg']:.2f} deg (limit {STILL_MAX_SPREAD_DEG}), "
            f"peak speed {m['max_dps']:.1f} deg/s (limit {STILL_MAX_DPS})")


# ============================================================================
# 5. PARTICIPANT FILES, ORDER AND CRASH RECOVERY
# ============================================================================

def williams_order(p_index):
    """10 rounds x 10 gesture ids for participant number p_index (0-based)."""
    rows = [[(b + i) % 10 for b in WILLIAMS_BASE] for i in range(10)]
    return [[GESTURES[g][0] for g in rows[(p_index + r) % 10]] for r in range(N_ROUNDS)]


def gesture_name(gid):
    return dict(GESTURES)[gid]


def participant_dirs(root, prefix):
    out = []
    for p in root.glob(prefix + "*"):
        if p.is_dir() and p.name[1:].isdigit() and (p / "order.csv").exists():
            out.append(p)
    return sorted(out, key=lambda p: int(p.name[1:]))


def is_complete(pdir):
    return any(r.get("event") == "PARTICIPANT_COMPLETE" for r in read_csv_dicts(pdir / "session_log.csv"))


def load_order(pdir):
    order = [[None] * len(GESTURES) for _ in range(N_ROUNDS)]
    for r in read_csv_dicts(pdir / "order.csv"):
        order[int(r["round"]) - 1][int(r["position"]) - 1] = r["gesture_id"]
    return order


def next_session_number(pdir):
    n = 0
    for row in read_csv_dicts(pdir / "session_log.csv"):
        s = row.get("session", "")
        if len(s) > 1 and s[0] == "S" and s[1:].isdigit():
            n = max(n, int(s[1:]))
    return n + 1


def unique_path(path):
    if not path.exists():
        return path
    k = 2
    while True:
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not cand.exists():
            return cand
        k += 1


def split_blocks(lines):
    """Split gesture-file lines into trial blocks (new block when the trial id
    changes or the timestamp restarts). Returns [(trial_id or None, [lines])]."""
    blocks, cur, cur_id, last_t = [], [], None, None
    for ln in lines:
        parts = ln.rstrip("\r\n").split(",")
        try:
            tid, t = int(parts[1]), int(parts[0])
        except (ValueError, IndexError):
            tid, t = None, None
        if cur and (tid != cur_id or t is None or last_t is None or t <= last_t):
            blocks.append((cur_id, cur))
            cur = []
        cur.append(ln)
        cur_id, last_t = tid, t
    if cur:
        blocks.append((cur_id, cur))
    return blocks


def block_is_valid(block_lines):
    if len(block_lines) != TRIAL_SAMPLES:
        return False
    for ln in block_lines:
        parts = ln.rstrip("\r\n").split(",")
        if len(parts) != len(CSV_HEADER):
            return False
        try:
            for i, v in enumerate(parts):
                int(v) if i in INT_COLUMNS else float(v)
        except ValueError:
            return False
    return True


def reconcile(pdir):
    """Make the gesture files and trial_log.csv agree after a crash or power cut.
    Nothing is deleted: removed rows and a full backup go to quarantine/."""
    logged = set()
    for r in read_csv_dicts(pdir / "trial_log.csv"):
        if r.get("result", "").startswith("ACCEPTED"):
            logged.add((r["gesture_id"], int(r["trial_id"])))
    present = set()
    for gid, gname in GESTURES:
        path = pdir / f"{gid}.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            lines = f.read().splitlines(keepends=True)
        if not lines:
            continue
        header_ok = lines[0].strip() == ",".join(CSV_HEADER)
        keep, removed, seen = [], [], set()
        for tid, blk in split_blocks(lines[1:] if header_ok else lines):
            if header_ok and tid is not None and tid not in seen and block_is_valid(blk):
                keep.append(blk)
                seen.add(tid)
                if (gid, tid) not in logged:      # written, but the log entry was lost in a crash
                    append_rows(pdir / "trial_log.csv", TRIAL_LOG_HEADER,
                                [[now_iso(), "-", tid, "", gid, gname, tid, "", "ACCEPTED",
                                  "recovered after an interrupted save"] + [""] * 13 + [f"{gid}.csv"]])
                    print(f"  Recovered {gid} repetition {tid} (saved just before an interruption).")
            else:
                removed.append(blk)
        if removed:
            qdir = pdir / "quarantine"
            qdir.mkdir(exist_ok=True)
            ts = stamp()
            shutil.copy2(path, qdir / f"{gid}_{ts}_full_backup.csv")
            with (qdir / f"{gid}_{ts}_removed_rows.csv").open("w", newline="", encoding="utf-8") as f:
                for blk in removed:
                    f.writelines(blk)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                f.write(",".join(CSV_HEADER) + "\r\n")
                for blk in keep:
                    f.writelines(blk)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            n_rows = sum(len(b) for b in removed)
            log_event(pdir, "-", "RECONCILE", f"{gid}: moved {n_rows} incomplete/duplicate rows to quarantine")
            print(f"  {gid}: {n_rows} rows from an interrupted save were moved to quarantine/.")
        present |= {(gid, t) for t in seen}
    for gid, tid in sorted(logged - present):
        log_event(pdir, "-", "RECONCILE_MISSING", f"{gid} repetition {tid} is logged but its data is missing")
        print(f"  WARNING: {gid} repetition {tid} is in the log but not in {gid}.csv; it will be recorded again.")
    return present


# ============================================================================
# 6. ONE PARTICIPANT
# ============================================================================

class Session:
    def __init__(self, sid, ref_id, ref_q, epoch, boot):
        self.sid, self.ref_id, self.ref_q, self.epoch, self.boot = sid, ref_id, ref_q, str(epoch), boot


RECAL_CODES = {"IMU_FAULT", "IMU_STALE", "IMU_EPOCH_CHANGED", "REF_INVALID", "NO_REFERENCE"}
ERROR_TEXT = {
    "IMU_FAULT": "The motion sensor had a fault; the glove is repairing it. Check the MPU6050 wires.",
    "IMU_STALE": "The motion sensor stopped sending data for a moment.",
    "IMU_MISSED_SAMPLE": "At least one IMU sample was missed during this capture; the attempt is rejected.",
    "IMU_EPOCH_CHANGED": "The motion sensor was interrupted while recording, so the orientation may have jumped.",
    "REF_INVALID": "The glove's calibration is no longer valid.",
    "NO_REFERENCE": "The glove has no calibration yet.",
    "CAL_MOVING": "The hand moved during the 5-second combined calibration; repeat it and keep completely still.",
    "CAL_RESEED_FAILED": "The glove could not establish the new orientation frame after calibration.",
    "CAL_BIAS_INCOMPLETE": "The glove did not receive enough full-rate gyro samples during the 5-second calibration.",
    "UNKNOWN_COMMAND": "The glove did not understand a command (firmware and program versions differ?).",
}


class Participant:
    def __init__(self, collector, pdir, kind):
        self.c = collector
        self.link = collector.link
        self.pdir = pdir
        self.pid = pdir.name
        self.kind = kind
        self.order = load_order(pdir)
        self.accepted = set()
        self.attempts = {}
        self.session = None
        self.check_done = set()
        for r in read_csv_dicts(pdir / "trial_log.csv"):
            try:
                key = (r["gesture_id"], int(r["trial_id"]))
            except (KeyError, ValueError):
                continue
            if r.get("attempt", "").isdigit():
                self.attempts[key] = max(self.attempts.get(key, 0), int(r["attempt"]))

    # ---------- logging helpers ----------
    @property
    def sid(self):
        return self.session.sid if self.session else "-"

    def event(self, name, details=""):
        log_event(self.pdir, self.sid, name, details)

    def next_attempt(self, key):
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]

    def log_trial(self, rnd, pos, gid, tid, attempt, result, reason, m=None, block=None, data_file=""):
        m = m or {}
        h = block.header if block is not None else {}

        def f(key, fmt):
            v = m.get(key)
            return "" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)
        append_rows(self.pdir / "trial_log.csv", TRIAL_LOG_HEADER, [[
            now_iso(), self.sid, rnd, pos, gid, gesture_name(gid), tid, attempt, result, reason,
            f("max_dps", "{:.2f}"), f("spread_deg", "{:.3f}"), f("flex_range_max", "{}"),
            f("saturated", "{}"), f("interval_min", "{}"), f("interval_max", "{}"),
            f("late_max_us", "{:.0f}"), f("imu_missed", "{:.0f}"), f("temp_c", "{:.2f}"),
            m.get("rail_mv", ""), h.get("epoch", ""), h.get("boot", ""), h.get("ref", ""), data_file]])

    def save_rejected(self, block, name):
        rdir = self.pdir / "rejected"
        rdir.mkdir(exist_ok=True)
        path = unique_path(rdir / name)
        write_new_file(path, CSV_HEADER, block.rows)
        return "rejected/" + path.name

    def report_glove_error(self, reply, context):
        text = ERROR_TEXT.get(reply.code, "The glove reported an error.")
        print(f"  GLOVE ERROR ({reply.code}): {text}")
        details = ",".join(f"{k}={v}" for k, v in reply.fields.items())
        self.event("GLOVE_ERROR", f"{context}: {reply.code} {details}")

    # ---------- link helpers ----------
    def on_link_lost(self, reason):
        self.link.close_conn()
        self.event("LINK_LOST", reason)
        print()
        print(f"  CONNECTION TO THE GLOVE LOST ({reason}).")
        hello = self.c.wait_for_glove(reconnecting=True)
        self.event("LINK_RESTORED", "boot={boot} reset={reset} sw={sw} epoch={epoch}".format(**defaults(hello)))

    def capture(self, cmd, arg=""):
        """Send a command. Returns the reply, or None if the link had to be restored."""
        try:
            return self.link.request(cmd, arg)
        except (LinkError, ProtocolError) as e:
            self.on_link_lost(str(e))
            return None

    def ensure_ready(self, collect_fist_after_recal=True):
        """Before a recording: fresh link, healthy sensor, still-valid calibration.

        If continuity was lost, a new neutral calibration is created. For normal
        trials/checkpoints we immediately collect the session's fist reference as
        well. fist_reference() calls this with collect_fist_after_recal=False to
        avoid recursion while recovering from a link loss during the fist hold.
        """
        if self.link.pending():                     # the glove reconnected while we were idle
            self.link.close_conn()
            hello = self.link.accept(timeout=3)
            if hello:
                self.c.show_hello(hello)
                self.event("LINK_RESTORED", "boot={boot} reset={reset} sw={sw} epoch={epoch}".format(**defaults(hello)))
        pong = self.capture("PING")
        if pong is None:
            pong = self.capture("PING")
            if pong is None:
                return False
        f = pong.fields
        if f.get("imu") != "OK":
            print("  The motion sensor is being repaired by the glove. Wait a few seconds and try again.")
            print("  If this repeats, check the MPU6050 wiring (see the troubleshooting page).")
            self.event("SENSOR_NOT_READY", ",".join(f"{k}={v}" for k, v in f.items()))
            time.sleep(1.0)
            return False
        s = self.session
        reasons = []
        if s is None:
            reasons.append("no calibration in this sitting")
        else:
            if f.get("boot") != s.boot:
                reasons.append("the glove restarted")
            elif f.get("epoch") != s.epoch:
                reasons.append("the motion sensor was interrupted")
            if f.get("ref") != s.ref_id or f.get("ref_ok") != "1":
                reasons.append("the glove's calibration is not the current one")
        if reasons:
            print()
            print("  A NEW 5-SECOND COMBINED CALIBRATION IS NEEDED: " + "; ".join(reasons) + ".")
            self.event("RECALIBRATION_NEEDED", "; ".join(reasons))
            self.calibrate("orientation continuity lost: " + "; ".join(reasons))
            if collect_fist_after_recal:
                print("  New session created; collecting its closed-hand reference before continuing.")
                self.fist_reference()
            print("  Calibration done. Now repeat the step you were on.")
            return False
        return True

    # ---------- steps ----------
    def sensor_check(self):
        """Short pre-participant health check - not a wiggle test and no data is saved."""
        print()
        line("-")
        print("QUICK SENSOR CHECK")
        line("-")
        print("This only checks that the glove is connected and the sensors are alive.")
        while True:
            r = self.capture("STATUS")
            if r is None:
                continue
            f = r.fields
            if f.get("imu") != "OK":
                print("  MPU6050 is not healthy. Check the wiring/power and retry.")
                if ask("  [R] retry or [Q] pause: ", ["r", "q"]) == "q":
                    raise PauseRequested()
                continue
            try:
                flex = [int(x) for x in f.get("flex", "").split("/")]
            except ValueError:
                flex = []
            if len(flex) != 5:
                print("  Could not read all five flex channels. Retry.")
                continue
            sat = [i for i, v in enumerate(flex) if v <= FLEX_LOW or v >= FLEX_HIGH]
            print(f"  MPU6050: OK | WHO_AM_I {f.get('who')} | temperature {f.get('temp_c')} C")
            print("  Flex now (thumb/index/middle/ring/little): " + "/".join(map(str, flex)))
            if sat:
                names = ["thumb", "index", "middle", "ring", "little"]
                print("  One or more flex channels are at the ADC end range: " + ", ".join(names[i] for i in sat))
                print("  Check that sensor/resistor wiring before collecting this participant.")
                if ask("  [R] retry or [Q] pause: ", ["r", "q"]) == "q":
                    raise PauseRequested()
                continue
            print("  If these values look normal and all five sensors are connected, continue.")
            choice = ask("  Sensor check OK? [Y] continue, [R] repeat, [Q] pause: ", ["y", "r", "q"])
            if choice == "y":
                self.event("SENSOR_CHECK", "PASS " + ",".join(f"{k}={v}" for k, v in f.items()))
                return
            if choice == "q":
                raise PauseRequested()

    def calibrate(self, reason):
        """One 5 s neutral hold creates all participant/session baselines.

        The same recording is used for open-hand flex baseline, accelerometer
        baseline, gyro zero-rate bias and the orientation reference. There is
        no separate gyro-bias pose.
        """
        sid = f"S{next_session_number(self.pdir):02d}"
        log_event(self.pdir, sid, "SESSION_START", reason)
        print()
        line("-")
        print(f"COMBINED NEUTRAL CALIBRATION  (session {sid}, 5 seconds)")
        line("-")
        print("Put the hand in the standard arm-wrestling neutral pose:")
        print("elbow on the table, forearm raised, wrist straight, hand sideways,")
        print("fingers open/straight. Keep the ENTIRE hand and arm completely still.")
        print("These same 5 seconds set the flex baseline, accel baseline, gyro bias")
        print("and neutral orientation reference.")
        attempt = 0
        while True:
            if ask("Press ENTER to start the 5 s calibration, Q to pause: ", ["", "q"]) == "q":
                raise PauseRequested()
            attempt += 1
            ref_id = f"{self.pid}_{sid}_A{attempt}"
            print("  CALIBRATING - stay completely still for 5 seconds...")
            r = self.capture("CAL", ref_id)
            if r is None:
                continue
            if r.kind == "ERR":
                if r.code == "CAL_MOVING":
                    print("  NOT STILL ENOUGH - gyro motion was detected during calibration.")
                    print(f"  bias noise {r.fields.get('bias_std_dps', '?')} deg/s; "
                          f"max deviation {r.fields.get('bias_max_dev_dps', '?')} deg/s")
                    log_event(self.pdir, sid, "CAL_REJECTED",
                              "firmware CAL_MOVING " + ",".join(f"{k}={v}" for k, v in r.fields.items()))
                else:
                    self.report_glove_error(r, f"combined calibration {ref_id}")
                continue
            if r.kind != "BLOCK" or r.code != "CAL":
                print("  Unexpected answer from the glove; trying again.")
                continue
            m = block_metrics(r)
            problems = technical_problems(r, m, CAL_SAMPLES)
            if r.ref is None or r.header.get("ref") != ref_id:
                problems.append("the glove did not confirm the calibration")
            if problems:
                print("  Calibration not usable (technical): " + "; ".join(problems))
                log_event(self.pdir, sid, "CAL_FAILED", "; ".join(problems))
                continue
            ok = still_ok(m)
            print(f"  Stillness: {describe_still(m)}")
            bias = r.stats.get("bias_dps", "?")
            print(f"  Gyro bias from the SAME 5 s: {bias} deg/s")
            if not ok:
                self.save_rejected(r, f"calibration_{sid}_A{attempt}.csv")
                log_event(self.pdir, sid, "CAL_REJECTED", describe_still(m))
                print("  NOT STILL ENOUGH - repeat the same 5-second neutral calibration.")
                continue

            self.session = Session(sid, ref_id, r.ref, r.header.get("epoch", ""), r.header.get("boot", ""))
            write_new_file(self.pdir / f"calibration_{sid}.csv", CSV_HEADER, r.rows)
            append_rows(self.pdir / "references.csv", REFERENCE_HEADER, [[
                now_iso(), sid, ref_id, attempt] + [f"{v:.8f}" for v in r.ref] + [
                f"{m['spread_deg']:.3f}", f"{m['max_dps']:.2f}", 1, 0,
                self.session.boot, self.session.epoch]])

            # Explicit summary of every feature baseline derived from this same 5 s hold.
            flex_means = [sum(int(row[c]) for row in r.rows) / len(r.rows) for c in range(2, 7)]
            ax = sum(int(row[14]) for row in r.rows) / len(r.rows) / 8192.0
            ay = sum(int(row[15]) for row in r.rows) / len(r.rows) / 8192.0
            az = sum(int(row[16]) for row in r.rows) / len(r.rows) / 8192.0
            try:
                bvals = [float(x) for x in r.stats.get("bias_dps", "nan/nan/nan").split("/")]
                if len(bvals) != 3:
                    raise ValueError
            except ValueError:
                bvals = [float("nan")] * 3
            append_rows(self.pdir / "calibration_summary.csv", CAL_SUMMARY_HEADER, [[
                now_iso(), sid, ref_id,
                *[f"{v:.3f}" for v in flex_means],
                *[f"{v:.6f}" for v in bvals],
                r.stats.get("bias_std_dps", ""), r.stats.get("bias_max_dev_dps", ""),
                f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
                *[f"{v:.8f}" for v in r.ref], f"{m['spread_deg']:.3f}", f"{m['max_dps']:.2f}",
                f"{m['temp_c']:.2f}", self.session.boot, self.session.epoch]])
            self.event("CALIBRATION_ACCEPTED",
                       f"ref={ref_id} attempt={attempt} bias_dps={bias} {describe_still(m)}")
            print(f"  Calibration saved (calibration_{sid}.csv + calibration_summary.csv).")
            return

    def fist_reference(self):
        print()
        line("-")
        print(f"CLOSED-HAND REFERENCE  (session {self.sid}, 3 seconds)")
        line("-")
        print("Same arm position. Make a tight fist and hold it completely still.")
        while True:
            if ask("Press ENTER to record, Q to pause: ", ["", "q"]) == "q":
                raise PauseRequested()
            if not self.ensure_ready(collect_fist_after_recal=False):
                print("  Session changed; record the fist reference for the new session now.")
                continue
            print("  RECORDING - hold the fist still for 3 seconds...")
            r = self.capture("FIST")
            if r is None:
                continue
            if r.kind == "ERR":
                self.report_glove_error(r, "fist reference")
                continue
            m = block_metrics(r)
            problems = technical_problems(r, m, FIST_SAMPLES, None, self.session)
            if problems:
                print("  Not usable (technical): " + "; ".join(problems))
                continue
            steady = still_ok(m) and m["flex_range_max"] <= FLEX_WARN_RANGE
            print(f"  Stillness: {describe_still(m)}; largest finger change {m['flex_range_max']}")
            if not steady:
                if self.kind == "real":
                    choice = ask("  Real participant: fist reference must pass. [R] redo or [Q] pause: ", ["r", "q"])
                    if choice == "q":
                        raise PauseRequested()
                    continue
                if ask("  Pilot only: fist was not steady. [R] redo, or [A] accept anyway: ", ["r", "a"]) == "r":
                    continue
            path = unique_path(self.pdir / f"fist_{self.sid}.csv")
            write_new_file(path, CSV_HEADER, r.rows)
            self.event("FIST_REFERENCE", f"file={path.name} steady={int(steady)} {describe_still(m)}")
            print(f"  Saved ({path.name}).")
            return

    def neutral_hold(self, cmd, n, title, name_fmt, seconds):
        """Checkpoint (2 s) or final drift check (5 s) in the neutral pose."""
        print()
        line("-")
        print(title)
        line("-")
        print("Back to the neutral pose: elbow on the table, arm-wrestling position, wrist")
        print(f"straight, hand sideways, fingers open. Stay completely still for {seconds} seconds.")
        while True:
            if ask("Press ENTER to record, Q to pause: ", ["", "q"]) == "q":
                raise PauseRequested()
            if not self.ensure_ready():
                continue
            print("  RECORDING - stay still...")
            r = self.capture(cmd)
            if r is None:
                continue
            if r.kind == "ERR":
                self.report_glove_error(r, title)
                continue
            m = block_metrics(r)
            problems = technical_problems(r, m, n, None, self.session)
            if problems:
                print("  Not usable (technical): " + "; ".join(problems))
                self.event("NEUTRAL_HOLD_FAILED", f"{cmd}: " + "; ".join(problems))
                continue
            drift = angle_deg(mean_quat(quats(r.rows)), self.session.ref_q)
            ok = still_ok(m)
            print(f"  Stillness: {describe_still(m)}")
            if not ok:
                if self.kind == "real":
                    choice = ask("  Real participant: neutral hold must pass. [R] redo or [Q] pause: ", ["r", "q"])
                    if choice == "q":
                        raise PauseRequested()
                    continue
                if ask("  Pilot only: not still enough. [R] redo, or [A] accept anyway: ", ["r", "a"]) == "r":
                    continue
            path = unique_path(self.pdir / name_fmt.format(sid=self.sid))
            write_new_file(path, CSV_HEADER, r.rows)
            print(f"  Orientation change since calibration {self.sid}: {drift:.2f} deg  (saved {path.name})")
            return path, drift, ok, m

    def checkpoint(self, rnd):
        path, drift, ok, m = self.neutral_hold("CHECK", CHECK_SAMPLES, f"NEUTRAL CHECKPOINT before round {rnd}",
                                               "checkpoint_{sid}_R%02d.csv" % rnd, 2)
        self.event("CHECKPOINT", f"round={rnd} drift_deg={drift:.3f} still={int(ok)} file={path.name} "
                                 f"spread_deg={m['spread_deg']:.3f} max_dps={m['max_dps']:.2f}")
        self.check_done.add((self.sid, rnd))

    def final_drift(self):
        path, drift, ok, m = self.neutral_hold("DRIFT", DRIFT_SAMPLES, "FINAL DRIFT CHECK (5 seconds)",
                                               "drift_check_{sid}.csv", 5)
        self.event("DRIFT_CHECK", f"drift_deg={drift:.3f} still={int(ok)} file={path.name}")
        print("\n  Drift through this participant (angle away from each session's calibration):")
        for r in read_csv_dicts(self.pdir / "session_log.csv"):
            if r.get("event") in ("CHECKPOINT", "DRIFT_CHECK"):
                d = parse_fields(r.get("details", "").split(" "))
                label = f"round {d.get('round')}" if r["event"] == "CHECKPOINT" else "final"
                print(f"    {r['session']}  {label:>9}: {float(d.get('drift_deg', 'nan')):6.2f} deg")

    def trial(self, rnd, pos, gid):
        name = gesture_name(gid)
        d = self.c.defs[gid]
        key = (gid, rnd)
        while True:
            print()
            print(f"ROUND {rnd}/10  -  trial {pos}/10 in this round  -  accepted {len(self.accepted)}/100")
            print(f"  GESTURE:  {name}  ({gid})" + (f"  -  {d['description']}" if d["description"] else ""))
            print(f"  Arm:      {d['arm_posture']}")
            print("  1) Relax the hand completely.  2) Form the gesture.  3) Hold it still.")
            if ask("  When the pose is stable, press ENTER to record (Q to pause): ", ["", "q"]) == "q":
                raise PauseRequested()
            if not self.ensure_ready():
                continue
            print("  RECORDING - hold still for 2 seconds...")
            r = self.capture("TRIAL", str(rnd))
            attempt = self.next_attempt(key)
            if r is None:
                self.log_trial(rnd, pos, gid, rnd, attempt, "FAILED", "connection lost during recording")
                print("  The trial was NOT saved. Record the same repetition again.")
                continue
            if r.kind == "ERR":
                self.report_glove_error(r, f"{gid} repetition {rnd}")
                self.log_trial(rnd, pos, gid, rnd, attempt, "FAILED", "glove error " + r.code)
                print("  The trial was NOT saved. Record the same repetition again.")
                continue
            if r.kind != "BLOCK" or r.code != "TRIAL":
                self.log_trial(rnd, pos, gid, rnd, attempt, "FAILED", "unexpected reply")
                continue
            m = block_metrics(r)
            problems = technical_problems(r, m, TRIAL_SAMPLES, rnd, self.session)
            if problems:
                f = self.save_rejected(r, f"{gid}_T{rnd:02d}_A{attempt}_FAILED.csv")
                self.log_trial(rnd, pos, gid, rnd, attempt, "FAILED", "; ".join(problems), m, r, f)
                print("  NOT USABLE (technical): " + "; ".join(problems))
                print("  The trial was NOT saved. Record the same repetition again.")
                continue
            warn = []
            if not math.isnan(m["max_dps"]) and m["max_dps"] > TRIAL_WARN_DPS:
                warn.append(f"peak rotation {m['max_dps']:.0f} deg/s")
            if m["spread_deg"] > TRIAL_WARN_SPREAD_DEG:
                warn.append(f"hand turned {m['spread_deg']:.1f} deg during the hold")
            if m["flex_range_max"] > FLEX_WARN_RANGE:
                warn.append(f"a finger reading changed by {m['flex_range_max']}")
            print(f"  Check: timing OK | movement {m['max_dps']:.1f} deg/s peak, {m['spread_deg']:.1f} deg turn"
                  f" | finger change max {m['flex_range_max']}"
                  + (f" | rail {m['rail_mv']} mV" if m["rail_mv"] and not m["rail_mv"].startswith("-1") else ""))
            if m["saturated"]:
                print("  NOTE: a flex reading is at the end of the ADC range (check that finger's wiring).")
            if warn:
                print("  WARNING: the hand may have moved (" + "; ".join(warn) + ").")
                print("           Discard only if you actually saw movement or the wrong gesture.")
            if ask("  Keep [K] or Discard [D]? ", ["k", "d"]) == "k":
                if key in self.accepted:        # should never happen; never write twice
                    print("  Already saved earlier - nothing written.")
                    return
                append_rows(self.pdir / f"{gid}.csv", CSV_HEADER, r.rows)
                self.log_trial(rnd, pos, gid, rnd, attempt, "ACCEPTED", "", m, r, f"{gid}.csv")
                self.accepted.add(key)
                print(f"  SAVED: {name}, repetition {rnd}/10.")
                return
            reason = discard_reason()
            f = self.save_rejected(r, f"{gid}_T{rnd:02d}_A{attempt}_DISCARDED.csv")
            self.log_trial(rnd, pos, gid, rnd, attempt, "DISCARDED", reason, m, r, f)
            print("  Discarded (kept in rejected/). The same repetition will be recorded again.")

    # ---------- the whole sitting ----------
    def run(self, resume):
        print()
        line("#")
        print(f"PARTICIPANT {self.pid}" + ("   (DRY RUN - pilot data)" if self.kind == "pilot" else "")
              + ("   - RESUMING" if resume else ""))
        line("#")
        self.accepted = reconcile(self.pdir)
        print(f"Accepted so far: {len(self.accepted)}/100")
        log_event(self.pdir, "-", "SITTING_START", "resume" if resume else "new participant")
        self.sensor_check()
        self.calibrate("resumed sitting" if resume else "start of sitting")
        self.fist_reference()
        for rnd in range(1, N_ROUNDS + 1):
            todo = [(pos, gid) for pos, gid in enumerate(self.order[rnd - 1], 1) if (gid, rnd) not in self.accepted]
            if not todo:
                continue
            if (self.sid, rnd) not in self.check_done:
                self.checkpoint(rnd)
            for pos, gid in todo:
                self.trial(rnd, pos, gid)
        self.final_drift()
        self.event("PARTICIPANT_COMPLETE", f"accepted={len(self.accepted)}")
        print()
        line("#")
        print(f"PARTICIPANT {self.pid} COMPLETE - 100/100 repetitions saved.")
        line("#")


def discard_reason():
    print("  Why? 1 wrong gesture, 2 participant moved, 3 recorded too early,")
    print("       4 glove/sensor problem, 5 other")
    a = ask("  Reason [1-5]: ", ["1", "2", "3", "4", "5"])
    reason = {"1": "wrong gesture", "2": "participant moved", "3": "recorded too early",
              "4": "glove/sensor problem", "5": "other"}[a]
    if a == "5":
        try:
            note = input("  Short note (optional): ").strip().replace(",", ";")
        except (KeyboardInterrupt, EOFError):
            note = ""
        if note:
            reason += ": " + note
    return reason


def defaults(h):
    h = h or {}
    return {k: h.get(k, "?") for k in ("boot", "reset", "sw", "epoch", "fw", "bias", "who", "rail")}


# ============================================================================
# 7. THE PROGRAM
# ============================================================================

class Collector:
    def __init__(self):
        ensure_dataset()
        self.defs = load_definitions()
        self.link = GloveLink()

    def show_hello(self, h):
        d = defaults(h)
        print(f"  Glove connected ({self.link.peer}): firmware {d['fw']}, boot id {d['boot']}, "
              f"last restart: {d['reset']}" + ("" if d["sw"] in ("NONE", "?") else f" ({d['sw']})"))
        if d["reset"] == "BROWNOUT":
            print("  WARNING: the glove restarted because its supply voltage dropped. Charge/replace the battery.")
        elif d["reset"] in ("TASK_WATCHDOG", "INT_WATCHDOG", "WATCHDOG", "CRASH") or d["sw"] == "IMU_TASK_STALL":
            print("  WARNING: the glove restarted after an internal fault. It is logged; if it repeats, report it.")
        if d["bias"] == "none":
            print("  Note: no gyro bias stored on the glove yet - it is measured before calibration.")

    def wait_for_glove(self, reconnecting=False, timeout=None):
        print("  Waiting for the glove to " + ("reconnect" if reconnecting else "connect") + " on port "
              f"{PORT} ... (glove powered on? hotspot on? LED blinking slowly = still searching)")
        print("  (Ctrl+C = pause the program)")
        try:
            hello = self.link.accept(timeout)
        except KeyboardInterrupt:
            print()
            raise PauseRequested()
        if hello:
            self.show_hello(hello)
        return hello

    def test_log(self, test, result, details=""):
        append_rows(DATASET_DIR / "glove_tests_log.csv", ["time", "test", "result", "details"],
                    [[now_iso(), test, result, details]])

    def request(self, cmd, arg=""):
        while True:
            try:
                return self.link.request(cmd, arg)
            except (LinkError, ProtocolError) as e:
                self.link.close_conn()
                print(f"\n  Connection problem ({e}).")
                self.wait_for_glove(reconnecting=True)

    def wiggle_test(self, pdir=None, optional=False):
        print()
        line("-")
        print(f"WIGGLE TEST ({WIGGLE_SECONDS:.0f} seconds)")
        line("-")
        print("The participant moves the wrist in all directions and bends and straightens")
        print("every finger again and again. Watch the numbers. Nothing is recorded.")
        while True:
            a = ask("Press ENTER to start" + (", S to skip" if optional else "") + ", Q to pause: ",
                    ["", "s", "q"] if optional else ["", "q"])
            if a == "q":
                raise PauseRequested()
            if a == "s":
                if pdir:
                    log_event(pdir, "-", "WIGGLE_TEST", "skipped")
                return True
            first = self.request("STATUS").fields
            fmin, fmax, faults_seen, link_lost = [9999] * 5, [-1] * 5, False, False
            t0 = time.monotonic()
            last = first
            while time.monotonic() - t0 < WIGGLE_SECONDS:
                try:
                    st = self.link.request("STATUS").fields
                except (LinkError, ProtocolError) as e:
                    link_lost = True
                    self.link.close_conn()
                    print(f"\n  Connection lost during the test ({e}).")
                    self.wait_for_glove(reconnecting=True)
                    break
                last = st
                flex = [int(x) for x in st.get("flex", "0/0/0/0/0").split("/")]
                for i, v in enumerate(flex):
                    fmin[i], fmax[i] = min(fmin[i], v), max(fmax[i], v)
                faults_seen |= st.get("imu") != "OK"
                de = int(st.get("i2c_err", 0)) - int(first.get("i2c_err", 0))
                left = WIGGLE_SECONDS - (time.monotonic() - t0)
                print(f"\r  {left:4.0f} s left | sensor {st.get('imu')} | bus errors +{de} | "
                      f"flex {'/'.join(str(v) for v in flex)}   ", end="", flush=True)
                time.sleep(0.25)
            print()
            fails, notes = [], []
            for k, label in (("i2c_err", "I2C bus errors"), ("faults", "sensor faults"),
                             ("reinits", "sensor restarts"), ("frozen", "frozen-data events"),
                             ("cfg_lost", "lost-settings events")):
                delta = int(last.get(k, 0)) - int(first.get(k, 0))
                if delta > 0:
                    fails.append(f"{label}: {delta}")
            if faults_seen:
                fails.append("sensor reported a fault")
            if link_lost:
                fails.append("connection lost")
            names = ["thumb", "index", "middle", "ring", "little"]
            for i in range(5):
                if fmax[i] >= FLEX_HIGH or (0 <= fmin[i] <= FLEX_LOW):
                    fails.append(f"{names[i]} flex reading hit the end of the range ({fmin[i]}-{fmax[i]})")
                elif fmax[i] >= 0 and fmax[i] - fmin[i] < WIGGLE_MIN_FLEX_RANGE:
                    notes.append(f"{names[i]} flex barely changed ({fmin[i]}-{fmax[i]}) - was it bent?")
            result = "PASS" if not fails else "FAIL"
            print(f"  WIGGLE TEST: {result}")
            for x in fails:
                print("    - " + x)
            for x in notes:
                print("    - note: " + x)
            details = "; ".join(fails + notes) or "no problems"
            if pdir:
                log_event(pdir, "-", "WIGGLE_TEST", f"{result}: {details}")
            else:
                self.test_log("wiggle", result, details)
            if result == "PASS":
                return True
            print("  Fix the glove (wires, connectors, sensor mounting) before real data.")
            a = ask("  [R] repeat the test, [C] continue anyway (logged), [Q] pause: ", ["r", "c", "q"])
            if a == "q":
                raise PauseRequested()
            if a == "c":
                if pdir:
                    log_event(pdir, "-", "WIGGLE_TEST_OVERRIDE", "operator continued after a failed test")
                return False

    def status(self):
        f = self.request("STATUS").fields
        print()
        print(f"  Firmware {f.get('fw')}  boot {f.get('boot')}  up {f.get('uptime_s')} s  Wi-Fi {f.get('rssi')} dBm")
        print(f"  Motion sensor: {f.get('imu')}  (WHO_AM_I {f.get('who')}, {f.get('temp_c')} C)  epoch {f.get('epoch')}")
        print(f"  Counters: bus errors {f.get('i2c_err')}, faults {f.get('faults')}, sensor recoveries {f.get('reinits')}, "
              f"frozen {f.get('frozen')}, lost settings {f.get('cfg_lost')}, gap resets {f.get('gap_inval')}")
        print(f"  Gyro bias {f.get('bias_dps')} deg/s ({f.get('bias_src')})   calibration {f.get('ref')} "
              f"(valid {f.get('ref_ok')})")
        rail = f.get("rail_mv", "-1")
        print(f"  Flex now {f.get('flex')}" + ("" if rail.startswith("-1") else f"   3.3 V rail {rail} mV"))

    def fault_test_sensor_reset(self):
        before = self.request("PING").fields
        self.request("TEST", "MPU_RESET")
        print("  Sensor reset sent. Waiting 2 seconds...")
        time.sleep(2.0)
        after = self.request("PING").fields
        ok = after.get("imu") == "OK" and after.get("epoch") != before.get("epoch")
        res = "PASS" if ok else "FAIL"
        print(f"  {res}: epoch {before.get('epoch')} -> {after.get('epoch')}, sensor {after.get('imu')}"
              + ("  (a new calibration would be forced)" if ok else ""))
        self.test_log("sensor_reset", res, f"epoch {before.get('epoch')}->{after.get('epoch')} imu={after.get('imu')}")

    def fault_test_reboot(self):
        before = self.request("PING").fields
        try:
            self.link.request("TEST", "REBOOT")
        except (LinkError, ProtocolError):
            pass
        self.link.close_conn()
        print("  Restart sent.")
        h = self.wait_for_glove(reconnecting=True, timeout=60)
        ok = bool(h) and h.get("boot") != before.get("boot")
        res = "PASS" if ok else "FAIL"
        print(f"  {res}: boot id {before.get('boot')} -> {h.get('boot') if h else '?'} "
              f"(reason {h.get('reset') if h else '?'}/{h.get('sw') if h else '?'})")
        self.test_log("glove_restart", res, f"boot {before.get('boot')}->{h.get('boot') if h else '?'}")
        if not h:
            self.wait_for_glove(reconnecting=True)

    def tests_menu(self):
        while True:
            print()
            line("-")
            print("GLOVE TESTS")
            line("-")
            print("  1  Show glove status")
            print("  2  Wiggle test")
            print("  3  Fault test: reset the motion sensor")
            print("  4  Fault test: restart the glove")
            print("  5  Back")
            c = ask("Choose 1-5: ", ["1", "2", "3", "4", "5"])
            if c == "1":
                self.status()
            elif c == "2":
                self.wiggle_test()
            elif c == "3":
                self.fault_test_sensor_reset()
            elif c == "4":
                self.fault_test_reboot()
            else:
                return

    def new_participant(self, kind):
        root, prefix = (REAL_DIR, "P") if kind == "real" else (PILOT_DIR, "X")
        nums = [int(p.name[1:]) for p in root.glob(prefix + "*") if p.is_dir() and p.name[1:].isdigit()]
        number = max(nums, default=0) + 1
        pid = f"{prefix}{number:02d}"
        pdir = root / pid
        print()
        print(f"New participant: {pid}")
        print("Short details (press ENTER to leave any of them empty):")
        try:
            length = input("  Hand length in mm (wrist crease to middle fingertip): ").strip()
            breadth = input("  Hand breadth in mm (across the knuckles): ").strip()
            age = input("  Age band (e.g. 18-24): ").strip()
            notes = input("  Glove fit notes: ").strip().replace(",", ";")
        except (KeyboardInterrupt, EOFError):
            print()
            raise PauseRequested()
        pdir.mkdir(parents=True, exist_ok=False)
        p_index = (number - 1) % 10
        order = williams_order(p_index)
        write_new_file(pdir / "order.csv", ["round", "position", "gesture_id", "gesture_name"],
                       [[r + 1, k + 1, gid, gesture_name(gid)] for r in range(N_ROUNDS)
                        for k, gid in enumerate(order[r])])
        meta_file = (DATASET_DIR if kind == "real" else PILOT_DIR) / "participants.csv"
        append_rows(meta_file, PARTICIPANT_HEADER, [[pid, kind, now_iso(), length, breadth, age, notes, p_index]])
        log_event(pdir, "-", "PARTICIPANT_CREATED", f"kind={kind} order_start_row={p_index}")
        return Participant(self, pdir, kind)

    def incomplete(self):
        out = []
        for root, prefix, kind in ((REAL_DIR, "P", "real"), (PILOT_DIR, "X", "pilot")):
            out += [(p, kind) for p in participant_dirs(root, prefix) if not is_complete(p)]
        return out

    def menu(self):
        while True:
            real = participant_dirs(REAL_DIR, "P")
            done = sum(is_complete(p) for p in real)
            todo = [(p, "real") for p in real if not is_complete(p)]
            print()
            line()
            print(f"MAIN MENU    participants complete: {done}    dataset: {DATASET_DIR}")
            line()
            print("  N  New participant")
            if todo:
                print("  R  Resume an unfinished participant: " + ", ".join(p.name for p, _ in todo))
            print("  S  Show current glove/sensor status")
            print("  Q  Quit")
            choice = ask("Choose: ", ["n", "s", "q"] + (["r"] if todo else []))
            if choice == "q":
                return
            part = None
            try:
                if choice == "s":
                    self.status()
                    continue
                if choice == "n":
                    part = self.new_participant("real")
                    part.run(resume=False)
                else:
                    if len(todo) == 1:
                        pdir, kind = todo[0]
                    else:
                        names = [p.name.lower() for p, _ in todo]
                        pick = ask("  Which one? Type its ID: ", names)
                        pdir, kind = todo[names.index(pick)]
                    part = Participant(self, pdir, kind)
                    part.run(resume=True)
            except PauseRequested:
                print()
                print("PAUSED. Everything accepted so far is saved. Choose R later to resume.")
                if part is not None:
                    part.event("PAUSED", "operator paused")


def main():
    print()
    line()
    print(f"SMART GLOVE DATA COLLECTOR V3 - {COLLECTOR_VERSION}")
    line()
    collector = Collector()
    try:
        collector.wait_for_glove()
        collector.menu()
    except PauseRequested:
        print("Stopped by the operator.")
    finally:
        collector.link.close_conn()
        collector.link.server.close()
    print("Data collector closed. All accepted data is saved in", DATASET_DIR)


if __name__ == "__main__":
    main()
