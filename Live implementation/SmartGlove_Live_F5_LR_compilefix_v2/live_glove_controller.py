"""
SmartGlove live controller for ESP32 F5 + Logistic Regression deployment.

Workflow:
1. Start Windows 2.4 GHz hotspot.
2. Run this script on the laptop.
3. Power the glove.
4. Perform one 5 s neutral calibration.
5. Repeat:
      FORM GESTURE -> press ENTER -> 2 s capture
      [P] predict or [D] discard
      if P: ESP32 performs F5 feature extraction + Logistic Regression
      if D: captured window is discarded
"""

import socket
import select
import sys
import time

HOST = "0.0.0.0"
PORT = 5000
SHOW_DEBUG = True

GESTURE_NAMES = {
    "G01": "Fist",
    "G02": "Excellent",
    "G03": "Stop",
    "G04": "Thumbs Up",
    "G05": "This Way",
    "G06": "Wait",
    "G07": "Hey You!",
    "G08": "Victory",
    "G09": "Call Me",
    "G10": "Attention",
}

FEATURE_NAMES = [
    "flex_thumb_centered_median",
    "flex_index_centered_median",
    "flex_middle_centered_median",
    "flex_ring_centered_median",
    "flex_little_centered_median",
    "gravity_rel_x",
    "gravity_rel_y",
    "gravity_rel_z",
    "acc_mag_median_g",
    "acc_mag_sd_g",
    "gyro_mag_median_dps",
    "gyro_mag_p95_dps",
]


class GloveLink:
    def __init__(self, host=HOST, port=PORT):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform.startswith("win") and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server.bind((host, port))
        self.server.listen(1)
        self.conn = None
        self.peer = None
        self.buf = b""
        self.seq = 0

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None
        try:
            self.server.close()
        except OSError:
            pass

    def read_line(self, timeout=15.0):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No reply from glove.")
            r, _, _ = select.select([self.conn], [], [], remaining)
            if not r:
                raise TimeoutError("No reply from glove.")
            chunk = self.conn.recv(4096)
            if not chunk:
                raise ConnectionError("Glove disconnected.")
            self.buf += chunk
        raw, _, self.buf = self.buf.partition(b"\n")
        return raw.decode("ascii", "replace").strip()

    def accept(self):
        print(f"Waiting for glove on TCP port {PORT} ...")
        conn, addr = self.server.accept()
        self.conn = conn
        self.peer = addr[0]
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = self.read_line(10.0)
        if not hello.startswith("HELLO,SMART_GLOVE_LIVE_F5_LR,"):
            raise RuntimeError(
                "Unexpected firmware greeting:\n"
                f"  {hello}\n"
                "Upload SmartGlove_Live_F5_LR.ino to the ESP32."
            )
        print(f"Connected to glove at {self.peer}")
        print(hello)

    def request(self, command, timeout=15.0):
        self.seq += 1
        seq = self.seq
        msg = f"{seq} {command}\n".encode("ascii")
        self.conn.sendall(msg)

        while True:
            line = self.read_line(timeout)
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                line_seq = int(parts[1])
            except ValueError:
                continue
            if line_seq != seq:
                continue
            return line


def parse_fields(line):
    out = {}
    parts = line.split(",")
    out["_tag"] = parts[0]
    out["_seq"] = parts[1] if len(parts) > 1 else ""
    if len(parts) > 2 and "=" not in parts[2]:
        out["_code"] = parts[2]
    for part in parts[2:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def require_ok(line, expected_tag):
    fields = parse_fields(line)
    if fields["_tag"] == "ERR":
        code = fields.get("_code", "UNKNOWN")
        raise RuntimeError(f"Glove error: {code}\n{line}")
    if fields["_tag"] != expected_tag:
        raise RuntimeError(f"Expected {expected_tag}, got:\n{line}")
    return fields


def calibrate(link):
    print()
    print("=" * 72)
    print("5 SECOND NEUTRAL CALIBRATION")
    print("=" * 72)
    print("Use the same neutral pose as data collection:")
    print("  - elbow on the marked/table position")
    print("  - forearm raised in the arm-wrestling posture")
    print("  - wrist straight")
    print("  - hand sideways")
    print("  - fingers naturally open/straight")
    print("  - keep the entire hand and arm still")
    input("\nPress ENTER when ready to calibrate...")

    print("CALIBRATING - stay still for 5 seconds...")
    line = link.request("LCAL", timeout=15.0)
    fields = require_ok(line, "CALOK")

    print("Calibration complete.")
    if SHOW_DEBUG:
        print("  neutral flex:", fields.get("flex"))
        print("  neutral accel:", fields.get("acc"))
        print("  gyro bias:", fields.get("bias"))
        print("  gyro bias std:", fields.get("bias_std"))
        print("  gyro bias max deviation:", fields.get("bias_dev"))
        print("  IMU epoch:", fields.get("epoch"))


def show_prediction(fields):
    gid = fields.get("gesture", "?")
    name = GESTURE_NAMES.get(gid, fields.get("class_name", "?"))
    print()
    print("=" * 72)
    print(f"PREDICTION: {gid} - {name}")
    print("=" * 72)

    if SHOW_DEBUG:
        ftext = fields.get("features", "")
        stext = fields.get("scores", "")
        try:
            values = [float(x) for x in ftext.split("/")]
        except ValueError:
            values = []
        if len(values) == 12:
            print("\nF5 features:")
            for n, v in zip(FEATURE_NAMES, values):
                print(f"  {n:32s} {v: .8g}")

        try:
            scores = [float(x) for x in stext.split("/")]
        except ValueError:
            scores = []
        if len(scores) == 10:
            print("\nClass scores:")
            ranked = sorted(
                zip(GESTURE_NAMES.keys(), scores),
                key=lambda x: x[1],
                reverse=True,
            )
            for g, s in ranked:
                print(f"  {g} {GESTURE_NAMES[g]:12s} {s: .6f}")


def prediction_loop(link):
    while True:
        print()
        print("FORM GESTURE.")
        print("Make the final pose stable before starting the recording.")
        action = input(
            "Press ENTER to record, C to recalibrate, S for status, or Q to quit: "
        ).strip().lower()

        if action == "q":
            return
        if action == "c":
            calibrate(link)
            continue
        if action == "s":
            line = link.request("STATUS", timeout=5.0)
            print(line)
            continue
        if action != "":
            print("Please press ENTER, C, S, or Q.")
            continue

        print("RECORDING - hold the gesture still for 2 seconds...")
        try:
            line = link.request("LCAP", timeout=8.0)
            fields = require_ok(line, "CAPTURED")
        except RuntimeError as exc:
            text = str(exc)
            print(text)
            if "NEED_CAL" in text:
                print("The current calibration is no longer valid.")
                calibrate(link)
            continue

        print("Recording complete.")
        if SHOW_DEBUG:
            print(
                f"  capture peak gyro rate: {fields.get('max_dps', '?')} deg/s"
                f" | max timing lateness: {fields.get('late_max_us', '?')} us"
            )

        while True:
            decision = input("[P] Predict or [D] Discard this recording? ").strip().lower()
            if decision == "d":
                line = link.request("LDISCARD", timeout=5.0)
                require_ok(line, "DISCARDED")
                print("Recording discarded. Nothing was predicted.")
                break
            if decision == "p":
                line = link.request("LPRED", timeout=5.0)
                fields = require_ok(line, "PRED")
                show_prediction(fields)
                break
            print("Please type P or D.")


def main():
    link = GloveLink()
    try:
        link.accept()

        # Optional initial status check.
        try:
            print("\nInitial glove status:")
            print(link.request("STATUS", timeout=5.0))
        except Exception as exc:
            print("Status request warning:", exc)

        calibrate(link)
        prediction_loop(link)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (ConnectionError, TimeoutError, OSError) as exc:
        print("\nConnection ended:", exc)
        print("Restart this script after the glove reconnects. Recalibrate before prediction.")
    finally:
        link.close()


if __name__ == "__main__":
    main()
