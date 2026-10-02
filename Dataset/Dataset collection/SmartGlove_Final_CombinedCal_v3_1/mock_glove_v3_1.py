"""
Mock SMART_GLOVE_V3 for testing collect_glove_data.py without hardware.
Speaks the same line protocol as SmartGlove_V3.ino (same field names/formats).

How to use (on the laptop, no glove needed):
  1) PowerShell window 1:  python collect_glove_data.py
  2) PowerShell window 2:  python mock_glove_v3.py            (a healthy glove)
     or for example:       python mock_glove_v3.py --reboot-at-trial 20 --drop-at-trial 5
  Use the D (dry run) option in the collector so test data goes to the pilot folder.
  Stop the mock with Ctrl+C.

Fault injection (counts are 1-based over all TRIAL commands received):
  --drop-at-trial N        close the TCP connection instead of answering, reconnect (same boot)
  --reboot-at-trial N      "brown-out": reconnect with a new boot id, no calibration
  --imu-fault-at-trial N   answer ERR IMU_EPOCH_CHANGED and bump the epoch
  --garbage-at-trial N     send a malformed data row inside the block
  --bad-timing-at-trial N  one 60 ms sample interval
  --moving-cal K           first K calibrations are shaky
  --moving-gyrocal K       first K gyro-bias measurements report movement
  --stale-pong             before every PONG, send a stale PONG with an old sequence number
  --hello NAME             firmware name in HELLO (test old-firmware rejection)
"""
import argparse
import math
import random
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5000)
ap.add_argument("--drop-at-trial", type=int, default=0)
ap.add_argument("--reboot-at-trial", type=int, default=0)
ap.add_argument("--imu-fault-at-trial", type=int, default=0)
ap.add_argument("--garbage-at-trial", type=int, default=0)
ap.add_argument("--bad-timing-at-trial", type=int, default=0)
ap.add_argument("--moving-cal", type=int, default=0)
ap.add_argument("--moving-gyrocal", type=int, default=0)
ap.add_argument("--stale-pong", action="store_true")
ap.add_argument("--hello", default="SMART_GLOVE_V3")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
rng = random.Random(args.seed)


def qmul(a, b):
    return (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0])


def axis_angle(ax, ay, az, deg):
    n = math.sqrt(ax*ax+ay*ay+az*az)
    h = math.radians(deg) / 2
    s = math.sin(h) / n
    return (math.cos(h), ax*s, ay*s, az*s)


def rpy(q):
    w, x, y, z = q
    r = math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    p = math.degrees(math.asin(max(-1, min(1, 2*(w*y-z*x)))))
    yw = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return r, p, yw


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


class Glove:
    def __init__(self):
        self.new_boot("POWER_ON", "NONE")
        self.trials = 0
        self.cals = 0
        self.gyrocals = 0
        self.t0 = time.time()
        self.i2c_err = 0
        self.reinits = 0

    def new_boot(self, reset, sw):
        self.boot = "%08X" % rng.getrandbits(32)
        self.reset, self.sw = reset, sw
        self.epoch = 1
        self.ref_valid, self.ref_id, self.ref_epoch, self.qref = False, "-", 0, (1, 0, 0, 0)
        self.bias = "flash"
        self.reinits = 0
        self.boot_time = time.time()

    def ref_ok(self):
        return self.ref_valid and self.ref_epoch == self.epoch

    def orientation(self, gesture_offset=None, shaky=False, i=0):
        # neutral pose + slow heading drift (0.02 deg/s about vertical) + noise
        drift = 0.02 * (time.time() - self.boot_time)
        q = qmul(axis_angle(0, 0, 1, drift), axis_angle(1, 0.2, 0, 80))
        if gesture_offset:
            q = qmul(q, axis_angle(*gesture_offset))
        amp = 3.0 if shaky else 0.15
        q = qmul(q, axis_angle(rng.random()-0.5, rng.random()-0.5, rng.random()-0.5,
                               amp * math.sin(i / 3.0) + rng.gauss(0, amp / 5)))
        n = math.sqrt(sum(c*c for c in q))
        return tuple(c / n for c in q)

    def block(self, kind, seq, n, trial_id=0, relative=True, gesture=None, shaky=False,
              garbage=False, bad_timing=False, with_ref=False):
        rows = []
        base_flex = [1200 + 150*k for k in range(5)]
        if gesture is not None:
            base_flex = [900 + ((gesture * 37 + k * 211) % 1800) for k in range(5)]
        off = None if gesture is None else (0.3, 1, 0.2, 5 + 3 * gesture)
        t = 0
        qs = []
        for i in range(n):
            q = self.orientation(off, shaky, i)
            qs.append(q)
            if relative:
                r, p, y = rpy(qmul(conj(self.qref), q))
            else:
                r = p = y = 0.0
            flex = [max(0, min(4095, int(b + rng.gauss(0, 4)))) for b in base_flex]
            g = [int(rng.gauss(0, 20 if not shaky else 400)) for _ in range(3)]
            a = [int(rng.gauss(0, 40)), int(rng.gauss(0, 40)), int(8192 + rng.gauss(0, 40))]
            rows.append("%d,%d,%d,%d,%d,%d,%d,%.7f,%.7f,%.7f,%.7f,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f" % (
                t, trial_id, *flex, *q, *g, *a, r, p, y))
            t += 60 if (bad_timing and i == 20) else 40
        if garbage:
            rows[10] = rows[10].replace(",", ";", 3)
        return rows, qs

    def send_block(self, sock, kind, seq, rows, max_dps, ref_line=None):
        out = ["BEGIN,%s,%d,n=%d,ref=%s,epoch=%d,boot=%s" % (kind, seq, len(rows),
                                                             self.ref_id if self.ref_valid else "-",
                                                             self.epoch, self.boot)]
        out += rows
        if ref_line:
            out.append(ref_line)
        if kind == "CAL":
            out.append("STATS,%d,max_dps=%.2f,imu_missed=0,late_max_us=%d,temp_c=31.20,rail_mv=-1/-1,"
                       "bias_dps=0.1000/-0.2000/0.0500,bias_std_dps=0.0810,bias_max_dev_dps=0.3200"
                       % (seq, max_dps, rng.randint(20, 400)))
        else:
            out.append("STATS,%d,max_dps=%.2f,imu_missed=0,late_max_us=%d,temp_c=31.20,rail_mv=-1/-1"
                       % (seq, max_dps, rng.randint(20, 400)))
        out.append("END,%s,%d" % (kind, seq))
        sock.sendall(("\n".join(out) + "\n").encode())


def err(sock, seq, code, g):
    sock.sendall(("ERR,%d,%s,sample=-1,age_us=0,epoch=%d,imu=OK,i2c_err=%d,faults=0,reinits=%d,"
                  "frozen=0,cfg_lost=0,gap_inval=0\n" % (seq, code, g.epoch, g.i2c_err, g.reinits)).encode())


def serve(g):
    """One TCP connection. Returns 'drop' or 'reboot' or 'closed'."""
    s = socket.create_connection(("127.0.0.1", args.port))
    s.sendall(("HELLO,%s,fw=3.0.1,boot=%s,reset=%s,sw=%s,epoch=%d,ref_ok=%d,bias=%s,who=0x68,rail=0,"
               "gyro_lsb_per_dps=16.4,acc_lsb_per_g=8192\n" % (args.hello, g.boot, g.reset, g.sw, g.epoch,
                                                               int(g.ref_ok()), g.bias)).encode())
    buf = b""
    while True:
        try:
            chunk = s.recv(4096)
        except OSError:
            return "closed"
        if not chunk:
            s.close()
            return "closed"
        buf += chunk
        while b"\n" in buf:
            raw, _, buf = buf.partition(b"\n")
            parts = raw.decode().strip().split(" ", 2)
            seq, cmd = int(parts[0]), parts[1]
            arg = parts[2] if len(parts) > 2 else ""
            if cmd == "PING":
                if args.stale_pong:
                    s.sendall(("PONG,%d,epoch=999,ref=STALE,ref_ok=0,imu=FAULT,boot=DEADBEEF\n" % (seq - 1)).encode())
                s.sendall(("PONG,%d,epoch=%d,ref=%s,ref_ok=%d,imu=OK,boot=%s\n" % (
                    seq, g.epoch, g.ref_id if g.ref_valid else "-", int(g.ref_ok()), g.boot)).encode())
            elif cmd == "STATUS":
                fl = "/".join(str(1000 + int(800 * (0.5 + 0.5 * math.sin(time.time() * 3 + k)))) for k in range(5))
                s.sendall(("STATUS,%d,fw=3.0.1,boot=%s,uptime_s=%d,epoch=%d,ref=%s,ref_ok=%d,imu=OK,i2c_err=%d,"
                           "faults=0,reinits=%d,frozen=0,cfg_lost=0,gap_inval=0,rec_fail=0,bias_src=%s,"
                           "bias_dps=0.100/-0.200/0.050,temp_c=31.20,rssi=-48,rail_mv=-1,flex=%s,heap=200000,who=0x68\n"
                           % (seq, g.boot, time.time() - g.boot_time, g.epoch, g.ref_id if g.ref_valid else "-",
                              int(g.ref_ok()), g.i2c_err, g.reinits, g.bias, fl)).encode())
            elif cmd == "GYROCAL":
                g.gyrocals += 1
                if g.gyrocals <= args.moving_gyrocal:
                    s.sendall(("ERR,%d,GYROCAL_MOVING,std_dps=2.410,max_dev_dps=9.100\n" % seq).encode())
                else:
                    g.bias = "measured"
                    g.ref_valid = False
                    s.sendall(("GYROCAL,%d,OK,bias_dps=0.1000/-0.2000/0.0500,std_dps=0.081,max_dev_dps=0.320\n" % seq).encode())
            elif cmd == "CAL":
                g.cals += 1
                shaky = g.cals <= args.moving_cal
                g.ref_valid = False
                rows, qs = g.block("CAL", seq, 125, relative=False, shaky=shaky)
                # sign-aligned mean as the reference
                acc = [0.0] * 4
                for q in qs:
                    sg = -1 if sum(a*b for a, b in zip(qs[0], q)) < 0 else 1
                    acc = [a + sg * c for a, c in zip(acc, q)]
                n = math.sqrt(sum(c*c for c in acc))
                g.qref = tuple(c / n for c in acc)
                if shaky:
                    s.sendall(("ERR,%d,CAL_MOVING,bias_std_dps=2.4100,bias_max_dev_dps=9.1000,epoch=%d\n"
                               % (seq, g.epoch)).encode())
                    continue
                g.bias = "cal5s"
                g.epoch += 1                         # combined calibration establishes a new orientation frame
                g.ref_id, g.ref_valid, g.ref_epoch = arg or "REF", True, g.epoch
                rows = [r[:r.rfind(",", 0, r.rfind(",", 0, r.rfind(",")))] + ",%.4f,%.4f,%.4f" % rpy(qmul(conj(g.qref), q))
                        for r, q in zip(rows, qs)]
                g.send_block(s, "CAL", seq, rows, 3.1,
                             "REF,%d,%.8f,%.8f,%.8f,%.8f" % (seq, *g.qref))
            elif cmd in ("FIST", "CHECK", "DRIFT", "TRIAL"):
                need = cmd != "FIST"
                if need and not g.ref_ok():
                    err(s, seq, "REF_INVALID" if g.ref_valid else "NO_REFERENCE", g)
                    continue
                if cmd == "TRIAL":
                    g.trials += 1
                    k = g.trials
                    if k == args.drop_at_trial:
                        s.close()
                        return "drop"
                    if k == args.reboot_at_trial:
                        s.close()
                        return "reboot"
                    if k == args.imu_fault_at_trial:
                        g.epoch += 1
                        err(s, seq, "IMU_EPOCH_CHANGED", g)
                        continue
                    tid = int(arg)
                    rows, _ = g.block("TRIAL", seq, 50, trial_id=tid, gesture=rng.randint(0, 9),
                                      garbage=(k == args.garbage_at_trial),
                                      bad_timing=(k == args.bad_timing_at_trial))
                    g.send_block(s, "TRIAL", seq, rows, rng.uniform(3, 12))
                else:
                    n = {"FIST": 75, "CHECK": 50, "DRIFT": 125}[cmd]
                    rows, _ = g.block(cmd, seq, n, relative=g.ref_ok(), gesture=None)
                    g.send_block(s, cmd, seq, rows, rng.uniform(2, 6))
            elif cmd == "TEST" and arg == "MPU_RESET":
                s.sendall(("TEST,%d,OK,MPU_RESET\n" % seq).encode())
                g.epoch += 1
                g.reinits += 1
            elif cmd == "TEST" and arg == "REBOOT":
                s.sendall(("TEST,%d,OK,REBOOT\n" % seq).encode())
                s.close()
                return "test_reboot"
            else:
                err(s, seq, "UNKNOWN_COMMAND", g)


def main():
    g = Glove()
    while True:
        try:
            how = serve(g)
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError):
            time.sleep(0.3)
            continue
        if how == "reboot":
            g.new_boot("BROWNOUT", "NONE")
        elif how == "test_reboot":
            g.new_boot("SOFTWARE", "TEST_REBOOT")
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
