"""In-memory telemetry store for the gateway: recent requests, lifecycle
events, rate series and day aggregates. One lock guards all state."""
import collections
import math
import os
import re
import subprocess
import threading
import time

_SERIES = ("answer_tps", "prompt_tps", "ttft_ms")
_SERIES_MAX = 2000


def client_tag(user_agent):
    ua = (user_agent or "").lower()
    if "opencode" in ua:
        return "opencode"
    if "hermes" in ua:
        return "hermes"
    if "curl/" in ua:
        return "curl"
    if "python" in ua:
        return "python"
    return "other"


def _n(v):
    return v if isinstance(v, (int, float)) else 0


class Telemetry:
    def __init__(self, recent_max=200, events_max=200, window_s=300, clock=time.time):
        self._clock = clock
        self._window_s = window_s
        self._lock = threading.Lock()
        self._since = int(clock())
        self._recent = collections.deque(maxlen=recent_max)
        self._events = collections.deque(maxlen=events_max)
        self._series = {k: collections.deque(maxlen=_SERIES_MAX) for k in _SERIES}
        self._totals = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
                        "output_tokens": 0, "errors": 0, "client_gone": 0}
        self._day = {"date": None, "cache_saved_tokens": 0, "clients": {}}
        self._next_id = 0

    def new_id(self):
        with self._lock:
            self._next_id = (self._next_id + 1) & 0xFFFFFFFF
            return "%08x" % self._next_id

    def _prune(self, now):
        cutoff = now - self._window_s
        for dq in self._series.values():
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def record(self, rec):
        with self._lock:
            now = self._clock()
            ts = rec.get("ts")
            if ts is None:
                ts = now
            self._recent.append(dict(rec))
            t = self._totals
            t["requests"] += 1
            t["prompt_tokens"] += _n(rec.get("prompt_tokens"))
            t["cached_tokens"] += _n(rec.get("cached_tokens"))
            t["output_tokens"] += _n(rec.get("output_tokens"))
            if rec.get("finish") == "error":
                t["errors"] += 1
            elif rec.get("finish") == "client_gone":
                t["client_gone"] += 1
            if rec.get("tps") is not None:
                self._series["answer_tps"].append((ts, rec["tps"]))
            if rec.get("ttft_ms") is not None:
                self._series["ttft_ms"].append((ts, rec["ttft_ms"]))
            pt, ct, pm = rec.get("prompt_tokens"), rec.get("cached_tokens"), rec.get("prompt_ms")
            if pt is not None and pm is not None and pm > 0:
                fresh = pt - _n(ct)
                if fresh > 0:
                    self._series["prompt_tps"].append((ts, round(fresh / (pm / 1000.0), 1)))
            date = time.strftime("%Y-%m-%d", time.localtime(ts))
            if self._day["date"] != date:
                self._day = {"date": date, "cache_saved_tokens": 0, "clients": {}}
            self._day["cache_saved_tokens"] += _n(rec.get("cached_tokens"))
            app = rec.get("client_app")
            if app:
                self._day["clients"][app] = self._day["clients"].get(app, 0) + 1
            self._prune(now)

    def event(self, kind, alias, **fields):
        with self._lock:
            self._events.append({"ts": self._clock(), "kind": kind, "alias": alias, **fields})

    def snapshot(self):
        with self._lock:
            self._prune(self._clock())
            series = {k: [[ts, v] for ts, v in dq] for k, dq in self._series.items()}
            vals = sorted(v for _, v in self._series["ttft_ms"])
            p95 = None
            if vals:
                p95 = int(vals[max(0, math.ceil(0.95 * len(vals)) - 1)])
            return {
                "totals": dict(self._totals, since=self._since),
                "recent": list(reversed(self._recent)),
                "events": list(reversed(self._events)),
                "series": series,
                "ttft_p95_ms": p95,
                "day": {"date": self._day["date"],
                        "cache_saved_tokens": self._day["cache_saved_tokens"],
                        "clients": dict(self._day["clients"])},
            }


def _int(pattern, text):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def parse_ioreg(text):
    return {"gpu_util_pct": _int(r'"Device Utilization %"\s*=\s*(\d+)', text),
            "gpu_inuse_bytes": _int(r'"In use system memory"\s*=\s*(\d+)', text)}


def parse_vm_stat(text, page_size, total_bytes):
    pages = {}
    for name in ("free", "inactive", "speculative"):
        pages[name] = _int(r"Pages %s:\s+(\d+)" % name, text)
    if any(v is None for v in pages.values()):
        return {"ram_used_bytes": None}
    return {"ram_used_bytes": total_bytes - sum(pages.values()) * page_size}


def parse_swapusage(text):
    out = {}
    for key, name in (("total", "swap_total_bytes"), ("used", "swap_used_bytes")):
        m = re.search(r"\b%s = ([\d.]+)([KMGT])" % key, text)
        out[name] = int(float(m.group(1)) * 1024 ** ("KMGT".index(m.group(2)) + 1)) if m else None
    return out


_PRESSURE = {1: "normal", 2: "warn", 4: "critical"}


class MachineSampler:
    """RAM, swap, pressure, GPU and disk without sudo. Each command has a
    2 second budget; a missing or slow command leaves its fields None."""

    def __init__(self, store_dir, ttl_s=1.0, run=subprocess.run):
        self._dir = store_dir
        self._ttl = ttl_s
        self._run = run
        self._lock = threading.Lock()
        self._at = None
        self._val = None

    def _cmd(self, *argv):
        try:
            r = self._run(list(argv), capture_output=True, text=True, timeout=2)
            if r.returncode != 0:
                return None
            return r.stdout
        except Exception:
            return None

    def _collect(self):
        s = {"ram_total_bytes": None, "ram_used_bytes": None,
             "swap_used_bytes": None, "swap_total_bytes": None, "pressure": None,
             "gpu_util_pct": None, "gpu_inuse_bytes": None,
             "disk_free_bytes": None, "disk_total_bytes": None}
        out = self._cmd("ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-w0")
        if out is not None:
            s.update(parse_ioreg(out))
        out = self._cmd("sysctl", "-n", "hw.memsize")
        try:
            s["ram_total_bytes"] = int(out.strip()) if out else None
        except ValueError:
            pass
        out = self._cmd("vm_stat")
        if out is not None and s["ram_total_bytes"]:
            ps = _int(r"page size of (\d+) bytes", out)
            if ps:
                s.update(parse_vm_stat(out, ps, s["ram_total_bytes"]))
        out = self._cmd("sysctl", "-n", "vm.swapusage")
        if out is not None:
            s.update(parse_swapusage(out))
        out = self._cmd("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
        try:
            s["pressure"] = _PRESSURE.get(int(out.strip())) if out else None
        except ValueError:
            pass
        try:
            st = os.statvfs(self._dir)
            s["disk_free_bytes"] = st.f_bavail * st.f_frsize
            s["disk_total_bytes"] = st.f_blocks * st.f_frsize
        except OSError:
            pass
        return s

    def sample(self):
        with self._lock:
            now = time.monotonic()
            if self._val is None or now - self._at >= self._ttl:
                self._val = self._collect()
                self._at = time.monotonic()
            return dict(self._val)
