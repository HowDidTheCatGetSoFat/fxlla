"""In-memory telemetry store for the gateway: recent requests, lifecycle
events, rate series and day aggregates. One lock guards all state."""
import collections
import math
import threading
import time
import uuid

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

    def new_id(self):
        return uuid.uuid4().hex[:8]

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
