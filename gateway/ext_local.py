"""Probes for Ollama, LM Studio and ComfyUI. Parsers never raise."""
import datetime
import json
import math
import re
import statistics
from pathlib import Path

from ext_http import ProbeError, Refused

GEN_PATHS = (
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "/api/embeddings",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
)

MAX_LATENCIES = 10000
_MB = 1024 * 1024
_GIB = 1024 ** 3

_GIN_RE = re.compile(
    r'^\[GIN\]\s+(\d{4})/(\d{2})/(\d{2})\s+-\s+\d{2}:\d{2}:\d{2}\s*\|\s*\d+\s*\|'
    r'\s*(\S+)\s*\|\s*[^|]*\|\s*[A-Z]+\s+"([^"]*)"'
)
_DUR_RE = re.compile("(\\d+(?:\\.\\d+)?)(ns|\u00b5s|\u03bcs|us|ms|h|m|s)")
_UNIT_MS = {
    "ns": 1e-6,
    "\u00b5s": 1e-3,
    "\u03bcs": 1e-3,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
    "m": 60000.0,
    "h": 3600000.0,
}

_LIMIT = 1e18


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float):
        return v if math.isfinite(v) and abs(v) < _LIMIT else None
    return v if abs(v) < _LIMIT else None


def _dict(x):
    return x if isinstance(x, dict) else {}


def _list(x):
    return x if isinstance(x, list) else []


def _duration_ms(text):
    pos = 0
    total = 0.0
    for m in _DUR_RE.finditer(text):
        if m.start() != pos:
            return None
        total += float(m.group(1)) * _UNIT_MS[m.group(2)]
        pos = m.end()
    if pos == 0 or pos != len(text):
        return None
    return total


def parse_gin(line):
    try:
        m = _GIN_RE.match(line)
        if not m:
            return None
        ms = _duration_ms(m.group(4))
        if ms is None:
            return None
        day = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        path = m.group(5).split("?", 1)[0]
        return day, round(ms, 6), path
    except Exception:
        return None


class OllamaLog:
    def __init__(self, path, today=datetime.date.today):
        self.path = Path(path)
        self._today = today
        self._offset = 0
        self._pending = b""
        self._date = None
        self._lat = []
        self._count = 0

    def _reset(self):
        self._offset = 0
        self._pending = b""
        self._lat = []
        self._count = 0

    def _ingest(self, line, today):
        parsed = parse_gin(line)
        if parsed is None:
            return
        day, ms, path = parsed
        if day != today or path not in GEN_PATHS:
            return
        self._count += 1
        self._lat.append(ms)
        if len(self._lat) > MAX_LATENCIES:
            del self._lat[: len(self._lat) - MAX_LATENCIES]

    def _refresh(self):
        today = self._today()
        if today != self._date:
            self._date = today
            self._reset()
        try:
            size = self.path.stat().st_size
        except OSError:
            self._reset()
            return
        if size < self._offset:
            self._reset()
        if size == self._offset:
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read(size - self._offset)
        self._offset += len(data)
        data = self._pending + data
        parts = data.split(b"\n")
        self._pending = parts.pop()
        for raw in parts:
            self._ingest(raw.decode("utf-8", "replace").rstrip("\r"), today)

    def today_stats(self):
        try:
            self._refresh()
            med = round(statistics.median(self._lat), 1) if self._lat else None
            return {"requests": self._count, "median_ms": med}
        except Exception:
            return {"requests": 0, "median_ms": None}


def _expires_in(text):
    if not isinstance(text, str):
        return None
    try:
        s = text.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        s = re.sub(r"(\.\d{6})\d+", r"\1", s)
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return int((dt - now).total_seconds())
    except Exception:
        return None


def _mb(x):
    n = _num(x)
    return None if n is None else int(round(n / _MB))


def parse_ollama(ps, tags, stats):
    loaded = []
    for m in _list(_dict(ps).get("models")):
        m = _dict(m)
        name = m.get("name") if isinstance(m.get("name"), str) else m.get("model")
        loaded.append({
            "name": name if isinstance(name, str) else None,
            "size_mb": _mb(m.get("size")),
            "vram_mb": _mb(m.get("size_vram")),
            "context": _num(m.get("context_length")),
            "expires_in_s": _expires_in(m.get("expires_at")),
        })
    installed = None
    if isinstance(tags, dict) and isinstance(tags.get("models"), list):
        models = tags["models"]
        total = sum(_num(_dict(m).get("size")) or 0 for m in models)
        installed = {"count": len(models), "size_gb": round(total / _GIB, 1)}
    return {
        "loaded": loaded,
        "installed": installed,
        "today": stats if isinstance(stats, dict) else None,
    }


def lmstudio_port(config_path=None):
    if config_path is None:
        config_path = Path.home() / ".lmstudio/.internal/http-server-config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            port = json.load(f).get("port")
        if isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536:
            return port
    except Exception:
        pass
    return 1234


def parse_lmstudio(models):
    items = models.get("data") if isinstance(models, dict) else models
    items = _list(items)
    loaded = []
    for m in items:
        m = _dict(m)
        if m.get("state") == "loaded":
            mid = m.get("id")
            loaded.append({
                "name": mid if isinstance(mid, str) else None,
                "context": _num(m.get("max_context_length")),
            })
    return {"loaded": loaded, "available": len(items)}


def _used_total(used_total_pairs):
    used = total = 0.0
    seen = False
    for t, f in used_total_pairs:
        t, f = _num(t), _num(f)
        if t is None or f is None:
            continue
        seen = True
        used += t - f
        total += t
    if not seen:
        return {"used_mb": None, "total_mb": None}
    return {"used_mb": int(round(used / _MB)), "total_mb": int(round(total / _MB))}


def parse_comfyui(stats, queue):
    system = _dict(_dict(stats).get("system"))
    devices = [_dict(d) for d in _list(_dict(stats).get("devices"))]
    q = _dict(queue)
    version = system.get("comfyui_version")
    return {
        "version": version if isinstance(version, str) else None,
        "queue": {
            "running": len(_list(q.get("queue_running"))),
            "pending": len(_list(q.get("queue_pending"))),
        },
        "vram": _used_total((d.get("vram_total"), d.get("vram_free")) for d in devices),
        "ram": _used_total([(system.get("ram_total"), system.get("ram_free"))]),
    }


def _base(url):
    return url.rstrip("/")


def ollama_probe(base_url, log):
    base = _base(base_url)

    def probe(get):
        ps = get(base + "/api/ps", None)
        try:
            tags = get(base + "/api/tags", None)
        except (Refused, ProbeError):
            tags = None
        stats = log.today_stats() if log is not None else None
        return {"up": True, "url": base, **parse_ollama(ps, tags, stats)}

    return probe


def lmstudio_probe(base_url):
    base = _base(base_url)

    def probe(get):
        models = get(base + "/api/v0/models", None)
        return {"up": True, "url": base, **parse_lmstudio(models)}

    return probe


def comfyui_probe(base_urls):
    bases = [_base(u) for u in base_urls]

    def probe(get):
        err = None
        for base in bases:
            try:
                stats = get(base + "/system_stats", None)
            except Refused:
                continue
            except ProbeError as e:
                err = err or e
                continue
            try:
                queue = get(base + "/queue", None)
            except (Refused, ProbeError):
                queue = None
            return {"up": True, "url": base, **parse_comfyui(stats, queue)}
        if err is not None:
            raise err
        raise Refused()

    return probe
