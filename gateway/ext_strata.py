"""Strata probe for /monitor externals. The key is sent only as a Bearer
header to the Strata base URL and never appears in an entry or error."""
import json
import math
import os
from pathlib import Path

import ext_http

MB = 1024 * 1024
ROW_KEYS = ("ts", "model", "engine", "ram_mb", "ttft_ms", "tps", "source",
            "prompt_tokens", "cached_tokens", "output_tokens",
            "reasoning_tokens", "prompt_ms", "decode_ms", "duration_ms",
            "finish", "client_app")
LIVE_KEYS = ("phase", "prompt_read", "prompt_total", "generated",
             "max_tokens", "elapsed_s", "tok_s")
LAUNCH_DIR = Path.home() / ".local/state/strata/launch"

_LIMIT = 1e18


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float):
        return v if math.isfinite(v) and abs(v) < _LIMIT else None
    return v if abs(v) < _LIMIT else None


def _dict(v):
    return v if isinstance(v, dict) else {}


def _mb(v):
    v = _num(v)
    return None if v is None else int(v) // MB


def find_keys(env=os.environ, launch_dir=LAUNCH_DIR):
    keys = []
    k = env.get("STRATA_API_KEY")
    if isinstance(k, str) and k:
        keys.append(k)
    files = []
    try:
        for p in Path(launch_dir).glob("*.json"):
            try:
                files.append((p.stat().st_mtime, p))
            except OSError:
                continue
    except OSError:
        pass
    files.sort(key=lambda t: t[0], reverse=True)
    for _, p in files:
        try:
            k = json.loads(p.read_text()).get("api_key")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(k, str) and k and k not in keys:
            keys.append(k)
    return keys


def row(rec, model, boot_time):
    if not isinstance(rec, dict):
        return None
    t = _num(rec.get("time"))
    if t is None:
        return None
    if _num(boot_time) is not None and t < boot_time:
        return None
    out = _num(rec.get("output_tokens"))
    dec = _num(rec.get("decode_ms"))
    tps = round(out / (dec / 1000), 1) if (dec and dec > 0 and out is not None) else None
    ttft = _num(rec.get("first_engine_token_s"))
    dur = _num(rec.get("duration_s"))
    fin = rec.get("finish")
    return {
        "ts": int(t), "model": model, "engine": "strata", "ram_mb": None,
        "ttft_ms": None if ttft is None else round(ttft * 1000),
        "tps": tps, "source": "strata",
        "prompt_tokens": _num(rec.get("prompt_tokens")),
        "cached_tokens": _num(rec.get("reused")),
        "output_tokens": out, "reasoning_tokens": None,
        "prompt_ms": _num(rec.get("prompt_ms")), "decode_ms": dec,
        "duration_ms": None if dur is None else int(dur * 1000),
        "finish": fin if isinstance(fin, str) else None,
        "client_app": None,
    }


def parse(health, telemetry, metrics, auth):
    try:
        return _parse(health, telemetry, metrics, auth)
    except Exception:
        return _parse(None, None, None, auth)


def _parse(health, telemetry, metrics, auth):
    h = _dict(health)
    t = _dict(telemetry)
    inst = _dict(t.get("instance"))
    mdl = _dict(t.get("model"))
    work = _dict(t.get("work"))
    rates = _dict(t.get("rates"))
    ram = _dict(t.get("ram"))
    tot = _dict(t.get("totals"))
    model = mdl.get("id") if isinstance(mdl.get("id"), str) else None
    if model is None and isinstance(h.get("model"), str):
        model = h["model"]
    context = _num(mdl.get("context"))
    if context is None:
        context = _num(h.get("max_context"))
    loaded = mdl.get("loaded") if isinstance(mdl.get("loaded"), bool) else None
    if loaded is None and isinstance(h.get("loaded"), bool):
        loaded = h["loaded"]
    state = inst.get("state") if isinstance(inst.get("state"), str) else None
    live = None
    lv = _dict(_dict(metrics).get("live"))
    if lv and lv.get("state") != "idle":
        live = {k: lv.get(k) for k in LIVE_KEYS}
    recent = []
    rows = t.get("recent")
    if isinstance(rows, list):
        for r in rows:
            try:
                e = row(r, model, inst.get("boot_time"))
            except Exception:
                e = None
            if e is not None:
                recent.append(e)
        recent.sort(key=lambda e: e["ts"], reverse=True)
        recent = recent[:20]
    return {
        "auth": auth, "model": model, "context": context, "loaded": loaded,
        "state": state,
        "work": {"active": _num(work.get("active")),
                 "queued": _num(work.get("queued"))},
        "live": live,
        "rates": {"answer_tok_s": _num(rates.get("answer_tok_s")),
                  "prefill_tok_s": _num(rates.get("prefill_tok_s"))},
        "ram": {"used_mb": _mb(ram.get("used")),
                "total_mb": _mb(ram.get("total"))},
        "totals": {"requests": _num(tot.get("requests")),
                   "prompt_tokens": _num(tot.get("prompt_tokens")),
                   "output_tokens": _num(tot.get("output_tokens")),
                   "since": _num(tot.get("since"))},
        "recent": recent,
    }


class StrataProbe:
    def __init__(self, base_url, key_source=find_keys):
        self.base_url = base_url.rstrip("/")
        self.key_source = key_source
        self._key = None

    def _auth_get(self, get, path, key):
        return get(self.base_url + path, {"Authorization": "Bearer " + key})

    def _telemetry(self, get):
        """Return (telemetry, key, auth)."""
        if self._key is not None:
            key = self._key
            try:
                return self._auth_get(get, "/v1/telemetry", key), key, "ok"
            except ext_http.ProbeError as e:
                self._key = None
                if e.reason != "http 401":
                    return None, key, "ok"
        keys = list(self.key_source())
        if not keys:
            return None, None, "missing"
        for key in keys:
            try:
                tel = self._auth_get(get, "/v1/telemetry", key)
            except ext_http.ProbeError as e:
                if e.reason == "http 401":
                    continue
                self._key = key
                return None, key, "ok"
            self._key = key
            return tel, key, "ok"
        return None, None, "rejected"

    def __call__(self, get):
        health = get(self.base_url + "/health", None)
        tel, key, auth = self._telemetry(get)
        metrics = None
        if key is not None and isinstance(tel, dict):
            active = _num(_dict(tel.get("work")).get("active"))
            if active and active > 0:
                try:
                    metrics = self._auth_get(get, "/metrics", key)
                except ext_http.ProbeError:
                    metrics = None
        return {"up": True, "url": self.base_url,
                **parse(health, tel, metrics, auth)}
