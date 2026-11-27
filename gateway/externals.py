"""Observer for external engines (Strata, Ollama, LM Studio, ComfyUI).

Probes run on demand from /monitor, in parallel, under a shared deadline.
Results are cached, served stale when a probe is late, and down engines are
not retried until a back-off passes. A probe that hangs is never resubmitted
while it is still running, so repeated snapshots cannot pile up threads.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import ext_http
import ext_local
import ext_strata

ENGINES = ("strata", "ollama", "lmstudio", "comfyui")


def _down(url, reason):
    return {"up": False, "url": url, "down_reason": reason}


class Externals:
    def __init__(self, probes, get=None, deadline_s=1.0, ttl_s=2.0,
                 down_retry_s=10.0, clock=time.monotonic):
        self.probes = dict(probes)
        self.get = get or (lambda url, headers=None:
                           ext_http.get_json(url, headers, timeout=0.5))
        self.deadline_s = deadline_s
        self.ttl_s = ttl_s
        self.down_retry_s = down_retry_s
        self.clock = clock
        self._round = threading.Lock()      # one snapshot round at a time
        self._state = threading.Lock()      # guards the dicts below
        self._res = {}       # name -> {"entry", "ts", "up", "served"}
        self._good = {}      # name -> (entry, ts) of the last successful probe
        self._running = set()
        self._snap = None
        self._snap_ts = None
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.probes)),
                                        thread_name_prefix="ext-probe")

    def _run(self, name):
        url, fn = self.probes[name]
        try:
            entry = fn(self.get)
            if not isinstance(entry, dict):
                raise ValueError("probe returned a non-dict")
            rec = {"entry": dict(entry), "up": True}
        except ext_http.Refused:
            rec = {"entry": _down(url, None), "up": False}
        except ext_http.ProbeError as e:
            rec = {"entry": _down(url, e.reason), "up": False}
        except Exception:
            rec = {"entry": _down(url, "error"), "up": False}
        with self._state:
            ts = self.clock()
            rec["ts"] = ts
            rec["served"] = False
            self._res[name] = rec
            if rec["up"]:
                self._good[name] = (rec["entry"], ts)
            self._running.discard(name)

    def snapshot(self):
        with self._round:
            now = self.clock()
            if self._snap is not None and now - self._snap_ts < self.ttl_s:
                return {k: dict(v) for k, v in self._snap.items()}
            futs = []
            with self._state:
                for name in self.probes:
                    if name in self._running:
                        continue
                    rec = self._res.get(name)
                    if (rec is not None and not rec["up"]
                            and now - rec["ts"] < self.down_retry_s):
                        continue
                    self._running.add(name)
                    futs.append(self._pool.submit(self._run, name))
            if futs:
                wait(futs, timeout=self.deadline_s)
            out = {}
            with self._state:
                end = self.clock()
                for name, (url, _fn) in self.probes.items():
                    rec = self._res.get(name)
                    if name in self._running:
                        good = self._good.get(name)
                        if good is not None:
                            e = dict(good[0])
                            e["stale"] = True
                            e["age_s"] = round(max(0.0, end - good[1]), 1)
                            out[name] = e
                        else:
                            out[name] = _down(url, "timeout")
                    elif rec is None:
                        out[name] = _down(url, "error")
                    elif rec["up"]:
                        e = dict(rec["entry"])
                        e["stale"] = False
                        e["age_s"] = 0.0
                        out[name] = e
                    else:
                        out[name] = dict(rec["entry"])
                    if rec is not None:
                        rec["served"] = True
            self._snap = out
            self._snap_ts = now
            return {k: dict(v) for k, v in out.items()}


def _env_url(name, default):
    return (os.environ.get(name) or default).rstrip("/")


def default_externals():
    strata_url = _env_url("FXLLA_STRATA_URL", "http://127.0.0.1:8096")
    ollama_url = _env_url("FXLLA_OLLAMA_URL", "http://127.0.0.1:11434")
    lms_url = _env_url("FXLLA_LMSTUDIO_URL",
                       "http://127.0.0.1:%d" % ext_local.lmstudio_port())
    comfy = os.environ.get("FXLLA_COMFYUI_URL")
    if comfy:
        comfy_urls = [comfy.rstrip("/")]
    else:
        comfy_urls = ["http://127.0.0.1:8000", "http://127.0.0.1:8188"]
    log = ext_local.OllamaLog(str(Path.home() / ".ollama/logs/server.log"))
    return Externals({
        "strata": (strata_url, ext_strata.StrataProbe(strata_url)),
        "ollama": (ollama_url, ext_local.ollama_probe(ollama_url, log)),
        "lmstudio": (lms_url, ext_local.lmstudio_probe(lms_url)),
        "comfyui": (comfy_urls[0], ext_local.comfyui_probe(comfy_urls)),
    })


def empty_externals():
    return {n: {"up": False, "url": None, "down_reason": "error"}
            for n in ENGINES}
