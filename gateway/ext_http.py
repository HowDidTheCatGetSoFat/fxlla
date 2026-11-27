"""HTTP helper for external engine probes. Never leaks URLs or headers."""
import json
import math
import socket
import time
import urllib.error
import urllib.request

MAX_BYTES = 4 * 1024 * 1024


class Refused(Exception):
    """Connection refused or no listener."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return ""


class ProbeError(Exception):
    """reason is one of: timeout, 'http <code>', bad json, error."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

    def __str__(self):
        return self.reason


def _is_timeout(e):
    return isinstance(e, (socket.timeout, TimeoutError))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a header never travels to another origin."""

    def redirect_request(self, *a, **kw):
        return None


# No env or system proxies, no redirects: a probe talks only to its URL.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                      _NoRedirect())


def _read_capped(r, deadline):
    chunks = []
    total = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError()
        chunk = r.read1(65536)
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
        if total >= MAX_BYTES:
            break
    return b"".join(chunks)[:MAX_BYTES]


def _no_constant(_name):
    return None


def _finite_float(text):
    v = float(text)
    return v if math.isfinite(v) else None


def get_json(url, headers=None, timeout=0.5):
    """GET and parse JSON. timeout is also a total wall-clock bound."""
    deadline = time.monotonic() + timeout
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = _read_capped(r, deadline)
    except urllib.error.HTTPError as e:
        code = e.code
        e.close()
        raise ProbeError("http %d" % code) from None
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, ConnectionRefusedError):
            raise Refused() from None
        if _is_timeout(reason):
            raise ProbeError("timeout") from None
        raise ProbeError("error") from None
    except ConnectionRefusedError:
        raise Refused() from None
    except Exception as e:
        if _is_timeout(e):
            raise ProbeError("timeout") from None
        raise ProbeError("error") from None
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_no_constant,
                          parse_float=_finite_float)
    except (ValueError, UnicodeDecodeError):
        raise ProbeError("bad json") from None


# Signature every probe receives: get(url, headers) -> parsed JSON.
GetFn = None
try:
    from typing import Callable, Optional
    GetFn = Callable[[str, Optional[dict]], object]
except Exception:  # pragma: no cover
    pass
