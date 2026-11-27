"""HTTP helper for external engine probes. Never leaks URLs or headers."""
import json
import socket
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


def get_json(url, headers=None, timeout=0.5):
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(MAX_BYTES)
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
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ProbeError("bad json") from None


# Signature every probe receives: get(url, headers) -> parsed JSON.
GetFn = None
try:
    from typing import Callable, Optional
    GetFn = Callable[[str, Optional[dict]], object]
except Exception:  # pragma: no cover
    pass
