"""Shared reader/writer for opencode's opencode.json, v2 format.

fxlla owns three kinds of entries in that file: the 'local' provider and the
'fxlla-rag', 'fxlla-graph' and 'fxlla-media' MCP servers. opencode v2 reads
the v1 layout but marks it deprecated, so these are written in the v2 layout
(top-level 'providers', 'mcp.servers') and any v1 copy of an entry fxlla owns
is moved over, never left beside its v2 twin. Everything else in the file is
the user's and is never touched.

The v1 to v2 field mapping follows opencode's own translation (v2.4.1,
packages/core/src/v1/config/migrate.ts), so a migrated entry means what the
v1 entry meant.
"""
import json
import math
import os
import re
import stat
import tempfile

SCHEMA = "https://opencode.ai/config.json"
PROVIDER = "local"
MCP_NAMES = ("fxlla-rag", "fxlla-graph", "fxlla-media")
# v1 npm package -> v2 package specifier (migrate.ts packageSpecifier).
AISDK = "aisdk:"
OPENAI_COMPAT = "@ai-sdk/openai-compatible"
# Model options an openai-compatible model reads itself; every other one was
# sent in the request body in v1 (provider-options.ts COMPATIBLE_KEYS).
COMPATIBLE_KEYS = frozenset((
    "apiKey", "baseURL", "headers", "body", "extraBody", "timeout",
    "chunkTimeout", "headerTimeout", "disabled", "reasoningEffort",
    "textVerbosity", "user", "strictJsonSchema"))
MAX_INT = 2 ** 53 - 1


class ConfigError(Exception):
    """The config file exists but cannot be read as JSON or JSONC."""


def _string_end(text, i):
    # Index just past the string that opens at text[i].
    j = i + 1
    while j < len(text) and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
    return j + 1


def _strip_jsonc(text):
    """Drop // and /* */ comments and trailing commas, leaving strings alone."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':
            j = _string_end(text, i)
            out.append(text[i:j])
            i = j
        elif text.startswith("//", i):
            while i < n and text[i] not in "\r\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            out.append(" ")
        else:
            out.append(text[i])
            i += 1
    text = "".join(out)
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':
            j = _string_end(text, i)
            out.append(text[i:j])
            i = j
        elif text[i] == "," and re.compile(r"\s*[\]}]").match(text, i + 1):
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def load(cfg):
    """The parsed file; {} when it is missing or empty. Raises ConfigError when
    it exists but is not a JSON (or JSONC) object, so it is never overwritten."""
    try:
        with open(cfg, encoding="utf-8-sig") as f:
            text = f.read()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError("%s: cannot read it (%s)" % (cfg, e.__class__.__name__))
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        try:
            data = json.loads(_strip_jsonc(text))
        except ValueError as e:
            raise ConfigError("%s: not valid JSON or JSONC (%s)" % (cfg, e))
    if not isinstance(data, dict):
        raise ConfigError("%s: the top level is not a JSON object" % cfg)
    return data


def save(cfg, data):
    """Atomic write: a temp file beside the target, then os.replace. The file
    holds API keys, so it is 0600, or an existing stricter mode if there is one."""
    cfg = os.path.realpath(cfg)
    mode = 0o600
    try:
        mode &= stat.S_IMODE(os.stat(cfg).st_mode)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cfg), prefix=".opencode.json.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, mode)
        os.replace(tmp, cfg)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _package(npm):
    return npm if npm.startswith(AISDK) else AISDK + npm


def _int(v):
    # migrate.ts int(): truncate and clamp; anything that is not a finite number
    # has no meaning as a limit or a price and is dropped by the callers.
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return max(-MAX_INT, min(MAX_INT, int(v)))


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return v


def _split(options, compatible):
    # provider-options.ts: an openai-compatible model keeps its known keys as
    # settings and sends the rest in the body; any other model keeps them all.
    if not compatible:
        return dict(options), None
    settings = {k: v for k, v in options.items() if k in COMPATIBLE_KEYS}
    body = {k: v for k, v in options.items() if k not in COMPATIBLE_KEYS}
    return settings, body or None


def _binding_off(options):
    return any(isinstance(options.get(k), dict) and options[k].get("blockBinding") is False
               for k in ("thinking", "reasoningConfig"))


def _cost(c):
    # One entry for the base price, one context tier above 200k.
    def entry(src):
        out = {}
        for key in ("input", "output"):
            if _num(src.get(key)) is not None:
                out[key] = src[key]
        cache = {k: src[f] for k, f in (("read", "cache_read"), ("write", "cache_write"))
                 if _num(src.get(f)) is not None}
        if cache:
            out["cache"] = cache
        return out
    costs = [entry(c)]
    tier = c.get("context_over_200k")
    if isinstance(tier, dict):
        costs.append(dict(entry(tier), tier={"type": "context", "size": 200000}))
    return costs


def _model(m, npm=None):
    # v1 model -> v2 model (migrate.ts migrateModel, lines 196-265).
    out = {}
    for src, dst in (("id", "modelID"), ("family", "family"), ("name", "name")):
        if isinstance(m.get(src), str):
            out[dst] = m[src]
    options = m.get("options") if isinstance(m.get("options"), dict) else None
    binding_off = options is not None and _binding_off(options)
    if binding_off:
        options = dict(options)
        for key in ("thinking", "reasoningConfig"):
            val = options.get(key)
            if not (isinstance(val, dict) and val.get("blockBinding") is False):
                continue
            rest = {k: v for k, v in val.items() if k != "blockBinding"}
            if rest:
                options[key] = rest
            else:
                del options[key]
    # compatibility: interleaved names the reasoning field (model.ts compatibility).
    compat = {}
    il = m.get("interleaved")
    if isinstance(il, dict):
        il = il.get("field")
    if isinstance(il, str):
        compat["reasoningField"] = il
    if binding_off:
        compat["supportsThinkingBlockBinding"] = False
    if compat:
        out["compatibility"] = compat
    prov = m.get("provider") if isinstance(m.get("provider"), dict) else {}
    mnpm = prov.get("npm") if isinstance(prov.get("npm"), str) else None
    if mnpm:
        out["package"] = _package(mnpm)
    compatible = (mnpm or npm) == OPENAI_COMPAT
    settings = body = None
    if options is not None:
        settings, body = _split(options, compatible)
    if isinstance(prov.get("api"), str) and prov["api"]:
        settings = dict(settings or {}, baseURL=prov["api"])
    if settings is not None:
        out["settings"] = settings
    mod = m.get("modalities") if isinstance(m.get("modalities"), dict) else {}
    caps = {}
    if isinstance(m.get("tool_call"), bool):
        caps["tools"] = m["tool_call"]
    for src, dst in (("input", "input"), ("output", "output")):
        if isinstance(mod.get(src), list):
            caps[dst] = mod[src]
    if caps:
        out["capabilities"] = caps
    if isinstance(m.get("headers"), dict):
        out["headers"] = m["headers"]
    if body is not None:
        out["body"] = body
    if isinstance(m.get("variants"), dict):
        out["variants"] = []
        for vid, vopts in m["variants"].items():
            if not isinstance(vopts, dict):
                continue
            vs, vb = _split(vopts, compatible)
            var = {"id": vid, "settings": vs}
            if vb is not None:
                var["body"] = vb
            out["variants"].append(var)
    if isinstance(m.get("cost"), dict):
        out["cost"] = _cost(m["cost"])
    if m.get("status") == "deprecated":
        out["disabled"] = True
    if isinstance(m.get("limit"), dict):
        limit = {}
        for key in ("context", "input", "output"):
            v = _int(m["limit"].get(key))
            if v is not None:
                limit[key] = v
        out["limit"] = limit
    return out


def _provider(v1):
    # v1 provider -> v2 provider (migrate.ts migrateStandardProvider, lines
    # 126-146): 'npm' becomes 'package', 'options' becomes 'settings' with
    # headers and body lifted out, setCacheKey moves to compatibility, and 'api'
    # is the fallback baseURL.
    out = {}
    if "name" in v1:
        out["name"] = v1["name"]
    if "env" in v1:
        out["env"] = v1["env"]
    npm = v1.get("npm") if isinstance(v1.get("npm"), str) else None
    if npm:
        out["package"] = _package(npm)
    has_opts = isinstance(v1.get("options"), dict)
    opts = dict(v1["options"]) if has_opts else {}
    headers = opts.pop("headers", None)
    body = opts.pop("body", None)
    if "setCacheKey" in opts:
        out["compatibility"] = {"supportsPromptCacheKey": opts.pop("setCacheKey")}
    api = v1.get("api") if isinstance(v1.get("api"), str) else None
    if api:
        opts["baseURL"] = opts.get("baseURL") or api
    if has_opts or api:
        out["settings"] = opts
    if has_opts and isinstance(headers, dict):
        out["headers"] = {k: h for k, h in headers.items() if isinstance(h, str)}
    if has_opts and isinstance(body, dict):
        out["body"] = dict(body)
    if isinstance(v1.get("models"), dict):
        out["models"] = {k: _model(m, npm) for k, m in v1["models"].items()
                         if isinstance(m, dict)}
    return out


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _timeout(t):
    # v1 had one MCP timeout; v2 splits it by phase (migrate.ts migrateMcpTimeout).
    if isinstance(t, int) and not isinstance(t, bool) and t > 0:
        return {"startup": t, "catalog": t, "execution": t}
    return None


def _is_v1_server(v):
    # normalize.ts isDirectLegacyMcp (line 850): a v1 entry, as opposed to the
    # v2 container, has a type of local or remote.
    return isinstance(v, dict) and v.get("type") in ("local", "remote")


def _container(mcp):
    """mcp['servers'] as the v2 container. normalize.ts (line 295) reads a
    'servers' key as the container unless its value is a v1 entry; then it is a
    v1 server NAMED 'servers' and lands inside the container like any other v1
    entry (legacyServers merged into servers, line 316). The file cannot hold
    both under one key, so such an entry is moved into a fresh container."""
    servers = mcp.get("servers")
    if _is_v1_server(servers):
        servers = {"servers": _server(servers)}
    elif not isinstance(servers, dict):
        servers = {}
    mcp["servers"] = servers
    return servers


def _server(v1):
    # v1 MCP entry -> v2 server (migrate.ts migrateMcp): 'enabled' becomes the
    # inverse 'disabled'.
    out = {"type": v1.get("type", "local")}
    keys = ("command", "cwd", "environment") if out["type"] == "local" else ("url", "headers")
    for key in keys:
        if key in v1:
            out[key] = v1[key]
    if "enabled" in v1:
        out["disabled"] = not v1["enabled"]
    if "codemode" in v1:
        out["codemode"] = v1["codemode"]
    t = _timeout(v1.get("timeout"))
    if t:
        out["timeout"] = t
    return out


def migrate(data):
    """Move the v1 entries fxlla owns to their v2 location, in place."""
    old = data.get("provider")
    if isinstance(old, dict) and isinstance(old.get(PROVIDER), dict):
        moved = _provider(old.pop(PROVIDER))
        provs = data.get("providers")
        if not isinstance(provs, dict):
            provs = data["providers"] = {}
        # An entry already in v2 form wins over the v1 copy it replaces.
        cur = provs.get(PROVIDER)
        provs[PROVIDER] = _merge(moved, cur) if isinstance(cur, dict) else moved
        if not old:
            del data["provider"]
    mcp = data.get("mcp")
    if isinstance(mcp, dict):
        for name in MCP_NAMES:
            v1 = mcp.get(name)
            # A v1 entry has its fields at the top.
            if isinstance(v1, dict) and "type" in v1:
                _container(mcp).setdefault(name, _server(v1))
                del mcp[name]
    return data


def open_config(cfg):
    """Load, migrate and stamp the schema: the starting point of every writer."""
    try:
        data = migrate(load(cfg))
    except ConfigError as e:
        raise SystemExit("fxlla: %s. The file was left as it is; fix or remove it and run again." % e)
    data.setdefault("$schema", SCHEMA)
    return data


def local_provider(data):
    provs = data.get("providers")
    if not isinstance(provs, dict):
        provs = data["providers"] = {}
    return provs.setdefault(PROVIDER, {})


def set_server(data, name, command, environment):
    mcp = data.get("mcp")
    if not isinstance(mcp, dict):
        mcp = data["mcp"] = {}
    servers = _container(mcp)
    servers[name] = {
        "type": "local",
        "command": command,
        "environment": environment,
        "disabled": False,
    }
