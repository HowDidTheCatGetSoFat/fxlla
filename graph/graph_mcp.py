#!/usr/bin/env python3
"""Minimal MCP server exposing the fxlla code graph over stdio.

Newline-delimited JSON-RPC, no dependencies. Three tools so opencode or Claude
Code can navigate a codebase structurally.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "codegraph.py")

TOOLS = [
    {"name": "find_definition",
     "description": "Find where a Python symbol (function, class, method) is defined.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}},
                     "required": ["name"]}},
    {"name": "find_references",
     "description": "Find where a Python symbol is referenced (called).",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}},
                     "required": ["name"]}},
    {"name": "find_callers",
     "description": "Find which functions call a Python symbol.",
     "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}},
                     "required": ["name"]}},
    {"name": "find_impact",
     "description": "Transitive callers (blast radius) of a Python symbol.",
     "inputSchema": {"type": "object",
                     "properties": {"name": {"type": "string"}, "depth": {"type": "integer"}},
                     "required": ["name"]}},
    {"name": "list_unused",
     "description": "List definitions never referenced by name (dead-code candidates).",
     "inputSchema": {"type": "object", "properties": {}}},
]
SUBCOMMAND = {"find_definition": "def", "find_references": "refs",
              "find_callers": "callers", "find_impact": "impact"}


def run_query(tool, args):
    if tool == "list_unused":
        cmd = [sys.executable, GRAPH, "unused", "--json"]
    else:
        cmd = [sys.executable, GRAPH, SUBCOMMAND[tool], args.get("name", ""), "--json"]
        if tool == "find_impact":
            cmd += ["--depth", str(int(args.get("depth", 5) or 5))]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if proc.returncode != 0:
        return "error: " + (proc.stderr.strip() or "query failed")
    return proc.stdout.strip() or "[]"


def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fxlla-graph", "version": "0.1.0"},
        })
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params", {})
        tool = params.get("name")
        if tool in SUBCOMMAND or tool == "list_unused":
            text = run_query(tool, params.get("arguments", {}))
            return _ok(mid, {"content": [{"type": "text", "text": text}]})
        return _err(mid, -32601, "unknown tool")
    if method and method.startswith("notifications/"):
        return None
    if mid is not None:
        return _err(mid, -32601, "method not found")
    return None


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main():
    os.environ["FXLLA_CALLER"] = "mcp"  # inherited by the subprocesses below
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
