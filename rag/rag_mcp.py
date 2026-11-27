#!/usr/bin/env python3
"""Minimal MCP server exposing fxlla RAG over stdio.

Speaks newline-delimited JSON-RPC (the MCP stdio transport) with no third-party
dependencies. Exposes one tool, rag_search, so opencode or Claude Code can query
a local knowledge base.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAG = os.path.join(HERE, "core.py")

TOOLS = [{
    "name": "rag_search",
    "description": "Search a local fxlla knowledge base and return the most "
                   "relevant text chunks with their source.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "kb": {"type": "string", "description": "knowledge base name"},
            "query": {"type": "string", "description": "the search query"},
            "k": {"type": "integer", "description": "number of chunks to return"},
        },
        "required": ["kb", "query"],
    },
}]


def rag_search(kb, query, k=5):
    proc = subprocess.run(
        [sys.executable, RAG, "search", kb, query, "-k", str(int(k or 5)), "--json"],
        capture_output=True, text=True, env=os.environ)
    if proc.returncode != 0:
        return "error: " + (proc.stderr.strip() or "search failed")
    return proc.stdout.strip() or "[]"


def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fxlla-rag", "version": "0.1.0"},
        })
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params", {})
        args = params.get("arguments", {})
        if params.get("name") == "rag_search":
            text = rag_search(args.get("kb", ""), args.get("query", ""), args.get("k", 5))
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
