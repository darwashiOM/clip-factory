#!/usr/bin/env python3
"""Standalone transcription runner — bypasses MCP server env issue."""
import sys, json
sys.path.insert(0, "mcp")
from bootstrap import resolve_root_and_load_env
ROOT = resolve_root_and_load_env()

# Import server functions after env is loaded
from transcribe_server import transcribe_file

result = transcribe_file("tbfpe_qimmw.mp4", language="ar", overwrite=True)
print(json.dumps(result, ensure_ascii=False, indent=2))
