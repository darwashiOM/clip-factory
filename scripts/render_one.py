#!/usr/bin/env python3
"""Direct render runner — stubs FastMCP so the renderer module loads without an MCP server."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MCP_DIR = ROOT / "mcp"

sys.path.insert(0, str(MCP_DIR))
sys.path.insert(0, str(ROOT))
os.chdir(MCP_DIR)

# --- Stub out FastMCP so the renderer module loads standalone ---
import types

_mcp_pkg = types.ModuleType("mcp")
_mcp_server = types.ModuleType("mcp.server")
_mcp_fastmcp = types.ModuleType("mcp.server.fastmcp")

class _FastMCPStub:
    def __init__(self, *a, **kw): pass
    def tool(self, *a, **kw):
        def decorator(fn): return fn
        return decorator
    def run(self, *a, **kw): pass

_mcp_fastmcp.FastMCP = _FastMCPStub
_mcp_pkg.server = _mcp_server
_mcp_server.fastmcp = _mcp_fastmcp
sys.modules["mcp"] = _mcp_pkg
sys.modules["mcp.server"] = _mcp_server
sys.modules["mcp.server.fastmcp"] = _mcp_fastmcp
# ---------------------------------------------------------------

import renderer_server_veo_timeline as renderer
import json

stem        = sys.argv[1] if len(sys.argv) > 1 else "tbfpe_qimmw"
clip_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
preset      = sys.argv[3] if len(sys.argv) > 3 else "dark-soft-recitation"

print(f"Rendering {stem} clip {clip_number} preset='{preset}' ...", flush=True)

result = renderer._render_one(
    stem=stem,
    clip_number=clip_number,
    preset=preset,
    burn_subtitles=True,
    auto_broll=True,
    overwrite=True,
)

print(json.dumps(result, indent=2, default=str))
