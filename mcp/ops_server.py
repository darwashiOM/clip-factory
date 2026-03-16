"""
clip-factory ops MCP server.

Provides health checks and directory listings.
Uses the same ffmpeg resolution as the renderer (prefers ffmpeg-full with libass).
"""

from pathlib import Path
import os
import subprocess
from mcp.server.fastmcp import FastMCP
from helpers import get_ffmpeg

from bootstrap import resolve_root_and_load_env

ROOT = resolve_root_and_load_env()

INCOMING = ROOT / "incoming"
FINAL = ROOT / "final"

mcp = FastMCP("clip-factory-ops", json_response=True)


def _safe_list(folder: Path, limit: int = 20):
    if not folder.exists():
        return []
    files = sorted(
        [p for p in folder.iterdir() if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
        }
        for p in files[:limit]
    ]


@mcp.tool()
def healthcheck() -> dict:
    """Return a simple health check for the clip factory."""
    return {
        "ok": True,
        "root": str(ROOT),
        "incoming_exists": INCOMING.exists(),
        "final_exists": FINAL.exists(),
    }


@mcp.tool()
def list_incoming(limit: int = 20) -> dict:
    """List the newest files in the incoming folder."""
    return {
        "folder": str(INCOMING),
        "files": _safe_list(INCOMING, limit),
    }


@mcp.tool()
def list_final(limit: int = 20) -> dict:
    """List the newest files in the final folder."""
    return {
        "folder": str(FINAL),
        "files": _safe_list(FINAL, limit),
    }


@mcp.tool()
def ffmpeg_version() -> dict:
    """Return the installed FFmpeg version and binary path."""
    ffmpeg = get_ffmpeg()
    try:
        result = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {
                "ok": False,
                "binary": ffmpeg,
                "error": result.stderr.splitlines()[0] if result.stderr else "non-zero exit",
            }
        first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        return {"ok": True, "binary": ffmpeg, "version": first_line}
    except FileNotFoundError:
        return {
            "ok": False,
            "binary": ffmpeg,
            "error": f"ffmpeg binary not found at '{ffmpeg}'",
        }
    except Exception as e:
        return {"ok": False, "binary": ffmpeg, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
