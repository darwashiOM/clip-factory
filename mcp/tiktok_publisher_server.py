"""
clip-factory TikTok publisher MCP server.

sys.path is patched at startup so that tiktok_poster (at the project root)
can be imported regardless of the working directory when this server starts.
"""

import sys
from pathlib import Path

# Ensure the project root is importable (tiktok_poster lives there, not in mcp/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from mcp.server.fastmcp import FastMCP
from tiktok_poster import (
    list_tiktok_accounts_core,
    list_due_tiktok_jobs_core,
    upload_due_tiktok_jobs_core,
)

mcp = FastMCP("clip-factory-tiktok-publisher", json_response=True)


@mcp.tool()
def list_tiktok_accounts() -> dict:
    """List locally configured TikTok accounts."""
    return list_tiktok_accounts_core()


@mcp.tool()
def list_due_tiktok_jobs(account: str = "", limit: int = 20) -> dict:
    """List queued TikTok jobs that are due for upload."""
    return list_due_tiktok_jobs_core(account=account, limit=limit)


@mcp.tool()
def upload_due_tiktok_jobs(
    account: str = "",
    limit: int = 3,
    dry_run: bool = False,
) -> dict:
    """
    Upload due queued TikTok jobs using the official inbox draft flow.
    Successful uploads are marked as draft_ready.
    """
    return upload_due_tiktok_jobs_core(account=account, limit=limit, dry_run=dry_run)


if __name__ == "__main__":
    mcp.run()
