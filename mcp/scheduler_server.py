"""
clip-factory scheduler MCP server.

sys.path is patched at startup so that scheduler_core (at the project root)
can be imported regardless of the working directory when this server starts.
"""

import sys
from pathlib import Path

# Ensure the project root is importable (scheduler_core lives there, not in mcp/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from mcp.server.fastmcp import FastMCP
from scheduler_core import (
    scheduler_summary_core,
    list_scheduled_jobs_core,
    plan_daily_schedule_core,
    schedule_next_days_core,
)

mcp = FastMCP("clip-factory-scheduler", json_response=True)


@mcp.tool()
def scheduler_summary() -> dict:
    """Return counts for the scheduler and publish queue."""
    return scheduler_summary_core()


@mcp.tool()
def list_scheduled_jobs(
    date_str: str = "",
    account: str = "",
    limit: int = 100,
) -> dict:
    """List scheduled jobs, optionally filtered by date or account."""
    return list_scheduled_jobs_core(date_str=date_str, account=account, limit=limit)


@mcp.tool()
def plan_daily_schedule(
    account: str,
    date_str: str = "",
    posts_per_day: int = 10,
    start_hour: int = 8,
    end_hour: int = 22,
    tag: str = "",
    platform: str = "tiktok",
    dry_run: bool = False,
) -> dict:
    """
    Create enough queued jobs to reach the daily target.
    Running it again is safe because it only fills missing slots.
    """
    return plan_daily_schedule_core(
        account=account,
        date_str=date_str,
        posts_per_day=posts_per_day,
        start_hour=start_hour,
        end_hour=end_hour,
        tag=tag,
        platform=platform,
        dry_run=dry_run,
    )


@mcp.tool()
def schedule_next_days(
    account: str,
    days: int = 1,
    posts_per_day: int = 10,
    start_hour: int = 8,
    end_hour: int = 22,
    tag: str = "",
    platform: str = "tiktok",
    dry_run: bool = False,
) -> dict:
    """Create daily schedules for the next N days."""
    return schedule_next_days_core(
        account=account,
        days=days,
        posts_per_day=posts_per_day,
        start_hour=start_hour,
        end_hour=end_hour,
        tag=tag,
        platform=platform,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    mcp.run()
