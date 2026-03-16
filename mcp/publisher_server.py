from pathlib import Path
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from helpers import atomic_write_json


from bootstrap import resolve_root_and_load_env

ROOT = resolve_root_and_load_env()

POOL_DIR = ROOT / "pool"
POOL_FILE = POOL_DIR / "video_pool.json"

PUBLISHER_DIR = ROOT / "publisher"
QUEUE_FILE = PUBLISHER_DIR / "publish_queue.json"


mcp = FastMCP("clip-factory-publisher", json_response=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_queue():
    PUBLISHER_DIR.mkdir(parents=True, exist_ok=True)
    if not QUEUE_FILE.exists():
        QUEUE_FILE.write_text(
            json.dumps(
                {
                    "jobs": {},
                    "history": [],
                    "updated_at": _now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _ensure_pool():
    if not POOL_FILE.exists():
        raise FileNotFoundError(
            f"Missing pool file: {POOL_FILE}. Run sync_final_to_pool in video-pool first."
        )


def _load_queue() -> dict:
    _ensure_queue()
    return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))


def _save_queue(data: dict):
    data["updated_at"] = _now_iso()
    atomic_write_json(QUEUE_FILE, data)


def _load_pool() -> dict:
    _ensure_pool()
    return json.loads(POOL_FILE.read_text(encoding="utf-8"))


def _save_pool(data: dict):
    data["updated_at"] = _now_iso()
    atomic_write_json(POOL_FILE, data)


def _normalize_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    out = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _get_pool_clip(file_name: str) -> dict:
    pool = _load_pool()
    clips = pool.get("clips", {})
    if file_name not in clips:
        raise FileNotFoundError(f"Clip not found in pool: {file_name}")
    return clips[file_name]


def _build_default_caption(clip: dict) -> str:
    tags = clip.get("tags", [])
    source = clip.get("source_stem", "")
    pieces = []
    if tags:
        pieces.append(" | ".join(tags[:3]))
    if source:
        pieces.append(source.replace("-", " ").replace("_", " "))
    return "\n".join([p for p in pieces if p]).strip()


def _build_default_hashtags(clip: dict) -> List[str]:
    base = ["#islam", "#muslim", "#reminder"]
    tag_map = {
        "dua": "#dua",
        "quran": "#quran",
        "dhikr": "#dhikr",
        "prayer": "#salah",
        "reflection": "#reminder",
        "tawakkul": "#tawakkul",
        "akhirah": "#akhirah",
    }
    for t in clip.get("tags", []):
        key = t.strip().lower()
        if key in tag_map and tag_map[key] not in base:
            base.append(tag_map[key])
    return base[:8]


def _job_sort_key(job: dict) -> str:
    return job.get("created_at", "")


@mcp.tool()
def queue_summary() -> dict:
    """Return simple counts for the publish queue."""
    data = _load_queue()
    jobs = list(data.get("jobs", {}).values())

    counts = {
        "queued": 0,
        "draft_ready": 0,
        "posted": 0,
        "failed": 0,
        "cancelled": 0,
    }
    for j in jobs:
        status = j.get("status", "queued")
        if status in counts:
            counts[status] += 1

    return {
        "ok": True,
        "queue_file": str(QUEUE_FILE),
        "total_jobs": len(jobs),
        **counts,
    }


@mcp.tool()
def list_publish_jobs(
    status: str = "",
    account: str = "",
    limit: int = 50,
) -> dict:
    """
    List publish jobs with optional filtering.
    status can be: queued, draft_ready, posted, failed, cancelled
    """
    data = _load_queue()
    jobs = list(data.get("jobs", {}).values())

    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    if account:
        jobs = [j for j in jobs if j.get("account") == account]

    jobs = sorted(jobs, key=_job_sort_key, reverse=True)

    return {
        "ok": True,
        "queue_file": str(QUEUE_FILE),
        "count": len(jobs[:limit]),
        "jobs": jobs[:limit],
    }


@mcp.tool()
def get_publish_job(job_id: str) -> dict:
    """Get one publish job by id."""
    data = _load_queue()
    jobs = data.get("jobs", {})
    if job_id not in jobs:
        raise FileNotFoundError(f"Job not found: {job_id}")
    return {
        "ok": True,
        "job": jobs[job_id],
    }


@mcp.tool()
def create_publish_job(
    file_name: str,
    account: str,
    caption_text: str = "",
    hashtags: Optional[List[str]] = None,
    platform: str = "tiktok",
    scheduled_for: str = "",
) -> dict:
    """
    Create a publish queue entry from a clip in the video pool.
    The clip should normally be approved first.
    """
    account = account.strip()
    if not account:
        raise ValueError("account is required")

    clip = _get_pool_clip(file_name)

    if clip.get("status") != "approved":
        return {
            "ok": False,
            "message": "Clip is not approved yet",
            "file_name": file_name,
            "clip_status": clip.get("status"),
        }

    blocked = clip.get("blocked_accounts", [])
    allowed = clip.get("allowed_accounts", [])
    posted_accounts = clip.get("posted_accounts", [])

    if account in blocked:
        return {
            "ok": False,
            "message": "Clip is blocked for this account",
            "file_name": file_name,
            "account": account,
        }

    if allowed and account not in allowed:
        return {
            "ok": False,
            "message": "Clip is not allowed for this account",
            "file_name": file_name,
            "account": account,
        }

    if account in posted_accounts:
        return {
            "ok": False,
            "message": "Clip was already posted for this account",
            "file_name": file_name,
            "account": account,
        }

    caption = caption_text.strip() or clip.get("caption_text", "").strip() or _build_default_caption(clip)
    hashtags_final = _normalize_list(hashtags) if hashtags is not None else _normalize_list(clip.get("hashtags")) or _build_default_hashtags(clip)

    queue = _load_queue()

    # Prevent duplicate active jobs for the same clip + account
    for existing_job in queue.get("jobs", {}).values():
        if (
            existing_job.get("file_name") == file_name
            and existing_job.get("account") == account
            and existing_job.get("status") not in {"cancelled", "failed", "posted"}
        ):
            return {
                "ok": False,
                "message": "An active job already exists for this clip and account",
                "file_name": file_name,
                "account": account,
                "existing_job_id": existing_job.get("job_id"),
                "existing_status": existing_job.get("status"),
            }
    job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "status": "queued",
        "platform": platform,
        "account": account,
        "file_name": file_name,
        "path": clip.get("path", ""),
        "source_stem": clip.get("source_stem", ""),
        "preset": clip.get("preset", ""),
        "caption_text": caption,
        "hashtags": hashtags_final,
        "scheduled_for": scheduled_for.strip() or None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "notes": "",
        "external_post_id": "",
    }

    queue.setdefault("jobs", {})[job_id] = job
    queue.setdefault("history", []).append(
        {
            "type": "job_created",
            "job_id": job_id,
            "file_name": file_name,
            "account": account,
            "timestamp": _now_iso(),
        }
    )
    _save_queue(queue)

    return {
        "ok": True,
        "job": job,
    }


@mcp.tool()
def create_publish_job_from_random_pool_clip(
    account: str,
    tag: str = "",
    platform: str = "tiktok",
    scheduled_for: str = "",
) -> dict:
    """
    Pick one approved eligible clip from the pool and create a queued publish job.
    """
    account = account.strip()
    if not account:
        raise ValueError("account is required")

    pool = _load_pool()
    clips = list(pool.get("clips", {}).values())

    eligible = []
    for clip in clips:
        if clip.get("status") != "approved":
            continue
        if tag and tag not in clip.get("tags", []):
            continue
        if account in clip.get("blocked_accounts", []):
            continue
        allowed = clip.get("allowed_accounts", [])
        if allowed and account not in allowed:
            continue
        if account in clip.get("posted_accounts", []):
            continue
        eligible.append(clip)

    if not eligible:
        return {
            "ok": False,
            "message": "No eligible approved clips found",
            "account": account,
            "tag": tag,
        }

    chosen = eligible[0] if len(eligible) == 1 else __import__("random").choice(eligible)

    return create_publish_job(
        file_name=chosen["file_name"],
        account=account,
        platform=platform,
        scheduled_for=scheduled_for,
    )


@mcp.tool()
def update_publish_job(
    job_id: str,
    caption_text: str = "",
    hashtags: Optional[List[str]] = None,
    scheduled_for: str = "",
    notes: str = "",
) -> dict:
    """Update caption, hashtags, schedule, or notes for a queued job."""
    queue = _load_queue()
    jobs = queue.get("jobs", {})
    if job_id not in jobs:
        raise FileNotFoundError(f"Job not found: {job_id}")

    job = jobs[job_id]

    if caption_text != "":
        job["caption_text"] = caption_text.strip()
    if hashtags is not None:
        job["hashtags"] = _normalize_list(hashtags)
    if scheduled_for != "":
        job["scheduled_for"] = scheduled_for.strip()
    if notes != "":
        job["notes"] = notes.strip()

    job["updated_at"] = _now_iso()

    queue.setdefault("history", []).append(
        {
            "type": "job_updated",
            "job_id": job_id,
            "timestamp": _now_iso(),
        }
    )
    _save_queue(queue)

    return {
        "ok": True,
        "job": job,
    }


@mcp.tool()
def set_publish_job_status(
    job_id: str,
    status: str,
    notes: str = "",
    external_post_id: str = "",
) -> dict:
    """
    Update publish job status.
    Allowed statuses: queued, draft_ready, posted, failed, cancelled
    If status=posted, this also updates the video pool record.
    """
    allowed = {"queued", "draft_ready", "posted", "failed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")

    queue = _load_queue()
    jobs = queue.get("jobs", {})
    if job_id not in jobs:
        raise FileNotFoundError(f"Job not found: {job_id}")

    job = jobs[job_id]
    job["status"] = status
    if notes:
        job["notes"] = notes.strip()
    if external_post_id:
        job["external_post_id"] = external_post_id.strip()
    job["updated_at"] = _now_iso()

    queue.setdefault("history", []).append(
        {
            "type": "job_status_changed",
            "job_id": job_id,
            "status": status,
            "timestamp": _now_iso(),
        }
    )
    _save_queue(queue)

    if status == "posted":
        pool = _load_pool()
        clips = pool.get("clips", {})
        if job["file_name"] in clips:
            clip = clips[job["file_name"]]
            posted_accounts = clip.setdefault("posted_accounts", [])
            if job["account"] not in posted_accounts:
                posted_accounts.append(job["account"])
            clip["post_count"] = int(clip.get("post_count", 0)) + 1
            clip["last_posted_at"] = _now_iso()
            clip["updated_at"] = _now_iso()

            pool.setdefault("history", []).append(
                {
                    "type": "posted",
                    "file_name": job["file_name"],
                    "account": job["account"],
                    "platform": job.get("platform", "tiktok"),
                    "caption_used": job.get("caption_text", ""),
                    "timestamp": _now_iso(),
                }
            )
            _save_pool(pool)

    return {
        "ok": True,
        "job": job,
    }


if __name__ == "__main__":
    mcp.run()
