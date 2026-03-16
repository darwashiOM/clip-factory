from pathlib import Path
import os
import json
import random
from datetime import datetime, timezone
from typing import Optional, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from helpers import atomic_write_json

from bootstrap import resolve_root_and_load_env

ROOT = resolve_root_and_load_env()

FINAL = ROOT / "final"
POOL_DIR = ROOT / "pool"
POOL_FILE = POOL_DIR / "video_pool.json"



mcp = FastMCP("clip-factory-video-pool", json_response=True)

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_pool():
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    if not POOL_FILE.exists():
        POOL_FILE.write_text(
            json.dumps(
                {
                    "clips": {},
                    "history": [],
                    "updated_at": _now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _load_pool() -> dict:
    _ensure_pool()
    return json.loads(POOL_FILE.read_text(encoding="utf-8"))


def _save_pool(data: dict):
    data["updated_at"] = _now_iso()
    atomic_write_json(POOL_FILE, data)


def _list_final_files():
    if not FINAL.exists():
        return []
    files = sorted(
        [p for p in FINAL.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def _default_clip_record(p: Path) -> dict:
    stat = p.stat()
    return {
        "file_name": p.name,
        "path": str(p),
        "stem": p.stem,
        "status": "pending",  # pending, approved, rejected
        "notes": "",
        "tags": [],
        "allowed_accounts": [],
        "blocked_accounts": [],
        "caption_text": "",
        "hashtags": [],
        "posted_accounts": [],
        "post_count": 0,
        "source_stem": _infer_source_stem_from_filename(p.name),
        "preset": _infer_preset_from_filename(p.name),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "last_posted_at": None,
        "size_bytes": stat.st_size,
    }


def _infer_source_stem_from_filename(file_name: str) -> str:
    # expected pattern: source__clip01__preset.mp4
    base = Path(file_name).stem
    parts = base.split("__clip")
    return parts[0] if parts else base


def _infer_preset_from_filename(file_name: str) -> str:
    base = Path(file_name).stem
    parts = base.split("__")
    if len(parts) >= 3:
        return parts[-1]
    return ""


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


def _get_clip(data: dict, file_name: str) -> dict:
    clips = data.get("clips", {})
    if file_name not in clips:
        raise FileNotFoundError(f"Clip not found in pool: {file_name}")
    return clips[file_name]


@mcp.tool()
def sync_final_to_pool() -> dict:
    """
    Scan final/ and add any missing rendered clips into the pool.
    Existing metadata is preserved.
    """
    data = _load_pool()
    clips = data.setdefault("clips", {})

    added = []
    updated = []

    for p in _list_final_files():
        if p.name not in clips:
            clips[p.name] = _default_clip_record(p)
            added.append(p.name)
        else:
            rec = clips[p.name]
            rec["path"] = str(p)
            rec["size_bytes"] = p.stat().st_size
            rec["updated_at"] = _now_iso()
            updated.append(p.name)

    _save_pool(data)

    return {
        "ok": True,
        "pool_file": str(POOL_FILE),
        "final_folder": str(FINAL),
        "added_count": len(added),
        "updated_count": len(updated),
        "added": added,
    }


@mcp.tool()
def list_pool_clips(
    status: str = "",
    account: str = "",
    only_unposted_for_account: bool = False,
    limit: int = 50,
) -> dict:
    """
    List clips in the pool with optional filtering.
    status can be: pending, approved, rejected, or blank for all.
    """
    data = _load_pool()
    clips = list(data.get("clips", {}).values())

    if status:
        clips = [c for c in clips if c.get("status") == status]

    if account:
        filtered = []
        for c in clips:
            allowed = c.get("allowed_accounts", [])
            blocked = c.get("blocked_accounts", [])
            posted = c.get("posted_accounts", [])
            if blocked and account in blocked:
                continue
            if allowed and account not in allowed:
                continue
            if only_unposted_for_account and account in posted:
                continue
            filtered.append(c)
        clips = filtered

    clips = sorted(
        clips,
        key=lambda c: c.get("updated_at", ""),
        reverse=True,
    )

    return {
        "ok": True,
        "pool_file": str(POOL_FILE),
        "count": len(clips[:limit]),
        "clips": clips[:limit],
    }


@mcp.tool()
def get_pool_clip(file_name: str) -> dict:
    """Get full metadata for one clip in the pool."""
    data = _load_pool()
    clip = _get_clip(data, file_name)
    return {
        "ok": True,
        "clip": clip,
    }


@mcp.tool()
def review_clip(
    file_name: str,
    status: str,
    notes: str = "",
) -> dict:
    """
    Set review status for a clip.
    status must be one of: pending, approved, rejected
    """
    allowed = {"pending", "approved", "rejected"}
    if status not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")

    data = _load_pool()
    clip = _get_clip(data, file_name)

    clip["status"] = status
    clip["notes"] = notes.strip()
    clip["updated_at"] = _now_iso()

    data.setdefault("history", []).append(
        {
            "type": "review",
            "file_name": file_name,
            "status": status,
            "notes": notes.strip(),
            "timestamp": _now_iso(),
        }
    )

    _save_pool(data)
    return {
        "ok": True,
        "file_name": file_name,
        "status": status,
        "notes": notes.strip(),
    }


@mcp.tool()
def update_clip_metadata(
    file_name: str,
    tags: Optional[List[str]] = None,
    allowed_accounts: Optional[List[str]] = None,
    blocked_accounts: Optional[List[str]] = None,
    caption_text: str = "",
    hashtags: Optional[List[str]] = None,
) -> dict:
    """
    Update tags, account rules, and posting text.
    Omitted lists remain unchanged only if passed as null from the client.
    """
    data = _load_pool()
    clip = _get_clip(data, file_name)

    if tags is not None:
        clip["tags"] = _normalize_list(tags)
    if allowed_accounts is not None:
        clip["allowed_accounts"] = _normalize_list(allowed_accounts)
    if blocked_accounts is not None:
        clip["blocked_accounts"] = _normalize_list(blocked_accounts)
    if caption_text != "":
        clip["caption_text"] = caption_text.strip()
    if hashtags is not None:
        clip["hashtags"] = _normalize_list(hashtags)

    clip["updated_at"] = _now_iso()

    data.setdefault("history", []).append(
        {
            "type": "metadata_update",
            "file_name": file_name,
            "timestamp": _now_iso(),
        }
    )

    _save_pool(data)
    return {
        "ok": True,
        "clip": clip,
    }


@mcp.tool()
def mark_clip_posted(
    file_name: str,
    account: str,
    platform: str = "tiktok",
    caption_used: str = "",
) -> dict:
    """
    Mark a clip as posted for one account.
    """
    account = account.strip()
    if not account:
        raise ValueError("account is required")

    data = _load_pool()
    clip = _get_clip(data, file_name)

    posted_accounts = clip.setdefault("posted_accounts", [])
    if account not in posted_accounts:
        posted_accounts.append(account)

    clip["post_count"] = int(clip.get("post_count", 0)) + 1
    clip["last_posted_at"] = _now_iso()
    clip["updated_at"] = _now_iso()

    data.setdefault("history", []).append(
        {
            "type": "posted",
            "file_name": file_name,
            "account": account,
            "platform": platform,
            "caption_used": caption_used.strip(),
            "timestamp": _now_iso(),
        }
    )

    _save_pool(data)
    return {
        "ok": True,
        "file_name": file_name,
        "account": account,
        "platform": platform,
        "post_count": clip["post_count"],
    }


@mcp.tool()
def pick_random_eligible_clip(
    account: str,
    require_approved: bool = True,
    exclude_already_posted_for_account: bool = True,
    tag: str = "",
) -> dict:
    """
    Pick one random eligible clip for an account.
    Eligibility rules:
    - approved if require_approved=true
    - not blocked for this account
    - if allowed_accounts is non-empty, account must be included
    - if exclude_already_posted_for_account=true, skip clips already posted there
    - optional tag filter
    """
    account = account.strip()
    if not account:
        raise ValueError("account is required")

    data = _load_pool()
    clips = list(data.get("clips", {}).values())

    eligible = []
    for c in clips:
        if require_approved and c.get("status") != "approved":
            continue

        if tag:
            if tag not in c.get("tags", []):
                continue

        blocked = c.get("blocked_accounts", [])
        if account in blocked:
            continue

        allowed = c.get("allowed_accounts", [])
        if allowed and account not in allowed:
            continue

        posted_accounts = c.get("posted_accounts", [])
        if exclude_already_posted_for_account and account in posted_accounts:
            continue

        eligible.append(c)

    if not eligible:
        return {
            "ok": False,
            "message": "No eligible clips found",
            "account": account,
            "eligible_count": 0,
        }

    chosen = random.choice(eligible)
    return {
        "ok": True,
        "account": account,
        "eligible_count": len(eligible),
        "clip": chosen,
    }


@mcp.tool()
def pool_summary() -> dict:
    """Return simple counts for the pool."""
    data = _load_pool()
    clips = list(data.get("clips", {}).values())

    pending = sum(1 for c in clips if c.get("status") == "pending")
    approved = sum(1 for c in clips if c.get("status") == "approved")
    rejected = sum(1 for c in clips if c.get("status") == "rejected")
    posted_any = sum(1 for c in clips if c.get("post_count", 0) > 0)

    return {
        "ok": True,
        "pool_file": str(POOL_FILE),
        "total_clips": len(clips),
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "posted_any": posted_any,
    }


if __name__ == "__main__":
    mcp.run()
