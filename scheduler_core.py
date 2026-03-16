from pathlib import Path
import os
import json
import uuid
import random
import tempfile
from datetime import datetime, timedelta, timezone, date
from typing import Optional, List
from zoneinfo import ZoneInfo

from bootstrap import resolve_root_and_load_env

ROOT = resolve_root_and_load_env()

POOL_FILE = ROOT / "pool" / "video_pool.json"
QUEUE_FILE = ROOT / "publisher" / "publish_queue.json"
SCHEDULER_DIR = ROOT / "scheduler"
SCHEDULES_FILE = SCHEDULER_DIR / "schedules.json"

TZ_NAME = os.environ.get("SCHEDULER_TIMEZONE", "America/New_York")
LOCAL_TZ = ZoneInfo(TZ_NAME)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (write temp, then os.replace) to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _ensure_paths():
    SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
    if not SCHEDULES_FILE.exists():
        SCHEDULES_FILE.write_text(
            json.dumps(
                {"plans": {}, "history": [], "updated_at": _now_iso()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict):
    data["updated_at"] = _now_iso()
    _atomic_write_json(path, data)


def _load_pool() -> dict:
    if not POOL_FILE.exists():
        raise FileNotFoundError(f"Missing pool file: {POOL_FILE}")
    return json.loads(POOL_FILE.read_text(encoding="utf-8"))


def _load_queue() -> dict:
    return _load_json(
        QUEUE_FILE,
        {"jobs": {}, "history": [], "updated_at": _now_iso()},
    )


def _save_queue(data: dict):
    _save_json(QUEUE_FILE, data)


def _load_schedules() -> dict:
    _ensure_paths()
    return json.loads(SCHEDULES_FILE.read_text(encoding="utf-8"))


def _save_schedules(data: dict):
    _save_json(SCHEDULES_FILE, data)


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


def _build_default_caption(clip: dict) -> str:
    tags = clip.get("tags", [])
    source = clip.get("source_stem", "")
    parts = []
    if tags:
        parts.append(" | ".join(tags[:3]))
    if source:
        parts.append(source.replace("-", " ").replace("_", " "))
    return "\n".join([p for p in parts if p]).strip()


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


def _parse_date_or_today(date_str: str) -> date:
    if date_str.strip():
        return date.fromisoformat(date_str.strip())
    return datetime.now(LOCAL_TZ).date()


def _local_day_bounds(target_date: date):
    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=LOCAL_TZ)
    end = start + timedelta(days=1)
    return start, end


def _job_belongs_to_local_date(job: dict, target_date: date) -> bool:
    scheduled_for = job.get("scheduled_for")
    if not scheduled_for:
        return False
    try:
        dt = datetime.fromisoformat(scheduled_for)
    except Exception:
        return False
    local_dt = dt.astimezone(LOCAL_TZ)
    return local_dt.date() == target_date


def _count_existing_jobs_for_date(queue: dict, account: str, target_date: date) -> int:
    count = 0
    for job in queue.get("jobs", {}).values():
        if job.get("account") != account:
            continue
        if job.get("status") in {"cancelled", "failed"}:
            continue
        if _job_belongs_to_local_date(job, target_date):
            count += 1
    return count


def _queued_file_names_for_date(queue: dict, account: str, target_date: date) -> set:
    out = set()
    for job in queue.get("jobs", {}).values():
        if job.get("account") != account:
            continue
        if job.get("status") in {"cancelled", "failed"}:
            continue
        if _job_belongs_to_local_date(job, target_date):
            out.add(job.get("file_name", ""))
    return out


def _eligible_clips(pool: dict, queue: dict, account: str, target_date: date, tag: str = "") -> List[dict]:
    already_queued_today = _queued_file_names_for_date(queue, account, target_date)

    eligible = []
    for clip in pool.get("clips", {}).values():
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

        if clip.get("file_name") in already_queued_today:
            continue

        eligible.append(clip)

    return eligible


def _pick_diverse_clips(clips: List[dict], count: int) -> List[dict]:
    if count <= 0:
        return []

    pool = clips[:]
    random.shuffle(pool)

    chosen = []
    used_sources = set()

    for clip in pool:
        source = clip.get("source_stem", "")
        if source and source in used_sources:
            continue
        chosen.append(clip)
        if source:
            used_sources.add(source)
        if len(chosen) >= count:
            return chosen

    for clip in pool:
        if clip in chosen:
            continue
        chosen.append(clip)
        if len(chosen) >= count:
            return chosen

    return chosen


def _generate_slots(target_date: date, posts_per_day: int, start_hour: int, end_hour: int) -> List[str]:
    if posts_per_day < 1:
        raise ValueError("posts_per_day must be at least 1")
    if not (0 <= start_hour <= 23 and 1 <= end_hour <= 24):
        raise ValueError("start_hour must be 0-23 and end_hour must be 1-24")
    if end_hour <= start_hour:
        raise ValueError("end_hour must be greater than start_hour")

    start_dt = datetime(target_date.year, target_date.month, target_date.day, start_hour, 0, 0, tzinfo=LOCAL_TZ)
    end_dt = datetime(target_date.year, target_date.month, target_date.day, end_hour, 0, 0, tzinfo=LOCAL_TZ)

    window_seconds = (end_dt - start_dt).total_seconds()
    step = window_seconds / posts_per_day

    slots = []
    for i in range(posts_per_day):
        dt = start_dt + timedelta(seconds=step * (i + 0.5))
        slots.append(dt.isoformat())
    return slots


def _create_job_from_clip(clip: dict, account: str, platform: str, scheduled_for: str) -> dict:
    return {
        "job_id": str(uuid.uuid4()),
        "status": "queued",
        "platform": platform,
        "account": account,
        "file_name": clip.get("file_name", ""),
        "path": clip.get("path", ""),
        "source_stem": clip.get("source_stem", ""),
        "preset": clip.get("preset", ""),
        "caption_text": clip.get("caption_text", "").strip() or _build_default_caption(clip),
        "hashtags": _normalize_list(clip.get("hashtags")) or _build_default_hashtags(clip),
        "scheduled_for": scheduled_for,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "notes": "",
        "external_post_id": "",
    }


def plan_daily_schedule_core(
    account: str,
    date_str: str = "",
    posts_per_day: int = 10,
    start_hour: int = 8,
    end_hour: int = 22,
    tag: str = "",
    platform: str = "tiktok",
    dry_run: bool = False,
) -> dict:
    account = account.strip()
    if not account:
        raise ValueError("account is required")

    target_date = _parse_date_or_today(date_str)
    pool = _load_pool()
    queue = _load_queue()
    schedules = _load_schedules()

    existing_count = _count_existing_jobs_for_date(queue, account, target_date)
    remaining_needed = max(0, posts_per_day - existing_count)

    eligible = _eligible_clips(pool, queue, account, target_date, tag=tag)
    chosen = _pick_diverse_clips(eligible, remaining_needed)

    all_slots = _generate_slots(target_date, posts_per_day, start_hour, end_hour)

    # keep slots stable and fill only the next missing positions
    existing_slots = []
    for job in queue.get("jobs", {}).values():
        if job.get("account") == account and _job_belongs_to_local_date(job, target_date):
            existing_slots.append(job.get("scheduled_for"))
    existing_slots = sorted([s for s in existing_slots if s])

    free_slots = [slot for slot in all_slots if slot not in existing_slots]
    free_slots = free_slots[:len(chosen)]

    created_jobs = []
    for clip, slot in zip(chosen, free_slots):
        job = _create_job_from_clip(clip, account, platform, slot)
        created_jobs.append(job)
        if not dry_run:
            queue.setdefault("jobs", {})[job["job_id"]] = job
            queue.setdefault("history", []).append(
                {
                    "type": "scheduled_job_created",
                    "job_id": job["job_id"],
                    "file_name": job["file_name"],
                    "account": account,
                    "scheduled_for": slot,
                    "timestamp": _now_iso(),
                }
            )

    plan_key = f"{account}__{target_date.isoformat()}"
    schedules.setdefault("plans", {})[plan_key] = {
        "account": account,
        "date": target_date.isoformat(),
        "posts_per_day_target": posts_per_day,
        "existing_count_before": existing_count,
        "created_count": len(created_jobs),
        "start_hour": start_hour,
        "end_hour": end_hour,
        "tag": tag,
        "platform": platform,
        "dry_run": dry_run,
        "jobs": created_jobs,
        "updated_at": _now_iso(),
    }
    schedules.setdefault("history", []).append(
        {
            "type": "daily_schedule_run",
            "account": account,
            "date": target_date.isoformat(),
            "created_count": len(created_jobs),
            "dry_run": dry_run,
            "timestamp": _now_iso(),
        }
    )

    if not dry_run:
        _save_queue(queue)
    _save_schedules(schedules)

    return {
        "ok": True,
        "timezone": TZ_NAME,
        "account": account,
        "date": target_date.isoformat(),
        "posts_per_day_target": posts_per_day,
        "existing_count_before": existing_count,
        "created_count": len(created_jobs),
        "remaining_unfilled": max(0, posts_per_day - existing_count - len(created_jobs)),
        "eligible_count": len(eligible),
        "created_jobs": created_jobs,
        "schedule_file": str(SCHEDULES_FILE),
        "queue_file": str(QUEUE_FILE),
    }


def schedule_next_days_core(
    account: str,
    days: int = 1,
    posts_per_day: int = 10,
    start_hour: int = 8,
    end_hour: int = 22,
    tag: str = "",
    platform: str = "tiktok",
    dry_run: bool = False,
) -> dict:
    if days < 1:
        raise ValueError("days must be at least 1")

    today = datetime.now(LOCAL_TZ).date()
    results = []
    total_created = 0

    for offset in range(days):
        target = today + timedelta(days=offset)
        res = plan_daily_schedule_core(
            account=account,
            date_str=target.isoformat(),
            posts_per_day=posts_per_day,
            start_hour=start_hour,
            end_hour=end_hour,
            tag=tag,
            platform=platform,
            dry_run=dry_run,
        )
        total_created += int(res.get("created_count", 0))
        results.append(res)

    return {
        "ok": True,
        "timezone": TZ_NAME,
        "account": account,
        "days": days,
        "total_created": total_created,
        "results": results,
    }


def list_scheduled_jobs_core(date_str: str = "", account: str = "", limit: int = 100) -> dict:
    queue = _load_queue()
    jobs = list(queue.get("jobs", {}).values())

    if date_str.strip():
        target_date = _parse_date_or_today(date_str)
        jobs = [j for j in jobs if _job_belongs_to_local_date(j, target_date)]

    if account.strip():
        jobs = [j for j in jobs if j.get("account") == account.strip()]

    jobs = sorted(jobs, key=lambda j: j.get("scheduled_for") or "", reverse=False)

    return {
        "ok": True,
        "timezone": TZ_NAME,
        "count": len(jobs[:limit]),
        "jobs": jobs[:limit],
        "queue_file": str(QUEUE_FILE),
    }


def scheduler_summary_core() -> dict:
    queue = _load_queue()
    jobs = list(queue.get("jobs", {}).values())

    upcoming = 0
    queued = 0
    draft_ready = 0
    posted = 0
    failed = 0
    cancelled = 0

    now_local = datetime.now(LOCAL_TZ)

    for job in jobs:
        status = job.get("status", "queued")
        if status == "queued":
            queued += 1
        elif status == "draft_ready":
            draft_ready += 1
        elif status == "posted":
            posted += 1
        elif status == "failed":
            failed += 1
        elif status == "cancelled":
            cancelled += 1

        scheduled_for = job.get("scheduled_for")
        if scheduled_for:
            try:
                dt = datetime.fromisoformat(scheduled_for).astimezone(LOCAL_TZ)
                if dt >= now_local and status in {"queued", "draft_ready"}:
                    upcoming += 1
            except Exception:
                pass

    return {
        "ok": True,
        "timezone": TZ_NAME,
        "queue_file": str(QUEUE_FILE),
        "upcoming_active_jobs": upcoming,
        "queued": queued,
        "draft_ready": draft_ready,
        "posted": posted,
        "failed": failed,
        "cancelled": cancelled,
        "total_jobs": len(jobs),
    }
