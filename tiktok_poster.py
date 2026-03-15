from pathlib import Path
import os
import json
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, List
import mimetypes

import httpx

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()

QUEUE_FILE = ROOT / "publisher" / "publish_queue.json"
ACCOUNTS_FILE = ROOT / "accounts" / "tiktok_accounts.json"

TZ_NAME = os.environ.get("SCHEDULER_TIMEZONE", "America/New_York")
LOCAL_TZ = ZoneInfo(TZ_NAME)

TIKTOK_INBOX_VIDEO_INIT = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically to prevent corruption on crash."""
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


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, default)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict):
    data["updated_at"] = _now_iso()
    _atomic_write_json(path, data)


def _load_queue() -> dict:
    return _load_json(
        QUEUE_FILE,
        {"jobs": {}, "history": [], "updated_at": _now_iso()},
    )


def _save_queue(data: dict):
    _save_json(QUEUE_FILE, data)


def _load_accounts() -> dict:
    return _load_json(
        ACCOUNTS_FILE,
        {"accounts": {}, "updated_at": _now_iso()},
    )


def _scheduled_is_due(job: dict) -> bool:
    scheduled_for = job.get("scheduled_for")
    if not scheduled_for:
        return True
    try:
        dt = datetime.fromisoformat(scheduled_for)
    except Exception:
        return False
    return dt.astimezone(LOCAL_TZ) <= datetime.now(LOCAL_TZ)


def _get_account_token(account_name: str) -> str:
    data = _load_accounts()
    account = data.get("accounts", {}).get(account_name)
    if not account:
        raise FileNotFoundError(f"TikTok account not found: {account_name}")
    token = (account.get("access_token") or "").strip()
    if not token:
        raise RuntimeError(f"Missing access_token for account: {account_name}")
    return token


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "video/mp4"


def _init_inbox_upload(access_token: str, file_size: int) -> dict:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": file_size,
            "total_chunk_count": 1,
        }
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(TIKTOK_INBOX_VIDEO_INIT, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    error = data.get("error", {})
    if error.get("code") not in (None, "", "ok"):
        raise RuntimeError(f"TikTok init upload failed: {error}")

    payload = data.get("data", {})
    upload_url = payload.get("upload_url")
    publish_id = payload.get("publish_id")
    if not upload_url or not publish_id:
        raise RuntimeError(f"Missing upload_url or publish_id in TikTok response: {data}")

    return {
        "publish_id": publish_id,
        "upload_url": upload_url,
        "raw_response": data,
    }


def _upload_video_bytes(upload_url: str, file_path: Path):
    file_size = file_path.stat().st_size
    headers = {
        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        "Content-Type": _guess_content_type(file_path),
        "Content-Length": str(file_size),
    }

    # Stream the file to avoid loading the entire video into memory.
    # httpx accepts a file-like object as content and reads it in chunks.
    with open(file_path, "rb") as f:
        with httpx.Client(timeout=None) as client:
            resp = client.put(upload_url, headers=headers, content=f)
            resp.raise_for_status()

    return {
        "ok": True,
        "http_status": resp.status_code,
    }


def list_tiktok_accounts_core() -> dict:
    data = _load_accounts()
    accounts = []
    for name, record in data.get("accounts", {}).items():
        accounts.append(
            {
                "account": name,
                "platform": record.get("platform", "tiktok"),
                "has_access_token": bool((record.get("access_token") or "").strip()),
                "has_refresh_token": bool((record.get("refresh_token") or "").strip()),
                "notes": record.get("notes", ""),
            }
        )
    return {
        "ok": True,
        "accounts_file": str(ACCOUNTS_FILE),
        "accounts": accounts,
    }


def list_due_tiktok_jobs_core(account: str = "", limit: int = 20) -> dict:
    queue = _load_queue()
    jobs = list(queue.get("jobs", {}).values())

    jobs = [
        j for j in jobs
        if j.get("platform", "tiktok") == "tiktok"
        and j.get("status") == "queued"
        and _scheduled_is_due(j)
    ]

    if account.strip():
        jobs = [j for j in jobs if j.get("account") == account.strip()]

    jobs = sorted(jobs, key=lambda j: j.get("scheduled_for") or "", reverse=False)

    return {
        "ok": True,
        "count": len(jobs[:limit]),
        "jobs": jobs[:limit],
    }


def upload_due_tiktok_jobs_core(
    account: str = "",
    limit: int = 3,
    dry_run: bool = False,
) -> dict:
    queue = _load_queue()
    jobs_map = queue.get("jobs", {})

    due_jobs = [
        j for j in jobs_map.values()
        if j.get("platform", "tiktok") == "tiktok"
        and j.get("status") == "queued"
        and _scheduled_is_due(j)
    ]

    if account.strip():
        due_jobs = [j for j in due_jobs if j.get("account") == account.strip()]

    due_jobs = sorted(due_jobs, key=lambda j: j.get("scheduled_for") or "", reverse=False)[:limit]

    results = []

    for job in due_jobs:
        job_id = job["job_id"]
        account_name = job["account"]
        path = Path(job["path"])

        if not path.exists():
            job["status"] = "failed"
            job["notes"] = f"Missing file: {path}"
            job["updated_at"] = _now_iso()
            results.append({"ok": False, "job_id": job_id, "error": f"Missing file: {path}"})
            continue

        try:
            access_token = _get_account_token(account_name)

            if dry_run:
                results.append(
                    {
                        "ok": True,
                        "job_id": job_id,
                        "account": account_name,
                        "file_name": job["file_name"],
                        "dry_run": True,
                    }
                )
                continue

            init_data = _init_inbox_upload(access_token, path.stat().st_size)
            upload_result = _upload_video_bytes(init_data["upload_url"], path)

            job["status"] = "draft_ready"
            job["external_post_id"] = init_data["publish_id"]
            job["notes"] = "Uploaded to TikTok inbox draft flow"
            job["updated_at"] = _now_iso()

            queue.setdefault("history", []).append(
                {
                    "type": "tiktok_upload_inbox_success",
                    "job_id": job_id,
                    "account": account_name,
                    "publish_id": init_data["publish_id"],
                    "timestamp": _now_iso(),
                }
            )

            results.append(
                {
                    "ok": True,
                    "job_id": job_id,
                    "account": account_name,
                    "file_name": job["file_name"],
                    "publish_id": init_data["publish_id"],
                    "http_status": upload_result["http_status"],
                    "status": "draft_ready",
                }
            )

        except Exception as e:
            job["status"] = "failed"
            job["notes"] = str(e)
            job["updated_at"] = _now_iso()

            queue.setdefault("history", []).append(
                {
                    "type": "tiktok_upload_inbox_failed",
                    "job_id": job_id,
                    "account": account_name,
                    "error": str(e),
                    "timestamp": _now_iso(),
                }
            )

            results.append(
                {
                    "ok": False,
                    "job_id": job_id,
                    "account": account_name,
                    "file_name": job.get("file_name", ""),
                    "error": str(e),
                }
            )

    if not dry_run:
        _save_queue(queue)

    return {
        "ok": True,
        "processed_count": len(results),
        "results": results,
        "queue_file": str(QUEUE_FILE),
    }
