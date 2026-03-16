#!/usr/bin/env python3
"""
stock_fetcher_server.py — MCP server for fetching and caching real stock footage.

Replaces AI video generation (Sora/LTX) with actual stock footage from:
  Primary:  Pexels Videos API
  Fallback: Pixabay Videos API

Downloads are cached in broll/stock/ keyed by (provider, video_id).
Clip-specific symlinks are placed in broll/ using the current stock-slot
naming convention:

    broll/{stem}__clip{N:02d}__ai{slot:02d}__stock.mp4

The ai{slot} token is kept only as a slot identifier for compatibility with
existing clip plans. These files are real stock footage only.

Environment variables
─────────────────────
  PEXELS_API_KEY              Pexels API key (required for Pexels)
  PIXABAY_API_KEY             Pixabay API key (required for Pixabay)
  STOCK_PROVIDER_PRIORITY     Comma-separated provider order  default: pexels,pixabay
  STOCK_CACHE_DIR             Override cache directory        default: broll/stock
  STOCK_PREFERRED_ORIENTATION portrait or landscape           default: portrait
  STOCK_MIN_DURATION_SEC      Minimum clip duration (float)   default: 4
  STOCK_MAX_DURATION_SEC      Maximum clip duration (float)   default: 60
  STOCK_ALLOW_LANDSCAPE_FALLBACK  true/false                  default: true
  STOCK_REQUIRE_SCENIC_FILTER true/false                      default: true
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from mcp.server.fastmcp import FastMCP

from bootstrap import resolve_root_and_load_env
from helpers import atomic_write_json

ROOT = resolve_root_and_load_env()
BROLL = ROOT / "broll"
CLIPS = ROOT / "clips"

mcp = FastMCP("clip-factory-stock-fetcher", json_response=True)


# ─── Config helpers ───────────────────────────────────────────────────────────

def _stock_cache_dir() -> Path:
    custom = os.environ.get("STOCK_CACHE_DIR", "").strip()
    d = Path(custom) if custom else BROLL / "stock"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _provider_priority() -> List[str]:
    raw = os.environ.get("STOCK_PROVIDER_PRIORITY", "pexels,pixabay").strip()
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _preferred_orientation() -> str:
    return os.environ.get("STOCK_PREFERRED_ORIENTATION", "portrait").strip().lower()


def _min_duration() -> float:
    try:
        return float(os.environ.get("STOCK_MIN_DURATION_SEC", "4"))
    except ValueError:
        return 4.0


def _max_duration() -> float:
    try:
        return float(os.environ.get("STOCK_MAX_DURATION_SEC", "60"))
    except ValueError:
        return 60.0


def _allow_landscape_fallback() -> bool:
    val = os.environ.get("STOCK_ALLOW_LANDSCAPE_FALLBACK", "true").strip().lower()
    return val in ("1", "true", "yes", "on")


def _require_scenic_filter() -> bool:
    val = os.environ.get("STOCK_REQUIRE_SCENIC_FILTER", "true").strip().lower()
    return val in ("1", "true", "yes", "on")


# ─── Scenic filter ────────────────────────────────────────────────────────────

# Terms that disqualify a result (people, animals, logos, presenters)
_REJECT_TERMS: frozenset[str] = frozenset({
    "people", "person", "woman", "man", "men", "women", "girl", "boy",
    "child", "children", "baby", "face", "portrait", "crowd", "group",
    "animal", "dog", "cat", "bird", "fish", "horse", "elephant",
    "wildlife", "insect", "butterfly", "pet", "zoo",
    "text", "logo", "watermark", "sign", "banner", "caption",
    "presenter", "speaker", "interview", "talking", "host",
    "hands", "hand", "fingers", "holding",
    "street", "pedestrian", "shopping", "market", "bazaar", "store",
    "protest", "event", "concert", "party", "celebration",
})

# Terms that indicate scenic / atmospheric content (used for query enrichment only,
# not as a hard filter — absence of these does not reject the result)
SCENIC_QUERY_HINTS: List[str] = [
    "desert", "ocean", "sea", "waves", "rain", "river", "mist", "clouds",
    "night sky", "mountains", "forest", "mosque", "architecture", "lantern",
    "stone corridor", "empty courtyard", "dawn", "dusk", "landscape",
    "nature", "waterfall", "lake", "fog", "canyon", "valley", "horizon",
    "stars", "moon", "sand", "dunes", "aerial",
]


def _tokenize(text: str) -> set[str]:
    return set(re.split(r"[\s,;|/()\[\]]+", text.lower()))


def _is_scenic(title: str, tags: str = "") -> bool:
    """Return True if the video title/tags pass the scenic filter."""
    if not _require_scenic_filter():
        return True
    tokens = _tokenize(title + " " + tags)
    return not bool(tokens & _REJECT_TERMS)


def _slug(s: str) -> str:
    out = []
    for ch in str(s or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "_", "-", "."):
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "stock"


def _sanitize_query(raw: str) -> str:
    """Strip Arabic text and truncate overly long prompt strings to a short search query."""
    # Drop any text that is mostly non-Latin (Arabic, etc.)
    if len(raw) > 80 or any(ord(c) > 0x06FF for c in raw):
        return "scenic nature landscape"
    return raw[:100].strip()


# ─── Pexels API ───────────────────────────────────────────────────────────────

def _pexels_key() -> str:
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PEXELS_API_KEY is not set in .env")
    return key


def _pexels_title_from_url(url: str) -> str:
    """Extract a human-readable title from a Pexels video URL slug."""
    # URL: https://www.pexels.com/video/ocean-waves-at-sunset-1234567/
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        slug = parts[-2]
        slug = re.sub(r"-\d+$", "", slug)
        return slug.replace("-", " ")
    return ""


def _pick_pexels_file(video_files: list, preferred_orientation: str) -> Optional[dict]:
    """Select the best-quality Pexels file for the desired orientation."""
    portrait = [f for f in video_files if f.get("height", 0) > f.get("width", 0) and f.get("link")]
    landscape = [f for f in video_files if f.get("width", 0) >= f.get("height", 0) and f.get("link")]
    all_files = [f for f in video_files if f.get("link")]

    def _area(f: dict) -> int:
        return f.get("height", 0) * f.get("width", 0)

    if preferred_orientation == "portrait" and portrait:
        return max(portrait, key=_area)
    if _allow_landscape_fallback() and landscape:
        return max(landscape, key=_area)
    if portrait:
        return max(portrait, key=_area)
    return all_files[0] if all_files else None


def _pexels_search(query: str, per_page: int = 5, orientation: str = "portrait") -> List[dict]:
    """Search Pexels Videos and return normalised result list."""
    url = "https://api.pexels.com/videos/search"
    resp = requests.get(
        url,
        headers={"Authorization": _pexels_key()},
        params={
            "query": query,
            "per_page": min(per_page, 20),
            "orientation": orientation,
            "size": "medium",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    min_dur = _min_duration()
    max_dur = _max_duration()
    results = []

    for video in data.get("videos", []):
        duration = video.get("duration", 0)
        if not (min_dur <= duration <= max_dur):
            continue

        title = _pexels_title_from_url(video.get("url", ""))
        tags = " ".join(str(t) for t in (video.get("tags") or []))

        if not _is_scenic(title, tags):
            continue

        chosen = _pick_pexels_file(video.get("video_files", []), orientation)
        if not chosen:
            continue

        results.append({
            "provider": "pexels",
            "id": str(video.get("id", "")),
            "title": title or query,
            "width": video.get("width", 0),
            "height": video.get("height", 0),
            "duration": duration,
            "download_url": chosen.get("link", ""),
            "file_width": chosen.get("width", 0),
            "file_height": chosen.get("height", 0),
            "source_page_url": video.get("url", ""),
            "query": query,
        })

    return results


# ─── Pixabay API ──────────────────────────────────────────────────────────────

def _pixabay_key() -> str:
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PIXABAY_API_KEY is not set in .env")
    return key


def _pick_pixabay_file(videos_dict: dict, preferred_orientation: str) -> Optional[dict]:
    """Select the best Pixabay video quality for the desired orientation."""
    tiers = ["large", "medium", "small", "tiny"]

    portrait: List[dict] = []
    landscape: List[dict] = []

    for tier in tiers:
        f = videos_dict.get(tier, {})
        if not f or not f.get("url"):
            continue
        w, h = f.get("width", 0), f.get("height", 0)
        (portrait if h > w else landscape).append(f)

    if preferred_orientation == "portrait" and portrait:
        return portrait[0]
    if _allow_landscape_fallback() and landscape:
        return landscape[0]
    return (portrait or landscape or [None])[0]


def _pixabay_search(query: str, per_page: int = 5, orientation: str = "portrait") -> List[dict]:
    """Search Pixabay Videos and return normalised result list."""
    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params={
            "key": _pixabay_key(),
            "q": query,
            "per_page": min(per_page, 20),
            "video_type": "film",
            "safesearch": "true",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    min_dur = _min_duration()
    max_dur = _max_duration()
    results = []

    for video in data.get("hits", []):
        duration = video.get("duration", 0)
        if not (min_dur <= duration <= max_dur):
            continue

        tags = video.get("tags", "")
        if not _is_scenic(tags, tags):
            continue

        chosen = _pick_pixabay_file(video.get("videos", {}), orientation)
        if not chosen:
            continue

        w, h = chosen.get("width", 0), chosen.get("height", 0)
        results.append({
            "provider": "pixabay",
            "id": str(video.get("id", "")),
            "title": tags or query,
            "width": w,
            "height": h,
            "duration": duration,
            "download_url": chosen.get("url", ""),
            "file_width": w,
            "file_height": h,
            "source_page_url": video.get("pageURL", ""),
            "query": query,
        })

    return results


# ─── Unified search ───────────────────────────────────────────────────────────

def _search_stock(
    query: str,
    per_page: int = 5,
    orientation: str = "portrait",
) -> Tuple[List[dict], List[str]]:
    """Try providers in priority order. Stop after first provider that returns results."""
    providers = _provider_priority()
    all_results: List[dict] = []
    errors: List[str] = []

    for provider in providers:
        try:
            if provider == "pexels":
                results = _pexels_search(query, per_page, orientation)
            elif provider == "pixabay":
                results = _pixabay_search(query, per_page, orientation)
            else:
                errors.append(f"Unknown provider: {provider}")
                continue
            all_results.extend(results)
            if all_results:
                break  # Primary delivered results — no need for fallback
        except RuntimeError as e:
            errors.append(f"{provider}: {e}")
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")

    return all_results, errors


# ─── Download & cache ─────────────────────────────────────────────────────────

def _cache_key(provider: str, video_id: str) -> str:
    return f"{provider}__{video_id}"

def _stock_manifest_path() -> Path:
    return _stock_cache_dir() / "manifest.json"


def _load_stock_manifest() -> dict:
    p = _stock_manifest_path()
    if not p.exists():
        return {"version": 1, "videos": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "videos": []}


def _save_stock_manifest(data: dict) -> None:
    atomic_write_json(_stock_manifest_path(), data, ensure_ascii=False)


def _normalize_cache_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _register_cached_video(result: dict, cached_path: Path) -> None:
    manifest = _load_stock_manifest()
    videos = manifest.setdefault("videos", [])

    provider = str(result.get("provider") or "")
    video_id = str(result.get("id") or "")
    local_path = str(cached_path)
    normalized_query = _normalize_cache_query(str(result.get("query") or ""))

    for item in videos:
        if (
            item.get("provider") == provider
            and item.get("video_id") == video_id
        ):
            item.update({
                "title": str(result.get("title") or ""),
                "query": str(result.get("query") or ""),
                "normalized_query": normalized_query,
                "orientation": "portrait" if result.get("file_height", 0) >= result.get("file_width", 0) else "landscape",
                "duration": result.get("duration", 0),
                "width": result.get("file_width", result.get("width", 0)),
                "height": result.get("file_height", result.get("height", 0)),
                "download_url": str(result.get("download_url") or ""),
                "source_page_url": str(result.get("source_page_url") or ""),
                "local_path": local_path,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            })
            _save_stock_manifest(manifest)
            return

    videos.append({
        "provider": provider,
        "video_id": video_id,
        "title": str(result.get("title") or ""),
        "query": str(result.get("query") or ""),
        "normalized_query": normalized_query,
        "orientation": "portrait" if result.get("file_height", 0) >= result.get("file_width", 0) else "landscape",
        "duration": result.get("duration", 0),
        "width": result.get("file_width", result.get("width", 0)),
        "height": result.get("file_height", result.get("height", 0)),
        "download_url": str(result.get("download_url") or ""),
        "source_page_url": str(result.get("source_page_url") or ""),
        "local_path": local_path,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_stock_manifest(manifest)


def _find_reusable_cached_video(query: str, orientation: str = "portrait") -> Optional[Path]:
    manifest = _load_stock_manifest()
    normalized_query = _normalize_cache_query(query)

    # exact query match first
    for item in manifest.get("videos", []):
        if item.get("normalized_query") != normalized_query:
            continue
        local_path = Path(str(item.get("local_path") or ""))
        if not local_path.exists():
            continue
        if orientation and item.get("orientation") not in ("", orientation):
            continue
        return local_path

    return None



def _download_to_cache(result: dict) -> Path:
    """Download a video file to broll/stock/. Returns local cached path."""
    cache_dir = _stock_cache_dir()
    fname = f"{_cache_key(result['provider'], result['id'])}.mp4"
    out_path = cache_dir / fname

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path  # Already cached

    url = result.get("download_url", "")
    if not url:
        raise ValueError(f"No download URL for {result['provider']} video {result['id']}")

    headers = {}
    if result["provider"] == "pexels":
        headers["Authorization"] = _pexels_key()

    tmp = out_path.with_suffix(".tmp")
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.rename(out_path)

    # Sidecar metadata
    meta = dict(result)
    meta["local_path"] = str(out_path)
    meta["cached_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(out_path.with_suffix(".json"), meta, ensure_ascii=False)

    _register_cached_video(result, out_path)

    return out_path


def _clip_asset_path(stem: str, clip_number: int, asset_slot: int) -> Path:
    """Path the renderer's stock-asset resolver will pick up for this clip+slot."""
    prefix = f"{_slug(stem)}__clip{clip_number:02d}__ai{asset_slot:02d}"
    return BROLL / f"{prefix}__stock.mp4"


def _link_to_clip(cached_path: Path, clip_path: Path, result: dict) -> None:
    """Create a symlink (or copy fallback) from clip asset path to cached file."""
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    if clip_path.exists() or clip_path.is_symlink():
        clip_path.unlink()

    try:
        rel = os.path.relpath(str(cached_path), str(clip_path.parent))
        clip_path.symlink_to(rel)
    except OSError:
        shutil.copy2(str(cached_path), str(clip_path))

    # Clip-specific sidecar
    meta = dict(result)
    meta["local_path"] = str(clip_path)
    meta["cached_source"] = str(cached_path)
    meta["cached_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(clip_path.with_suffix(".json"), meta, ensure_ascii=False)


# ─── Candidate helpers ────────────────────────────────────────────────────────

def _load_candidates(stem: str) -> dict:
    p = CLIPS / f"{stem}.candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing candidate file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _scenic_beats(stem: str, clip_number: int) -> List[dict]:
    """Return stock_video beats from a clip's visual_plan."""
    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if not (1 <= clip_number <= len(clips)):
        raise ValueError(f"clip_number must be 1–{len(clips)}")
    clip = clips[clip_number - 1]
    return [b for b in (clip.get("visual_plan") or []) if b.get("type") == "stock_video"]

# ─── MCP tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def healthcheck() -> dict:
    """Check stock fetcher configuration and API key availability."""
    providers = _provider_priority()
    provider_status: Dict[str, str] = {}

    for p in providers:
        try:
            if p == "pexels":
                _pexels_key()
                provider_status[p] = "key_present"
            elif p == "pixabay":
                _pixabay_key()
                provider_status[p] = "key_present"
            else:
                provider_status[p] = "unknown_provider"
        except RuntimeError as e:
            provider_status[p] = f"missing_key: {e}"

    cache_dir = _stock_cache_dir()
    cached = list(cache_dir.glob("*.mp4")) if cache_dir.exists() else []

    return {
        "ok": True,
        "providers": provider_status,
        "cache_dir": str(cache_dir),
        "cached_files": len(cached),
        "config": {
            "provider_priority": providers,
            "preferred_orientation": _preferred_orientation(),
            "min_duration_sec": _min_duration(),
            "max_duration_sec": _max_duration(),
            "allow_landscape_fallback": _allow_landscape_fallback(),
            "require_scenic_filter": _require_scenic_filter(),
        },
    }


@mcp.tool()
def list_stock_files(limit: int = 50) -> dict:
    """List cached stock video files in broll/stock/."""
    cache_dir = _stock_cache_dir()
    files = sorted(
        [p for p in cache_dir.glob("*.mp4") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "folder": str(cache_dir),
        "files": [
            {"name": p.name, "path": str(p), "size_bytes": p.stat().st_size}
            for p in files[:limit]
        ],
    }


@mcp.tool()
def search_stock_videos(
    query: str,
    per_page: int = 5,
    orientation: str = "portrait",
) -> dict:
    """Search for scenic stock videos. Tries providers in STOCK_PROVIDER_PRIORITY order."""
    results, errors = _search_stock(query, per_page, orientation)
    return {
        "query": query,
        "orientation": orientation,
        "count": len(results),
        "results": results,
        "errors": errors,
    }


@mcp.tool()
def download_stock_video(
    url_or_id: str,
    provider: str,
    prefix: str = "",
    overwrite: bool = False,
) -> dict:
    """Download a stock video directly by URL to the cache (broll/stock/)."""
    if not url_or_id.startswith("http"):
        return {
            "ok": False,
            "message": (
                "Pass a direct download URL (starts with http). "
                "For ID-based lookup, use search_stock_videos first to get the download_url."
            ),
        }

    video_id = hashlib.md5(url_or_id.encode()).hexdigest()[:12]
    result: dict = {
        "provider": provider or "direct",
        "id": video_id,
        "title": prefix or "stock",
        "width": 0,
        "height": 0,
        "duration": 0,
        "download_url": url_or_id,
        "file_width": 0,
        "file_height": 0,
        "source_page_url": "",
        "query": prefix,
    }

    out_path = _stock_cache_dir() / f"{_cache_key(result['provider'], result['id'])}.mp4"
    if out_path.exists() and not overwrite:
        return {"ok": True, "saved_file": str(out_path), "message": "Already cached"}

    cached = _download_to_cache(result)
    return {"ok": True, "saved_file": str(cached), "provider": result["provider"], "video_id": video_id}


@mcp.tool()
def fetch_stock_for_candidate(
    stem: str,
    clip_number: int,
    overwrite: bool = False,
) -> dict:
    """
    Fetch and cache real stock footage for all scenic beats in a clip's visual_plan.

    For each stock_video beat:
      1. Extract the prompt/notes as a search query (Arabic stripped automatically).
      2. Search Pexels → Pixabay in priority order.
      3. Download to broll/stock/{provider}__{id}.mp4 (cached; re-used across renders).
      4. Create a symlink broll/{stem}__clip{N}__ai{slot}__stock.mp4 for the renderer.
    """
    BROLL.mkdir(parents=True, exist_ok=True)

    beats = _scenic_beats(stem, clip_number)
    if not beats:
        return {
            "ok": False,
            "message": "No stock_video beats in visual_plan for this clip.",
            "clip_number": clip_number,
        }

    orientation = _preferred_orientation()
    fetched: List[dict] = []

    for beat in beats:
        slot = int(beat.get("asset_slot") or 0)
        if slot < 1:
            continue

        clip_path = _clip_asset_path(stem, clip_number, slot)

        if (clip_path.exists() or clip_path.is_symlink()) and not overwrite:
            fetched.append({"asset_slot": slot, "saved_file": str(clip_path), "status": "existing"})
            continue

        # Build a clean search query from the beat's prompt or notes
        raw_query = str(beat.get("prompt") or beat.get("notes") or "scenic nature landscape")
        query = _sanitize_query(raw_query)

        reusable = _find_reusable_cached_video(query, orientation=orientation)
        if reusable and reusable.exists():
            pseudo_result = {
                "provider": "cache",
                "id": reusable.stem,
                "title": query,
                "duration": 0,
                "query": query,
                "download_url": "",
                "source_page_url": "",
            }
            _link_to_clip(reusable, clip_path, pseudo_result)
            fetched.append({
                "asset_slot": slot,
                "saved_file": str(clip_path),
                "cached_source": str(reusable),
                "provider": "cache",
                "video_id": reusable.stem,
                "query": query,
                "status": "reused_from_manifest",
            })
            continue

        results, errors = _search_stock(query, per_page=5, orientation=orientation)

        # Landscape fallback
        if not results and _allow_landscape_fallback() and orientation != "landscape":
            r2, e2 = _search_stock(query, per_page=5, orientation="landscape")
            results.extend(r2)
            errors.extend(e2)

        if not results:
            fetched.append({
                "asset_slot": slot,
                "status": "no_results",
                "query": query,
                "errors": errors,
            })
            continue

        chosen = results[0]
        try:
            cached_path = _download_to_cache(chosen)
            _link_to_clip(cached_path, clip_path, chosen)
            fetched.append({
                "asset_slot": slot,
                "saved_file": str(clip_path),
                "cached_source": str(cached_path),
                "provider": chosen["provider"],
                "video_id": chosen["id"],
                "duration": chosen["duration"],
                "query": query,
                "status": "downloaded",
            })
        except Exception as e:
            fetched.append({
                "asset_slot": slot,
                "status": "error",
                "query": query,
                "error": str(e),
            })

    ok_count = sum(1 for f in fetched if f.get("status") in ("downloaded", "existing"))
    return {
        "ok": True,
        "stem": stem,
        "clip_number": clip_number,
        "fetched_count": ok_count,
        "results": fetched,
    }


@mcp.tool()
def fetch_stock_for_visual_plan(
    stem: str,
    clip_number: int,
    overwrite: bool = False,
) -> dict:
    """Alias for fetch_stock_for_candidate — fetches stock footage for all scenic beats."""
    return fetch_stock_for_candidate(stem=stem, clip_number=clip_number, overwrite=overwrite)


if __name__ == "__main__":
    mcp.run()
