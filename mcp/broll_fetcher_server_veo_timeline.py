from pathlib import Path
import os
import json
import time
from typing import Optional, List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

from helpers import atomic_write_json

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()
BROLL = ROOT / "broll"
CLIPS = ROOT / "clips"

load_dotenv(ROOT / ".env")

mcp = FastMCP("clip-factory-broll-fetcher", json_response=True)


def _slug(s: str) -> str:
    out = []
    for ch in str(s or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in [" ", "_", "-", "."]:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "broll"


def _client():
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    return OpenAI(api_key=api_key)

def _video_model(default: str = "sora-2") -> str:
    return (os.environ.get("OPENAI_VIDEO_MODEL") or default).strip()


def _size_from_params(aspect_ratio: str, resolution: str) -> str:
    ar = (aspect_ratio or "").strip()
    res = (resolution or "").strip().lower()

    if ar == "9:16":
        if res == "720p":
            return "720x1280"
        if res in {"1024p", "1024"}:
            return "1024x1792"
    elif ar == "16:9":
        if res == "720p":
            return "1280x720"
        if res in {"1024p", "1024"}:
            return "1792x1024"

    raise ValueError(f"Unsupported OpenAI video size mapping for aspect_ratio={aspect_ratio}, resolution={resolution}")

def _seconds_param(duration_seconds: int) -> str:
    if duration_seconds <= 4:
        return "4"
    if duration_seconds <= 8:
        return "8"
    return "12"


def _load_candidates(stem: str) -> dict:
    p = CLIPS / f"{stem}.candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing candidate file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _list_broll_files(limit: int = 50) -> List[dict]:
    if not BROLL.exists():
        return []
    files = sorted([p for p in BROLL.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "name": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
        }
        for p in files[:limit]
    ]


def _negative_prompt() -> str:
    return (
        "no people, no human face, no portrait, no hands, no body, no crowd, no silhouettes, "
        "no animals, no birds, no insects, no living creatures, no text, no logos, no watermark, "
        "no presenter, no talking head, no interview, no stage, no microphone"
    )


def _poll_video(client, video, poll_seconds: int = 10, max_wait_seconds: int = 1800):
    started = time.time()
    while getattr(video, "status", None) in ("queued", "in_progress"):
        if time.time() - started > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for Sora generation after {max_wait_seconds} seconds")
        time.sleep(poll_seconds)
        video = client.videos.retrieve(video.id)

    if getattr(video, "status", None) != "completed":
        err = getattr(video, "error", None)
        raise RuntimeError(f"Sora generation failed: status={getattr(video, 'status', None)} error={err}")

    return video

def _save_generated_video(client, video, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.videos.download_content(video.id)
    with open(out_path, "wb") as f:
        f.write(content.read())

def _build_prompt(query: str, variant_index: int = 1) -> str:
    query = " ".join(str(query or "").split()).strip() or "serene cinematic natural scenery"
    variants = [
        f"Cinematic vertical scenic video of {query}. Subtle natural motion, photorealistic, beautiful light, quiet atmosphere. No people, no human face, no hands, no animals, no birds, no insects, no text, no logos.",
        f"Atmospheric environmental video of {query}. Slow camera drift, rich detail, elegant composition, photorealistic, vertical 9:16. Empty scene only. No humans, no faces, no animals, no text, no logos.",
        f"Beautiful moody cutaway video of {query}. Natural motion, cinematic realism, photorealistic, vertical 9:16. Scenery only. No people, no human face, no living creatures, no text, no logos.",
    ]
    return variants[(max(1, variant_index) - 1) % len(variants)]


def _generate_video_file(
    prompt: str,
    out_path: Path,
    model: str,
    aspect_ratio: str,
    resolution: str,
    duration_seconds: int,
    seed: Optional[int] = None,  # kept for compatibility, ignored
) -> Dict:
    client = _client()

    size = _size_from_params(aspect_ratio, resolution)
    seconds = _seconds_param(duration_seconds)

    video = client.videos.create(
        model=model,
        prompt=prompt,
        size=size,
        seconds=seconds,
    )

    video = _poll_video(client, video)
    _save_generated_video(client, video, out_path)

    return {
        "model": model,
        "prompt": prompt,
        "size": size,
        "seconds": seconds,
        "saved_to": str(out_path),
        "video_id": getattr(video, "id", ""),
        "status": getattr(video, "status", ""),
    }
def _expected_asset_path(stem: str, clip_number: int, asset_slot: int) -> Path:
    return BROLL / f"{_slug(stem)}__clip{clip_number:02d}__ai{asset_slot:02d}__veo.mp4"


def _candidate_visual_beats(stem: str, clip_number: int) -> List[dict]:
    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if clip_number < 1 or clip_number > len(clips):
        raise ValueError(f"clip_number must be between 1 and {len(clips)}")
    clip = clips[clip_number - 1]
    visual_plan = clip.get("visual_plan") or []
    return [beat for beat in visual_plan if str(beat.get("type", "")).lower() == "ai_video"]


@mcp.tool()
def list_broll_files(limit: int = 50) -> dict:
    return {"folder": str(BROLL), "files": _list_broll_files(limit)}


@mcp.tool()
def search_broll(query: str, per_page: int = 3) -> dict:
    prompts = [_build_prompt(query, idx) for idx in range(1, max(1, per_page) + 1)]
    return {
        "provider": "openai-sora",
        "query": query,
        "count": len(prompts),
        "results": [
            {
                "index": idx,
                "prompt": prompt,
            }
            for idx, prompt in enumerate(prompts, start=1)
        ],
    }


@mcp.tool()
def download_broll(
    query: str,
    pick_index: int = 1,
    prefix: str = "",
    model: str = "sora-2",
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    duration_seconds: int = 4,
    overwrite: bool = False,
    seed: Optional[int] = None,
) -> dict:
    BROLL.mkdir(parents=True, exist_ok=True)
    model = model or _video_model()
    prompt = _build_prompt(query, pick_index)
    base = _slug(prefix) + "__" if str(prefix).strip() else ""
    out_path = BROLL / f"{base}veo__{_slug(query)}__v{pick_index:02d}.mp4"

    if out_path.exists() and not overwrite:
        meta_path = out_path.with_suffix(".json")
        return {
            "ok": True,
            "saved_file": str(out_path),
            "metadata_file": str(meta_path),
            "message": "File already exists",
            "prompt": prompt,
        }

    meta = _generate_video_file(
        prompt=prompt,
        out_path=out_path,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        duration_seconds=duration_seconds,
    )
    meta.update(
        {
            "provider": "openai-sora",
            "query": query,
            "pick_index": pick_index,
            "prefix": prefix,
        }
    )
    meta_path = out_path.with_suffix(".json")
    atomic_write_json(meta_path, meta, ensure_ascii=False)

    return {
        "ok": True,
        "saved_file": str(out_path),
        "metadata_file": str(meta_path),
        "prompt": prompt,
        "result": meta,
    }


@mcp.tool()
def fetch_broll_for_candidate(
    stem: str,
    clip_number: int,
    model: str = "sora-2",
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    duration_seconds: int = 4,
    overwrite: bool = False,
) -> dict:
    BROLL.mkdir(parents=True, exist_ok=True)
    model = model or _video_model()

    beats = _candidate_visual_beats(stem, clip_number)
    if not beats:
        return {
            "ok": False,
            "message": "This clip has no ai_video beats in visual_plan",
            "clip_number": clip_number,
        }

    generated = []
    for beat in beats:
        slot = int(beat.get("asset_slot") or 0)
        if slot < 1:
            continue
        prompt = str(beat.get("prompt") or "").strip()
        if not prompt:
            prompt = _build_prompt("serene cinematic natural scenery", slot)
        out_path = _expected_asset_path(stem, clip_number, slot)

        if out_path.exists() and not overwrite:
            generated.append(
                {
                    "asset_slot": slot,
                    "prompt": prompt,
                    "saved_file": str(out_path),
                    "status": "existing",
                }
            )
            continue

        meta = _generate_video_file(
            prompt=prompt,
            out_path=out_path,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
        )
        meta.update(
            {
                "provider": "openai-sora",
                "stem": stem,
                "clip_number": clip_number,
                "asset_slot": slot,
                "clip_relative_start_sec": beat.get("start_offset_sec", 0.0),
                "clip_relative_end_sec": beat.get("end_offset_sec", 0.0),
                "clip_relative_duration_sec": beat.get("duration_sec", 0.0),
            }
        )
        meta_path = out_path.with_suffix(".json")
        atomic_write_json(meta_path, meta, ensure_ascii=False)
        generated.append(
            {
                "asset_slot": slot,
                "prompt": prompt,
                "saved_file": str(out_path),
                "metadata_file": str(meta_path),
                "status": "generated",
            }
        )

    return {
        "ok": True,
        "provider": "openai-sora",
        "stem": stem,
        "clip_number": clip_number,
        "model": model,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "duration_seconds": duration_seconds,
        "generated_count": len(generated),
        "results": generated,
    }


if __name__ == "__main__":
    mcp.run()
