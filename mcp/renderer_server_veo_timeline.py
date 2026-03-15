"""
clip-factory renderer MCP server with alternating scenic insert support.

Behavior:
- prefers clip-specific transcript artifacts first, then stem-level transcript artifacts
- honors refined/quran boundary suggestions when present
- uses clip.visual_plan to alternate between the original source and AI scenic videos
- keeps the original clip audio throughout the whole short
- burns subtitles after the visual timeline is assembled
"""

from pathlib import Path
import os
import json
import subprocess
import tempfile
from typing import Optional, List, Tuple, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from helpers import (
    generate_clip_ass,
    parse_srt_to_segments,
    get_ffmpeg,
    atomic_write_json,
)

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()
INCOMING = ROOT / "incoming"
TRANSCRIPTS = ROOT / "transcripts"
CLIPS = ROOT / "clips"
FINAL = ROOT / "final"
BROLL = ROOT / "broll"

load_dotenv(ROOT / ".env")

mcp = FastMCP("clip-factory-renderer", json_response=True)

VIDEO_EXTS = [".mp4", ".mov", ".m4v", ".webm", ".mkv"]
AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]

SUBTITLE_FONT = os.environ.get("SUBTITLE_FONT", "Geeza Pro")
SUBTITLE_FONTSIZE = int(os.environ.get("SUBTITLE_FONTSIZE", "60"))
SUBTITLE_MARGIN_V = int(os.environ.get("SUBTITLE_MARGIN_V", "110"))


def _find_source_video(stem: str) -> Path:
    for ext in VIDEO_EXTS:
        p = INCOMING / f"{stem}{ext}"
        if p.exists():
            return p
    for p in INCOMING.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.stem == stem:
            return p
    for ext in AUDIO_EXTS:
        p = INCOMING / f"{stem}{ext}"
        if p.exists():
            return p
    for p in INCOMING.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.stem == stem:
            return p
    raise FileNotFoundError(f"No source video found in incoming/ for stem: {stem}")


def _is_audio_only(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def _load_candidates(stem: str) -> dict:
    p = CLIPS / f"{stem}.candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing candidate file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_transcript_segments(stem: str, clip_number: Optional[int] = None) -> List[dict]:
    preferred = []

    if clip_number is not None:
        clip_stem = f"{stem}__clip{clip_number:02d}"
        preferred += [
            TRANSCRIPTS / f"{clip_stem}.quran_guard.verbose.json",
            TRANSCRIPTS / f"{clip_stem}.refined.verbose.json",
            TRANSCRIPTS / f"{clip_stem}.quran_guard.srt",
            TRANSCRIPTS / f"{clip_stem}.refined.captions.srt",
            TRANSCRIPTS / f"{clip_stem}.refined.srt",
        ]

    preferred += [
        TRANSCRIPTS / f"{stem}.quran_guard.verbose.json",
        TRANSCRIPTS / f"{stem}.refined.verbose.json",
        TRANSCRIPTS / f"{stem}.verbose.json",
        TRANSCRIPTS / f"{stem}.quran_guard.srt",
        TRANSCRIPTS / f"{stem}.refined.captions.srt",
        TRANSCRIPTS / f"{stem}.captions.srt",
        TRANSCRIPTS / f"{stem}.refined.srt",
        TRANSCRIPTS / f"{stem}.srt",
    ]

    for path in preferred:
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                segments = data.get("segments") or []
                if segments:
                    return segments
            else:
                segments = parse_srt_to_segments(path.read_text(encoding="utf-8"))
                if segments:
                    return segments
        except Exception:
            continue

    return []


def _make_clip_ass_tempfile(stem: str, clip_number: int, clip_start: float, clip_end: float) -> Optional[Path]:
    segments = _load_transcript_segments(stem, clip_number=clip_number)
    if not segments:
        return None

    ass_content = generate_clip_ass(
        segments=segments,
        clip_start=clip_start,
        clip_end=clip_end,
        font=SUBTITLE_FONT,
        fontsize=SUBTITLE_FONTSIZE,
        margin_v=SUBTITLE_MARGIN_V,
        clean_arabic=False,
    )

    if "Dialogue:" not in ass_content:
        return None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ass", prefix="clipfactory_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(ass_content)
        return Path(tmp_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _escape_filter_path(path: Path) -> str:
    s = str(path)
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    return s


def _preset_filter(preset: str) -> str:
    presets = {
        "clean-warm": "eq=contrast=1.08:brightness=0.02:saturation=1.10",
        "cinematic-soft": "eq=contrast=1.12:brightness=0.02:saturation=1.06,unsharp=5:5:0.7:5:5:0.0",
        "high-contrast": "eq=contrast=1.22:brightness=-0.01:saturation=1.15",
        "golden-islamic": "eq=contrast=1.11:brightness=0.03:saturation=1.12,colorbalance=rs=0.05:gs=0.01:bs=-0.02,unsharp=5:5:0.5:5:5:0.0",
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(presets)}")
    return presets[preset]


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
    return slug.strip("-") or "clip"


def _list_media(folder: Path, limit: int = 50):
    if not folder.exists():
        return []
    files = sorted([p for p in folder.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "path": str(p), "size_bytes": p.stat().st_size} for p in files[:limit]]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_boundary_suggestion(payload: Optional[dict]) -> Optional[dict]:
    if not payload:
        return None
    boundary = payload.get("boundary_suggestion")
    if isinstance(boundary, dict):
        return boundary
    return None


def _resolve_clip_timing(stem: str, clip_number: int, clip: dict) -> Tuple[float, float, dict]:
    original_start = round(_safe_float(clip.get("start_sec")), 2)
    original_end = round(_safe_float(clip.get("end_sec")), 2)
    source = "candidate_json"
    reason = "original_candidate"

    override_start = clip.get("adjusted_start_sec")
    override_end = clip.get("adjusted_end_sec")
    if override_start is not None and override_end is not None:
        s = round(_safe_float(override_start), 2)
        e = round(_safe_float(override_end), 2)
        if e > s:
            return s, e, {
                "source": "candidate_adjusted_fields",
                "reason": "candidate_contains_adjusted_boundary_fields",
                "changed": bool(abs(s - original_start) > 0.01 or abs(e - original_end) > 0.01),
                "original_start_sec": original_start,
                "original_end_sec": original_end,
            }

    clip_stem = f"{stem}__clip{clip_number:02d}"
    summary_paths = [
        TRANSCRIPTS / f"{clip_stem}.refined.summary.json",
        TRANSCRIPTS / f"{clip_stem}.quran_guard.summary.json",
    ]

    best_boundary = None
    best_confidence = -1.0
    for path in summary_paths:
        payload = _load_json_if_exists(path)
        boundary = _extract_boundary_suggestion(payload)
        if not boundary:
            continue
        s = round(_safe_float(boundary.get("suggested_start_sec"), original_start), 2)
        e = round(_safe_float(boundary.get("suggested_end_sec"), original_end), 2)
        if e <= s:
            continue
        confidence = _safe_float(boundary.get("confidence"), 0.0)
        if confidence > best_confidence:
            best_confidence = confidence
            best_boundary = {
                "start": s,
                "end": e,
                "source": path.name,
                "reason": boundary.get("reason", "boundary_suggestion"),
                "confidence": round(confidence, 4),
                "changed": bool(abs(s - original_start) > 0.01 or abs(e - original_end) > 0.01),
            }

    if best_boundary:
        return best_boundary["start"], best_boundary["end"], {
            "source": best_boundary["source"],
            "reason": best_boundary["reason"],
            "confidence": best_boundary["confidence"],
            "changed": best_boundary["changed"],
            "original_start_sec": original_start,
            "original_end_sec": original_end,
        }

    return original_start, original_end, {
        "source": source,
        "reason": reason,
        "changed": False,
        "original_start_sec": original_start,
        "original_end_sec": original_end,
    }


def _middle_insert_plan(duration: float, ai_duration: float = 1.8) -> List[dict]:
    if duration <= ai_duration + 3.0:
        return [{"type": "original", "start_offset_sec": 0.0, "end_offset_sec": round(duration, 2), "duration_sec": round(duration, 2), "asset_slot": 0, "prompt": "", "notes": "fallback_original_only"}]
    start = max(1.4, min(duration / 2.0 - ai_duration / 2.0, duration - ai_duration - 1.6))
    end = min(duration - 1.4, start + ai_duration)
    return [
        {"type": "original", "start_offset_sec": 0.0, "end_offset_sec": round(start, 2), "duration_sec": round(start, 2), "asset_slot": 0, "prompt": "", "notes": "original_intro"},
        {"type": "ai_video", "start_offset_sec": round(start, 2), "end_offset_sec": round(end, 2), "duration_sec": round(end - start, 2), "asset_slot": 1, "prompt": "", "notes": "manual_middle_insert"},
        {"type": "original", "start_offset_sec": round(end, 2), "end_offset_sec": round(duration, 2), "duration_sec": round(duration - end, 2), "asset_slot": 0, "prompt": "", "notes": "original_outro"},
    ]


def _normalize_visual_plan(plan: List[dict], duration: float) -> List[dict]:
    beats: List[dict] = []
    for beat in plan or []:
        beat_type = str(beat.get("type") or "").lower().strip()
        if beat_type not in {"original", "ai_video"}:
            continue
        s = max(0.0, min(duration, _safe_float(beat.get("start_offset_sec"), 0.0)))
        e = max(0.0, min(duration, _safe_float(beat.get("end_offset_sec"), 0.0)))
        if e - s <= 0.2:
            continue
        beats.append(
            {
                "type": beat_type,
                "start_offset_sec": round(s, 2),
                "end_offset_sec": round(e, 2),
                "duration_sec": round(e - s, 2),
                "asset_slot": int(beat.get("asset_slot") or 0),
                "prompt": str(beat.get("prompt") or ""),
                "notes": str(beat.get("notes") or ""),
            }
        )
    beats.sort(key=lambda b: b["start_offset_sec"])
    return beats or [{"type": "original", "start_offset_sec": 0.0, "end_offset_sec": round(duration, 2), "duration_sec": round(duration, 2), "asset_slot": 0, "prompt": "", "notes": "full_original"}]


def _find_ai_asset(stem: str, clip_number: int, asset_slot: int) -> Optional[Path]:
    prefix = f"{_slug(stem)}__clip{clip_number:02d}__ai{asset_slot:02d}"
    for ext in VIDEO_EXTS:
        exact = BROLL / f"{prefix}__veo{ext}"
        if exact.exists():
            return exact
    candidates = sorted(
        [p for p in BROLL.glob(f"{prefix}*.mp4") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_visual_plan_for_render(clip: dict, duration: float, broll_file: Optional[str]) -> List[dict]:
    visual_plan = clip.get("visual_plan") or []
    if visual_plan:
        return _normalize_visual_plan(visual_plan, duration)

    if broll_file:
        return _middle_insert_plan(duration)

    return [{"type": "original", "start_offset_sec": 0.0, "end_offset_sec": round(duration, 2), "duration_sec": round(duration, 2), "asset_slot": 0, "prompt": "", "notes": "full_original"}]


def _build_video_segment_chain(input_label: str, trim_start: float, trim_end: float, preset: str, label: str) -> str:
    return (
        f"{input_label}trim=start={trim_start}:end={trim_end},"
        f"setpts=PTS-STARTPTS,"
        f"scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,"
        f"fps=30,"
        f"setsar=1,"
        f"format=yuv420p,"
        f"{_preset_filter(preset)}"
        f"[{label}]"
    )


def _build_visual_timeline_filter(
    visual_plan: List[dict],
    start: float,
    duration: float,
    preset: str,
    audio_only: bool,
    ai_input_index_map: Dict[int, int],
) -> Tuple[str, List[str]]:
    parts: List[str] = []
    concat_labels: List[str] = []
    used_assets: List[str] = []

    for idx, beat in enumerate(visual_plan):
        label = f"vseg{idx}"
        beat_type = beat["type"]
        seg_start = _safe_float(beat["start_offset_sec"])
        seg_end = _safe_float(beat["end_offset_sec"])
        if beat_type == "original":
            if audio_only:
                trim_start = seg_start
                trim_end = seg_end
                parts.append(_build_video_segment_chain("[0:v]", trim_start, trim_end, preset, label))
            else:
                trim_start = start + seg_start
                trim_end = start + seg_end
                parts.append(_build_video_segment_chain("[0:v]", trim_start, trim_end, preset, label))
        else:
            slot = int(beat.get("asset_slot") or 0)
            if slot < 1 or slot not in ai_input_index_map:
                fallback_label = f"vfallback{idx}"
                if audio_only:
                    trim_start = seg_start
                    trim_end = seg_end
                    parts.append(_build_video_segment_chain("[0:v]", trim_start, trim_end, preset, fallback_label))
                else:
                    trim_start = start + seg_start
                    trim_end = start + seg_end
                    parts.append(_build_video_segment_chain("[0:v]", trim_start, trim_end, preset, fallback_label))
                label = fallback_label
            else:
                input_idx = ai_input_index_map[slot]
                used_assets.append(f"slot:{slot}@input:{input_idx}")
                parts.append(_build_video_segment_chain(f"[{input_idx}:v]", 0.0, beat["duration_sec"], preset, label))
        concat_labels.append(f"[{label}]")

    parts.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=1:a=0[vcat]")
    return ";".join(parts), used_assets


def _render_one(
    stem: str,
    clip_number: int,
    preset: str = "golden-islamic",
    burn_subtitles: bool = True,
    broll_file: Optional[str] = None,
    auto_broll: bool = False,
    broll_start_in_source_sec: float = 0.0,
    broll_duration_sec: float = 1.8,
    broll_insert_at_clip_sec: float = -1.0,
    overwrite: bool = True,
) -> dict:
    FINAL.mkdir(parents=True, exist_ok=True)

    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if clip_number < 1 or clip_number > len(clips):
        raise ValueError(f"clip_number must be between 1 and {len(clips)}")

    clip = clips[clip_number - 1]
    start, end, timing_meta = _resolve_clip_timing(stem=stem, clip_number=clip_number, clip=clip)
    duration = round(end - start, 2)
    if duration <= 0:
        raise ValueError(f"Invalid clip duration: {duration}s")

    source_video = _find_source_video(stem)
    out_name = f"{_slug(stem)}__clip{clip_number:02d}__{_slug(preset)}.mp4"
    out_path = FINAL / out_name
    if out_path.exists() and not overwrite:
        return {"ok": True, "message": "Rendered file already exists", "output_file": str(out_path)}

    audio_only = _is_audio_only(source_video)
    ass_tempfile: Optional[Path] = None
    subtitle_filter_str = ""
    subtitle_source = ""

    if burn_subtitles:
        try:
            ass_tempfile = _make_clip_ass_tempfile(stem, clip_number, start, end)
        except Exception as e:
            subtitle_source = f"subtitle generation failed: {e}"
        if ass_tempfile:
            escaped = _escape_filter_path(ass_tempfile)
            subtitle_filter_str = f"subtitles=filename='{escaped}':wrap_unicode=1"
            subtitle_source = str(ass_tempfile)
        elif not subtitle_source:
            subtitle_source = "no transcript found for clip"

    visual_plan = _build_visual_plan_for_render(clip=clip, duration=duration, broll_file=broll_file)

    ai_paths: Dict[int, Path] = {}
    if auto_broll:
        for beat in visual_plan:
            if beat["type"] != "ai_video":
                continue
            slot = int(beat.get("asset_slot") or 0)
            if slot < 1 or slot in ai_paths:
                continue
            asset = _find_ai_asset(stem, clip_number, slot)
            if asset:
                ai_paths[slot] = asset

    if broll_file:
        selected = Path(broll_file)
        if not selected.is_absolute():
            selected = BROLL / broll_file
        if not selected.exists():
            raise FileNotFoundError(f"B-roll file not found: {selected}")
        ai_paths[1] = selected

    ffmpeg = get_ffmpeg()
    used_broll = any(beat["type"] == "ai_video" for beat in visual_plan) and bool(ai_paths)

    cmd: List[str]
    if audio_only:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:size=1080x1920:rate=30:duration={duration}",
            "-i",
            str(source_video),
        ]
        ai_input_start = 2
        audio_input_label = "[1:a]"
    else:
        cmd = [ffmpeg, "-y", "-i", str(source_video)]
        ai_input_start = 1
        audio_input_label = "[0:a]"

    ai_input_index_map: Dict[int, int] = {}
    for offset, slot in enumerate(sorted(ai_paths)):
        cmd += ["-i", str(ai_paths[slot])]
        ai_input_index_map[slot] = ai_input_start + offset

    audio_chain = f"{audio_input_label}atrim=start={start}:end={end},asetpts=PTS-STARTPTS,loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    visual_chain, used_assets = _build_visual_timeline_filter(
        visual_plan=visual_plan,
        start=start,
        duration=duration,
        preset=preset,
        audio_only=audio_only,
        ai_input_index_map=ai_input_index_map,
    )

    filter_parts = [visual_chain, audio_chain]
    if subtitle_filter_str:
        filter_parts.append(f"[vcat]{subtitle_filter_str}[v]")
    else:
        filter_parts.append("[vcat]null[v]")

    filter_complex = ";".join(filter_parts)

    try:
        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if ass_tempfile and ass_tempfile.exists():
            try:
                ass_tempfile.unlink()
            except OSError:
                pass

    if result.returncode != 0:
        return {
            "ok": False,
            "clip_number": clip_number,
            "error": "ffmpeg render failed",
            "stderr_tail": "\n".join(result.stderr.splitlines()[-30:]),
            "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
            "command": cmd,
            "timing": timing_meta,
            "visual_plan": visual_plan,
        }

    effective_candidate = dict(clip)
    effective_candidate["effective_start_sec"] = start
    effective_candidate["effective_end_sec"] = end
    effective_candidate["effective_duration_sec"] = duration

    return {
        "ok": True,
        "output_file": str(out_path),
        "preset": preset,
        "burn_subtitles": burn_subtitles,
        "subtitle_source": subtitle_source,
        "used_broll": used_broll,
        "selected_broll": [str(path) for _, path in sorted(ai_paths.items())],
        "ffmpeg_binary": ffmpeg,
        "candidate": effective_candidate,
        "timing": timing_meta,
        "visual_plan": visual_plan,
        "used_assets": used_assets,
    }


@mcp.tool()
def list_filter_presets() -> dict:
    return {
        "presets": [
            {"name": "clean-warm", "description": "Balanced warm look"},
            {"name": "cinematic-soft", "description": "Softer cinematic contrast"},
            {"name": "high-contrast", "description": "Punchier separation"},
            {"name": "golden-islamic", "description": "Warm reflective Islamic tone"},
        ]
    }


@mcp.tool()
def list_broll_files(limit: int = 50) -> dict:
    return {"folder": str(BROLL), "files": _list_media(BROLL, limit)}


@mcp.tool()
def list_final_renders(limit: int = 50) -> dict:
    return {"folder": str(FINAL), "files": _list_media(FINAL, limit)}


@mcp.tool()
def suggest_broll_for_clip(stem: str, clip_number: int) -> dict:
    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if clip_number < 1 or clip_number > len(clips):
        raise ValueError(f"clip_number must be between 1 and {len(clips)}")

    clip = clips[clip_number - 1]
    duration = round(_safe_float(clip.get("end_sec")) - _safe_float(clip.get("start_sec")), 2)
    visual_plan = _build_visual_plan_for_render(clip, duration, None)
    assets = []
    for beat in visual_plan:
        if beat["type"] != "ai_video":
            continue
        slot = int(beat.get("asset_slot") or 0)
        asset = _find_ai_asset(stem, clip_number, slot)
        assets.append(
            {
                "asset_slot": slot,
                "prompt": beat.get("prompt", ""),
                "expected_file": str(BROLL / f"{_slug(stem)}__clip{clip_number:02d}__ai{slot:02d}__veo.mp4"),
                "existing_file": str(asset) if asset else "",
                "start_offset_sec": beat.get("start_offset_sec", 0.0),
                "end_offset_sec": beat.get("end_offset_sec", 0.0),
            }
        )

    return {
        "clip_number": clip_number,
        "title": clip.get("title", ""),
        "visual_mode": clip.get("visual_mode", "alternate_scenic"),
        "visual_plan": visual_plan,
        "assets": assets,
    }


@mcp.tool()
def render_clip_from_candidate(
    stem: str,
    clip_number: int,
    preset: str = "golden-islamic",
    burn_subtitles: bool = True,
    broll_file: Optional[str] = None,
    auto_broll: bool = False,
    broll_start_in_source_sec: float = 0.0,
    broll_duration_sec: float = 1.8,
    broll_insert_at_clip_sec: float = -1.0,
    overwrite: bool = True,
) -> dict:
    return _render_one(
        stem=stem,
        clip_number=clip_number,
        preset=preset,
        burn_subtitles=burn_subtitles,
        broll_file=broll_file,
        auto_broll=auto_broll,
        broll_start_in_source_sec=broll_start_in_source_sec,
        broll_duration_sec=broll_duration_sec,
        broll_insert_at_clip_sec=broll_insert_at_clip_sec,
        overwrite=overwrite,
    )


@mcp.tool()
def batch_render_from_candidates(
    stem: str,
    clip_numbers: Optional[List[int]] = None,
    preset: str = "golden-islamic",
    burn_subtitles: bool = True,
    auto_broll: bool = True,
    max_renders: int = 5,
    overwrite: bool = True,
) -> dict:
    data = _load_candidates(stem)
    clips = data.get("clips") or []

    if not clips:
        return {"ok": False, "error": "No clip candidates found"}

    targets = clip_numbers[:max_renders] if clip_numbers else list(range(1, min(len(clips), max_renders) + 1))

    results = []
    for clip_number in targets:
        try:
            results.append(
                _render_one(
                    stem=stem,
                    clip_number=clip_number,
                    preset=preset,
                    burn_subtitles=burn_subtitles,
                    auto_broll=auto_broll,
                    overwrite=overwrite,
                )
            )
        except Exception as e:
            results.append({"ok": False, "clip_number": clip_number, "error": str(e)})

    success_count = sum(1 for r in results if r.get("ok"))
    manifest_path = FINAL / f"{_slug(stem)}__batch_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "stem": stem,
            "preset": preset,
            "burn_subtitles": burn_subtitles,
            "auto_broll": auto_broll,
            "results": results,
        },
    )

    return {
        "ok": True,
        "stem": stem,
        "requested": targets,
        "success_count": success_count,
        "total_count": len(results),
        "manifest": str(manifest_path),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run()
