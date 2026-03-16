"""
clip-factory renderer MCP server with alternating scenic insert support.

Behavior:
- prefers clip-specific transcript artifacts first, then stem-level transcript artifacts
- honors refined/quran boundary suggestions when present
- uses clip.visual_plan to alternate between the original source and stock scenic videos
- keeps the original clip audio throughout the whole short
- burns subtitles/recitation text after the visual timeline is assembled

Text styling
────────────
All text styling is controlled by a single TextStyleConfig loaded from text_config.py.
Change TEXT_STYLE_MODE in .env to switch between "subtitle" and "center_recitation"
modes.  Individual style values (font, size, colours, animation) can be overridden
via ASS_* env vars — no code changes needed.  See text_config.py for the full list.
"""

from pathlib import Path
import dataclasses
import os
import json
import subprocess
import tempfile
from typing import Optional, List, Tuple, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── Resolve ROOT and load .env BEFORE importing helpers / text_config ──────────
# Two-phase bootstrap so that CLIP_FACTORY_ROOT can be set inside .env:
#
#   Phase 1 — Load .env from the best-guess location (shell CLIP_FACTORY_ROOT
#             or the default ~/clip-factory).  This makes any .env-defined
#             CLIP_FACTORY_ROOT visible to os.environ before Phase 2 reads it.
#
#   Phase 2 — Re-read CLIP_FACTORY_ROOT from the now-updated environment.
#             If ROOT moved, reload the new ROOT's .env with override=True
#             so app-level vars in the real deployment root take precedence.
_INITIAL_ROOT = Path(
    os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
).resolve()
load_dotenv(_INITIAL_ROOT / ".env")

ROOT = Path(
    os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
).resolve()
if ROOT != _INITIAL_ROOT and (ROOT / ".env").exists():
    load_dotenv(ROOT / ".env", override=True)

INCOMING = ROOT / "incoming"
TRANSCRIPTS = ROOT / "transcripts"
CLIPS = ROOT / "clips"
FINAL = ROOT / "final"
BROLL = ROOT / "broll"

from helpers import (  # noqa: E402 — intentionally after dotenv
    generate_clip_ass,
    parse_srt_to_segments,
    get_ffmpeg,
    atomic_write_json,
)
from text_config import load_text_config, TextStyleConfig  # noqa: E402

mcp = FastMCP("clip-factory-renderer", json_response=True)

VIDEO_EXTS = [".mp4", ".mov", ".m4v", ".webm", ".mkv"]
AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]

# Single source of truth for all text styling in this server.
# Loaded once after dotenv is resolved.  Reads TEXT_STYLE_MODE and ASS_* env vars.
# To change style: edit .env and restart the server — no code changes needed.
_text_cfg: TextStyleConfig = load_text_config()

# ── Visual presentation env knobs ─────────────────────────────────────────────
# All read once at startup.  Change in .env and restart to take effect.

# Duration (seconds) of crossfade between consecutive video segments.
# Set to 0.0 to use a hard cut (concat) instead of an xfade dissolve.
RENDER_TRANSITION_DURATION: float = float(
    os.environ.get("RENDER_TRANSITION_DURATION", "0.4")
)

# FFmpeg xfade transition type — see `ffmpeg -filters | grep xfade` for options.
# Common: dissolve (default), fade, wipeleft, wiperight, slideleft, slideright.
RENDER_TRANSITION_TYPE: str = (
    os.environ.get("RENDER_TRANSITION_TYPE", "dissolve").strip().lower() or "dissolve"
)

# Minimum seconds of the original speaker footage to show before any scenic insert
# begins.  Prevents opening immediately on scenic B-roll when the speaker is
# still establishing the recitation.
SPEAKER_HOLD_SECS: float = float(os.environ.get("SPEAKER_HOLD_SECS", "3.0"))

# When true, ffmpeg samples a frame from the source video at the clip start to
# measure mean scene brightness and automatically picks a contrasting text color
# (white on dark scenes, dark on very bright scenes).
# Overrides ASS_PRIMARY_COLOR for the brightness-selected colour only.
ADAPTIVE_TEXT_COLOR: bool = (
    os.environ.get("ADAPTIVE_TEXT_COLOR", "true").strip().lower()
    in ("1", "true", "yes")
)

# Optional path to a directory containing font files for the libass subtitles
# filter.  When set, added as `fontsdir=...` to the ffmpeg subtitles filter so
# libass looks there BEFORE the system fontconfig path.
# Example: ASS_FONTSDIR=/Users/darwashi/clip-factory/fonts
# Leave unset to rely on fontconfig alone (works on macOS with system fonts).
ASS_FONTSDIR: str = os.environ.get("ASS_FONTSDIR", "").strip()


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


def _load_transcript_segments(stem: str, clip_number: Optional[int] = None) -> Tuple[List[dict], str]:
    """
    Load the best available transcript segments for a clip, in deterministic
    priority order.

    Priority rationale (highest → lowest quality):
    ┌─ clip-level artifacts (preferred — scoped to this exact clip) ──────────┐
    │ 1. {clip}.quran_guard.verbose.json  Corpus-canonical Quran text with    │
    │                                     quran_guard metadata.  Multi-segment │
    │                                     window timing is correct after the   │
    │                                     timing fix in quran_guard_server.py. │
    │ 2. {clip}.refined.verbose.json      Gemini-corrected ASR text, absolute  │
    │                                     timestamps, no quran_guard metadata. │
    │ 3. {clip}.quran_guard.srt           Correct cue timing; less metadata    │
    │                                     than verbose JSON.                   │
    │ 4. {clip}.refined.captions.srt      clean_arabic_for_captions applied —  │
    │                                     acceptable for non-Quran text.       │
    │ 5. {clip}.refined.srt               Raw Gemini-corrected SRT.            │
    └─────────────────────────────────────────────────────────────────────────┘
    ┌─ stem-level artifacts (fallback — covers full source, not just clip) ───┐
    │ 6–11. Same cascade as above for the full stem, then raw SRT fallbacks.  │
    └─────────────────────────────────────────────────────────────────────────┘

    The list is evaluated top-to-bottom; the first path that exists AND parses
    to at least one segment wins.  This is deterministic: the result depends
    only on which pipeline stages have been run, not on file-system ordering.

    generate_clip_ass is always called with clean_arabic=False, so text content
    from any of these sources is preserved as-is for rendering.

    Returns (segments, source_filename) — source_filename is the basename of
    the winning artifact (e.g. "stem__clip01.quran_guard.verbose.json") or ""
    when no transcript was found.  The caller includes this in render metadata
    so it is visible which pipeline stage's output is being rendered.
    """
    preferred: List[Path] = []

    if clip_number is not None:
        clip_stem = f"{stem}__clip{clip_number:02d}"
        preferred += [
            TRANSCRIPTS / f"{clip_stem}.quran_guard.verbose.json",
            TRANSCRIPTS / f"{clip_stem}.refined.verbose.json",
            TRANSCRIPTS / f"{clip_stem}.quran_guard.srt",
            TRANSCRIPTS / f"{clip_stem}.refined.srt",
        ]

    # Stem-level fallbacks (6–11).
    preferred += [
        TRANSCRIPTS / f"{stem}.quran_guard.verbose.json",
        TRANSCRIPTS / f"{stem}.refined.verbose.json",
        TRANSCRIPTS / f"{stem}.verbose.json",
        TRANSCRIPTS / f"{stem}.quran_guard.srt",
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
                    return segments, path.name
            else:
                segments = parse_srt_to_segments(path.read_text(encoding="utf-8"))
                if segments:
                    return segments, path.name
        except Exception:
            continue

    return [], ""


def _make_clip_ass_tempfile(
    stem: str,
    clip_number: int,
    clip_start: float,
    clip_end: float,
    source_video: Optional[Path] = None,
    ai_paths: Optional[Dict[int, Path]] = None,
) -> Tuple[Optional[Path], dict]:
    """
    Generate an ASS text file for one clip and write it to a temp file.

    Text source: _load_transcript_segments (deterministic priority cascade).
    Style source: module-level _text_cfg (loaded from .env at server startup).

    clean_arabic=False: render_text from quran_guard is already corpus-canonical.
    Applying clean_arabic_for_captions would strip valid Quranic characters
    (e.g. ٱ alef wasla U+0671).

    Adaptive text color (ADAPTIVE_TEXT_COLOR=true):
    Samples the source video at clip_start for mean luma. If ai_paths are
    provided, also samples frame 0 of each broll file. The HIGHEST brightness
    across all sampled frames determines the chosen text color — this is a
    worst-case approach that ensures legibility on the brightest beat in the
    timeline, not just the opening frame.

    Returns (ass_path_or_None, render_meta_dict).  render_meta_dict always
    contains: font, text_color, adaptive_triggered, brightness_sampled,
    transcript_artifact.  It is included verbatim in the render result so
    the caller can inspect every styling decision made at render time.
    """
    meta: dict = {
        "font": _text_cfg.font,
        "text_color": _text_cfg.primary_color,
        "adaptive_triggered": False,
        "brightness_sampled": None,
        "transcript_artifact": "",
    }

    segments, artifact_name = _load_transcript_segments(stem, clip_number=clip_number)
    meta["transcript_artifact"] = artifact_name

    if not segments:
        return None, meta

    # Apply adaptive text color if requested and we have a video to sample.
    cfg = _text_cfg
    if ADAPTIVE_TEXT_COLOR and source_video is not None and not _is_audio_only(source_video):
        try:
            ffmpeg_bin = get_ffmpeg()
            # Sample source video at clip_start (speaker footage brightness).
            brightnesses = [_sample_clip_brightness(source_video, clip_start, ffmpeg_bin)]
            # Also sample each broll file at frame 0 (scenic insert brightness).
            # Worst-case: use the highest brightness across all sources so text
            # is legible on the brightest beat in the timeline.
            for broll_path in (ai_paths or {}).values():
                if broll_path.exists():
                    brightnesses.append(_sample_clip_brightness(broll_path, 0.0, ffmpeg_bin))
            worst = max(brightnesses)
            color = _choose_text_color(worst)
            meta["adaptive_triggered"] = True
            meta["brightness_sampled"] = round(worst, 1)
            meta["text_color"] = color
            if color != cfg.primary_color:
                cfg = dataclasses.replace(cfg, primary_color=color)
        except Exception:
            pass  # sampling failed — keep the preset color unchanged

    meta["font"] = cfg.font

    ass_content = generate_clip_ass(
        segments=segments,
        clip_start=clip_start,
        clip_end=clip_end,
        config=cfg,
        clean_arabic=False,  # preserve corpus-canonical text; see docstring
    )

    if "Dialogue:" not in ass_content:
        return None, meta

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ass", prefix="clipfactory_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(ass_content)
        return Path(tmp_path), meta
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None, meta


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
        "golden-islamic": (
            "eq=contrast=1.11:brightness=0.03:saturation=1.12,"
            "colorbalance=rs=0.05:gs=0.01:bs=-0.02,"
            "unsharp=5:5:0.5:5:5:0.0"
        ),
        # Recommended preset for center_recitation mode: pulls exposure down,
        # desaturates slightly, adds a cool-neutral balance and a gentle edge
        # sharpen — creates the dark cinematic backdrop that makes white Arabic
        # text pop without color fighting the recitation.
        "dark-soft-recitation": (
            "eq=contrast=1.06:brightness=-0.06:saturation=0.82,"
            "colorbalance=rs=-0.02:gs=-0.01:bs=0.04,"
            "unsharp=5:5:0.35:5:5:0.0"
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(presets)}")
    return presets[preset]


def _validate_font(font_name: str) -> dict:
    """
    Check whether font_name is reachable by libass.

    Two resolution paths, checked in the order libass itself uses them:

    Path 1 — ASS_FONTSDIR is set:
        libass searches this directory before touching fontconfig.  We scan for
        a .ttf/.otf/.ttc file whose stem contains font_name (case-insensitive).
        If found → exact_match=True, source="ASS_FONTSDIR".
        If the directory is missing or contains no matching file → exact_match=False
        with a clear warning so the operator knows to add the font file.
        NOTE: when ASS_FONTSDIR is set, the fc-match result is irrelevant because
        libass bypasses fontconfig for this directory lookup.

    Path 2 — ASS_FONTSDIR not set:
        fontconfig (fc-match) determines what libass uses.  Amiri Quran is a macOS
        system font and resolves correctly.  On Linux it is not bundled; set
        ASS_FONTSDIR to a directory containing the .ttf to guarantee the right font.

    Returns:
      font_name     — configured font family name
      resolved      — font or file libass will actually use
      exact_match   — True when the intended font is confirmed available
      source        — "ASS_FONTSDIR" | "fontconfig" | "none"
      warning       — non-empty when a substitution or missing-font is likely
    """
    # ── Path 1: explicit fonts directory ──────────────────────────────────────
    if ASS_FONTSDIR:
        fd = Path(ASS_FONTSDIR)
        if not fd.is_dir():
            return {
                "font_name": font_name,
                "resolved": "unknown",
                "exact_match": False,
                "source": "ASS_FONTSDIR",
                "warning": (
                    f"ASS_FONTSDIR='{ASS_FONTSDIR}' is set but the directory does "
                    f"not exist.  libass will fall back to fontconfig with no font "
                    f"guarantee.  Create the directory and add a .ttf/.otf for "
                    f"'{font_name}', or unset ASS_FONTSDIR."
                ),
            }
        _font_exts = {".ttf", ".otf", ".ttc"}
        font_files = [p for p in fd.iterdir() if p.suffix.lower() in _font_exts]
        match = next(
            (p for p in font_files if font_name.lower().replace(" ", "") in p.stem.lower().replace(" ", "")),
            None,
        )
        return {
            "font_name": font_name,
            "resolved": match.name if match else f"not found in {fd.name}/",
            "exact_match": bool(match),
            "source": "ASS_FONTSDIR",
            "warning": (
                ""
                if match
                else (
                    f"ASS_FONTSDIR='{ASS_FONTSDIR}' contains no file matching "
                    f"'{font_name}' (.ttf/.otf/.ttc).  Add the font file or unset "
                    f"ASS_FONTSDIR to fall back to fontconfig."
                )
            ),
        }

    # ── Path 2: fontconfig (fc-match) ─────────────────────────────────────────
    try:
        result = subprocess.run(
            ["fc-match", "--format=%{family}", font_name],
            capture_output=True, text=True, timeout=5,
        )
        raw = result.stdout.strip().split("\n")[0].strip()
        # Localised family names are comma-separated inside the format string.
        
        resolved_names = [n.strip() for n in raw.split(",") if n.strip()]
        primary = resolved_names[0] if resolved_names else "unknown"
        exact = any(n.lower() == font_name.lower() for n in resolved_names)
        return {
            "font_name": font_name,
            "resolved": primary,
            "exact_match": exact,
            "source": "fontconfig",
            "warning": (
                ""
                if exact
                else (
                    f"fontconfig resolved '{font_name}' to '{primary}'. "
                    f"Arabic glyphs may render with a fallback font causing uneven "
                    f"letter sizes.  Fix: install '{font_name}' system-wide, or set "
                    f"ASS_FONTSDIR to a directory containing its .ttf/.otf file."
                )
            ),
        }
    except FileNotFoundError:
        return {
            "font_name": font_name,
            "resolved": "unknown",
            "exact_match": False,
            "source": "none",
            "warning": (
                "fc-match not found — cannot validate font via fontconfig.  "
                "Set ASS_FONTSDIR to a directory containing the font file to "
                "guarantee libass uses the correct Arabic font."
            ),
        }
    except Exception as e:
        return {
            "font_name": font_name,
            "resolved": "unknown",
            "exact_match": False,
            "source": "none",
            "warning": f"font validation error: {e}",
        }


def _sample_clip_brightness(source: Path, start: float, ffmpeg_bin: str) -> float:
    """
    Extract the mean luma (0–255) of a single frame from `source` at `start` seconds.

    Uses a 1×1 pixel scale to get a fast average, then reads the RGB value from
    the raw pipe and converts to luma with the Rec.601 coefficients.

    Returns 128.0 (mid-grey) on any error so callers always get a usable value.
    """
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-ss", str(start),
                "-i", str(source),
                "-vframes", "1",
                "-vf", "scale=1:1:flags=area,format=rgb24",
                "-f", "rawvideo",
                "pipe:1",
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0 or len(result.stdout) < 3:
            return 128.0
        r, g, b = result.stdout[0], result.stdout[1], result.stdout[2]
        # Rec.601 luma
        return 0.299 * r + 0.587 * g + 0.114 * b
    except Exception:
        return 128.0


def _choose_text_color(brightness: float) -> str:
    """
    Return an ASS primary_color string (&HAABBGGRR) that suits the scene
    brightness (0–255 mean luma).

    Three-zone palette matching the target recitation style:

      Zone 1  brightness < 80    → pure white    &H00FFFFFF
              Very dark scenes (night, deep shadow, black studio).
              Maximum contrast.

      Zone 2  80 ≤ brightness < 170 → warm cream  &H00D5E8F5
              Medium scenes (indoor natural light, overcast sky, golden-hour
              stone, mosque interior).  Softer than pure white — looks
              premium rather than clinical while still highly legible.
              ASS AABBGGRR: A=00, B=D5, G=E8, R=F5 → RGB(245,232,213).

      Zone 3  brightness ≥ 170   → dark grey     &H00202020
              Very bright scenes (clear sky, noon desert, white sand).
              Dark text only when necessary — avoids blinding contrast flip.
    """
    if brightness >= 170:
        return "&H00202020"   # near-black on very bright scenes
    if brightness >= 80:
        return "&H00D5E8F5"   # warm cream on medium-brightness scenes
    return "&H00FFFFFF"       # pure white on dark scenes


def _build_transition_chain(
    seg_labels: List[str],
    seg_durations: List[float],
    transition_type: str,
    transition_duration: float,
) -> List[str]:
    """
    Build xfade filter parts to join N video segment labels into [vcat].

    For N segments with a dissolve of duration T:
      [seg0][seg1]xfade=transition=dissolve:duration=T:offset=(d0-T)[xf1];
      [xf1][seg2]xfade=...offset=(d0+d1-2T)[xf2];
      ...
      [xf(N-2)][segN-1]xfade=...[vcat]

    The offset for the i-th xfade is: cumulative_duration_so_far - T
    (starts the dissolve T seconds before the cut point).

    Returns an empty list if len(seg_labels) <= 1 or transition_duration <= 0,
    so the caller can fall back to plain concat.

    The final output label is always [vcat] so no relabeling step is needed.
    """
    if len(seg_labels) <= 1 or transition_duration <= 0.0:
        return []

    parts: List[str] = []
    cumulative: float = seg_durations[0]
    prev_label: str = seg_labels[0]

    for i in range(1, len(seg_labels)):
        next_lbl = seg_labels[i]
        next_dur = seg_durations[i] if i < len(seg_durations) else transition_duration * 2

        # Cap transition so it never exceeds half of either adjacent segment.
        safe_t = min(
            transition_duration,
            seg_durations[i - 1] * 0.5 if i - 1 < len(seg_durations) else transition_duration,
            next_dur * 0.5,
        )
        safe_t = max(0.05, round(safe_t, 4))  # floor at 50 ms — avoids near-zero edge case

        offset = max(0.0, round(cumulative - safe_t, 4))
        # Final xfade outputs directly to [vcat]; intermediate ones use [xfN].
        out_lbl = "vcat" if i == len(seg_labels) - 1 else f"xf{i}"
        parts.append(
            f"[{prev_label}][{next_lbl}]"
            f"xfade=transition={transition_type}:duration={safe_t:.4f}:offset={offset:.4f}"
            f"[{out_lbl}]"
        )
        cumulative = cumulative + next_dur - safe_t
        prev_label = out_lbl

    return parts


def _enforce_speaker_hold(plan: List[dict], duration: float) -> List[dict]:
    """
    Enforce SPEAKER_HOLD_SECS on any pre-computed visual plan.

    This is the single enforcement point for all plan sources — whether the
    plan came from scene_director, clip_finder, or a manual visual_plan in
    the candidate JSON.  _middle_insert_plan already respects SPEAKER_HOLD_SECS
    internally so it does not need to go through here.

    Algorithm:
      1. Clamp each ai_video beat to [SPEAKER_HOLD_SECS, duration - SPEAKER_HOLD_SECS].
      2. Drop any ai_video beat whose clamped duration is < 0.5 s.
      3. Rebuild the full timeline, filling gaps with original-footage beats.

    The result always covers [0, duration] continuously with no gaps.
    """
    hold = SPEAKER_HOLD_SECS

    # Step 1 & 2: clamp and filter scenic insert beats (ai_video or stock_video).
    ai_beats: List[dict] = []
    for beat in plan:
        if beat["type"] != "stock_video":
            continue
        b = dict(beat)
        s = max(b["start_offset_sec"], hold)
        e = min(b["end_offset_sec"], duration - hold)
        if e - s < 0.5:
            continue   # too short after clamping; discard
        b["start_offset_sec"] = round(s, 2)
        b["end_offset_sec"] = round(e, 2)
        b["duration_sec"] = round(e - s, 2)
        ai_beats.append(b)

    if not ai_beats:
        # All inserts were removed by the hold; return full-original timeline.
        return [
            {
                "type": "original",
                "start_offset_sec": 0.0,
                "end_offset_sec": round(duration, 2),
                "duration_sec": round(duration, 2),
                "asset_slot": 0,
                "prompt": "",
                "notes": "speaker_hold_enforced_full_original",
            }
        ]

    # Step 3: rebuild the timeline by filling gaps with original beats.
    ai_beats.sort(key=lambda b: b["start_offset_sec"])
    result: List[dict] = []
    cursor = 0.0
    for ab in ai_beats:
        gap = ab["start_offset_sec"] - cursor
        if gap >= 0.2:
            result.append(
                {
                    "type": "original",
                    "start_offset_sec": round(cursor, 2),
                    "end_offset_sec": ab["start_offset_sec"],
                    "duration_sec": round(gap, 2),
                    "asset_slot": 0,
                    "prompt": "",
                    "notes": "original_segment",
                }
            )
        result.append(ab)
        cursor = ab["end_offset_sec"]

    tail = duration - cursor
    if tail >= 0.2:
        result.append(
            {
                "type": "original",
                "start_offset_sec": round(cursor, 2),
                "end_offset_sec": round(duration, 2),
                "duration_sec": round(tail, 2),
                "asset_slot": 0,
                "prompt": "",
                "notes": "original_outro",
            }
        )

    return result or [
        {
            "type": "original",
            "start_offset_sec": 0.0,
            "end_offset_sec": round(duration, 2),
            "duration_sec": round(duration, 2),
            "asset_slot": 0,
            "prompt": "",
            "notes": "speaker_hold_enforced_full_original",
        }
    ]


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

    use_boundary_suggestion = str(
        os.environ.get("RENDER_USE_BOUNDARY_SUGGESTION", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}

    if use_boundary_suggestion:
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
        "source": "candidate_json",
        "reason": "original_candidate",
        "changed": False,
        "original_start_sec": original_start,
        "original_end_sec": original_end,
    }




def _middle_insert_plan(duration: float) -> List[dict]:
    intro_sec = min(max(SPEAKER_HOLD_SECS, 2.5), max(1.2, duration - 1.2))

    if duration - intro_sec < 1.2:
        return [
            {
                "type": "original",
                "start_offset_sec": 0.0,
                "end_offset_sec": round(duration, 2),
                "duration_sec": round(duration, 2),
                "asset_slot": 0,
                "prompt": "",
                "notes": "full_original_fallback",
            }
        ]

    return [
        {
            "type": "original",
            "start_offset_sec": 0.0,
            "end_offset_sec": round(intro_sec, 2),
            "duration_sec": round(intro_sec, 2),
            "asset_slot": 0,
            "prompt": "",
            "notes": "speaker_intro",
        },
        {
            "type": "stock_video",
            "start_offset_sec": round(intro_sec, 2),
            "end_offset_sec": round(duration, 2),
            "duration_sec": round(duration - intro_sec, 2),
            "asset_slot": 1,
            "prompt": "",
            "notes": "stock_takeover_full_remainder",
        },
    ]

def _normalize_visual_plan(plan: List[dict], duration: float) -> List[dict]:
    beats: List[dict] = []
    for beat in plan or []:
        beat_type = str(beat.get("type") or "").lower().strip()
        if beat_type not in {"original", "stock_video"}:
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


def _find_stock_asset(stem: str, clip_number: int, asset_slot: int) -> Optional[Path]:
    prefix = f"{_slug(stem)}__clip{clip_number:02d}__ai{asset_slot:02d}"
    for ext in VIDEO_EXTS:
        exact = BROLL / f"{prefix}__stock{ext}"
        if exact.exists():
            return exact
    return None


def _build_visual_plan_for_render(clip: dict, duration: float, broll_file: Optional[str]) -> List[dict]:
    visual_plan = clip.get("visual_plan") or []
    if visual_plan:
        # _normalize_visual_plan validates and sorts; _enforce_speaker_hold
        # then clamps any ai_video beat that starts too early (e.g. from
        # scene_director using a different intro_guard value than SPEAKER_HOLD_SECS).
        plan = _normalize_visual_plan(visual_plan, duration)
        return _enforce_speaker_hold(plan, duration)

    if broll_file:
        # _middle_insert_plan already uses SPEAKER_HOLD_SECS internally.
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

def _safe_round3(value: float) -> float:
    try:
        return round(float(value), 3)
    except Exception:
        return 0.0


def _visual_plan_total_duration(plan: List[dict]) -> float:
    total = 0.0
    for beat in plan or []:
        total += max(0.0, _safe_float(beat.get("duration_sec"), 0.0))
    return _safe_round3(total)


def _visual_overlap_duration(
    seg_durations: List[float],
    transition_duration: float,
) -> float:
    if len(seg_durations) <= 1 or transition_duration <= 0.0:
        return 0.0

    overlap = 0.0
    for i in range(1, len(seg_durations)):
        prev_dur = seg_durations[i - 1] if i - 1 < len(seg_durations) else transition_duration
        next_dur = seg_durations[i] if i < len(seg_durations) else transition_duration
        safe_t = min(
            transition_duration,
            prev_dur * 0.5,
            next_dur * 0.5,
        )
        safe_t = max(0.05, round(safe_t, 4))
        overlap += safe_t
    return _safe_round3(overlap)


def _expected_visual_output_duration(
    plan: List[dict],
    transition_duration: float,
) -> float:
    seg_durations = [max(0.0, _safe_float(b.get("duration_sec"), 0.0)) for b in (plan or [])]
    total = sum(seg_durations)
    overlap = _visual_overlap_duration(seg_durations, transition_duration)
    return _safe_round3(total - overlap)



def _build_visual_timeline_filter(
    visual_plan: List[dict],
    start: float,
    duration: float,
    preset: str,
    audio_only: bool,
    ai_input_index_map: Dict[int, int],
    transition_type: str = "dissolve",
    transition_duration: float = 0.0,
) -> Tuple[str, List[str]]:
    """
    Build the video filter_complex chain for the visual timeline.

    When transition_duration > 0, consecutive segments are joined with
    xfade filters (smooth dissolve by default) instead of a hard concat cut.
    When transition_duration == 0 or there is only one segment, a plain
    concat is used — identical to the previous behaviour.

    Each segment is trimmed, scaled to 1080×1920, colour-graded with `preset`,
    and given the `vsegN` label.  The final composite is labelled [vcat].
    """
    seg_parts: List[str] = []   # per-segment filter chains
    seg_labels: List[str] = []  # vseg0, vseg1, …
    seg_durations: List[float] = []
    used_assets: List[str] = []

    for idx, beat in enumerate(visual_plan):
        label = f"vseg{idx}"
        beat_type = beat["type"]
        seg_start = _safe_float(beat["start_offset_sec"])
        seg_end = _safe_float(beat["end_offset_sec"])
        seg_dur = _safe_float(beat["duration_sec"])

        if beat_type == "original":
            if audio_only:
                trim_start, trim_end = seg_start, seg_end
            else:
                trim_start, trim_end = start + seg_start, start + seg_end
            seg_parts.append(_build_video_segment_chain("[0:v]", trim_start, trim_end, preset, label))
        else:
            slot = int(beat.get("asset_slot") or 0)

            if slot < 1 or slot not in ai_input_index_map:
                label = f"vfallback{idx}"

                if ai_input_index_map:
                    # Never fall back to black/original when we have at least one real scenic asset.
                    fallback_slot = sorted(ai_input_index_map.keys())[0]
                    input_idx = ai_input_index_map[fallback_slot]
                    used_assets.append(f"fallback_slot:{fallback_slot}@input:{input_idx}")
                    seg_parts.append(
                        _build_video_segment_chain(
                            f"[{input_idx}:v]",
                            0.0,
                            seg_dur,
                            preset,
                            label,
                        )
                    )
                else:
                    # If there is truly no scenic asset at all, only then use original footage.
                    # This avoids black gaps caused by partially missing stock windows.
                    if audio_only:
                        trim_start, trim_end = seg_start, seg_end
                    else:
                        trim_start, trim_end = start + seg_start, start + seg_end
                    seg_parts.append(
                        _build_video_segment_chain("[0:v]", trim_start, trim_end, preset, label)
                    )
            else:
                input_idx = ai_input_index_map[slot]
                used_assets.append(f"slot:{slot}@input:{input_idx}")
                seg_parts.append(
                    _build_video_segment_chain(
                        f"[{input_idx}:v]",
                        0.0,
                        beat["duration_sec"],
                        preset,
                        label,
                    )
                )
        seg_labels.append(label)
        seg_durations.append(seg_dur)

    # Join segments — xfade (smooth dissolve) or concat (hard cut).
    xfade_parts = _build_transition_chain(
        seg_labels, seg_durations, transition_type, transition_duration
    )
    if xfade_parts:
        # xfade chain — final filter already outputs [vcat].
        all_parts = seg_parts + xfade_parts
    else:
        labels_str = "".join(f"[{lbl}]" for lbl in seg_labels)
        all_parts = seg_parts + [f"{labels_str}concat=n={len(seg_labels)}:v=1:a=0[vcat]"]

    return ";".join(all_parts), used_assets


def _render_one(
    stem: str,
    clip_number: int,
    preset: str = "dark-soft-recitation",
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

    # ── Resolve visual plan and ai_paths FIRST so we can pass them to
    # _make_clip_ass_tempfile for worst-case brightness sampling. ─────────────
    visual_plan = _build_visual_plan_for_render(clip=clip, duration=duration, broll_file=broll_file)

    ai_paths: Dict[int, Path] = {}
    if auto_broll:
        for beat in visual_plan:
            if beat["type"] != "stock_video":
                continue
            slot = int(beat.get("asset_slot") or 0)
            if slot < 1 or slot in ai_paths:
                continue
            asset = _find_stock_asset(stem, clip_number, slot)
            if asset:
                ai_paths[slot] = asset

    if broll_file:
        selected = Path(broll_file)
        if not selected.is_absolute():
            selected = BROLL / broll_file
        if not selected.exists():
            raise FileNotFoundError(f"B-roll file not found: {selected}")
        ai_paths[1] = selected

    # ── Subtitle generation ───────────────────────────────────────────────────
    ass_tempfile: Optional[Path] = None
    subtitle_filter_str = ""
    subtitle_source = ""
    text_meta: dict = {}

    if burn_subtitles:
        try:
            ass_tempfile, text_meta = _make_clip_ass_tempfile(
                stem, clip_number, start, end,
                source_video=source_video,
                ai_paths=ai_paths,
            )
        except Exception as e:
            subtitle_source = f"subtitle generation failed: {e}"
            text_meta = {}
        if ass_tempfile:
            escaped = _escape_filter_path(ass_tempfile)
            fontsdir_part = (
                f":fontsdir='{_escape_filter_path(Path(ASS_FONTSDIR))}'"
                if ASS_FONTSDIR else ""
            )
            subtitle_filter_str = f"ass=filename='{escaped}'{fontsdir_part}:shaping=complex"
            subtitle_source = str(ass_tempfile)
        elif not subtitle_source:
            subtitle_source = "no transcript found for clip"

    ffmpeg = get_ffmpeg()
    used_broll = any(beat["type"] == "stock_video" for beat in visual_plan) and bool(ai_paths)

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
        transition_type=RENDER_TRANSITION_TYPE,
        transition_duration=RENDER_TRANSITION_DURATION,
    )

    visual_plan_total = _visual_plan_total_duration(visual_plan)
    visual_expected_output = _expected_visual_output_duration(
        visual_plan,
        RENDER_TRANSITION_DURATION,
    )
    visual_expected_overlap = _safe_round3(
        visual_plan_total - visual_expected_output
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
            "timing_debug": {
                "clip_start_sec": start,
                "clip_end_sec": end,
                "clip_duration_sec": duration,
                "audio_trim_start_sec": start,
                "audio_trim_end_sec": end,
                "audio_expected_duration_sec": duration,
                "visual_plan_total_duration_sec": visual_plan_total,
                "visual_expected_output_duration_sec": visual_expected_output,
                "visual_expected_overlap_loss_sec": visual_expected_overlap,
                "transition_type": RENDER_TRANSITION_TYPE,
                "transition_duration_sec": RENDER_TRANSITION_DURATION,
                "segment_count": len(visual_plan),
            },
        }

    effective_candidate = dict(clip)
    effective_candidate["effective_start_sec"] = start
    effective_candidate["effective_end_sec"] = end
    effective_candidate["effective_duration_sec"] = duration

    xfade_used = RENDER_TRANSITION_DURATION > 0 and len(visual_plan) > 1

    return {
        "ok": True,
        "output_file": str(out_path),
        "preset": preset,
        "burn_subtitles": burn_subtitles,
        "subtitle_source": subtitle_source,
        # text_style_mode confirms which style config was active at render time.
        # "center_recitation" = centered large Arabic text (cinematic mode).
        # "subtitle"          = bottom-center standard subtitle mode.
        "text_style_mode": _text_cfg.mode,
        "text_visibility": text_meta,
        "font_validation": _validate_font(_text_cfg.font),
        "xfade_used": xfade_used,
        "used_broll": used_broll,
        "selected_broll": [str(path) for _, path in sorted(ai_paths.items())],
        "ffmpeg_binary": ffmpeg,
        "candidate": effective_candidate,
        "timing": timing_meta,
        "visual_plan": visual_plan,
        "used_assets": used_assets,
        "timing_debug": {
            "clip_start_sec": start,
            "clip_end_sec": end,
            "clip_duration_sec": duration,
            "audio_trim_start_sec": start,
            "audio_trim_end_sec": end,
            "audio_expected_duration_sec": duration,
            "visual_plan_total_duration_sec": visual_plan_total,
            "visual_expected_output_duration_sec": visual_expected_output,
            "visual_expected_overlap_loss_sec": visual_expected_overlap,
            "transition_type": RENDER_TRANSITION_TYPE,
            "transition_duration_sec": RENDER_TRANSITION_DURATION,
            "segment_count": len(visual_plan),
        },
    }


# ── Startup font check ────────────────────────────────────────────────────────
# Runs once when the server loads, after all helper functions are defined.
# Emits a visible stderr warning when fontconfig cannot find the configured
# Arabic font — this is the most common silent failure mode on Linux/Docker.
# Set ASS_FONT_STRICT=true to abort startup instead of just warning.
_startup_font_check = _validate_font(_text_cfg.font)
if not _startup_font_check["exact_match"]:
    import sys as _sys
    _warn = _startup_font_check.get("warning", f"Font '{_text_cfg.font}' not found.")
    print(
        f"\nWARNING[clip-factory-renderer]: {_warn}\n"
        f"Arabic glyphs will render with the fallback font and may appear misshapen.\n"
        f"To abort on startup instead of warning: set ASS_FONT_STRICT=true in .env.\n",
        file=_sys.stderr,
        flush=True,
    )
    if os.environ.get("ASS_FONT_STRICT", "").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(
            f"Font '{_text_cfg.font}' not available "
            f"(source={_startup_font_check['source']}, "
            f"resolved='{_startup_font_check['resolved']}'). "
            f"Set ASS_FONTSDIR or unset ASS_FONT_STRICT to allow fallback."
        )


@mcp.tool()
def list_filter_presets() -> dict:
    return {
        "presets": [
            {"name": "clean-warm", "description": "Balanced warm look"},
            {"name": "cinematic-soft", "description": "Softer cinematic contrast"},
            {"name": "high-contrast", "description": "Punchier separation"},
            {"name": "golden-islamic", "description": "Warm reflective Islamic tone"},
            {
                "name": "dark-soft-recitation",
                "description": (
                    "Dark cinematic grade for center_recitation mode: lowers exposure, "
                    "desaturates slightly, adds cool-neutral balance and gentle edge sharpen. "
                    "Creates a dark backdrop that makes white Arabic text pop without "
                    "color competing with the recitation."
                ),
            },
        ],
        "transition_config": {
            "RENDER_TRANSITION_TYPE": RENDER_TRANSITION_TYPE,
            "RENDER_TRANSITION_DURATION": RENDER_TRANSITION_DURATION,
            "SPEAKER_HOLD_SECS": SPEAKER_HOLD_SECS,
            "ADAPTIVE_TEXT_COLOR": ADAPTIVE_TEXT_COLOR,
        },
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
        if beat["type"] != "stock_video":
            continue
        slot = int(beat.get("asset_slot") or 0)
        asset = _find_stock_asset(stem, clip_number, slot)
        assets.append(
            {
                "asset_slot": slot,
                "prompt": beat.get("prompt", ""),
                "expected_prefix": str(BROLL / f"{_slug(stem)}__clip{clip_number:02d}__ai{slot:02d}") + "__stock.*",
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
    preset: str = "dark-soft-recitation",
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

def _verse_key_sort(key: str):
    return tuple(map(int, key.split(":")))


@mcp.tool()
def batch_render_from_candidates(
    stem: str,
    clip_numbers: Optional[List[int]] = None,
    preset: str = "dark-soft-recitation",
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
