from pathlib import Path
import os
import json
import shlex
import subprocess
from typing import Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from helpers import get_ffmpeg, atomic_write_json

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()
LOOKLAB = ROOT / "looklab"
PRESETS_DIR = LOOKLAB / "presets"
PREVIEWS_DIR = LOOKLAB / "previews"
RENDERS_DIR = LOOKLAB / "renders"

load_dotenv(ROOT / ".env")

mcp = FastMCP("clip-factory-look-lab", json_response=True)

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_FOLDERS = [ROOT / "incoming", ROOT / "broll", ROOT / "final", ROOT / "clips", ROOT]

BASE_PRESETS: Dict[str, Dict[str, float | str]] = {
    "clean-warm": {
        "contrast": 1.08,
        "brightness": 0.02,
        "saturation": 1.10,
        "warmth": 0.02,
        "sharpness": 0.35,
        "grain": 0.00,
        "vignette": 0.00,
        "glow": 0.00,
        "notes": "Clean bright social look with mild warmth.",
    },
    "cinematic-soft": {
        "contrast": 1.12,
        "brightness": 0.02,
        "saturation": 1.06,
        "warmth": 0.01,
        "sharpness": 0.55,
        "grain": 0.02,
        "vignette": 0.08,
        "glow": 0.04,
        "notes": "Soft premium contrast for reflective shorts.",
    },
    "deep-night": {
        "contrast": 1.16,
        "brightness": -0.02,
        "saturation": 0.96,
        "warmth": -0.02,
        "sharpness": 0.40,
        "grain": 0.03,
        "vignette": 0.12,
        "glow": 0.02,
        "notes": "Moody low-light look for night scenes.",
    },
    "desert-gold": {
        "contrast": 1.10,
        "brightness": 0.03,
        "saturation": 1.12,
        "warmth": 0.05,
        "sharpness": 0.45,
        "grain": 0.02,
        "vignette": 0.06,
        "glow": 0.03,
        "notes": "Warm golden tone for desert and dawn visuals.",
    },
    "emerald-teal": {
        "contrast": 1.14,
        "brightness": 0.00,
        "saturation": 1.04,
        "warmth": -0.03,
        "sharpness": 0.50,
        "grain": 0.02,
        "vignette": 0.08,
        "glow": 0.03,
        "notes": "Cool premium look for ocean, rain, and blue-hour shots.",
    },
}


def _ensure_dirs() -> None:
    for folder in [LOOKLAB, PRESETS_DIR, PREVIEWS_DIR, RENDERS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def _safe_slug(value: str) -> str:
    out = []
    for ch in str(value or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "_", "-", "."}:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "asset"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_custom_presets() -> Dict[str, Dict[str, float | str]]:
    _ensure_dirs()
    out: Dict[str, Dict[str, float | str]] = {}
    for path in PRESETS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = str(data.get("name") or path.stem).strip()
            if name:
                out[name] = data
        except Exception:
            continue
    return out


def _all_presets() -> Dict[str, Dict[str, float | str]]:
    presets = dict(BASE_PRESETS)
    presets.update(_load_custom_presets())
    return presets


def _resolve_input_path(source_path: str) -> Path:
    raw = str(source_path or "").strip()
    if not raw:
        raise ValueError("source_path is required")

    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()

    candidate = (ROOT / raw).resolve()
    if candidate.exists():
        return candidate

    for folder in MEDIA_FOLDERS:
        probe = folder / raw
        if probe.exists():
            return probe.resolve()

    raise FileNotFoundError(f"Could not find media path: {source_path}")


def _media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    raise ValueError(f"Unsupported media type for {path.name}")


def _ffprobe_json(path: Path) -> dict:
    ffmpeg_path = Path(get_ffmpeg())
    ffprobe_candidate = ffmpeg_path.with_name("ffprobe")
    ffprobe = str(ffprobe_candidate) if ffprobe_candidate.exists() else "ffprobe"
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")
    return json.loads(result.stdout or "{}")


def _duration_sec(path: Path) -> float:
    info = _ffprobe_json(path)
    fmt = info.get("format") or {}
    return _safe_float(fmt.get("duration"), 0.0)


def _has_audio(path: Path) -> bool:
    info = _ffprobe_json(path)
    for stream in info.get("streams") or []:
        if stream.get("codec_type") == "audio":
            return True
    return False


def _preset_values(name: str) -> Dict[str, float | str]:
    presets = _all_presets()
    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {sorted(presets)}")
    return dict(presets[name])


def _build_filter_chain(
    preset: str,
    *,
    grain_strength: float = 0.0,
    vignette_strength: float = 0.0,
    glow_strength: float = 0.0,
    warmth_shift: float = 0.0,
    saturation_boost: float = 0.0,
    contrast_boost: float = 0.0,
) -> tuple[str, Dict[str, float | str]]:
    p = _preset_values(preset)

    contrast = _clamp(_safe_float(p.get("contrast"), 1.0) + contrast_boost, 0.6, 1.8)
    brightness = _clamp(_safe_float(p.get("brightness"), 0.0), -0.3, 0.3)
    saturation = _clamp(_safe_float(p.get("saturation"), 1.0) + saturation_boost, 0.2, 2.0)
    warmth = _clamp(_safe_float(p.get("warmth"), 0.0) + warmth_shift, -0.25, 0.25)
    sharpness = _clamp(_safe_float(p.get("sharpness"), 0.0), 0.0, 2.0)
    grain = _clamp(_safe_float(p.get("grain"), 0.0) + grain_strength, 0.0, 0.2)
    vignette = _clamp(_safe_float(p.get("vignette"), 0.0) + vignette_strength, 0.0, 0.45)
    glow = _clamp(_safe_float(p.get("glow"), 0.0) + glow_strength, 0.0, 0.25)

    filters: List[str] = [f"eq=contrast={contrast:.3f}:brightness={brightness:.3f}:saturation={saturation:.3f}"]

    if abs(warmth) > 0.001:
        rs = _clamp(warmth, -0.2, 0.2)
        bs = _clamp(-warmth * 0.75, -0.2, 0.2)
        filters.append(f"colorbalance=rs={rs:.3f}:gs=0.000:bs={bs:.3f}")

    if sharpness > 0.001:
        filters.append(f"unsharp=5:5:{sharpness:.3f}:5:5:0.0")

    if glow > 0.001:
        filters.append(
            f"split=2[base][blur];[blur]gblur=sigma={2.0 + glow * 8.0:.2f}[soft];"
            f"[base][soft]blend=all_mode=screen:all_opacity={0.10 + glow * 0.55:.3f}"
        )

    if grain > 0.001:
        strength = int(round(3 + grain * 80))
        filters.append(f"noise=alls={strength}:allf=t")

    if vignette > 0.001:
        angle = _clamp(vignette * 2.4, 0.05, 1.2)
        filters.append(f"vignette=angle={angle:.3f}")

    filter_chain = ",".join(filters)
    settings = {
        "preset": preset,
        "contrast": contrast,
        "brightness": brightness,
        "saturation": saturation,
        "warmth": warmth,
        "sharpness": sharpness,
        "grain": grain,
        "vignette": vignette,
        "glow": glow,
        "filter_chain": filter_chain,
        "notes": p.get("notes", ""),
    }
    return filter_chain, settings


def _run_ffmpeg(cmd: List[str]) -> dict:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "command": " ".join(shlex.quote(part) for part in cmd),
    }


def _default_output_name(path: Path, suffix: str, ext: str = ".mp4") -> str:
    return f"{_safe_slug(path.stem)}{suffix}{ext}"


def _write_sidecar(output_path: Path, payload: dict) -> Path:
    sidecar = output_path.with_suffix(output_path.suffix + ".look.json")
    atomic_write_json(sidecar, payload)
    return sidecar


@mcp.tool()
def healthcheck() -> dict:
    """Return a simple health check for look-lab."""
    _ensure_dirs()
    ffmpeg = get_ffmpeg()
    ffmpeg_path = Path(ffmpeg)
    ffprobe_candidate = ffmpeg_path.with_name("ffprobe")
    ffprobe = str(ffprobe_candidate) if ffprobe_candidate.exists() else "ffprobe"
    return {
        "ok": True,
        "root": str(ROOT),
        "looklab_exists": LOOKLAB.exists(),
        "presets_dir": str(PRESETS_DIR),
        "previews_dir": str(PREVIEWS_DIR),
        "renders_dir": str(RENDERS_DIR),
        "ffmpeg": ffmpeg,
        "ffprobe_guess": ffprobe,
        "preset_count": len(_all_presets()),
    }


@mcp.tool()
def list_presets() -> dict:
    """List built-in and custom look presets."""
    presets = _all_presets()
    return {
        "folder": str(PRESETS_DIR),
        "presets": [
            {
                "name": name,
                "notes": str(payload.get("notes", "")),
                "contrast": _safe_float(payload.get("contrast"), 1.0),
                "brightness": _safe_float(payload.get("brightness"), 0.0),
                "saturation": _safe_float(payload.get("saturation"), 1.0),
                "warmth": _safe_float(payload.get("warmth"), 0.0),
                "grain": _safe_float(payload.get("grain"), 0.0),
                "vignette": _safe_float(payload.get("vignette"), 0.0),
                "glow": _safe_float(payload.get("glow"), 0.0),
            }
            for name, payload in sorted(presets.items())
        ],
    }


@mcp.tool()
def inspect_media(source_path: str) -> dict:
    """Inspect a media file and return ffprobe metadata."""
    path = _resolve_input_path(source_path)
    info = _ffprobe_json(path)
    return {
        "path": str(path),
        "kind": _media_kind(path),
        "duration_sec": _safe_float((info.get("format") or {}).get("duration"), 0.0),
        "has_audio": _has_audio(path) if _media_kind(path) == "video" else False,
        "ffprobe": info,
    }


@mcp.tool()
def create_viral_look_preset(
    name: str,
    base_preset: str = "cinematic-soft",
    contrast: Optional[float] = None,
    brightness: Optional[float] = None,
    saturation: Optional[float] = None,
    warmth: Optional[float] = None,
    sharpness: Optional[float] = None,
    grain: Optional[float] = None,
    vignette: Optional[float] = None,
    glow: Optional[float] = None,
    notes: str = "",
) -> dict:
    """Create and save a reusable custom look preset."""
    _ensure_dirs()
    base = _preset_values(base_preset)
    payload = {
        "name": name,
        "base_preset": base_preset,
        "contrast": _safe_float(contrast, _safe_float(base.get("contrast"), 1.0)),
        "brightness": _safe_float(brightness, _safe_float(base.get("brightness"), 0.0)),
        "saturation": _safe_float(saturation, _safe_float(base.get("saturation"), 1.0)),
        "warmth": _safe_float(warmth, _safe_float(base.get("warmth"), 0.0)),
        "sharpness": _safe_float(sharpness, _safe_float(base.get("sharpness"), 0.0)),
        "grain": _safe_float(grain, _safe_float(base.get("grain"), 0.0)),
        "vignette": _safe_float(vignette, _safe_float(base.get("vignette"), 0.0)),
        "glow": _safe_float(glow, _safe_float(base.get("glow"), 0.0)),
        "notes": notes or base.get("notes", ""),
    }
    path = PRESETS_DIR / f"{_safe_slug(name)}.json"
    atomic_write_json(path, payload)
    return {
        "ok": True,
        "preset_path": str(path),
        "preset": payload,
    }


@mcp.tool()
def preview_grade(
    source_path: str,
    preset: str = "cinematic-soft",
    start_sec: float = 0.0,
    duration_sec: float = 6.0,
    grain_strength: float = 0.0,
    vignette_strength: float = 0.0,
    glow_strength: float = 0.0,
    warmth_shift: float = 0.0,
    saturation_boost: float = 0.0,
    contrast_boost: float = 0.0,
    output_name: str = "",
) -> dict:
    """Render a short graded preview clip for quick look testing."""
    _ensure_dirs()
    path = _resolve_input_path(source_path)
    if _media_kind(path) != "video":
        raise ValueError("preview_grade currently expects a video input")

    total_duration = _duration_sec(path)
    start_sec = _clamp(_safe_float(start_sec), 0.0, max(0.0, total_duration))
    duration_sec = _clamp(_safe_float(duration_sec, 6.0), 1.0, 20.0)
    if start_sec + duration_sec > total_duration and total_duration > 0:
        duration_sec = max(1.0, total_duration - start_sec)

    filter_chain, settings = _build_filter_chain(
        preset,
        grain_strength=grain_strength,
        vignette_strength=vignette_strength,
        glow_strength=glow_strength,
        warmth_shift=warmth_shift,
        saturation_boost=saturation_boost,
        contrast_boost=contrast_boost,
    )

    out_name = output_name.strip() or _default_output_name(path, f"__preview__{_safe_slug(preset)}")
    output_path = PREVIEWS_DIR / out_name
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(path),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
    ]
    if _has_audio(path):
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd.append(str(output_path))

    result = _run_ffmpeg(cmd)
    if not result["ok"]:
        return result

    sidecar = _write_sidecar(output_path, {
        "tool": "preview_grade",
        "source_path": str(path),
        "output_path": str(output_path),
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "settings": settings,
    })
    return {
        "ok": True,
        "source_path": str(path),
        "output_path": str(output_path),
        "sidecar_path": str(sidecar),
        "duration_sec": duration_sec,
        "settings": settings,
    }


@mcp.tool()
def apply_look_preset(
    source_path: str,
    preset: str = "cinematic-soft",
    grain_strength: float = 0.0,
    vignette_strength: float = 0.0,
    glow_strength: float = 0.0,
    warmth_shift: float = 0.0,
    saturation_boost: float = 0.0,
    contrast_boost: float = 0.0,
    output_name: str = "",
) -> dict:
    """Apply a full look preset to a video or image and save the graded result."""
    _ensure_dirs()
    path = _resolve_input_path(source_path)
    kind = _media_kind(path)
    filter_chain, settings = _build_filter_chain(
        preset,
        grain_strength=grain_strength,
        vignette_strength=vignette_strength,
        glow_strength=glow_strength,
        warmth_shift=warmth_shift,
        saturation_boost=saturation_boost,
        contrast_boost=contrast_boost,
    )

    if kind == "video":
        ext = ".mp4"
    else:
        ext = path.suffix.lower()

    out_name = output_name.strip() or _default_output_name(path, f"__{_safe_slug(preset)}", ext=ext)
    output_path = RENDERS_DIR / out_name
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vf",
        filter_chain,
    ]

    if kind == "video":
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
        ]
        if _has_audio(path):
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
    else:
        cmd += ["-frames:v", "1"]

    cmd.append(str(output_path))
    result = _run_ffmpeg(cmd)
    if not result["ok"]:
        return result

    sidecar = _write_sidecar(output_path, {
        "tool": "apply_look_preset",
        "source_path": str(path),
        "output_path": str(output_path),
        "kind": kind,
        "settings": settings,
    })
    return {
        "ok": True,
        "source_path": str(path),
        "output_path": str(output_path),
        "sidecar_path": str(sidecar),
        "kind": kind,
        "settings": settings,
    }


@mcp.tool()
def add_film_grain(source_path: str, strength: float = 0.03, output_name: str = "") -> dict:
    """Convenience wrapper to add subtle film grain."""
    return apply_look_preset(
        source_path=source_path,
        preset="cinematic-soft",
        grain_strength=_clamp(strength, 0.0, 0.2),
        output_name=output_name,
    )


@mcp.tool()
def add_vignette(source_path: str, strength: float = 0.08, output_name: str = "") -> dict:
    """Convenience wrapper to add a subtle vignette."""
    return apply_look_preset(
        source_path=source_path,
        preset="cinematic-soft",
        vignette_strength=_clamp(strength, 0.0, 0.35),
        output_name=output_name,
    )


@mcp.tool()
def apply_glow(source_path: str, strength: float = 0.05, output_name: str = "") -> dict:
    """Convenience wrapper to add a soft cinematic glow."""
    return apply_look_preset(
        source_path=source_path,
        preset="cinematic-soft",
        glow_strength=_clamp(strength, 0.0, 0.25),
        output_name=output_name,
    )


@mcp.tool()
def match_broll_to_main(
    main_source_path: str,
    broll_source_path: str,
    preset: str = "",
    extra_warmth_shift: float = 0.0,
    extra_saturation_boost: float = 0.0,
    extra_contrast_boost: float = 0.0,
    output_name: str = "",
) -> dict:
    """Apply a shared look to a scenic insert so it matches the main footage better."""
    main_path = _resolve_input_path(main_source_path)
    broll_path = _resolve_input_path(broll_source_path)

    chosen_preset = preset.strip()
    inferred_settings = None
    if not chosen_preset:
        sidecar_candidates = [
            main_path.with_suffix(main_path.suffix + ".look.json"),
            RENDERS_DIR / f"{main_path.stem}__cinematic-soft.mp4.look.json",
            RENDERS_DIR / f"{main_path.stem}__desert-gold.mp4.look.json",
            RENDERS_DIR / f"{main_path.stem}__emerald-teal.mp4.look.json",
        ]
        for candidate in sidecar_candidates:
            if candidate.exists():
                try:
                    inferred_settings = json.loads(candidate.read_text(encoding="utf-8"))
                    settings = inferred_settings.get("settings") or {}
                    chosen_preset = str(settings.get("preset") or "").strip()
                    if chosen_preset:
                        break
                except Exception:
                    continue

    if not chosen_preset:
        chosen_preset = "cinematic-soft"

    result = apply_look_preset(
        source_path=str(broll_path),
        preset=chosen_preset,
        warmth_shift=extra_warmth_shift,
        saturation_boost=extra_saturation_boost,
        contrast_boost=extra_contrast_boost,
        output_name=output_name,
    )
    if not result.get("ok"):
        return result

    return {
        "ok": True,
        "main_source_path": str(main_path),
        "broll_source_path": str(broll_path),
        "matched_output_path": result.get("output_path"),
        "sidecar_path": result.get("sidecar_path"),
        "chosen_preset": chosen_preset,
        "inferred_from_main": bool(inferred_settings),
        "settings": result.get("settings"),
    }


if __name__ == "__main__":
    mcp.run()
