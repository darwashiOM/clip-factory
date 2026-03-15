from pathlib import Path
import os
import json
import subprocess
import tempfile
from typing import Optional, List, Tuple, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from helpers import get_ffmpeg, atomic_write_json

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()
BROLL = ROOT / "broll"
FINAL = ROOT / "final"

load_dotenv(ROOT / ".env")

mcp = FastMCP("asset-guard", json_response=True)

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


class DetectionDecision(BaseModel):
    detected: bool = Field(description="Whether the target issue was detected")
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
    evidence: List[str] = Field(default_factory=list, description="Short evidence bullets")


class LowQualityResult(BaseModel):
    detected: bool
    confidence: float = Field(ge=0, le=1)
    issues: List[str] = Field(default_factory=list)
    technical_score: int = Field(ge=0, le=100)


class ScenicResult(BaseModel):
    rejected: bool
    confidence: float = Field(ge=0, le=1)
    reasons: List[str] = Field(default_factory=list)


class CinematicQualityResult(BaseModel):
    score: int = Field(ge=0, le=100)
    reasons: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


class VisionInspection(BaseModel):
    human_faces: DetectionDecision
    animal_faces: DetectionDecision
    text_or_logo: DetectionDecision
    visual_artifacts: DetectionDecision
    low_quality_generation: LowQualityResult
    non_scenic: ScenicResult
    cinematic_quality: CinematicQualityResult
    summary: str = ""


class TechnicalInfo(BaseModel):
    path: str
    exists: bool
    media_type: str
    width: int = 0
    height: int = 0
    duration_sec: float = 0.0
    frame_rate: float = 0.0
    file_size_bytes: int = 0
    portrait: bool = False
    technical_flags: List[str] = Field(default_factory=list)
    technical_score: int = Field(ge=0, le=100, default=0)


class AssetGuardReport(BaseModel):
    ok: bool
    source_path: str
    report_file: str = ""
    model: str = ""
    technical: TechnicalInfo
    frame_samples: List[str] = Field(default_factory=list)
    vision: VisionInspection
    auto_reject: bool = False
    auto_reject_reasons: List[str] = Field(default_factory=list)


def _client():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment or .env")
    return genai.Client(api_key=api_key)


def _vision_model_name() -> str:
    return (
        os.environ.get("GEMINI_VISION_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-2.5-pro"
    )


def _resolve_media_path(source_path: str) -> Path:
    p = Path(str(source_path or "")).expanduser()
    if p.is_absolute() and p.exists():
        return p
    for base in [ROOT, BROLL, FINAL]:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Media file not found: {source_path}")


def _list_media(folder: Path, limit: int = 50):
    if not folder.exists():
        return []
    files = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
        }
        for p in files[:limit]
    ]


def _ffprobe_json(path: Path) -> dict:
    ffmpeg_path = Path(get_ffmpeg())
    ffprobe_candidate = ffmpeg_path.with_name("ffprobe")
    ffprobe = str(ffprobe_candidate) if ffprobe_candidate.exists() else "ffprobe"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return json.loads(result.stdout or "{}")


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in IMAGE_EXTS:
        return "image"
    return "unknown"


def _parse_frame_rate(rate_str: str) -> float:
    text = str(rate_str or "0/0")
    if "/" in text:
        a, b = text.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            return 0.0 if den == 0 else round(num / den, 3)
        except Exception:
            return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _technical_info(path: Path) -> TechnicalInfo:
    info = _ffprobe_json(path)
    streams = info.get("streams") or []
    format_info = info.get("format") or {}
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    duration = float(video_stream.get("duration") or format_info.get("duration") or 0.0)
    fps = _parse_frame_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/0")
    size_bytes = int(format_info.get("size") or path.stat().st_size)
    media_type = _guess_media_type(path)
    portrait = height > width if width and height else False

    flags: List[str] = []
    score = 100

    if width <= 0 or height <= 0:
        flags.append("could_not_read_dimensions")
        score -= 30
    else:
        if max(width, height) < 720:
            flags.append("resolution_below_720")
            score -= 25
        elif max(width, height) < 1080:
            flags.append("resolution_below_1080")
            score -= 10

        if media_type == "video" and not portrait:
            flags.append("not_portrait")
            score -= 10

    if media_type == "video":
        if duration <= 0:
            flags.append("duration_missing")
            score -= 20
        elif duration < 1.2:
            flags.append("very_short_video")
            score -= 15
        elif duration > 12.0:
            flags.append("long_for_insert")
            score -= 5

        if fps and fps < 20:
            flags.append("low_frame_rate")
            score -= 15

    if size_bytes < 120_000:
        flags.append("small_file_size")
        score -= 10

    score = max(0, min(100, score))

    return TechnicalInfo(
        path=str(path),
        exists=path.exists(),
        media_type=media_type,
        width=width,
        height=height,
        duration_sec=round(duration, 3),
        frame_rate=fps,
        file_size_bytes=size_bytes,
        portrait=portrait,
        technical_flags=flags,
        technical_score=score,
    )


def _sample_times(duration_sec: float, max_frames: int) -> List[float]:
    if duration_sec <= 0:
        return [0.0]
    if duration_sec <= 2.0:
        return [0.0, max(0.0, duration_sec * 0.5)]
    count = max(2, min(max_frames, 6))
    step = duration_sec / (count + 1)
    times = [round(step * i, 3) for i in range(1, count + 1)]
    return sorted(set(max(0.0, min(duration_sec - 0.05, t)) for t in times))


def _extract_sample_frames(path: Path, technical: TechnicalInfo, max_frames: int = 4) -> List[Path]:
    ffmpeg = get_ffmpeg()
    media_type = technical.media_type

    if media_type == "image":
        return [path]

    if media_type != "video":
        raise ValueError(f"Unsupported media type for asset guard: {path.suffix}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="asset_guard_frames_"))
    out_paths: List[Path] = []
    for idx, t in enumerate(_sample_times(technical.duration_sec, max_frames), start=1):
        out = tmp_dir / f"frame_{idx:02d}.jpg"
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-vf",
            "scale='if(gt(iw,720),720,iw)':-2",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if result.returncode == 0 and out.exists():
            out_paths.append(out)

    if not out_paths:
        raise RuntimeError("Failed to extract sample frames for vision inspection")
    return out_paths


def _mime_type_for(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


def _vision_prompt(technical: TechnicalInfo) -> str:
    return f"""
You are an asset quality gate for short-form scenic inserts.

Analyze the provided image frames from one generated scenic asset.
The intended policy is:
- no humans
- no human faces
- no body parts like hands if they are prominent
- no animals or animal faces
- no text overlays
- no logos, watermarks, or brand marks
- should be scenery / environment / architecture / nature / abstract atmospheric scene
- should feel cinematic and usable in a premium short-form religious/reflection video

Use the technical metadata too:
- media_type: {technical.media_type}
- width: {technical.width}
- height: {technical.height}
- duration_sec: {technical.duration_sec}
- frame_rate: {technical.frame_rate}
- portrait: {technical.portrait}
- technical_flags: {technical.technical_flags}
- technical_score: {technical.technical_score}

Be strict. If there is even one visible face, obvious animal, or obvious text/logo, mark it detected.
If the asset is not clearly scenery, reject it as non_scenic.
If it has AI defects like warped objects, duplicate structures, smeared texture, mushy detail, temporal inconsistency, bad composition, or obvious generation artifacts, flag them.

Return JSON matching the requested schema.
""".strip()


def _vision_inspect(frames: List[Path], technical: TechnicalInfo) -> VisionInspection:
    client = _client()
    parts: List[object] = [_vision_prompt(technical)]
    for frame in frames:
        parts.append(types.Part.from_bytes(data=frame.read_bytes(), mime_type=_mime_type_for(frame)))

    response = client.models.generate_content(
        model=_vision_model_name(),
        contents=parts,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": VisionInspection.model_json_schema(),
            "temperature": 0.1,
        },
    )

    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, VisionInspection):
            return parsed
        if isinstance(parsed, dict):
            return VisionInspection.model_validate(parsed)

    text = getattr(response, "text", "") or "{}"
    return VisionInspection.model_validate_json(text)


def _report_path_for(path: Path) -> Path:
    return path.parent / f"{path.name}.asset_guard.json"


def _build_auto_reject(vision: VisionInspection, technical: TechnicalInfo) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if vision.human_faces.detected:
        reasons.append("human_faces_detected")
    if vision.animal_faces.detected:
        reasons.append("animal_faces_detected")
    if vision.text_or_logo.detected:
        reasons.append("text_or_logo_detected")
    if vision.non_scenic.rejected:
        reasons.append("asset_is_not_scenic")

    if vision.low_quality_generation.detected and vision.low_quality_generation.confidence >= 0.7:
        reasons.append("low_quality_generation")
    if vision.visual_artifacts.detected and vision.visual_artifacts.confidence >= 0.7:
        reasons.append("visual_artifacts_detected")

    if technical.technical_score < 55:
        reasons.append("technical_quality_too_low")

    return (len(reasons) > 0, reasons)


def _inspect_asset(source_path: str, max_frames: int = 4, save_report: bool = True, use_cache: bool = True) -> AssetGuardReport:
    path = _resolve_media_path(source_path)
    report_path = _report_path_for(path)

    if use_cache and report_path.exists():
        try:
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            return AssetGuardReport.model_validate(cached)
        except Exception:
            pass

    technical = _technical_info(path)
    frames = _extract_sample_frames(path, technical=technical, max_frames=max_frames)
    vision = _vision_inspect(frames, technical=technical)
    auto_reject, auto_reject_reasons = _build_auto_reject(vision, technical)

    report = AssetGuardReport(
        ok=True,
        source_path=str(path),
        report_file=str(report_path),
        model=_vision_model_name(),
        technical=technical,
        frame_samples=[str(p) for p in frames],
        vision=vision,
        auto_reject=auto_reject,
        auto_reject_reasons=auto_reject_reasons,
    )

    if save_report:
        atomic_write_json(report_path, report.model_dump(), ensure_ascii=False)

    return report


@mcp.tool()
def healthcheck() -> dict:
    """Return a simple health check for asset guard."""
    return {
        "ok": True,
        "root": str(ROOT),
        "broll_exists": BROLL.exists(),
        "final_exists": FINAL.exists(),
        "vision_model": _vision_model_name(),
        "ffmpeg": get_ffmpeg(),
        "has_gemini_key": bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()),
    }


@mcp.tool()
def list_assets(limit: int = 50) -> dict:
    """List recent media assets from broll/ and final/."""
    return {
        "broll_folder": str(BROLL),
        "final_folder": str(FINAL),
        "broll": _list_media(BROLL, limit),
        "final": _list_media(FINAL, limit),
    }


@mcp.tool()
def inspect_asset(source_path: str, max_frames: int = 4, save_report: bool = True, use_cache: bool = True) -> dict:
    """Run the full asset guard inspection on one media file."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=save_report, use_cache=use_cache)
    return report.model_dump()


@mcp.tool()
def detect_human_faces(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Detect visible human faces or prominent human presence in an asset."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "human_faces": report.vision.human_faces.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


@mcp.tool()
def detect_animal_faces(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Detect visible animals or animal faces in an asset."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "animal_faces": report.vision.animal_faces.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


@mcp.tool()
def detect_text_or_logo(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Detect visible text overlays, watermarks, or logos in an asset."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "text_or_logo": report.vision.text_or_logo.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


@mcp.tool()
def detect_low_quality_generation(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Detect low-quality generated assets using technical and vision checks."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "technical": report.technical.model_dump(),
        "low_quality_generation": report.vision.low_quality_generation.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


@mcp.tool()
def detect_visual_artifacts(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Detect AI artifacts, warped shapes, texture smearing, or visual defects."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "visual_artifacts": report.vision.visual_artifacts.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


@mcp.tool()
def reject_non_scenic_asset(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Reject assets that are not clearly scenery-based and safe for your visual policy."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "non_scenic": report.vision.non_scenic.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


@mcp.tool()
def score_cinematic_quality(source_path: str, max_frames: int = 4, use_cache: bool = True) -> dict:
    """Score the cinematic quality of the asset for premium short-form use."""
    report = _inspect_asset(source_path, max_frames=max_frames, save_report=True, use_cache=use_cache)
    return {
        "source_path": report.source_path,
        "report_file": report.report_file,
        "technical": {
            "technical_score": report.technical.technical_score,
            "technical_flags": report.technical.technical_flags,
            "width": report.technical.width,
            "height": report.technical.height,
            "frame_rate": report.technical.frame_rate,
            "duration_sec": report.technical.duration_sec,
        },
        "cinematic_quality": report.vision.cinematic_quality.model_dump(),
        "auto_reject": report.auto_reject,
        "auto_reject_reasons": report.auto_reject_reasons,
    }


if __name__ == "__main__":
    mcp.run()
