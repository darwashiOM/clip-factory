from pathlib import Path
import os
import json
import math
import subprocess
import tempfile
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from helpers import clean_arabic_for_captions

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()
INCOMING = ROOT / "incoming"
TRANSCRIPTS = ROOT / "transcripts"

load_dotenv(ROOT / ".env")

mcp = FastMCP("clip-factory-transcribe", json_response=True)


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")
    return OpenAI(api_key=api_key)


def _safe_media_files(limit: int = 50) -> List[Dict[str, Any]]:
    allowed = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    if not INCOMING.exists():
        return []

    files = sorted(
        [p for p in INCOMING.iterdir() if p.is_file() and p.suffix.lower() in allowed],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return [
        {
            "name": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        }
        for p in files[:limit]
    ]


def _sec_to_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def _segments_to_srt(segments: List[Dict[str, Any]], clean: bool = False) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _sec_to_srt_time(float(seg["start"]))
        end = _sec_to_srt_time(float(seg["end"]))
        text = str(seg["text"]).strip()

        if clean:
            text = clean_arabic_for_captions(text)

        if not text:
            continue

        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines).strip() + "\n" if lines else ""


def _ffprobe_duration(src: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _build_chunk_plan(total_duration: float, chunk_seconds: int, overlap_seconds: int) -> List[Dict[str, Any]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    if total_duration <= 0:
        return []

    chunk_count = max(1, math.ceil(total_duration / chunk_seconds))
    plan = []
    for index in range(chunk_count):
        nominal_start = index * chunk_seconds
        nominal_end = min((index + 1) * chunk_seconds, total_duration)
        extract_start = nominal_start if index == 0 else max(0.0, nominal_start - overlap_seconds)
        extract_end = nominal_end if index == chunk_count - 1 else min(total_duration, nominal_end + overlap_seconds)
        keep_start = nominal_start
        keep_end = nominal_end
        plan.append(
            {
                "index": index,
                "nominal_start": round(nominal_start, 3),
                "nominal_end": round(nominal_end, 3),
                "extract_start": round(extract_start, 3),
                "extract_end": round(extract_end, 3),
                "keep_start": round(keep_start, 3),
                "keep_end": round(keep_end, 3),
                "duration": round(extract_end - extract_start, 3),
            }
        )
    return plan


def _extract_chunk_audio(src: Path, out_path: Path, start: float, duration: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            str(out_path),
        ],
        check=True,
    )


def _normalize_verbose_response(verbose_resp: Any) -> Dict[str, Any]:
    if hasattr(verbose_resp, "model_dump"):
        return verbose_resp.model_dump()
    if isinstance(verbose_resp, dict):
        return verbose_resp
    return json.loads(json.dumps(verbose_resp, default=lambda o: getattr(o, "__dict__", str(o))))


def _merge_chunk_segments(
    chunk_data: List[Dict[str, Any]],
    total_duration: float,
    language: Optional[str],
) -> Dict[str, Any]:
    merged_segments: List[Dict[str, Any]] = []
    chunk_reports: List[Dict[str, Any]] = []

    for chunk in chunk_data:
        verbose_data = chunk["verbose_data"]
        raw_segments = verbose_data.get("segments") or []
        accepted = 0

        for seg in raw_segments:
            try:
                rel_start = float(seg.get("start", 0.0))
                rel_end = float(seg.get("end", rel_start))
            except (TypeError, ValueError):
                continue

            abs_start = chunk["extract_start"] + rel_start
            abs_end = chunk["extract_start"] + rel_end
            midpoint = (abs_start + abs_end) / 2.0
            keep_end = chunk["keep_end"]
            keep_start = chunk["keep_start"]
            is_last_chunk = chunk["index"] == len(chunk_data) - 1

            if midpoint < keep_start:
                continue
            if midpoint >= keep_end and not is_last_chunk:
                continue

            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            merged_seg = dict(seg)
            merged_seg["start"] = round(max(0.0, abs_start), 3)
            merged_seg["end"] = round(min(total_duration, max(abs_start, abs_end)), 3)
            merged_seg["text"] = text
            merged_segments.append(merged_seg)
            accepted += 1

        chunk_reports.append(
            {
                "index": chunk["index"],
                "chunk_file": chunk["chunk_file"],
                "extract_start": chunk["extract_start"],
                "extract_end": chunk["extract_end"],
                "keep_start": chunk["keep_start"],
                "keep_end": chunk["keep_end"],
                "raw_segment_count": len(raw_segments),
                "accepted_segment_count": accepted,
            }
        )

    merged_segments.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))

    compact_segments: List[Dict[str, Any]] = []
    for seg in merged_segments:
        if compact_segments:
            prev = compact_segments[-1]
            prev_text = str(prev.get("text", "")).strip()
            cur_text = str(seg.get("text", "")).strip()
            start_gap = abs(float(seg["start"]) - float(prev["start"]))
            end_gap = abs(float(seg["end"]) - float(prev["end"]))
            if cur_text == prev_text and start_gap < 1.0 and end_gap < 1.0:
                continue
        compact_segments.append(seg)

    text = " ".join(str(seg.get("text", "")).strip() for seg in compact_segments if str(seg.get("text", "")).strip()).strip()

    merged_verbose = {
        "task": "transcribe",
        "language": language,
        "duration": round(total_duration, 3),
        "text": text,
        "segments": compact_segments,
        "chunking": {
            "enabled": True,
            "chunk_count": len(chunk_data),
            "chunks": chunk_reports,
        },
    }
    return merged_verbose


@mcp.tool()
def list_transcribable_files(limit: int = 50) -> dict:
    """List compatible media files in the incoming folder."""
    return {
        "folder": str(INCOMING),
        "files": _safe_media_files(limit),
    }


@mcp.tool()
def inspect_chunk_plan(
    file_name: str,
    chunk_seconds: int = 600,
    overlap_seconds: int = 8,
) -> dict:
    """Inspect how a media file would be split before transcription."""
    src = (INCOMING / file_name).resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"File not found: {src}")

    duration = _ffprobe_duration(src)
    plan = _build_chunk_plan(duration, chunk_seconds, overlap_seconds)
    return {
        "ok": True,
        "source": str(src),
        "duration_seconds": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "chunk_count": len(plan),
        "plan": plan,
    }


@mcp.tool()
def transcribe_file(
    file_name: str,
    language: Optional[str] = "ar",
    overwrite: bool = False,
    chunk_seconds: int = 600,
    overlap_seconds: int = 8,
    keep_chunk_files: bool = False,
) -> dict:
    """
    Chunked transcription for one media file from incoming/ using OpenAI Whisper.

    Saves:
    - transcripts/<stem>.json
    - transcripts/<stem>.verbose.json
    - transcripts/<stem>.srt
    - transcripts/<stem>.captions.srt
    - transcripts/<stem>.txt
    - transcripts/<stem>.chunks.json
    """
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)

    src = (INCOMING / file_name).resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"File not found: {src}")

    txt_path = TRANSCRIPTS / f"{src.stem}.txt"
    json_path = TRANSCRIPTS / f"{src.stem}.json"
    verbose_path = TRANSCRIPTS / f"{src.stem}.verbose.json"
    srt_path = TRANSCRIPTS / f"{src.stem}.srt"
    captions_path = TRANSCRIPTS / f"{src.stem}.captions.srt"
    chunks_manifest_path = TRANSCRIPTS / f"{src.stem}.chunks.json"

    if not overwrite and all(
        p.exists() for p in [txt_path, json_path, verbose_path, srt_path, captions_path, chunks_manifest_path]
    ):
        return {
            "ok": True,
            "message": "Transcript files already exist",
            "source": str(src),
            "outputs": {
                "txt": str(txt_path),
                "json": str(json_path),
                "verbose_json": str(verbose_path),
                "srt": str(srt_path),
                "captions_srt": str(captions_path),
                "chunks_manifest": str(chunks_manifest_path),
            },
        }

    client = _client()
    duration = _ffprobe_duration(src)
    plan = _build_chunk_plan(duration, chunk_seconds, overlap_seconds)
    if not plan:
        raise RuntimeError("Unable to build chunk plan for transcription")

    chunk_data: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix=f"{src.stem}_chunks_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        persisted_chunk_dir = TRANSCRIPTS / f"{src.stem}.chunks"
        if keep_chunk_files:
            persisted_chunk_dir.mkdir(parents=True, exist_ok=True)

        for chunk in plan:
            chunk_name = f"{src.stem}__chunk{chunk['index'] + 1:03d}.mp3"
            chunk_path = tmpdir_path / chunk_name
            _extract_chunk_audio(
                src=src,
                out_path=chunk_path,
                start=float(chunk["extract_start"]),
                duration=float(chunk["duration"]),
            )

            if keep_chunk_files:
                target_chunk_path = persisted_chunk_dir / chunk_name
                target_chunk_path.write_bytes(chunk_path.read_bytes())
                saved_chunk_path = str(target_chunk_path)
            else:
                saved_chunk_path = str(chunk_path)

            with open(chunk_path, "rb") as f:
                verbose_resp = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-1",
                    response_format="verbose_json",
                    language=language,
                )

            verbose_data = _normalize_verbose_response(verbose_resp)
            chunk_data.append(
                {
                    **chunk,
                    "chunk_file": saved_chunk_path,
                    "verbose_data": verbose_data,
                }
            )

    merged_verbose = _merge_chunk_segments(chunk_data, duration, language)
    text = (merged_verbose.get("text") or "").strip()
    segments = merged_verbose.get("segments") or []

    srt_text = _segments_to_srt(segments) if segments else ""
    captions_text = _segments_to_srt(segments, clean=True) if segments else ""

    txt_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps({"text": text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verbose_path.write_text(
        json.dumps(merged_verbose, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    srt_path.write_text(srt_text, encoding="utf-8")
    captions_path.write_text(captions_text, encoding="utf-8")

    chunks_manifest = {
        "source": str(src),
        "duration_seconds": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "chunk_count": len(plan),
        "keep_chunk_files": keep_chunk_files,
        "chunks": [
            {
                "index": c["index"],
                "chunk_file": c["chunk_file"],
                "nominal_start": c["nominal_start"],
                "nominal_end": c["nominal_end"],
                "extract_start": c["extract_start"],
                "extract_end": c["extract_end"],
                "keep_start": c["keep_start"],
                "keep_end": c["keep_end"],
                "raw_segment_count": len((c.get("verbose_data") or {}).get("segments") or []),
            }
            for c in chunk_data
        ],
    }
    chunks_manifest_path.write_text(
        json.dumps(chunks_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "source": str(src),
        "language": language,
        "duration_seconds": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "chunk_count": len(plan),
        "text_preview": text[:300],
        "segment_count": len(segments),
        "outputs": {
            "txt": str(txt_path),
            "json": str(json_path),
            "verbose_json": str(verbose_path),
            "srt": str(srt_path),
            "captions_srt": str(captions_path),
            "chunks_manifest": str(chunks_manifest_path),
        },
    }


@mcp.tool()
def list_transcripts(limit: int = 50) -> dict:
    """List transcript artifacts in transcripts/."""
    if not TRANSCRIPTS.exists():
        return {"folder": str(TRANSCRIPTS), "files": []}

    files = sorted(
        [p for p in TRANSCRIPTS.iterdir() if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return {
        "folder": str(TRANSCRIPTS),
        "files": [
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
            for p in files[:limit]
        ],
    }


if __name__ == "__main__":
    mcp.run()
