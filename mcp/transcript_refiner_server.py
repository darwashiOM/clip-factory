"""
clip-factory transcript refiner MCP server.

Refines only the selected clip windows by default, keeps absolute timestamps,
and now also emits a boundary suggestion for cleaner render timing.
"""

from pathlib import Path
import sys
import os
import json
import time
from typing import List, Tuple, Dict

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from helpers import clean_arabic_for_captions, get_ffmpeg, atomic_write_json

from bootstrap import resolve_root_and_load_env
import llm_client

ROOT = resolve_root_and_load_env()

TRANSCRIPTS = ROOT / "transcripts"
CLIPS = ROOT / "clips"



mcp = FastMCP("clip-factory-transcript-refiner", json_response=True)

_CONTINUATION_START_WORDS = {
    "و", "ف", "ثم", "لكن", "لأن", "لان", "إذا", "اذا", "يعني", "ولهذا", "ولذلك", "أما", "اما", "بل", "كما",
}
_CONTINUATION_END_WORDS = {
    "و", "ف", "ثم", "لكن", "لأن", "لان", "إذا", "اذا", "يعني", "مثلا", "مثلًا", "وهو", "وهي",
}
_PAYOFF_WORDS = {
    "لذلك", "ولهذا", "فالخلاصة", "الخلاصة", "النتيجة", "إذن", "اذن", "لهذا", "انتبه", "احذر",
    "جنة", "النار", "رحمة", "مغفرة", "دعاء", "قرآن", "التوبة", "الخشوع",
}


class CorrectedSegment(BaseModel):
    index: int = Field(ge=0, description="Zero-based index of the segment within the current chunk")
    text: str = Field(description="Corrected Arabic text for this exact segment only")


class SegmentRefineResult(BaseModel):
    corrected_segments: List[CorrectedSegment]


def _model_name() -> str:
    # Legacy Gemini env vars take priority so existing configs keep working.
    legacy = (os.environ.get("GEMINI_REFINER_MODEL") or os.environ.get("GEMINI_MODEL") or "").strip()
    if legacy:
        return legacy
    return llm_client._text_model_name()


def _load_verbose(stem: str):
    preferred = [
        TRANSCRIPTS / f"{stem}.refined.verbose.json",
        TRANSCRIPTS / f"{stem}.verbose.json",
    ]
    p = next((x for x in preferred if x.exists()), None)
    if not p:
        raise FileNotFoundError(f"Missing verbose transcript for stem {stem}")
    data = json.loads(p.read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    if not segments:
        raise RuntimeError(f"No segments found in {p}")
    return p, data, segments


def _load_candidates(stem: str) -> dict:
    p = CLIPS / f"{stem}.candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing candidate file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_clip_segments(stem: str, clip_number: int, padding_segments: int = 1) -> Tuple[dict, List[dict], int, int]:
    _, _, segments = _load_verbose(stem)
    candidates = _load_candidates(stem)
    clips = candidates.get("clips") or []
    if clip_number < 1 or clip_number > len(clips):
        raise ValueError(f"clip_number must be between 1 and {len(clips)}")
    clip = clips[clip_number - 1]
    start = float(clip["start_sec"])
    end = float(clip["end_sec"])

    overlapping = [
        i for i, seg in enumerate(segments)
        if float(seg.get("end", 0)) > start and float(seg.get("start", 0)) < end
    ]
    if not overlapping:
        return clip, [], -1, -1

    lo = max(0, overlapping[0] - padding_segments)
    hi = min(len(segments), overlapping[-1] + 1 + padding_segments)
    clip_segments = [dict(seg) for seg in segments[lo:hi]]
    local_start_idx = overlapping[0] - lo
    local_end_idx = overlapping[-1] - lo
    return clip, clip_segments, local_start_idx, local_end_idx


def _sec_to_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def _segments_to_srt(segments, clean: bool = False) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _sec_to_srt_time(float(seg["start"]))
        end = _sec_to_srt_time(float(seg["end"]))
        text = str(seg.get("text", "")).strip()
        if clean:
            text = clean_arabic_for_captions(text)
        if not text:
            continue
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return ("\n".join(lines).strip() + "\n") if lines else ""


def _chunk_segments(segments: List[dict], chunk_size: int = 12, max_chars: int = 1800):
    chunks = []
    current = []
    current_chars = 0

    for global_index, seg in enumerate(segments):
        text = str(seg.get("text", "")).strip()
        line = f"[{len(current)}] {text}"
        if current and (len(current) >= chunk_size or current_chars + len(line) > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0

        current.append({
            "global_index": global_index,
            "local_index": len(current),
            "text": text,
        })
        current_chars += len(line)

    if current:
        chunks.append(current)
    return chunks


def _build_chunk_prompt(chunk: List[dict]) -> str:
    prompt_lines = [f"[{item['local_index']}] {item['text']}" for item in chunk]
    return f"""
You are correcting Arabic speech-to-text transcript segments for final subtitles and clip selection.

Return ONLY valid JSON matching the schema.
Do not include markdown.
Do not include commentary.

NON-NEGOTIABLE RULES:
- Keep the SAME number of segments.
- Keep each segment aligned 1-to-1 with its LOCAL chunk index.
- Correct Arabic spelling, obvious ASR mistakes, spacing, and punctuation.
- Do not summarize.
- Do not merge segments.
- Do not split segments.
- Do not add any text that is not supported by the original segment.
- Preserve meaning as closely as possible.
- Keep Islamic terminology natural in Arabic.
- If a phrase might be Quran and you are not absolutely sure, stay very close to the original wording instead of guessing.
- If a segment is already good, keep it nearly unchanged.
- Preserve Arabic diacritics and Quranic marks when they already exist in the source text.
- If the source text has no diacritics, do not add them unless the segment is an exact Quran match from a trusted Quran source.

Output one corrected text entry for every local index shown below.

Segments:
{chr(10).join(prompt_lines)}
""".strip()


def _coerce_refine_response(response_text: str, chunk: List[dict]) -> List[str]:
    parsed = SegmentRefineResult.model_validate_json(response_text)
    corrected_by_local = {item.index: item.text.strip() for item in parsed.corrected_segments}

    results = []
    for item in chunk:
        original = item["text"].strip()
        corrected = corrected_by_local.get(item["local_index"], original).strip() or original
        results.append(corrected)
    return results


def _refine_chunk(chunk: List[dict], retries: int = 1, retry_sleep_sec: float = 1.0) -> List[str]:
    """Call the configured LLM to correct a chunk of transcript segments.
    When TRANSCRIPT_REFINER_USE_LLM=false, returns originals unchanged (no API call)."""
    originals = [item["text"].strip() for item in chunk]

    if not llm_client.transcript_refiner_use_llm():
        return originals

    prompt = _build_chunk_prompt(chunk)
    llm = llm_client.get_text_llm()

    for attempt in range(retries + 1):
        try:
            raw = llm.generate_json(prompt)
            return _coerce_refine_response(raw, chunk)
        except Exception:
            if attempt < retries:
                time.sleep(retry_sleep_sec)

    return originals


def _text(seg: dict) -> str:
    return str(seg.get("text", "")).strip().replace("\n", " ")


def _starts_with_continuation(text: str) -> bool:
    words = [w for w in text.split() if w][:3]
    if not words:
        return True
    if words[0] in _CONTINUATION_START_WORDS:
        return True
    return len(words) >= 2 and " ".join(words[:2]) in _CONTINUATION_START_WORDS


def _ends_with_continuation(text: str) -> bool:
    words = [w for w in text.split() if w][-3:]
    if not words:
        return True
    if words[-1] in _CONTINUATION_END_WORDS:
        return True
    return len(words) >= 2 and " ".join(words[-2:]) in _CONTINUATION_END_WORDS


def _ends_with_stop(text: str) -> bool:
    return text.strip().endswith((".", "؟", "!", ":", "…"))


def _has_payoff(text: str) -> bool:
    words = {w.strip("،؛:.?!؟") for w in text.split() if w.strip("،؛:.?!؟")}
    return any(w in words for w in _PAYOFF_WORDS)


def _duration_target_score(duration: float) -> float:
    target = 28.0
    score = 1.0 - abs(duration - target) / 10.0
    return max(0.0, min(1.0, score))


def _score_local_window(segments: List[dict], start_idx: int, end_idx: int) -> Tuple[float, dict]:
    if start_idx < 0 or end_idx >= len(segments) or end_idx < start_idx:
        return -1.0, {}

    first = _text(segments[start_idx])
    last = _text(segments[end_idx])
    next_text = _text(segments[end_idx + 1]) if end_idx + 1 < len(segments) else ""
    duration = float(segments[end_idx].get("end", 0)) - float(segments[start_idx].get("start", 0))
    if duration <= 0:
        return -1.0, {}

    opening = 0.70
    if _starts_with_continuation(first):
        opening -= 0.36
    if len(first.split()) <= 8:
        opening += 0.06

    ending = 0.62
    if _ends_with_stop(last):
        ending += 0.20
    if _has_payoff(last):
        ending += 0.08
    if _ends_with_continuation(last):
        ending -= 0.30
    if next_text and _starts_with_continuation(next_text):
        ending -= 0.10

    standalone = 0.66
    if _starts_with_continuation(first):
        standalone -= 0.22
    if _ends_with_continuation(last):
        standalone -= 0.18

    duration_score = _duration_target_score(duration)
    total = 0.30 * opening + 0.34 * ending + 0.22 * standalone + 0.14 * duration_score
    metrics = {
        "opening": round(max(0.0, min(1.0, opening)), 4),
        "ending": round(max(0.0, min(1.0, ending)), 4),
        "standalone": round(max(0.0, min(1.0, standalone)), 4),
        "duration": round(max(0.0, min(1.0, duration_score)), 4),
        "score": round(max(0.0, min(1.0, total)), 4),
        "duration_sec": round(duration, 2),
    }
    return total, metrics


def _suggest_boundaries(clip: dict, refined_segments: List[dict], local_start_idx: int, local_end_idx: int) -> dict:
    if not refined_segments or local_start_idx < 0 or local_end_idx < 0:
        start = round(float(clip["start_sec"]), 2)
        end = round(float(clip["end_sec"]), 2)
        return {
            "suggested_start_sec": start,
            "suggested_end_sec": end,
            "start_segment_index": local_start_idx,
            "end_segment_index": local_end_idx,
            "changed": False,
            "reason": "no_local_boundary_context",
            "confidence": 0.0,
        }

    best_key = None
    best_payload = None
    for s_shift in (-1, 0, 1):
        for e_shift in (-1, 0, 1, 2):
            s_idx = max(0, min(len(refined_segments) - 1, local_start_idx + s_shift))
            e_idx = max(0, min(len(refined_segments) - 1, local_end_idx + e_shift))
            if e_idx < s_idx:
                continue

            score, metrics = _score_local_window(refined_segments, s_idx, e_idx)
            if score < 0:
                continue

            start = round(float(refined_segments[s_idx].get("start", 0)), 2)
            end = round(float(refined_segments[e_idx].get("end", 0)), 2)
            reason_parts = []
            if s_idx != local_start_idx:
                reason_parts.append("start_adjusted_to_segment_boundary")
            if e_idx != local_end_idx:
                reason_parts.append("end_adjusted_to_complete_thought")
            if metrics.get("ending", 0.0) >= 0.78:
                reason_parts.append("ending_scores_strong")
            if metrics.get("opening", 0.0) >= 0.72:
                reason_parts.append("opening_scores_clean")
            if not reason_parts:
                reason_parts.append("kept_original_boundaries")

            key = (
                round(metrics.get("score", 0.0), 6),
                round(metrics.get("ending", 0.0), 6),
                round(metrics.get("opening", 0.0), 6),
                -abs(metrics.get("duration_sec", 0.0) - 28.0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_payload = {
                    "suggested_start_sec": start,
                    "suggested_end_sec": end,
                    "start_segment_index": s_idx,
                    "end_segment_index": e_idx,
                    "changed": bool(s_idx != local_start_idx or e_idx != local_end_idx),
                    "reason": "; ".join(reason_parts),
                    "confidence": round(metrics.get("score", 0.0), 4),
                    "metrics": metrics,
                }

    if best_payload is None:
        start = round(float(clip["start_sec"]), 2)
        end = round(float(clip["end_sec"]), 2)
        return {
            "suggested_start_sec": start,
            "suggested_end_sec": end,
            "start_segment_index": local_start_idx,
            "end_segment_index": local_end_idx,
            "changed": False,
            "reason": "fallback_to_original_candidate",
            "confidence": 0.0,
        }
    return best_payload


def _save_clip_outputs(
    clip_stem: str,
    source_stem: str,
    clip_number: int,
    clip: dict,
    refined_segments: List[dict],
    corrected_count: int,
    chunk_count: int,
    failed_chunk_count: int,
    boundary_suggestion: dict,
):
    refined_verbose_path = TRANSCRIPTS / f"{clip_stem}.refined.verbose.json"
    refined_txt_path = TRANSCRIPTS / f"{clip_stem}.refined.txt"
    refined_srt_path = TRANSCRIPTS / f"{clip_stem}.refined.srt"
    refined_captions_srt_path = TRANSCRIPTS / f"{clip_stem}.refined.captions.srt"
    summary_path = TRANSCRIPTS / f"{clip_stem}.refined.summary.json"

    refined_verbose = {
        "source_stem": source_stem,
        "clip_number": clip_number,
        "clip": clip,
        "segments": refined_segments,
        "text": " ".join(
            str(seg.get("text", "")).strip()
            for seg in refined_segments
            if str(seg.get("text", "")).strip()
        ),
        "refiner": {
            "enabled": llm_client.transcript_refiner_use_llm(),
            "provider": llm_client._text_provider(),
            "model": _model_name(),
            "segment_count": len(refined_segments),
            "chunk_count": chunk_count,
            "failed_chunk_count": failed_chunk_count,
            "corrected_segment_count": corrected_count,
        },
        "boundary_suggestion": boundary_suggestion,
    }

    refined_text = refined_verbose["text"].strip()
    refined_srt = _segments_to_srt(refined_segments, clean=False)
    refined_captions = _segments_to_srt(refined_segments, clean=True)

    atomic_write_json(refined_verbose_path, refined_verbose, ensure_ascii=False)
    refined_txt_path.write_text(refined_text + "\n", encoding="utf-8")
    refined_srt_path.write_text(refined_srt, encoding="utf-8")
    refined_captions_srt_path.write_text(refined_captions, encoding="utf-8")
    atomic_write_json(
        summary_path,
        {
            "source_stem": source_stem,
            "clip_number": clip_number,
            "clip_stem": clip_stem,
            "model": _model_name(),
            "segment_count": len(refined_segments),
            "chunk_count": chunk_count,
            "failed_chunk_count": failed_chunk_count,
            "corrected_segment_count": corrected_count,
            "boundary_suggestion": boundary_suggestion,
            "outputs": {
                "verbose_json": str(refined_verbose_path),
                "txt": str(refined_txt_path),
                "srt": str(refined_srt_path),
                "captions_srt": str(refined_captions_srt_path),
                "summary_json": str(summary_path),
            },
        },
        ensure_ascii=False,
    )
    return {
        "verbose_json": str(refined_verbose_path),
        "txt": str(refined_txt_path),
        "srt": str(refined_srt_path),
        "captions_srt": str(refined_captions_srt_path),
        "summary_json": str(summary_path),
    }


@mcp.tool()
def refine_clip_candidate(
    stem: str,
    clip_number: int,
    overwrite: bool = True,
    padding_segments: int = 1,
    chunk_size: int = 12,
    max_chars: int = 1800,
) -> dict:
    """
    Refine only the transcript text needed for one selected clip candidate.
    Outputs are written under a clip-specific stem:
    transcripts/<stem>__clipNN.refined.*
    """
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)

    clip_stem = f"{stem}__clip{clip_number:02d}"
    outputs = {
        "verbose_json": str(TRANSCRIPTS / f"{clip_stem}.refined.verbose.json"),
        "txt": str(TRANSCRIPTS / f"{clip_stem}.refined.txt"),
        "srt": str(TRANSCRIPTS / f"{clip_stem}.refined.srt"),
        "captions_srt": str(TRANSCRIPTS / f"{clip_stem}.refined.captions.srt"),
        "summary_json": str(TRANSCRIPTS / f"{clip_stem}.refined.summary.json"),
    }
    if not overwrite and all(Path(p).exists() for p in outputs.values()):
        return {"ok": True, "message": "Refined clip artifacts already exist", "clip_stem": clip_stem, "outputs": outputs}

    clip, clip_segments, local_start_idx, local_end_idx = _extract_clip_segments(stem, clip_number, padding_segments=padding_segments)
    if not clip_segments:
        raise RuntimeError(f"No transcript segments overlap clip {clip_number} for stem {stem}")

    chunks = _chunk_segments(clip_segments, chunk_size=chunk_size, max_chars=max_chars)
    corrected_count = 0
    refined_segments = []
    failed_chunks = 0

    for chunk in chunks:
        originals = [item["text"].strip() for item in chunk]
        refined_texts = _refine_chunk(chunk)
        if refined_texts == originals:
            failed_chunks += 1

        for item, refined_text in zip(chunk, refined_texts):
            seg = dict(clip_segments[item["global_index"]])
            original = str(seg.get("text", "")).strip()
            cleaned = (refined_text or original).strip()
            if cleaned != original:
                corrected_count += 1
            seg["text"] = cleaned
            refined_segments.append(seg)

    boundary_suggestion = _suggest_boundaries(
        clip=clip,
        refined_segments=refined_segments,
        local_start_idx=local_start_idx,
        local_end_idx=local_end_idx,
    )

    output_paths = _save_clip_outputs(
        clip_stem=clip_stem,
        source_stem=stem,
        clip_number=clip_number,
        clip=clip,
        refined_segments=refined_segments,
        corrected_count=corrected_count,
        chunk_count=len(chunks),
        failed_chunk_count=failed_chunks,
        boundary_suggestion=boundary_suggestion,
    )

    return {
        "ok": True,
        "source_stem": stem,
        "clip_number": clip_number,
        "clip_stem": clip_stem,
        "model": _model_name(),
        "segment_count": len(refined_segments),
        "chunk_count": len(chunks),
        "failed_chunk_count": failed_chunks,
        "corrected_segment_count": corrected_count,
        "boundary_suggestion": boundary_suggestion,
        "outputs": output_paths,
    }


@mcp.tool()
def show_refined_summary(clip_stem: str) -> dict:
    summary_path = TRANSCRIPTS / f"{clip_stem}.refined.summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    mcp.run()
