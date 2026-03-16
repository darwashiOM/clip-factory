from pathlib import Path
import os
import json
import re
from typing import List, Optional, Tuple, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from helpers import atomic_write_json
from bootstrap import resolve_root_and_load_env
import llm_client

ROOT = resolve_root_and_load_env()

TRANSCRIPTS = ROOT / "transcripts"
CLIPS = ROOT / "clips"

# ── Visual plan timing ────────────────────────────────────────────────────────
# Minimum seconds of speaker at the START before the first scenic insert.
# Also used as the outro guard — minimum speaker seconds at the END.
# Both are configurable via .env so operators can tune without touching code.
# These default values are deliberately generous (calm Islamic-reflection style):
#   3.0 s intro  — establish the speaker before any scenic cut
#   2.5 s outro  — always land back on the speaker for the close
_SPEAKER_HOLD_SECS: float = float(os.environ.get("SPEAKER_HOLD_SECS", "3.0"))
_OUTRO_GUARD_SECS: float = float(os.environ.get("OUTRO_GUARD_SECS", "2.5"))


mcp = FastMCP("clip-finder", json_response=True)

_PUNCT_END_RE = re.compile(r"[.؟!?!۔…:؛]$")
_CONTINUATION_START_WORDS = {
    "و", "ف", "ثم", "لكن", "لأن", "لان", "إذا", "اذا", "يعني", "ولهذا", "ولذلك",
    "فهذا", "وهذا", "وهذه", "فهذه", "أما", "اما", "بل", "حتى", "كما", "أي", "اي",
}
_CONTINUATION_END_WORDS = {
    "و", "ف", "ثم", "لكن", "لأن", "لان", "إذا", "اذا", "يعني", "مثلا", "مثلًا", "مثال", "وهو", "وهي",
}
_PAYOFF_WORDS = {
    "لذلك", "ولهذا", "فلهذا", "فالخلاصة", "الخلاصة", "فالنتيجة", "النتيجة", "فهذا",
    "إذن", "اذن", "لهذا", "فانظر", "فتأمل", "فتامل", "انتبه", "احذر", "فاحذر",
    "أبشر", "وابشر", "رحمة", "مغفرة", "جنة", "النار", "دعاء", "قرآن", "التوبة", "الخشوع",
    "فاستعد", "فتب", "فارجع", "فاصبر", "فتوكل", "فتذكر", "فاسمع",
}
_CONTEXT_DEPENDENT_WORDS = {
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "كما", "أيضا", "ايضا", "كذلك", "حينها", "وقتها",
}


class VisualBeat(BaseModel):
    type: str = Field(description="'original' or 'stock_video' (real scenic clip)")
    start_offset_sec: float = Field(ge=0, description="Segment start relative to the clip start in seconds")
    end_offset_sec: float = Field(gt=0, description="Segment end relative to the clip start in seconds")
    duration_sec: float = Field(gt=0, description="Segment duration in seconds")
    asset_slot: int = Field(default=0, ge=0, description="1-based generated asset slot for stock_video beats, or 0 for original")
    prompt: str = Field(default="", description="Scenic prompt or stock search hint for stock_video beats")
    notes: str = Field(default="", description="Optional notes for the renderer or generation layer")


class ClipCandidate(BaseModel):
    title: str = Field(description="Short clip title")
    hook: str = Field(description="Opening line or first complete idea actually said in the clip")
    start_sec: float = Field(ge=0, description="Clip start time in seconds")
    end_sec: float = Field(gt=0, description="Clip end time in seconds")
    duration_sec: float = Field(gt=0, description="Clip duration in seconds")
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
    why_it_works: str = Field(description="Why this clip works as a complete standalone short")
    add_broll: bool = Field(description="Whether adding scenery or B-roll is a good idea")
    broll_query: str = Field(description="Short B-roll idea or empty string if none")
    editorial_score: float = Field(default=0.0, ge=0, le=1, description="Deterministic editorial score after boundary repair")
    opening_score: float = Field(default=0.0, ge=0, le=1, description="Opening completeness score, not hook score")
    ending_score: float = Field(default=0.0, ge=0, le=1, description="Ending completeness score")
    standalone_score: float = Field(default=0.0, ge=0, le=1, description="Standalone clarity score")
    target_duration_score: float = Field(default=0.0, ge=0, le=1, description="Duration fitness score")
    completeness_of_beginning: float = Field(default=0.0, ge=0, le=1)
    completeness_of_ending: float = Field(default=0.0, ge=0, le=1)
    self_contained_meaning: float = Field(default=0.0, ge=0, le=1)
    dependency_on_prior_context: float = Field(default=0.0, ge=0, le=1)
    dependency_on_following_context: float = Field(default=0.0, ge=0, le=1)
    emotional_or_spiritual_resolution: float = Field(default=0.0, ge=0, le=1)
    overall_standalone_quality: float = Field(default=0.0, ge=0, le=1)
    start_segment_index: int = Field(default=-1, ge=-1, description="Transcript segment index for clip start")
    end_segment_index: int = Field(default=-1, ge=-1, description="Transcript segment index for clip end")
    boundary_notes: List[str] = Field(default_factory=list, description="Boundary validation notes")
    visual_mode: str = Field(default="alternate_scenic", description="Renderer hint for how to use original video and scenic inserts")
    visual_plan: List[VisualBeat] = Field(default_factory=list, description="Alternating original/scenic beat plan for the renderer")


class ClipPlan(BaseModel):
    source_stem: str
    language: str
    clips: List[ClipCandidate]


def _selection_model_name() -> str:
    # Legacy Gemini env vars keep working so existing configs aren't broken.
    legacy = (
        os.environ.get("GEMINI_SELECTION_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or ""
    ).strip()
    return legacy or llm_client._text_model_name()


def _rerank_model_name() -> str:
    legacy = (
        os.environ.get("GEMINI_RERANK_MODEL")
        or os.environ.get("GEMINI_SELECTION_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or ""
    ).strip()
    return legacy or llm_client._text_model_name()


def _list_verbose_sources(limit: int = 50):
    if not TRANSCRIPTS.exists():
        return []

    files = sorted(
        [p for p in TRANSCRIPTS.glob("*.verbose.json") if p.is_file() and "__clip" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    items = []
    for p in files[:limit]:
        stem = p.name.removesuffix(".verbose.json")
        txt_path = TRANSCRIPTS / f"{stem}.txt"
        srt_path = TRANSCRIPTS / f"{stem}.srt"
        items.append(
            {
                "stem": stem,
                "verbose_json": str(p),
                "txt_exists": txt_path.exists(),
                "srt_exists": srt_path.exists(),
                "size_bytes": p.stat().st_size,
            }
        )
    return items


def _load_source(stem: str):
    preferred_verbose = [
        TRANSCRIPTS / f"{stem}.quran_guard.verbose.json",
        TRANSCRIPTS / f"{stem}.refined.verbose.json",
        TRANSCRIPTS / f"{stem}.verbose.json",
    ]
    preferred_txt = [
        TRANSCRIPTS / f"{stem}.refined.txt",
        TRANSCRIPTS / f"{stem}.txt",
    ]

    verbose_path = next((p for p in preferred_verbose if p.exists()), None)
    txt_path = next((p for p in preferred_txt if p.exists()), None)

    if not verbose_path:
        raise FileNotFoundError(
            f"Missing transcript source. Tried: {[str(p) for p in preferred_verbose]}"
        )
    if not txt_path:
        raise FileNotFoundError(
            f"Missing text transcript. Tried: {[str(p) for p in preferred_txt]}"
        )

    verbose_data = json.loads(verbose_path.read_text(encoding="utf-8"))
    text_data = txt_path.read_text(encoding="utf-8").strip()
    segments = verbose_data.get("segments") or []
    if not segments:
        raise RuntimeError(f"No transcript segments found in {verbose_path}")
    return verbose_path, txt_path, verbose_data, text_data, segments


def _segment_text(seg: dict) -> str:
    return str(seg.get("text", "")).strip().replace("\n", " ")


def _segments_to_prompt_text(segments: List[dict], start_index: int = 0) -> str:
    lines = []
    for local_idx, seg in enumerate(segments):
        global_idx = start_index + local_idx
        start = round(float(seg.get("start", 0)), 2)
        end = round(float(seg.get("end", 0)), 2)
        text = _segment_text(seg)
        lines.append(f"#{global_idx} [{start} -> {end}] {text}")
    return "\n".join(lines)


def _segment_windows(segments: List[dict], window_size: int = 220, stride: int = 140) -> List[Tuple[int, int, List[dict]]]:
    windows: List[Tuple[int, int, List[dict]]] = []
    n = len(segments)
    if n == 0:
        return windows
    start = 0
    while start < n:
        end = min(n, start + window_size)
        windows.append((start, end, segments[start:end]))
        if end >= n:
            break
        start += stride
    return windows


def _window_label(start_idx: int, end_idx: int, segments: List[dict]) -> str:
    if not segments:
        return f"segments {start_idx}-{max(start_idx, end_idx - 1)}"
    start_t = round(_safe_float(segments[0].get("start")), 2)
    end_t = round(_safe_float(segments[-1].get("end")), 2)
    return f"segments {start_idx}-{max(start_idx, end_idx - 1)} ({start_t}s->{end_t}s)"


def _trim_text(text: str, max_chars: int = 18000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].strip()
    tail = text[-(max_chars // 2) :].strip()
    return head + "\n\n[... transcript omitted for length ...]\n\n" + tail


def _build_generation_prompt(
    stem: str,
    transcript_context: str,
    segment_text: str,
    max_candidates: int,
    min_seconds: int,
    max_seconds: int,
    window_label: str,
) -> str:
    ideal_low = max(min_seconds, 30)
    ideal_high = min(max_seconds, 59)
    return f"""
You are an extremely strict Arabic short-form editor.

Reason carefully and deliberately before choosing anything.
Your top goal is NOT hooks. Your top goal is finding clips that feel like a full finished unit.

Return ONLY valid JSON matching the schema.
Do not include markdown.
Do not include explanations outside the JSON.

Your job is to propose up to {max_candidates} candidate clips from this Arabic transcript window.
This window is part of a larger talk, so you must be extra careful not to choose clips that feel like middle excerpts.

MOST IMPORTANT RULE:
Only choose clips that have a clear beginning, middle, and ending within under 60 seconds.
Completeness matters much more than virality, suspense, or dramatic openings.

NON-NEGOTIABLE RULES:
- Use ONLY timestamps that already exist in the segment list below.
- Every clip must start exactly at a segment start time and end exactly at a segment end time.
- Every clip must be understandable to a cold viewer with zero prior context.
- Every clip must not start in the middle of a sentence, explanation, story, warning, or emotional build-up.
- Every clip must not end abruptly or feel cut off.
- The full lesson, reflection, mini-story, warning, reminder, question-answer, or argument must begin and finish inside the clip.
- If a promising moment only works because of earlier context, reject it.
- If a promising moment needs more time after the clip to land fully, reject it.
- If the complete thought cannot fit in the allowed duration range, reject it.
- Never cut a story before its payoff, conclusion, warning, dua, command, or final statement is complete.
- Never start on a continuation word or a dangling thought.
- Never end on a continuation word or obvious unfinished sentence.
- Do not invent timestamps.
- Do not invent text.
- It is better to return fewer excellent complete clips than many flashy incomplete ones.

PRIORITY ORDER:
1. Complete beginning inside the clip.
2. Complete ending inside the clip.
3. Self-contained meaning with minimal dependency on before/after context.
4. Natural stopping point and emotional or spiritual resolution.
5. Only after that, general quality and memorability.

DURATION RULES:
- Allowed duration range is {min_seconds} to {max_seconds} seconds.
- Prefer clips around {ideal_low} to {ideal_high} seconds when possible.
- Never exceed {max_seconds} seconds.
- Shorter is okay only if the thought is fully complete.
- Do not force a shorter clip if it removes necessary setup or payoff.

SELECTION RULES:
- Prefer complete mini-stories, complete reflections, complete reminders, complete warnings, and complete Q/A units.
- Prefer clips that begin with a natural setup or opening statement.
- Prefer clips that end with a natural conclusion or landing point.
- Reject vague or context-dependent excerpts even if they sound strong.
- Reject clips that feel like part 2 of something.
- Reject clips that feel like they are still building after the clip ends.

B-ROLL RULES:
- add_broll should be true when scenic visual cutaways would improve pacing or emphasis.
- Never suggest humans, faces, hands, children, crowds, silhouettes of people, animals, birds, insects, or any living creature.
- Never suggest dialogue scenes, presenters, talking heads, or character action.
- Prefer symbolic or environmental scenery only: empty mosque interiors, desert wind over dunes, moonlit courtyards, rain on windows, ocean waves, mountain mist, lantern light, sunrise clouds, empty roads, prayer hall architecture, drifting dust motes, calm water, dramatic skies.
- broll_query must be short, concrete, and scenery-only.
- If no useful scenic insert is needed, set add_broll to false and broll_query to an empty string.

FIELD GUIDANCE:
- title must be short and specific.
- hook should contain the first real idea or opening line actually said in the clip, not a manufactured hook.
- why_it_works must explain why the clip is complete and self-contained.
- confidence should reflect your confidence in completeness, not in virality.
- Fill the completeness and dependency fields honestly.

Source stem: {stem}
Language: Arabic
Window searched: {window_label}
Allowed duration range: {min_seconds} to {max_seconds} seconds

High-level transcript context:
{transcript_context}

Timestamped segments for exact timestamp selection:
{segment_text}
""".strip()


def _build_rerank_prompt(stem: str, candidate_json: str, keep_count: int) -> str:
    return f"""
You are reranking Arabic short-form clips with one dominant goal:
keep only the clips that feel fully complete and self-contained under 60 seconds.

Reason carefully and deliberately before ranking.
Return ONLY valid JSON matching the same schema.
Do not include markdown.
Do not include commentary.
Keep at most {keep_count} clips.

RERANKING RULES:
- Put completeness above hooks.
- Put self-contained meaning above virality.
- Prioritize clips with a real beginning and a real ending inside the chosen range.
- Prioritize clips with low dependency on prior and following context.
- Reject clips that feel like a middle excerpt.
- Reject clips that feel even slightly cut off at the end.
- Reject clips that depend on earlier unseen context.
- Reject clips that still depend on what comes after.
- If two clips overlap heavily, keep the one that feels more complete, even if the other sounds more dramatic.
- If a clip is flashy but incomplete, reject it.
- If only a few clips truly meet this standard, keep only those few.

Source stem: {stem}

Candidate plan to rerank:
{candidate_json}
""".strip()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _find_first_overlap(segments: List[dict], t: float) -> int:
    for idx, seg in enumerate(segments):
        if _safe_float(seg.get("end")) > t:
            return idx
    return max(0, len(segments) - 1)


def _find_last_overlap(segments: List[dict], t: float) -> int:
    for idx in range(len(segments) - 1, -1, -1):
        if _safe_float(segments[idx].get("start")) < t:
            return idx
    return 0


def _first_words(text: str, n: int = 3) -> List[str]:
    return [w for w in text.split() if w][:n]


def _last_words(text: str, n: int = 3) -> List[str]:
    words = [w for w in text.split() if w]
    return words[-n:]


def _starts_with_continuation(text: str) -> bool:
    words = _first_words(text, 3)
    if not words:
        return True
    first = words[0]
    if first in _CONTINUATION_START_WORDS:
        return True
    if len(words) >= 2 and " ".join(words[:2]) in _CONTINUATION_START_WORDS:
        return True
    return False


def _ends_with_continuation(text: str) -> bool:
    words = _last_words(text, 3)
    if not words:
        return True
    last = words[-1]
    if last in _CONTINUATION_END_WORDS:
        return True
    tail2 = " ".join(words[-2:]) if len(words) >= 2 else last
    return tail2 in _CONTINUATION_END_WORDS


def _ends_with_stop(text: str) -> bool:
    text = text.strip()
    return bool(_PUNCT_END_RE.search(text))


def _has_payoff_signal(text: str) -> bool:
    words = {w.strip("،؛:.?!؟") for w in text.split() if w.strip("،؛:.?!؟")}
    return any(w in words for w in _PAYOFF_WORDS)


def _context_dependency_penalty(text: str) -> float:
    words = [w.strip("،؛:.?!؟") for w in _first_words(text, 6)]
    if not words:
        return 0.32
    penalty = 0.0
    if words[0] in _CONTEXT_DEPENDENT_WORDS:
        penalty += 0.14
    pronouns = sum(1 for w in words if w in _CONTEXT_DEPENDENT_WORDS)
    penalty += min(0.12, pronouns * 0.035)
    return min(0.26, penalty)


def _duration_target_score(duration: float, min_seconds: int, max_seconds: int) -> float:
    target = min(max(42.0, float(min_seconds) + 8.0), float(max_seconds) - 3.0)
    width = max(8.0, min(16.0, (max_seconds - min_seconds) / 2.0))
    score = 1.0 - (abs(duration - target) / width)
    return max(0.0, min(1.0, score))


def _previous_segment_text(segments: List[dict], idx: int) -> str:
    if idx <= 0:
        return ""
    return _segment_text(segments[idx - 1])


def _next_segment_text(segments: List[dict], idx: int) -> str:
    if idx + 1 >= len(segments):
        return ""
    return _segment_text(segments[idx + 1])


def _beginning_completeness_score(first_segment_text: str, previous_segment_text: str) -> float:
    if not first_segment_text.strip():
        return 0.0
    score = 0.74
    if _starts_with_continuation(first_segment_text):
        score -= 0.32
    score -= _context_dependency_penalty(first_segment_text)
    if previous_segment_text.strip() and not _ends_with_stop(previous_segment_text):
        score -= 0.08
    if len(first_segment_text.split()) < 5:
        score -= 0.04
    if first_segment_text.strip().endswith(":"):
        score -= 0.06
    return max(0.0, min(1.0, score))


def _ending_completeness_score(last_segment_text: str, next_segment_text: str) -> float:
    if not last_segment_text.strip():
        return 0.0
    score = 0.70
    if _ends_with_stop(last_segment_text):
        score += 0.18
    if _has_payoff_signal(last_segment_text):
        score += 0.16
    if _ends_with_continuation(last_segment_text):
        score -= 0.28
    if next_segment_text.strip() and not _ends_with_stop(last_segment_text):
        score -= 0.08
    return max(0.0, min(1.0, score))


def _prior_context_dependency(first_segment_text: str, previous_segment_text: str) -> float:
    dep = 0.08 + _context_dependency_penalty(first_segment_text)
    if _starts_with_continuation(first_segment_text):
        dep += 0.22
    if previous_segment_text.strip() and not _ends_with_stop(previous_segment_text):
        dep += 0.08
    return max(0.0, min(1.0, dep))


def _following_context_dependency(last_segment_text: str, next_segment_text: str) -> float:
    dep = 0.06
    if _ends_with_continuation(last_segment_text):
        dep += 0.28
    if next_segment_text.strip() and not _ends_with_stop(last_segment_text):
        dep += 0.14
    if not _has_payoff_signal(last_segment_text) and not _ends_with_stop(last_segment_text):
        dep += 0.06
    return max(0.0, min(1.0, dep))


def _resolution_score(last_segment_text: str) -> float:
    score = 0.48
    if _ends_with_stop(last_segment_text):
        score += 0.16
    if _has_payoff_signal(last_segment_text):
        score += 0.22
    if _ends_with_continuation(last_segment_text):
        score -= 0.22
    return max(0.0, min(1.0, score))


def _standalone_meaning_score(
    candidate_text: str,
    beginning: float,
    ending: float,
    prior_dep: float,
    following_dep: float,
) -> float:
    if not candidate_text.strip():
        return 0.0
    score = 0.24 + (0.25 * beginning) + (0.25 * ending) + (0.14 * (1.0 - prior_dep)) + (0.12 * (1.0 - following_dep))
    if len(candidate_text.split()) < 16:
        score -= 0.08
    return max(0.0, min(1.0, score))


def _score_variant(
    clip: ClipCandidate,
    segments: List[dict],
    start_idx: int,
    end_idx: int,
    min_seconds: int,
    max_seconds: int,
) -> Tuple[float, Dict[str, float], List[str]]:
    start_seg = segments[start_idx]
    end_seg = segments[end_idx]
    start = round(_safe_float(start_seg.get("start")), 2)
    end = round(_safe_float(end_seg.get("end")), 2)
    duration = round(end - start, 2)
    if duration < min_seconds or duration > max_seconds:
        return -1.0, {}, ["duration_out_of_range"]

    selected = segments[start_idx:end_idx + 1]
    candidate_text = " ".join(_segment_text(seg) for seg in selected if _segment_text(seg))
    first_text = _segment_text(selected[0]) if selected else ""
    last_text = _segment_text(selected[-1]) if selected else ""
    prev_text = _previous_segment_text(segments, start_idx)
    next_text = _next_segment_text(segments, end_idx)

    beginning = _beginning_completeness_score(first_text, prev_text)
    ending = _ending_completeness_score(last_text, next_text)
    prior_dep = _prior_context_dependency(first_text, prev_text)
    following_dep = _following_context_dependency(last_text, next_text)
    self_contained = _standalone_meaning_score(candidate_text, beginning, ending, prior_dep, following_dep)
    resolution = _resolution_score(last_text)
    duration_score = _duration_target_score(duration, min_seconds=min_seconds, max_seconds=max_seconds)
    overall_standalone = max(0.0, min(1.0, (0.30 * beginning) + (0.30 * ending) + (0.25 * self_contained) + (0.15 * resolution)))

    notes: List[str] = []
    if _starts_with_continuation(first_text):
        notes.append("starts_like_continuation")
    if prev_text.strip() and not _ends_with_stop(prev_text):
        notes.append("previous_segment_not_closed")
    if _ends_with_continuation(last_text):
        notes.append("ends_like_continuation")
    if _ends_with_stop(last_text):
        notes.append("clean_terminal_punctuation")
    if _has_payoff_signal(last_text):
        notes.append("ending_contains_payoff_signal")
    if duration >= 30 and duration <= 59:
        notes.append("complete_under_60")

    model_conf = max(0.0, min(1.0, _safe_float(clip.confidence, 0.0)))
    editorial = (
        0.10 * model_conf +
        0.23 * beginning +
        0.23 * ending +
        0.18 * self_contained +
        0.10 * (1.0 - prior_dep) +
        0.08 * (1.0 - following_dep) +
        0.05 * resolution +
        0.03 * duration_score
    )

    if beginning < 0.52:
        editorial -= 0.12
        notes.append("weak_beginning_completeness")
    if ending < 0.54:
        editorial -= 0.14
        notes.append("weak_ending_completeness")
    if self_contained < 0.52:
        editorial -= 0.12
        notes.append("not_fully_self_contained")
    if prior_dep > 0.48:
        editorial -= 0.12
        notes.append("depends_on_prior_context")
    if following_dep > 0.48:
        editorial -= 0.14
        notes.append("depends_on_following_context")
    if resolution < 0.46:
        editorial -= 0.06
        notes.append("limited_resolution")
    if duration > 59:
        editorial -= 0.50
        notes.append("over_59_seconds")

    metrics = {
        "beginning": round(max(0.0, min(1.0, beginning)), 4),
        "ending": round(max(0.0, min(1.0, ending)), 4),
        "self_contained": round(max(0.0, min(1.0, self_contained)), 4),
        "prior_dep": round(max(0.0, min(1.0, prior_dep)), 4),
        "following_dep": round(max(0.0, min(1.0, following_dep)), 4),
        "resolution": round(max(0.0, min(1.0, resolution)), 4),
        "duration": round(max(0.0, min(1.0, duration_score)), 4),
        "overall_standalone": round(max(0.0, min(1.0, overall_standalone)), 4),
        "editorial": round(max(0.0, min(1.0, editorial)), 4),
    }
    return editorial, metrics, notes


def _repair_clip_candidate(
    clip: ClipCandidate,
    segments: List[dict],
    min_seconds: int,
    max_seconds: int,
) -> Optional[ClipCandidate]:
    if not segments:
        return None

    start_guess = max(0.0, _safe_float(clip.start_sec, 0.0))
    end_guess = max(0.0, _safe_float(clip.end_sec, 0.0))
    if end_guess <= start_guess:
        return None

    base_start_idx = _find_first_overlap(segments, start_guess)
    base_end_idx = _find_last_overlap(segments, end_guess)
    if base_end_idx < base_start_idx:
        return None

    candidate_pairs = set()
    for s_shift in (-4, -3, -2, -1, 0, 1, 2):
        for e_shift in (-2, -1, 0, 1, 2, 3, 4):
            s_idx = max(0, min(len(segments) - 1, base_start_idx + s_shift))
            e_idx = max(0, min(len(segments) - 1, base_end_idx + e_shift))
            if e_idx < s_idx:
                continue
            candidate_pairs.add((s_idx, e_idx))

    best = None

    for s_idx, e_idx in sorted(candidate_pairs):
        editorial, metrics, notes = _score_variant(
            clip=clip,
            segments=segments,
            start_idx=s_idx,
            end_idx=e_idx,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )
        if editorial < 0:
            continue

        start = round(_safe_float(segments[s_idx].get("start")), 2)
        end = round(_safe_float(segments[e_idx].get("end")), 2)
        duration = round(end - start, 2)
        key = (
            round(editorial, 6),
            round(metrics.get("ending", 0.0), 6),
            round(metrics.get("beginning", 0.0), 6),
            round(metrics.get("self_contained", 0.0), 6),
            -abs(duration - 42.0),
            -s_idx,
        )
        if best is None or key > best[0]:
            repaired = ClipCandidate(
                title=clip.title.strip(),
                hook=clip.hook.strip(),
                start_sec=start,
                end_sec=end,
                duration_sec=duration,
                confidence=max(0.0, min(1.0, _safe_float(clip.confidence, 0.0))),
                why_it_works=clip.why_it_works.strip(),
                add_broll=bool(clip.add_broll),
                broll_query=clip.broll_query.strip(),
                editorial_score=round(metrics.get("editorial", 0.0), 4),
                opening_score=round(metrics.get("beginning", 0.0), 4),
                ending_score=round(metrics.get("ending", 0.0), 4),
                standalone_score=round(metrics.get("self_contained", 0.0), 4),
                target_duration_score=round(metrics.get("duration", 0.0), 4),
                completeness_of_beginning=round(metrics.get("beginning", 0.0), 4),
                completeness_of_ending=round(metrics.get("ending", 0.0), 4),
                self_contained_meaning=round(metrics.get("self_contained", 0.0), 4),
                dependency_on_prior_context=round(metrics.get("prior_dep", 0.0), 4),
                dependency_on_following_context=round(metrics.get("following_dep", 0.0), 4),
                emotional_or_spiritual_resolution=round(metrics.get("resolution", 0.0), 4),
                overall_standalone_quality=round(metrics.get("overall_standalone", 0.0), 4),
                start_segment_index=s_idx,
                end_segment_index=e_idx,
                boundary_notes=notes,
            )
            best = (key, repaired)

    if best is None:
        return None

    repaired = best[1]
    if repaired.duration_sec > 59:
        return None
    if repaired.completeness_of_beginning < 0.50:
        return None
    if repaired.completeness_of_ending < 0.52:
        return None
    if repaired.self_contained_meaning < 0.50:
        return None
    if repaired.dependency_on_prior_context > 0.54:
        return None
    if repaired.dependency_on_following_context > 0.54:
        return None
    if repaired.overall_standalone_quality < 0.52:
        return None
    return repaired


def _heavy_overlap_ratio(a: ClipCandidate, b: ClipCandidate) -> float:
    overlap = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    shorter = min(a.duration_sec, b.duration_sec)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _sanitize_visual_topic(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw or "").strip())
    text = text.replace("_", " ").replace("-", " ")
    lowered = text.lower()

    if any(token in lowered for token in ["dua", "دعاء", "prayer", "salah", "salat", "خشوع", "tawbah", "repent", "quran", "قرآن", "mushaf", "mosque", "masjid", "مسجد"]):
        return "empty mosque interior with warm dawn light and floating dust motes"
    if any(token in lowered for token in ["جنة", "jannah", "paradise", "garden", "رحمة", "mercy"]):
        return "lush garden with flowing water at sunrise"
    if any(token in lowered for token in ["نار", "hell", "warning", "punishment", "fear", "storm"]):
        return "dramatic storm clouds over a barren desert landscape"
    if any(token in lowered for token in ["صبر", "sabr", "patience", "trust", "tawakkul", "journey", "path"]):
        return "vast desert dunes at sunrise with gentle wind"
    if any(token in lowered for token in ["tears", "cry", "repentance", "توبة", "night"]):
        return "rain on a window at night with soft city lights blurred in the distance"

    cleaned = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text).strip()
    return cleaned or "serene natural landscape with soft cinematic motion"


def _stock_query_for_clip(clip: "ClipCandidate", slot: int) -> str:
    """
    Return a short, search-friendly query for a stock footage provider.

    Uses broll_query first (set by the Gemini clip finder); falls back to title.
    Arabic text is replaced with a themed scenic fallback via _sanitize_visual_topic.
    Slot is used to slightly diversify multi-slot queries (title vs broll_query).
    """
    _FALLBACK_QUERIES = [
        "scenic nature landscape",
        "misty mountains dawn",
        "ocean waves shore",
        "desert dunes sunset",
        "river forest mist",
    ]
    if slot == 1 or not clip.title:
        base = (clip.broll_query or clip.title or "").strip()
    else:
        # Second/third slot: prefer title for variety
        base = (clip.title or clip.broll_query or "").strip()

    # Strip clips that are fully Arabic — use themed mapping instead
    if not base or all(ord(c) > 0x06FF or c.isspace() for c in base):
        return _FALLBACK_QUERIES[(slot - 1) % len(_FALLBACK_QUERIES)]

    # Run through the themed mapper so Quran/Islamic keywords map to good visuals
    query = _sanitize_visual_topic(base)
    # Keep it concise: stock search works better with shorter, focused phrases
    return query[:80].strip()


# Keep old name as alias so any external code that references it still works.
def _scenic_prompt_for_slot(base_topic: str, slot: int) -> str:
    """Legacy alias — use _stock_query_for_clip for new code."""
    topic = _sanitize_visual_topic(base_topic)
    return topic[:80].strip()


def _planned_insert_count(duration: float) -> int:
    if duration < 28:
        return 1
    if duration < 44:
        return 2
    return 3


def _build_visual_plan_for_clip(clip: ClipCandidate) -> List[VisualBeat]:
    duration = max(0.0, float(clip.duration_sec))
    if duration <= 0:
        return []

    if not clip.add_broll:
        return [
            VisualBeat(
                type="original",
                start_offset_sec=0.0,
                end_offset_sec=round(duration, 2),
                duration_sec=round(duration, 2),
                asset_slot=0,
                prompt="",
                notes="full_original",
            )
        ]

    insert_count = _planned_insert_count(duration)
    ai_len = 2.0 if insert_count <= 2 else 1.7
    if duration > 45:
        ai_len = 2.1 if insert_count == 2 else 1.8

    # Use env-based guards so timing matches the renderer's SPEAKER_HOLD_SECS.
    # intro_guard  = minimum speaker footage before the FIRST scenic insert.
    # outro_guard  = minimum speaker footage held at the END of the clip.
    intro_guard = _SPEAKER_HOLD_SECS
    outro_guard = _OUTRO_GUARD_SECS
    centers_map = {1: [0.52], 2: [0.38, 0.72], 3: [0.28, 0.55, 0.80]}
    centers = centers_map.get(insert_count, [0.52])

    ai_beats: List[VisualBeat] = []
    last_end = max(0.0, intro_guard - 0.6)
    latest_start = max(intro_guard, duration - outro_guard - ai_len)
    for idx, frac in enumerate(centers, start=1):
        desired_start = max(intro_guard, duration * frac - ai_len / 2.0)
        start_offset = max(last_end + 0.8, desired_start)
        start_offset = min(start_offset, latest_start)
        end_offset = min(duration - outro_guard, start_offset + ai_len)
        if end_offset - start_offset < 1.2:
            continue
        beat = VisualBeat(
            type="stock_video",
            start_offset_sec=round(start_offset, 2),
            end_offset_sec=round(end_offset, 2),
            duration_sec=round(end_offset - start_offset, 2),
            asset_slot=idx,
            prompt=_stock_query_for_clip(clip, idx),
            notes=f"scenic_insert_{idx}",
        )
        ai_beats.append(beat)
        last_end = end_offset

    if not ai_beats:
        return [
            VisualBeat(
                type="original",
                start_offset_sec=0.0,
                end_offset_sec=round(duration, 2),
                duration_sec=round(duration, 2),
                asset_slot=0,
                prompt="",
                notes="fallback_original_only",
            )
        ]

    plan: List[VisualBeat] = []
    cursor = 0.0
    for beat in ai_beats:
        if beat.start_offset_sec - cursor >= 0.35:
            plan.append(
                VisualBeat(
                    type="original",
                    start_offset_sec=round(cursor, 2),
                    end_offset_sec=round(beat.start_offset_sec, 2),
                    duration_sec=round(beat.start_offset_sec - cursor, 2),
                    asset_slot=0,
                    prompt="",
                    notes="original_segment",
                )
            )
        plan.append(beat)
        cursor = beat.end_offset_sec

    if duration - cursor >= 0.35:
        plan.append(
            VisualBeat(
                type="original",
                start_offset_sec=round(cursor, 2),
                end_offset_sec=round(duration, 2),
                duration_sec=round(duration - cursor, 2),
                asset_slot=0,
                prompt="",
                notes="original_outro",
            )
        )

    return [p for p in plan if p.duration_sec > 0.2]


def _normalize_plan(plan: ClipPlan, segments: List[dict], min_seconds: int, max_seconds: int) -> ClipPlan:
    cleaned: List[ClipCandidate] = []
    for clip in plan.clips:
        repaired = _repair_clip_candidate(clip, segments=segments, min_seconds=min_seconds, max_seconds=max_seconds)
        if repaired is not None:
            repaired.visual_mode = "alternate_scenic"
            repaired.visual_plan = _build_visual_plan_for_clip(repaired)
            cleaned.append(repaired)

    cleaned.sort(
        key=lambda x: (
            -x.editorial_score,
            -x.completeness_of_ending,
            -x.completeness_of_beginning,
            -x.self_contained_meaning,
            x.duration_sec,
            x.start_sec,
        )
    )

    deduped: List[ClipCandidate] = []
    for clip in cleaned:
        keep = True
        for idx, existing in enumerate(deduped):
            overlap_ratio = _heavy_overlap_ratio(clip, existing)
            if overlap_ratio >= 0.68:
                existing_key = (
                    existing.editorial_score,
                    existing.completeness_of_ending,
                    existing.completeness_of_beginning,
                    existing.self_contained_meaning,
                    -existing.duration_sec,
                )
                new_key = (
                    clip.editorial_score,
                    clip.completeness_of_ending,
                    clip.completeness_of_beginning,
                    clip.self_contained_meaning,
                    -clip.duration_sec,
                )
                if new_key > existing_key:
                    deduped[idx] = clip
                keep = False
                break
        if keep:
            deduped.append(clip)

    deduped.sort(key=lambda x: (-x.editorial_score, x.start_sec))
    return ClipPlan(source_stem=plan.source_stem, language=plan.language, clips=deduped)


def _coerce_clip_plan(data: dict, stem: str) -> ClipPlan:
    if not isinstance(data, dict):
        return ClipPlan(source_stem=stem, language="Arabic", clips=[])
    if "source_stem" not in data:
        data["source_stem"] = stem
    if "language" not in data:
        data["language"] = "Arabic"
    if "clips" not in data or not isinstance(data.get("clips"), list):
        data["clips"] = []
    return ClipPlan.model_validate(data)


def _generate_window_candidates(
    llm: "llm_client.TextLLM",
    stem: str,
    transcript_context: str,
    window_segments: List[dict],
    window_start_idx: int,
    window_end_idx: int,
    max_candidates: int,
    min_seconds: int,
    max_seconds: int,
) -> ClipPlan:
    segment_text = _segments_to_prompt_text(window_segments, start_index=window_start_idx)
    prompt = _build_generation_prompt(
        stem=stem,
        transcript_context=transcript_context,
        segment_text=segment_text,
        max_candidates=max_candidates,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        window_label=_window_label(window_start_idx, window_end_idx, window_segments),
    )
    raw = llm.generate_json(prompt)
    return _coerce_clip_plan(json.loads(raw), stem=stem)


def _heuristic_find_clips(
    stem: str,
    segments: List[dict],
    min_seconds: int,
    max_seconds: int,
    max_clips: int,
) -> ClipPlan:
    """
    Find clip candidates using pure heuristic scoring — no LLM required.
    Used when CLIP_FINDER_USE_LLM=false.

    Generates all valid (start, end) segment pairs in the duration window,
    scores each with the existing deterministic scorer, deduplicates overlapping
    results, and returns the top max_clips candidates.
    """
    n = len(segments)
    scored: List[tuple] = []

    for start_idx in range(n):
        first_text = _segment_text(segments[start_idx])
        # Skip windows that clearly continue a previous sentence
        if _starts_with_continuation(first_text):
            continue
        start_t = _safe_float(segments[start_idx].get("start"))

        for end_idx in range(start_idx, n):
            end_t = _safe_float(segments[end_idx].get("end"))
            duration = end_t - start_t

            if duration < min_seconds:
                continue
            if duration > max_seconds:
                break  # All further end_idx values will be longer

            # Minimal stub so _score_variant can read clip.confidence
            dummy = ClipCandidate(
                title="",
                hook="",
                start_sec=start_t,
                end_sec=end_t,
                duration_sec=duration,
                confidence=0.5,
                why_it_works="",
                add_broll=False,
                broll_query="",
            )
            editorial, metrics, notes = _score_variant(
                clip=dummy,
                segments=segments,
                start_idx=start_idx,
                end_idx=end_idx,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            )
            if editorial < 0:
                continue
            scored.append((editorial, metrics, notes, start_idx, end_idx, start_t, end_t, duration))

    # Best first
    scored.sort(key=lambda x: (-x[0], -x[7]))

    # Build ClipCandidates, deduplicate heavy overlaps
    raw_candidates: List[ClipCandidate] = []
    for editorial, metrics, notes, start_idx, end_idx, start_t, end_t, duration in scored:
        selected = segments[start_idx:end_idx + 1]
        texts = [_segment_text(s) for s in selected if _segment_text(s)]
        first_text = texts[0] if texts else ""

        # Simple title: first 5 Arabic words
        title_words = first_text.split()[:5]
        title = " ".join(title_words) if title_words else f"clip_{len(raw_candidates)+1}"

        broll_query = _sanitize_visual_topic(first_text)

        cand = ClipCandidate(
            title=title,
            hook=(first_text[:100] or title),
            start_sec=start_t,
            end_sec=end_t,
            duration_sec=duration,
            confidence=round(min(1.0, max(0.0, editorial)), 3),
            why_it_works=(
                f"heuristic: editorial={editorial:.3f} "
                f"begin={metrics.get('beginning', 0):.2f} "
                f"end={metrics.get('ending', 0):.2f}"
            ),
            add_broll=(duration >= 18 and editorial > 0.55),
            broll_query=broll_query,
        )
        raw_candidates.append(cand)

    # Deduplicate: drop any candidate that heavily overlaps a higher-scoring one
    deduped: List[ClipCandidate] = []
    for cand in raw_candidates:
        keep = True
        for existing in deduped:
            overlap = max(0.0, min(cand.end_sec, existing.end_sec) - max(cand.start_sec, existing.start_sec))
            shorter = min(cand.duration_sec, existing.duration_sec)
            if shorter > 0 and overlap / shorter >= 0.70:
                keep = False
                break
        if keep:
            deduped.append(cand)
        if len(deduped) >= max_clips * 4:
            break  # Have enough to work with

    return ClipPlan(source_stem=stem, language="Arabic", clips=deduped)


def _merge_candidate_plans(stem: str, plans: List[ClipPlan]) -> ClipPlan:
    merged: List[ClipCandidate] = []
    for plan in plans:
        merged.extend(plan.clips)
    return ClipPlan(source_stem=stem, language="Arabic", clips=merged)


@mcp.tool()
def list_clip_sources(limit: int = 50) -> dict:
    """List transcript sources that are ready for clip finding."""
    return {
        "folder": str(TRANSCRIPTS),
        "sources": _list_verbose_sources(limit),
    }


@mcp.tool()
def find_clips_from_stem(
    stem: str,
    max_clips: int = 15,
    min_seconds: int = 20,
    max_seconds: int = 59,
    generation_count: int = 60,
    window_size_segments: int = 220,
    window_stride_segments: int = 140,
) -> dict:
    """
    Use Gemini to find self-contained short-form clip candidates from a transcript.

    The server uses a broader search process:
    1) search the full transcript in overlapping timestamped windows
    2) merge all proposed candidates
    3) rerank them with completeness prioritized over hooks
    4) repair boundaries and filter weak clips deterministically
    """
    if max_clips < 1:
        raise ValueError("max_clips must be at least 1")
    if generation_count < max_clips:
        raise ValueError("generation_count must be >= max_clips")
    if min_seconds < 1:
        raise ValueError("min_seconds must be at least 1")
    if max_seconds <= min_seconds:
        raise ValueError("max_seconds must be greater than min_seconds")
    if max_seconds > 59:
        raise ValueError("max_seconds must be <= 59 for complete under-60 clips")
    if window_size_segments < 40:
        raise ValueError("window_size_segments must be at least 40")
    if window_stride_segments < 20:
        raise ValueError("window_stride_segments must be at least 20")

    CLIPS.mkdir(parents=True, exist_ok=True)

    _, _, _, text_data, segments = _load_source(stem)
    use_llm = llm_client.clip_finder_use_llm()

    window_failures: List[dict] = []
    searched_windows: List[dict] = []
    windows: List = []
    per_window_candidates: int = 0

    if use_llm:
        # ── LLM-assisted path (DeepSeek / OpenAI / Gemini) ───────────────────
        transcript_context = _trim_text(text_data, max_chars=18000)
        windows = _segment_windows(segments, window_size=window_size_segments, stride=window_stride_segments)
        llm = llm_client.get_text_llm()
        per_window_candidates = max(4, min(8, (generation_count // max(1, len(windows))) + 2))

        raw_plans: List[ClipPlan] = []

        for start_idx, end_idx, window_segments in windows:
            searched_windows.append({
                "start_segment": start_idx,
                "end_segment": end_idx - 1,
                "segment_count": len(window_segments),
                "label": _window_label(start_idx, end_idx, window_segments),
            })
            try:
                plan = _generate_window_candidates(
                    llm=llm,
                    stem=stem,
                    transcript_context=transcript_context,
                    window_segments=window_segments,
                    window_start_idx=start_idx,
                    window_end_idx=end_idx,
                    max_candidates=per_window_candidates,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                )
                raw_plans.append(plan)
            except Exception as exc:
                window_failures.append({
                    "start_segment": start_idx,
                    "end_segment": end_idx - 1,
                    "error": str(exc),
                })

        merged_generated_plan = _merge_candidate_plans(stem=stem, plans=raw_plans)
        generated_plan = _normalize_plan(merged_generated_plan, segments=segments, min_seconds=min_seconds, max_seconds=max_seconds)

        if generated_plan.clips:
            try:
                rerank_raw = llm.generate_json(
                    _build_rerank_prompt(
                        stem=stem,
                        candidate_json=json.dumps(generated_plan.model_dump(), ensure_ascii=False, indent=2),
                        keep_count=max_clips,
                    )
                )
                final_plan = _coerce_clip_plan(json.loads(rerank_raw), stem=stem)
                final_plan = _normalize_plan(final_plan, segments=segments, min_seconds=min_seconds, max_seconds=max_seconds)
            except Exception:
                final_plan = generated_plan
        else:
            final_plan = ClipPlan(source_stem=stem, language="Arabic", clips=[])

        if not final_plan.clips and generated_plan.clips:
            final_plan = generated_plan

    else:
        # ── Heuristic path (no LLM — zero API cost) ──────────────────────────
        raw_heuristic = _heuristic_find_clips(
            stem=stem,
            segments=segments,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            max_clips=max_clips * 4,  # over-generate, then filter
        )
        generated_plan = _normalize_plan(raw_heuristic, segments=segments, min_seconds=min_seconds, max_seconds=max_seconds)
        final_plan = generated_plan

    final_plan.clips = final_plan.clips[:max_clips]

    output = final_plan.model_dump()
    output["selection_model"] = _selection_model_name() if use_llm else "heuristic"
    output["rerank_model"] = _rerank_model_name() if use_llm else "heuristic"
    output["llm_provider"] = llm_client._text_provider() if use_llm else "none"
    output["clip_finder_mode"] = "llm" if use_llm else "heuristic"
    output["generated_candidate_count"] = len(generated_plan.clips)
    output["boundary_validator"] = "windowed-complete-under60-v2"
    output["requested_duration_range"] = {"min_seconds": min_seconds, "max_seconds": max_seconds}
    output["selection_priority"] = [
        "complete_beginning",
        "complete_ending",
        "self_contained_meaning",
        "low_context_dependency",
        "under_60_seconds",
    ]
    output["window_search"] = {
        "window_count": len(windows),
        "window_size_segments": window_size_segments,
        "window_stride_segments": window_stride_segments,
        "per_window_candidates": per_window_candidates,
        "searched_windows": searched_windows,
        "window_failures": window_failures,
    }

    output_path = CLIPS / f"{stem}.candidates.json"
    atomic_write_json(output_path, output, ensure_ascii=False)

    return {
        "ok": True,
        "source_stem": stem,
        "selection_model": _selection_model_name(),
        "rerank_model": _rerank_model_name(),
        "segment_count": len(segments),
        "window_count": len(windows),
        "window_failures": len(window_failures),
        "generated_candidate_count": len(generated_plan.clips),
        "clip_count": len(final_plan.clips),
        "output_json": str(output_path),
        "clips": output["clips"],
    }


@mcp.tool()
def list_saved_clip_plans(limit: int = 50) -> dict:
    """List saved clip candidate JSON files."""
    if not CLIPS.exists():
        return {"folder": str(CLIPS), "files": []}

    files = sorted(
        [p for p in CLIPS.glob("*.candidates.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "folder": str(CLIPS),
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
