from pathlib import Path
import os
import json
import re
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from mcp.server.fastmcp import FastMCP

from helpers import atomic_write_json, merge_refined_and_quran_segments, segment_render_text

ROOT = Path(os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))).resolve()
TRANSCRIPTS = ROOT / "transcripts"
CLIPS = ROOT / "clips"

load_dotenv(ROOT / ".env")

mcp = FastMCP("clip-factory-scene-director", json_response=True)

BLOCKED_VISUAL_TERMS = {
    "people", "person", "human", "humans", "man", "men", "woman", "women", "child", "children",
    "boy", "girl", "face", "faces", "hand", "hands", "body", "bodies", "crowd", "crowds", "silhouette",
    "speaker", "presenter", "talking head", "portrait", "selfie", "animal", "animals", "bird", "birds",
    "cat", "cats", "dog", "dogs", "horse", "horses", "camel", "camels", "insect", "insects",
    "logo", "logos", "text", "words", "letters", "caption", "captions",
    "شخص", "اشخاص", "إنسان", "انسان", "ناس", "رجل", "امرأة", "امراه", "طفل", "أطفال", "اطفال",
    "وجه", "وجوه", "يد", "أيدي", "ايدي", "حيوان", "حيوانات", "طير", "طيور", "شعار", "نص"
}

SCENIC_FALLBACKS = [
    "empty mosque courtyard at dawn",
    "desert dunes at sunrise",
    "moonlit prayer hall interior",
    "ocean waves at dusk",
    "rain on window at night",
    "mountain mist at sunrise",
    "lantern-lit stone alley with no people",
    "empty road under dramatic clouds",
]


class PromptIdea(BaseModel):
    topic: str = Field(description="Short scenic topic")
    prompt: str = Field(description="Final scenic video prompt")
    reason: str = Field(description="Why this scene matches the clip")
    energy: str = Field(description="One-word pacing feel such as calm, intense, reflective")


class PromptSet(BaseModel):
    summary: str = Field(description="One-sentence summary of the clip's emotional center")
    mood: str = Field(description="Short mood label")
    scene_prompts: List[PromptIdea] = Field(default_factory=list)


class VisualBeat(BaseModel):
    type: str = Field(description="Either 'original' or 'ai_video'")
    start_offset_sec: float = Field(ge=0, description="Segment start relative to clip start")
    end_offset_sec: float = Field(gt=0, description="Segment end relative to clip start")
    duration_sec: float = Field(gt=0, description="Segment duration")
    asset_slot: int = Field(default=0, ge=0, description="1-based slot for ai_video beats, else 0")
    prompt: str = Field(default="", description="Prompt for ai_video beat")
    notes: str = Field(default="", description="Notes for renderer or fetcher")


class RatioDecision(BaseModel):
    duration_sec: float
    mode: str
    insert_count: int
    original_ratio: float
    scenic_ratio: float
    intro_guard_sec: float
    outro_guard_sec: float
    ai_segment_length_sec: float


class VisualPlanEnvelope(BaseModel):
    source_stem: str
    clip_number: int
    clip_title: str
    clip_hook: str
    clip_start_sec: float
    clip_end_sec: float
    clip_duration_sec: float
    mode: str
    summary: str
    mood: str
    prompts: List[PromptIdea]
    visual_plan: List[VisualBeat]


class MatchScore(BaseModel):
    score: float = Field(ge=0, le=1)
    matched_terms: List[str] = Field(default_factory=list)
    notes: str = Field(default="")


class TimelinePreview(BaseModel):
    duration_sec: float
    mode: str
    beats: List[VisualBeat]


class ClipContext(BaseModel):
    source_stem: str
    clip_number: int
    clip_title: str
    clip_hook: str
    start_sec: float
    end_sec: float
    duration_sec: float
    add_broll: bool
    broll_query: str
    transcript_excerpt: str
    segment_count: int


class CandidateFileList(BaseModel):
    folder: str
    files: List[Dict[str, object]]


class HealthResponse(BaseModel):
    ok: bool
    root: str
    transcripts_exists: bool
    clips_exists: bool
    gemini_key_present: bool


_SENTENCE_BREAK_RE = re.compile(r"([،,:;؛.!?؟])")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "is", "are",
    "هذا", "هذه", "ذلك", "تلك", "في", "من", "على", "إلى", "الى", "عن", "ثم", "او", "أو", "و",
}


def _client() -> genai.Client:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env")
    return genai.Client(api_key=api_key)


def _model_name() -> str:
    return (
        os.environ.get("GEMINI_SCENE_DIRECTOR_MODEL")
        or os.environ.get("GEMINI_SELECTION_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-2.5-pro"
    )


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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip_duration(clip: dict) -> float:
    start = _safe_float(clip.get("start_sec"), 0.0)
    end = _safe_float(clip.get("end_sec"), 0.0)
    duration = _safe_float(clip.get("duration_sec"), max(0.0, end - start))
    if duration <= 0 and end > start:
        duration = end - start
    return round(duration, 2)


def _load_candidates(stem: str) -> dict:
    path = CLIPS / f"{stem}.candidates.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_file_path(stem: str) -> Path:
    return CLIPS / f"{stem}.candidates.json"


def _load_verbose_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_transcript_segments(stem: str) -> List[dict]:
    refined = _load_verbose_json(TRANSCRIPTS / f"{stem}.refined.verbose.json")
    quran = _load_verbose_json(TRANSCRIPTS / f"{stem}.quran_guard.verbose.json")
    plain = _load_verbose_json(TRANSCRIPTS / f"{stem}.verbose.json")

    refined_segments = (refined or {}).get("segments") or []
    quran_segments = (quran or {}).get("segments") or []
    plain_segments = (plain or {}).get("segments") or []

    if refined_segments and quran_segments:
        merged = merge_refined_and_quran_segments(refined_segments, quran_segments)
        if merged:
            return merged
    if quran_segments:
        return quran_segments
    if refined_segments:
        return refined_segments
    return plain_segments


def _segments_for_clip(stem: str, clip: dict) -> List[dict]:
    start = _safe_float(clip.get("start_sec"), 0.0)
    end = _safe_float(clip.get("end_sec"), 0.0)
    segments = _load_transcript_segments(stem)
    selected = []
    for seg in segments:
        seg_start = _safe_float(seg.get("start"), 0.0)
        seg_end = _safe_float(seg.get("end"), 0.0)
        if seg_end <= start or seg_start >= end:
            continue
        selected.append(seg)
    return selected


def _clip_text(stem: str, clip: dict) -> str:
    parts = []
    for seg in _segments_for_clip(stem, clip):
        txt = str(segment_render_text(seg) or "").replace("\n", " ").strip()
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()


def _trim_excerpt(text: str, max_chars: int = 1600) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _tokenize(text: str) -> List[str]:
    text = str(text or "").lower()
    text = _SENTENCE_BREAK_RE.sub(" ", text)
    text = re.sub(r"[^\w\u0600-\u06FF\s-]", " ", text)
    tokens = []
    for tok in text.split():
        tok = tok.strip("-_")
        if not tok or len(tok) < 3 or tok in _STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def _sanitize_visual_topic(topic: str) -> str:
    topic = str(topic or "").strip()
    if not topic:
        return SCENIC_FALLBACKS[0]
    lowered = topic.lower()
    for blocked in BLOCKED_VISUAL_TERMS:
        lowered = lowered.replace(blocked, " ")
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,.-")
    if not lowered:
        return SCENIC_FALLBACKS[0]
    return lowered


def _enforce_prompt(prompt: str) -> str:
    prompt = str(prompt or "").strip()
    prompt = re.sub(r"\s+", " ", prompt)
    lowered = prompt.lower()
    for blocked in sorted(BLOCKED_VISUAL_TERMS, key=len, reverse=True):
        lowered = lowered.replace(blocked, " ")
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,.-")
    if not lowered:
        lowered = SCENIC_FALLBACKS[0]
    suffix = (
        " photorealistic, cinematic, subtle natural motion, vertical 9:16, "
        "empty environment only, no people, no human faces, no hands, no animals, "
        "no birds, no insects, no text, no logos."
    )
    if lowered.endswith("."):
        lowered = lowered[:-1]
    return lowered + "," + suffix if not lowered.endswith(",") else lowered + suffix


def _fallback_prompt_set(clip: dict, beat_count: int) -> PromptSet:
    bases = [
        _sanitize_visual_topic(clip.get("broll_query") or ""),
        _sanitize_visual_topic(clip.get("title") or ""),
        _sanitize_visual_topic(clip.get("hook") or ""),
    ]
    bases = [b for b in bases if b]
    if not bases:
        bases = SCENIC_FALLBACKS[:]

    prompts: List[PromptIdea] = []
    seen = set()
    idx = 0
    while len(prompts) < beat_count:
        base = bases[idx % len(bases)] if idx < len(bases) else SCENIC_FALLBACKS[idx % len(SCENIC_FALLBACKS)]
        topic = _sanitize_visual_topic(base)
        if topic in seen:
            idx += 1
            continue
        seen.add(topic)
        prompts.append(
            PromptIdea(
                topic=topic,
                prompt=_enforce_prompt(topic),
                reason="Fallback scenic prompt derived from clip metadata",
                energy="reflective",
            )
        )
        idx += 1

    return PromptSet(
        summary=str(clip.get("why_it_works") or clip.get("title") or "Scenic reinforcement").strip() or "Scenic reinforcement",
        mood="reflective",
        scene_prompts=prompts,
    )


def _build_prompt_generation_request(stem: str, clip_number: int, clip: dict, clip_text: str, beat_count: int, mode: str) -> str:
    return f"""
You are a scene director for Arabic short-form videos.

Return ONLY valid JSON matching the schema.
No markdown.
No explanations outside JSON.

Your job is to create {beat_count} scenic insert ideas for a short vertical video.
These inserts must be symbolic or environmental only.

NON-NEGOTIABLE RULES:
- Never include humans, faces, hands, children, crowds, silhouettes, presenters, speakers, or any character action.
- Never include animals, birds, insects, or any living creature.
- Never include text, logos, signage text, UI, captions, calligraphy, or readable writing.
- Prefer empty architecture, weather, sky, water, dunes, roads, windows, light, mist, shadows, lanterns, and atmosphere.
- Every prompt must be suitable for cinematic short video generation.
- Prompts must be specific enough to visualize, but not cluttered.
- Prompts must be in English because the generation layer uses English prompts best.

EDITORIAL GOAL:
- Keep the original sermon footage as the backbone.
- Scenic inserts should amplify emotion, reflection, warning, awe, calm, or payoff.
- Do not summarize the speech literally. Find visual metaphors and matching atmosphere.
- Make adjacent prompts feel visually different from each other.

OUTPUT RULES:
- summary: 1 sentence about the emotional center of the clip
- mood: 1 to 3 words only
- scene_prompts: exactly {beat_count} items
- topic: 2 to 6 words
- prompt: 1 sentence, English, scenic only, no blocked content
- reason: very short
- energy: one word like calm, intense, reflective, urgent, hopeful

Mode: {mode}
Source stem: {stem}
Clip number: {clip_number}
Clip title: {clip.get('title', '')}
Clip hook: {clip.get('hook', '')}
Clip why_it_works: {clip.get('why_it_works', '')}
Clip broll_query: {clip.get('broll_query', '')}
Clip duration: {_clip_duration(clip)}

Clip transcript excerpt:
{_trim_excerpt(clip_text, 2400)}
""".strip()


def _generate_prompt_set(stem: str, clip_number: int, clip: dict, beat_count: int, mode: str) -> PromptSet:
    clip_text = _clip_text(stem, clip)
    if not clip_text:
        return _fallback_prompt_set(clip, beat_count)

    try:
        client = _client()
        response = client.models.generate_content(
            model=_model_name(),
            contents=_build_prompt_generation_request(stem, clip_number, clip, clip_text, beat_count, mode),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": PromptSet.model_json_schema(),
            },
        )
        prompt_set = PromptSet.model_validate_json(response.text)
        cleaned: List[PromptIdea] = []
        for idea in prompt_set.scene_prompts[:beat_count]:
            topic = _sanitize_visual_topic(idea.topic)
            prompt = _enforce_prompt(idea.prompt or topic)
            cleaned.append(
                PromptIdea(
                    topic=topic,
                    prompt=prompt,
                    reason=str(idea.reason or "").strip() or "Scene supports the spoken meaning",
                    energy=str(idea.energy or "reflective").strip() or "reflective",
                )
            )
        if not cleaned:
            return _fallback_prompt_set(clip, beat_count)
        while len(cleaned) < beat_count:
            fallback = _fallback_prompt_set(clip, beat_count).scene_prompts[len(cleaned)]
            cleaned.append(fallback)
        return PromptSet(
            summary=str(prompt_set.summary or clip.get("why_it_works") or clip.get("title") or "Scenic reinforcement").strip() or "Scenic reinforcement",
            mood=str(prompt_set.mood or "reflective").strip() or "reflective",
            scene_prompts=cleaned[:beat_count],
        )
    except Exception:
        return _fallback_prompt_set(clip, beat_count)


def _decide_ratio(duration_sec: float, mode: str = "balanced") -> RatioDecision:
    duration_sec = max(1.0, float(duration_sec))
    mode = (mode or "balanced").strip().lower()

    if mode not in {"light", "balanced", "cinematic"}:
        mode = "balanced"

    if duration_sec < 22:
        insert_count = 1
    elif duration_sec < 38:
        insert_count = 2
    else:
        insert_count = 3

    if mode == "light":
        insert_count = max(1, insert_count - 1)
        ai_len = 1.6
        intro_guard = 2.1
        outro_guard = 2.1
    elif mode == "cinematic":
        ai_len = 2.2 if insert_count <= 2 else 1.8
        intro_guard = 1.7
        outro_guard = 1.8
    else:
        ai_len = 1.9 if insert_count <= 2 else 1.7
        intro_guard = 1.9
        outro_guard = 2.0

    scenic_total = min(duration_sec * 0.42, insert_count * ai_len)
    original_total = max(0.0, duration_sec - scenic_total)
    return RatioDecision(
        duration_sec=round(duration_sec, 2),
        mode=mode,
        insert_count=insert_count,
        original_ratio=round(original_total / duration_sec, 4),
        scenic_ratio=round(scenic_total / duration_sec, 4),
        intro_guard_sec=round(intro_guard, 2),
        outro_guard_sec=round(outro_guard, 2),
        ai_segment_length_sec=round(ai_len, 2),
    )


def _schedule_visual_plan(duration_sec: float, prompts: List[PromptIdea], mode: str = "balanced") -> List[VisualBeat]:
    ratio = _decide_ratio(duration_sec, mode=mode)
    duration = ratio.duration_sec
    prompt_count = min(len(prompts), ratio.insert_count)
    if prompt_count <= 0:
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

    centers_map = {1: [0.50], 2: [0.36, 0.70], 3: [0.24, 0.52, 0.78]}
    centers = centers_map.get(prompt_count, [0.50])
    ai_len = ratio.ai_segment_length_sec
    intro_guard = ratio.intro_guard_sec
    outro_guard = ratio.outro_guard_sec
    latest_start = max(intro_guard, duration - outro_guard - ai_len)

    ai_beats: List[VisualBeat] = []
    last_end = max(0.0, intro_guard - 0.65)
    for idx, center in enumerate(centers, start=1):
        desired_start = max(intro_guard, duration * center - ai_len / 2.0)
        start_offset = max(last_end + 0.7, desired_start)
        start_offset = min(start_offset, latest_start)
        end_offset = min(duration - outro_guard, start_offset + ai_len)
        if end_offset - start_offset < 1.1:
            continue
        prompt = prompts[idx - 1]
        ai_beats.append(
            VisualBeat(
                type="ai_video",
                start_offset_sec=round(start_offset, 2),
                end_offset_sec=round(end_offset, 2),
                duration_sec=round(end_offset - start_offset, 2),
                asset_slot=idx,
                prompt=prompt.prompt,
                notes=f"{prompt.energy}:{prompt.reason}",
            )
        )
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

    beats: List[VisualBeat] = []
    cursor = 0.0
    for beat in ai_beats:
        if beat.start_offset_sec - cursor >= 0.3:
            beats.append(
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
        beats.append(beat)
        cursor = beat.end_offset_sec

    if duration - cursor >= 0.3:
        beats.append(
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

    return [beat for beat in beats if beat.duration_sec > 0.2]


def _score_prompt_against_text(prompt: str, clip_text: str) -> MatchScore:
    prompt_tokens = set(_tokenize(prompt))
    clip_tokens = set(_tokenize(clip_text))
    matched = sorted(prompt_tokens & clip_tokens)

    score = 0.35
    score += min(0.35, len(matched) * 0.08)
    if any(tok in prompt.lower() for tok in ["dawn", "night", "rain", "ocean", "desert", "mist", "mosque", "courtyard"]):
        score += 0.1
    if "no people" in prompt.lower() and "no animals" in prompt.lower():
        score += 0.1
    if len(prompt) > 200:
        score -= 0.08
    score = max(0.0, min(1.0, score))

    note = "Good symbolic fit" if score >= 0.7 else "Usable but generic" if score >= 0.5 else "Weak semantic fit"
    return MatchScore(score=round(score, 4), matched_terms=matched[:12], notes=note)


def _get_clip(stem: str, clip_number: int) -> Tuple[dict, dict, List[dict]]:
    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if clip_number < 1 or clip_number > len(clips):
        raise ValueError(f"clip_number must be between 1 and {len(clips)}")
    clip = clips[clip_number - 1]
    clip_segments = _segments_for_clip(stem, clip)
    return data, clip, clip_segments


@mcp.tool()
def healthcheck() -> dict:
    return HealthResponse(
        ok=True,
        root=str(ROOT),
        transcripts_exists=TRANSCRIPTS.exists(),
        clips_exists=CLIPS.exists(),
        gemini_key_present=bool((os.environ.get("GEMINI_API_KEY") or "").strip()),
    ).model_dump()


@mcp.tool()
def list_candidate_plans(limit: int = 50) -> dict:
    CLIPS.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in CLIPS.glob("*.candidates.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return CandidateFileList(
        folder=str(CLIPS),
        files=[
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
            for p in files[:limit]
        ],
    ).model_dump()


@mcp.tool()
def inspect_clip_context(stem: str, clip_number: int) -> dict:
    _, clip, clip_segments = _get_clip(stem, clip_number)
    excerpt = _trim_excerpt(_clip_text(stem, clip), 1800)
    return ClipContext(
        source_stem=stem,
        clip_number=clip_number,
        clip_title=str(clip.get("title") or ""),
        clip_hook=str(clip.get("hook") or ""),
        start_sec=round(_safe_float(clip.get("start_sec"), 0.0), 2),
        end_sec=round(_safe_float(clip.get("end_sec"), 0.0), 2),
        duration_sec=_clip_duration(clip),
        add_broll=bool(clip.get("add_broll", False)),
        broll_query=str(clip.get("broll_query") or ""),
        transcript_excerpt=excerpt,
        segment_count=len(clip_segments),
    ).model_dump()


@mcp.tool()
def enforce_no_faces_policy(prompt: str) -> dict:
    cleaned = _enforce_prompt(prompt)
    removed = sorted([term for term in BLOCKED_VISUAL_TERMS if term in str(prompt or "").lower()])
    return {
        "ok": True,
        "original_prompt": prompt,
        "cleaned_prompt": cleaned,
        "removed_terms": removed,
    }


@mcp.tool()
def decide_original_vs_scenic_ratio(duration_sec: float, mode: str = "balanced") -> dict:
    return _decide_ratio(duration_sec, mode=mode).model_dump()


@mcp.tool()
def suggest_cut_timeline(duration_sec: float, mode: str = "balanced") -> dict:
    ratio = _decide_ratio(duration_sec, mode=mode)
    dummy_prompts = [
        PromptIdea(topic=f"scene {i}", prompt=_enforce_prompt(f"scenic environment {i}"), reason="preview", energy="reflective")
        for i in range(1, ratio.insert_count + 1)
    ]
    beats = _schedule_visual_plan(duration_sec, dummy_prompts, mode=mode)
    return TimelinePreview(duration_sec=round(duration_sec, 2), mode=ratio.mode, beats=beats).model_dump()


@mcp.tool()
def generate_scenic_prompts(stem: str, clip_number: int, mode: str = "balanced") -> dict:
    _, clip, _ = _get_clip(stem, clip_number)
    ratio = _decide_ratio(_clip_duration(clip), mode=mode)
    prompt_set = _generate_prompt_set(stem, clip_number, clip, beat_count=ratio.insert_count, mode=ratio.mode)
    matches = [
        _score_prompt_against_text(idea.prompt, _clip_text(stem, clip)).model_dump()
        for idea in prompt_set.scene_prompts
    ]
    return {
        "ok": True,
        "source_stem": stem,
        "clip_number": clip_number,
        "mode": ratio.mode,
        "summary": prompt_set.summary,
        "mood": prompt_set.mood,
        "prompts": [idea.model_dump() for idea in prompt_set.scene_prompts],
        "prompt_match_scores": matches,
    }


@mcp.tool()
def score_visual_match_to_script(stem: str, clip_number: int, prompt: str) -> dict:
    _, clip, _ = _get_clip(stem, clip_number)
    score = _score_prompt_against_text(prompt, _clip_text(stem, clip))
    return {
        "ok": True,
        "source_stem": stem,
        "clip_number": clip_number,
        "prompt": prompt,
        "score": score.model_dump(),
    }


@mcp.tool()
def build_visual_plan(
    stem: str,
    clip_number: int,
    mode: str = "balanced",
    overwrite: bool = True,
    update_candidates: bool = True,
) -> dict:
    data, clip, _ = _get_clip(stem, clip_number)
    duration = _clip_duration(clip)
    ratio = _decide_ratio(duration, mode=mode)
    prompt_set = _generate_prompt_set(stem, clip_number, clip, beat_count=ratio.insert_count, mode=ratio.mode)
    plan = _schedule_visual_plan(duration, prompt_set.scene_prompts, mode=ratio.mode)

    envelope = VisualPlanEnvelope(
        source_stem=stem,
        clip_number=clip_number,
        clip_title=str(clip.get("title") or ""),
        clip_hook=str(clip.get("hook") or ""),
        clip_start_sec=round(_safe_float(clip.get("start_sec"), 0.0), 2),
        clip_end_sec=round(_safe_float(clip.get("end_sec"), 0.0), 2),
        clip_duration_sec=duration,
        mode=ratio.mode,
        summary=prompt_set.summary,
        mood=prompt_set.mood,
        prompts=prompt_set.scene_prompts,
        visual_plan=plan,
    )

    plan_path = CLIPS / f"{stem}__clip{clip_number:02d}.scene_plan.json"
    if plan_path.exists() and not overwrite:
        return {
            "ok": True,
            "message": "Scene plan already exists",
            "scene_plan_file": str(plan_path),
        }

    atomic_write_json(plan_path, envelope.model_dump(), ensure_ascii=False)

    if update_candidates:
        clips = data.get("clips") or []
        clip_ref = clips[clip_number - 1]
        clip_ref["visual_mode"] = "alternate_scenic"
        clip_ref["visual_summary"] = prompt_set.summary
        clip_ref["visual_mood"] = prompt_set.mood
        clip_ref["visual_ratio"] = ratio.model_dump()
        clip_ref["visual_prompts"] = [idea.model_dump() for idea in prompt_set.scene_prompts]
        clip_ref["visual_plan"] = [beat.model_dump() for beat in plan]
        if not clip_ref.get("broll_query") and prompt_set.scene_prompts:
            clip_ref["broll_query"] = prompt_set.scene_prompts[0].topic
        atomic_write_json(_candidate_file_path(stem), data, ensure_ascii=False)

    return {
        "ok": True,
        "source_stem": stem,
        "clip_number": clip_number,
        "mode": ratio.mode,
        "scene_plan_file": str(plan_path),
        "updated_candidate_file": str(_candidate_file_path(stem)) if update_candidates else "",
        "insert_count": ratio.insert_count,
        "summary": prompt_set.summary,
        "mood": prompt_set.mood,
        "visual_plan": [beat.model_dump() for beat in plan],
        "prompts": [idea.model_dump() for idea in prompt_set.scene_prompts],
    }


@mcp.tool()
def build_visual_plans_for_stem(
    stem: str,
    mode: str = "balanced",
    only_add_broll: bool = True,
    overwrite: bool = True,
) -> dict:
    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if not clips:
        return {"ok": False, "error": "No clip candidates found"}

    results = []
    targets = []
    for idx, clip in enumerate(clips, start=1):
        if only_add_broll and not bool(clip.get("add_broll", False)):
            continue
        targets.append(idx)

    if not targets:
        return {
            "ok": True,
            "source_stem": stem,
            "message": "No clips matched the requested filter",
            "results": [],
        }

    for clip_number in targets:
        try:
            results.append(
                build_visual_plan(
                    stem=stem,
                    clip_number=clip_number,
                    mode=mode,
                    overwrite=overwrite,
                    update_candidates=True,
                )
            )
        except Exception as e:
            results.append({"ok": False, "clip_number": clip_number, "error": str(e)})

    success_count = sum(1 for r in results if r.get("ok"))
    manifest_path = CLIPS / f"{stem}.scene_director.manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "stem": stem,
            "mode": mode,
            "only_add_broll": only_add_broll,
            "results": results,
        },
        ensure_ascii=False,
    )

    return {
        "ok": True,
        "source_stem": stem,
        "mode": mode,
        "requested": targets,
        "success_count": success_count,
        "total_count": len(results),
        "manifest": str(manifest_path),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run()
