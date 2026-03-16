from pathlib import Path
import os
import json
import re
import datetime
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# ── Bootstrap: resolve ROOT and load .env before any other import ─────────────
# helpers must be imported AFTER dotenv so that any env-var-gated behaviour
# in helpers (e.g. module-level constants) sees the correct values.
# Two-phase so that CLIP_FACTORY_ROOT inside .env is respected.
_INITIAL_ROOT = Path(
    os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
).resolve()
load_dotenv(_INITIAL_ROOT / ".env")

ROOT = Path(
    os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
).resolve()
if ROOT != _INITIAL_ROOT and (ROOT / ".env").exists():
    load_dotenv(ROOT / ".env", override=True)

# helpers imported here — AFTER dotenv — so text_config env vars are visible.
from helpers import atomic_write_json, segment_render_text  # noqa: E402
import llm_client

TRANSCRIPTS = ROOT / "transcripts"
CLIPS = ROOT / "clips"

mcp = FastMCP("clip-factory-scene-director", json_response=True)

# ── Plan versioning ────────────────────────────────────────────────────────────
# Bump when scheduling logic, style targets, or prompt policy changes so callers
# can detect stale plans in candidates.json and know to re-run build_visual_plan.
SCENE_PLAN_POLICY_VERSION: str = "2"

# ── Renderer-compatible knobs ──────────────────────────────────────────────────
# These mirror the renderer's own env vars so plans carry the rendering intent
# they were designed for.  The renderer does NOT read these from scene plans
# (it uses its own env), but storing them in the plan makes the design intent
# explicit and allows future tooling to validate plan/renderer alignment.

# Transition settings — must match RENDER_TRANSITION_TYPE / RENDER_TRANSITION_DURATION
# in the renderer's .env for dissolves to feel intentional.
_RENDER_TRANSITION_T: float = float(
    os.environ.get("RENDER_TRANSITION_DURATION", "0.4")
)
_RENDER_TRANSITION_TYPE: str = (
    os.environ.get("RENDER_TRANSITION_TYPE", "dissolve").strip().lower() or "dissolve"
)

# Recommended renderer color-grade preset for this planning style.
# Matches the "dark-soft-recitation" preset in renderer_server_veo_timeline.py.
# Override via SCENE_RECOMMENDED_PRESET in .env when using a different preset.
_RECOMMENDED_PRESET: str = (
    os.environ.get("SCENE_RECOMMENDED_PRESET", "dark-soft-recitation").strip()
    or "dark-soft-recitation"
)

# Text readability hint.  "dark" tells the renderer the scenic beats are intended
# to be underexposed so white/cream text is readable without adaptive sampling.
# Values: dark | neutral | bright.
_DARKNESS_INTENT: str = (
    os.environ.get("SCENE_DARKNESS_INTENT", "dark").strip().lower() or "dark"
)

# ── Blocked visual terms ───────────────────────────────────────────────────────
BLOCKED_VISUAL_TERMS = {
    "people", "person", "human", "humans", "man", "men", "woman", "women",
    "child", "children", "boy", "girl", "face", "faces", "hand", "hands",
    "body", "bodies", "crowd", "crowds", "silhouette", "speaker", "presenter",
    "talking head", "portrait", "selfie",
    "animal", "animals", "bird", "birds", "cat", "cats", "dog", "dogs",
    "horse", "horses", "camel", "camels", "insect", "insects",
    "logo", "logos", "text", "words", "letters", "caption", "captions",
    "شخص", "اشخاص", "إنسان", "انسان", "ناس", "رجل", "امرأة", "امراه",
    "طفل", "أطفال", "اطفال", "وجه", "وجوه", "يد", "أيدي", "ايدي",
    "حيوان", "حيوانات", "طير", "طيور", "شعار", "نص",
}

# ── Fallback prompts ───────────────────────────────────────────────────────────
# Aligned with the target look: dark, soft, slow-moving, spiritually resonant.
# These are used when Gemini is unavailable or the clip has no usable text.
SCENIC_FALLBACKS = [
    "dark stone mosque corridor at night, warm amber lantern light on ancient carved walls",
    "slow-moving river at deep blue dusk, mist settling on still dark water",
    "moonlit desert at night, long dune shadows, absolute stillness, no horizon line",
    "empty prayer hall at pre-dawn, dim hanging lanterns casting soft circles of light",
    "rain-streaked dark window at night, soft street light refracting through drops",
    "ancient stone minaret against deep violet twilight sky, no movement except clouds",
    "candle flame in a dark stone alcove, soft gold light on Islamic geometric carving",
    "mist rolling slowly over mountain ridgeline at dawn, dark foreground, pale sky",
]


# ── Pydantic models ────────────────────────────────────────────────────────────

class PromptIdea(BaseModel):
    topic: str = Field(description="Short scenic topic")
    prompt: str = Field(description="Final scenic video prompt")
    reason: str = Field(description="Why this scene matches the clip")
    energy: str = Field(description="One-word pacing feel: calm, reflective, awe, solemn, hopeful")


class PromptSet(BaseModel):
    summary: str = Field(description="One-sentence summary of the clip's emotional center")
    mood: str = Field(description="Short mood label — 1 to 3 words")
    scene_prompts: List[PromptIdea] = Field(default_factory=list)


class VisualBeat(BaseModel):
    # type is open to three values so stock footage can be planned alongside AI.
    type: str = Field(
        description=(
            "Beat source: 'original' (speaker/reciter footage) or "
            "'stock_video' (real scenic stock clip)"
        )
    )
    start_offset_sec: float = Field(ge=0, description="Segment start relative to clip start")
    end_offset_sec: float = Field(gt=0, description="Segment end relative to clip start")
    duration_sec: float = Field(gt=0, description="Segment duration in seconds")
    asset_slot: int = Field(default=0, ge=0, description="1-based slot for stock_video beats; 0 for original")
    prompt: str = Field(default="", description="Stock search hint for stock_video beats")
    notes: str = Field(default="", description="Renderer notes: energy:reason or fetcher instructions")


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
    """
    Complete scene plan written to {stem}__clip{N}.scene_plan.json.

    Fields added in plan_policy_version 2:
      visual_mode, plan_policy_version, transition_style,
      transition_duration_sec, recommended_preset, darkness_intent.

    These are advisory — the renderer uses its own .env values — but storing
    them in the plan makes the design intent explicit and detectable.
    """
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
    # ── renderer-compatible metadata (v2) ─────────────────────────────────────
    visual_mode: str = Field(
        default="mixed_stock",
        description="Derived from beat types: speaker_only | mixed_stock",
    )
    plan_policy_version: str = Field(
        default="1",
        description="Bumped when scheduling logic changes; helps detect stale plans",
    )
    transition_style: str = Field(
        default="dissolve",
        description="Recommended xfade type for the renderer (dissolve, fade, …)",
    )
    transition_duration_sec: float = Field(
        default=0.4,
        description="Recommended xfade duration the plan was sized for",
    )
    recommended_preset: str = Field(
        default="dark-soft-recitation",
        description="Recommended renderer color-grade preset",
    )
    darkness_intent: str = Field(
        default="dark",
        description="Text readability hint: dark | neutral | bright",
    )


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


def _model_name() -> str:
    legacy = (
        os.environ.get("GEMINI_SCENE_DIRECTOR_MODEL")
        or os.environ.get("GEMINI_SELECTION_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or ""
    ).strip()
    return legacy or llm_client._text_model_name()


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


def _load_transcript_segments(stem: str, clip_number: Optional[int] = None) -> List[dict]:
    """
    Load the best available transcript segments for a stem (and optionally a
    specific clip).

    Priority mirrors renderer_server_veo_timeline._load_transcript_segments:
      clip-level (if clip_number is given):
        1. {clip}.quran_guard.verbose.json  — corpus-canonical text
        2. {clip}.refined.verbose.json      — Gemini-corrected ASR
      stem-level fallback:
        3. {stem}.quran_guard.verbose.json
        4. {stem}.refined.verbose.json
        5. {stem}.verbose.json              — raw Whisper transcription

    Clip-level files are checked first so scene_director uses the same
    canonical text that the renderer will use for subtitle generation.
    """
    candidates = []
    if clip_number is not None:
        clip_stem = f"{stem}__clip{clip_number:02d}"
        candidates += [
            TRANSCRIPTS / f"{clip_stem}.quran_guard.verbose.json",
            TRANSCRIPTS / f"{clip_stem}.refined.verbose.json",
        ]
    candidates += [
        TRANSCRIPTS / f"{stem}.quran_guard.verbose.json",
        TRANSCRIPTS / f"{stem}.refined.verbose.json",
        TRANSCRIPTS / f"{stem}.verbose.json",
    ]
    for candidate in candidates:
        data = _load_verbose_json(candidate)
        if data:
            segs = data.get("segments") or []
            if segs:
                return segs
    return []


def _segments_for_clip(stem: str, clip: dict, clip_number: Optional[int] = None) -> List[dict]:
    start = _safe_float(clip.get("start_sec"), 0.0)
    end = _safe_float(clip.get("end_sec"), 0.0)
    segments = _load_transcript_segments(stem, clip_number=clip_number)
    selected = []
    for seg in segments:
        seg_start = _safe_float(seg.get("start"), 0.0)
        seg_end = _safe_float(seg.get("end"), 0.0)
        if seg_end <= start or seg_start >= end:
            continue
        selected.append(seg)
    return selected


def _clip_text(stem: str, clip: dict, clip_number: Optional[int] = None) -> str:
    parts = []
    for seg in _segments_for_clip(stem, clip, clip_number=clip_number):
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
        # Word-boundary match so "man-made" doesn't become "-made".
        lowered = re.sub(r"\b" + re.escape(blocked) + r"\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,.-")
    if not lowered:
        return SCENIC_FALLBACKS[0]
    return lowered


def _enforce_prompt(prompt: str) -> str:
    """
    Sanitize a scenic prompt and append the mandatory cinematic suffix.

    Uses word-boundary matching for blocked terms so compound words like
    "man-made" are not corrupted.  The suffix encodes the target look:
    dark, soft, slightly underexposed, slow natural motion.
    """
    prompt = str(prompt or "").strip()
    prompt = re.sub(r"\s+", " ", prompt)
    lowered = prompt.lower()
    # Sort by length descending so longer phrases match before their subwords.
    for blocked in sorted(BLOCKED_VISUAL_TERMS, key=len, reverse=True):
        lowered = re.sub(r"\b" + re.escape(blocked) + r"\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,.-")
    if not lowered:
        lowered = SCENIC_FALLBACKS[0]
    # Suffix encodes the target visual style so downstream generators receive
    # it even if the upstream Gemini call did not include it explicitly.
    suffix = (
        " photorealistic, cinematic, dark and soft, slightly underexposed, "
        "slow subtle natural motion, vertical 9:16, empty environment only, "
        "no people, no human faces, no hands, no animals, no birds, "
        "no insects, no text, no logos, smooth gradual movement only."
    )
    if lowered.endswith("."):
        lowered = lowered[:-1]
    return lowered + "," + suffix if not lowered.endswith(",") else lowered + suffix


def _fallback_prompt_set(clip: dict, beat_count: int) -> PromptSet:
    """
    Build a PromptSet from clip metadata when Gemini is unavailable.

    Falls back to SCENIC_FALLBACKS (dark, atmospheric) rather than generic
    landscape terms.  Tries clip broll_query and title first for relevance,
    then rotates through SCENIC_FALLBACKS to fill remaining beats.
    """
    bases = [
        _sanitize_visual_topic(clip.get("broll_query") or ""),
        _sanitize_visual_topic(clip.get("title") or ""),
        _sanitize_visual_topic(clip.get("hook") or ""),
    ]
    bases = [b for b in bases if b]
    if not bases:
        bases = list(SCENIC_FALLBACKS)

    prompts: List[PromptIdea] = []
    seen: set = set()
    idx = 0
    fallback_idx = 0
    while len(prompts) < beat_count:
        if idx < len(bases):
            topic = _sanitize_visual_topic(bases[idx])
            idx += 1
        else:
            topic = _sanitize_visual_topic(SCENIC_FALLBACKS[fallback_idx % len(SCENIC_FALLBACKS)])
            fallback_idx += 1
        if not topic or topic in seen:
            continue
        seen.add(topic)
        prompts.append(
            PromptIdea(
                topic=topic,
                prompt=_enforce_prompt(topic),
                reason="Atmospheric fallback aligned with dark-soft cinematic style",
                energy="reflective",
            )
        )

    return PromptSet(
        summary=str(clip.get("why_it_works") or clip.get("title") or "Scenic reinforcement").strip()
        or "Scenic reinforcement",
        mood="reflective",
        scene_prompts=prompts,
    )


def _build_prompt_generation_request(
    stem: str, clip_number: int, clip: dict, clip_text: str, beat_count: int, mode: str
) -> str:
    return f"""
You are a scene director for Arabic short-form vertical videos.

Return ONLY valid JSON matching the schema.
No markdown. No explanations outside JSON.

Your job: create {beat_count} scenic insert ideas for a short vertical video of Arabic recitation or Islamic teaching.

═══════════════════════════════════════════════════
NON-NEGOTIABLE SAFETY RULES
═══════════════════════════════════════════════════
- Never include humans, faces, hands, children, crowds, silhouettes, or any character action.
- Never include animals, birds, insects, or any living creature.
- Never include text, logos, signage, calligraphy, readable writing, or UI elements.
- If a prompt breaks these rules, replace it with a dark atmospheric scene from nature or architecture.

═══════════════════════════════════════════════════
TARGET VISUAL STYLE — READ CAREFULLY
═══════════════════════════════════════════════════
The final video has Arabic text burned over the footage in cream/white.
The scenic inserts must act as a DARK CANVAS for that text.

This means:
- Scenes must be dark, soft, and slightly underexposed — never bright or saturated.
- Preferred lighting: blue hour, deep dusk, pre-dawn, night, interior lantern light.
- Preferred subjects: empty mosque interiors, water at night, mist, stone architecture,
  desert at night, dark skies, rain on glass, candle or lantern light, deep shadows.
- Avoid: noon daylight, bright sky, white sand in sun, bright green nature, busy patterns.
- Motion must be SLOW and SUBTLE — drifting mist, gentle water, rising smoke, drifting clouds.
  Fast motion, rapid cuts, or chaotic movement are wrong for this context.
- Every scene should feel spiritually weighty and contemplative — not decorative.

═══════════════════════════════════════════════════
EDITORIAL GOAL
═══════════════════════════════════════════════════
- The speaker/reciter footage is the backbone — inserts should deepen the feeling, not distract.
- Each insert should be a visual metaphor for the spoken meaning, NOT a literal illustration.
- Adjacent inserts must feel visually different from each other (different subject, different light).
- Match the emotional arc: awe → reflection → payoff. Do not frontload the heaviest imagery.

═══════════════════════════════════════════════════
OUTPUT RULES
═══════════════════════════════════════════════════
- summary: 1 sentence about the emotional center of this clip
- mood: 1 to 3 words, lowercase
- scene_prompts: exactly {beat_count} items
- topic: 2 to 6 words, English
- prompt: 1 sentence, English, scenic only, dark-soft style, no blocked content
- reason: very short, why this scene fits this specific moment
- energy: one word — calm, awe, solemn, reflective, hopeful, or urgent

Mode: {mode}
Source stem: {stem}
Clip number: {clip_number}
Clip title: {clip.get('title', '')}
Clip hook: {clip.get('hook', '')}
Clip why_it_works: {clip.get('why_it_works', '')}
Clip broll_query: {clip.get('broll_query', '')}
Clip duration: {_clip_duration(clip)}s

Clip transcript excerpt (Arabic — use for emotional/thematic context only):
{_trim_excerpt(clip_text, 2400)}
""".strip()


def _generate_prompt_set(stem: str, clip_number: int, clip: dict, beat_count: int, mode: str) -> PromptSet:
    clip_text = _clip_text(stem, clip, clip_number=clip_number)
    if not clip_text:
        return _fallback_prompt_set(clip, beat_count)

    # When SCENE_DIRECTOR_USE_LLM=false (the default), skip the expensive LLM call
    # and fall through to the deterministic fallback immediately.
    if not llm_client.scene_director_use_llm():
        return _fallback_prompt_set(clip, beat_count)

    try:
        llm = llm_client.get_text_llm()
        raw = llm.generate_json(
            _build_prompt_generation_request(stem, clip_number, clip, clip_text, beat_count, mode)
        )
        prompt_set = PromptSet.model_validate_json(raw)
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
            summary=str(
                prompt_set.summary or clip.get("why_it_works") or clip.get("title") or "Scenic reinforcement"
            ).strip() or "Scenic reinforcement",
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

    # Pad beat length so the fully-visible scenic window equals the intended
    # ai_len after both dissolve edges are consumed by xfade.
    # Example: ai_len=1.7, T=0.4 → stored beat=2.5s → visible=1.7s ✓
    # When _RENDER_TRANSITION_T=0.0 (default), no padding is applied.
    ai_len = round(ai_len + 2.0 * _RENDER_TRANSITION_T, 2)

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


def _schedule_visual_plan(
    duration_sec: float, prompts: List[PromptIdea], mode: str = "balanced"
) -> List[VisualBeat]:
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
                type="stock_video",
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


def _derive_visual_mode(plan: List[VisualBeat]) -> str:
    """
    Derive the visual_mode string from the actual beat types in a plan.

    Replaces the old hardcoded "alternate_scenic" with a value that describes
    what the plan actually contains:
      speaker_only  - all original footage
      mixed_stock   - original + real stock scenic clips
    """
    types = {b.type for b in plan}
    has_stock = "stock_video" in types
    if has_stock:
        return "mixed_stock"
    return "speaker_only"


def _score_prompt_against_text(prompt: str, clip_text: str) -> MatchScore:
    prompt_tokens = set(_tokenize(prompt))
    clip_tokens = set(_tokenize(clip_text))
    matched = sorted(prompt_tokens & clip_tokens)

    score = 0.35
    score += min(0.35, len(matched) * 0.08)
    if any(
        tok in prompt.lower()
        for tok in ["dawn", "night", "rain", "ocean", "desert", "mist", "mosque", "courtyard", "dusk", "dark"]
    ):
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
    clip_segments = _segments_for_clip(stem, clip, clip_number=clip_number)
    return data, clip, clip_segments


# ── MCP tools ──────────────────────────────────────────────────────────────────

@mcp.tool()
def healthcheck() -> dict:
    result = HealthResponse(
        ok=True,
        root=str(ROOT),
        transcripts_exists=TRANSCRIPTS.exists(),
        clips_exists=CLIPS.exists(),
        gemini_key_present=bool((os.environ.get("GEMINI_API_KEY") or "").strip()),
    ).model_dump()
    result["provider"] = llm_client.provider_summary()
    result["scene_director_use_llm"] = llm_client.scene_director_use_llm()
    return result


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
            {"name": p.name, "path": str(p), "size_bytes": p.stat().st_size}
            for p in files[:limit]
        ],
    ).model_dump()


@mcp.tool()
def inspect_clip_context(stem: str, clip_number: int) -> dict:
    _, clip, clip_segments = _get_clip(stem, clip_number)
    # Pass clip_number so clip-level transcript is preferred over stem-level.
    excerpt = _trim_excerpt(_clip_text(stem, clip, clip_number=clip_number), 1800)
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
        PromptIdea(
            topic=f"scene {i}",
            prompt=_enforce_prompt(f"dark atmospheric scenic environment {i}"),
            reason="preview",
            energy="reflective",
        )
        for i in range(1, ratio.insert_count + 1)
    ]
    beats = _schedule_visual_plan(duration_sec, dummy_prompts, mode=mode)
    return TimelinePreview(
        duration_sec=round(duration_sec, 2), mode=ratio.mode, beats=beats
    ).model_dump()


@mcp.tool()
def generate_scenic_prompts(stem: str, clip_number: int, mode: str = "balanced") -> dict:
    _, clip, _ = _get_clip(stem, clip_number)
    ratio = _decide_ratio(_clip_duration(clip), mode=mode)
    prompt_set = _generate_prompt_set(
        stem, clip_number, clip, beat_count=ratio.insert_count, mode=ratio.mode
    )
    matches = [
        _score_prompt_against_text(
            idea.prompt, _clip_text(stem, clip, clip_number=clip_number)
        ).model_dump()
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
    score = _score_prompt_against_text(
        prompt, _clip_text(stem, clip, clip_number=clip_number)
    )
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
    """
    Generate a visual scene plan for one clip and write it to disk.

    Writes two artifacts:
      {stem}__clip{N}.scene_plan.json   — full VisualPlanEnvelope (always)
      {stem}.candidates.json            — mutates the clip entry (if update_candidates=True)

    Candidate mutation writes only the fields owned by scene_director:
      visual_mode, visual_summary, visual_mood, visual_ratio, visual_prompts,
      visual_plan, plan_policy_version, plan_generated_at.

    broll_query is left unchanged if already present — it is set by the upstream
    clip selection pipeline and scene_director should not overwrite it.
    """
    data, clip, _ = _get_clip(stem, clip_number)
    duration = _clip_duration(clip)
    ratio = _decide_ratio(duration, mode=mode)
    prompt_set = _generate_prompt_set(
        stem, clip_number, clip, beat_count=ratio.insert_count, mode=ratio.mode
    )
    plan = _schedule_visual_plan(duration, prompt_set.scene_prompts, mode=ratio.mode)
    derived_mode = _derive_visual_mode(plan)

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
        visual_mode=derived_mode,
        plan_policy_version=SCENE_PLAN_POLICY_VERSION,
        transition_style=_RENDER_TRANSITION_TYPE,
        transition_duration_sec=_RENDER_TRANSITION_T,
        recommended_preset=_RECOMMENDED_PRESET,
        darkness_intent=_DARKNESS_INTENT,
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
        # Write only scene_director-owned fields.  Do not blast unrelated fields.
        clip_ref["visual_mode"] = derived_mode          # was hardcoded "alternate_scenic"
        clip_ref["visual_summary"] = prompt_set.summary
        clip_ref["visual_mood"] = prompt_set.mood
        clip_ref["visual_ratio"] = ratio.model_dump()
        clip_ref["visual_prompts"] = [idea.model_dump() for idea in prompt_set.scene_prompts]
        clip_ref["visual_plan"] = [beat.model_dump() for beat in plan]
        clip_ref["plan_policy_version"] = SCENE_PLAN_POLICY_VERSION
        clip_ref["plan_generated_at"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        # Only set broll_query if the clip does not already have one.
        # broll_query is owned by the upstream clip-selection pipeline.
        if not clip_ref.get("broll_query") and prompt_set.scene_prompts:
            clip_ref["broll_query"] = prompt_set.scene_prompts[0].topic
        atomic_write_json(_candidate_file_path(stem), data, ensure_ascii=False)

    return {
        "ok": True,
        "source_stem": stem,
        "clip_number": clip_number,
        "mode": ratio.mode,
        "derived_visual_mode": derived_mode,
        "scene_plan_file": str(plan_path),
        "updated_candidate_file": str(_candidate_file_path(stem)) if update_candidates else "",
        "insert_count": ratio.insert_count,
        "summary": prompt_set.summary,
        "mood": prompt_set.mood,
        "visual_ratio": ratio.model_dump(),
        "visual_plan": [beat.model_dump() for beat in plan],
        "prompts": [idea.model_dump() for idea in prompt_set.scene_prompts],
        "plan_metadata": {
            "plan_policy_version": SCENE_PLAN_POLICY_VERSION,
            "transition_style": _RENDER_TRANSITION_TYPE,
            "transition_duration_sec": _RENDER_TRANSITION_T,
            "recommended_preset": _RECOMMENDED_PRESET,
            "darkness_intent": _DARKNESS_INTENT,
        },
    }


@mcp.tool()
def build_visual_plans_for_stem(
    stem: str,
    mode: str = "balanced",
    only_add_broll: bool = True,
    overwrite: bool = True,
) -> dict:
    """
    Build scene plans for all matching clips in a stem.

    Candidate JSON is written once per clip (sequential, not batched) because
    each build_visual_plan call loads the latest on-disk state.  This is safe
    for sequential execution and avoids accumulating stale in-memory state
    across multiple planning calls.
    """
    data = _load_candidates(stem)
    clips = data.get("clips") or []
    if not clips:
        return {"ok": False, "error": "No clip candidates found"}

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

    results = []
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
            "plan_policy_version": SCENE_PLAN_POLICY_VERSION,
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
