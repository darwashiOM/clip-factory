"""
Shared helpers for clip-factory MCP servers.

Covers:
- Arabic caption cleaning (tashkeel / diacritic stripping, Unicode normalization)
- Per-clip ASS text generation supporting both subtitle and center-recitation modes
- Atomic JSON / text file writes to prevent corruption on crash
- FFmpeg binary resolution (prefers ffmpeg-full which has libass support)

Text-config single source of truth
───────────────────────────────────
All styling decisions (font, size, colour, alignment, animation) live in
text_config.TextStyleConfig.  This module no longer declares any module-level
style constants.  Call text_config.load_text_config() after dotenv is loaded to
obtain a config instance, then pass it to generate_clip_ass().

Arabic rendering note
─────────────────────
arabic_reshaper and python-bidi are NOT imported at module level.  They are
loaded lazily inside generate_clip_ass only when config.use_reshaper_bidi is
True.  The default is False because libass uses HarfBuzz + FriBidi natively;
pre-processing the text here causes double-shaping which breaks glyph connections.
See text_config.py for the full explanation.
"""

from pathlib import Path
import os
import json
import re
import unicodedata
import tempfile
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from text_config import TextStyleConfig


_SENTENCE_BREAK_RE = re.compile(r"([،,:;؛.!?؟])")


# ─── Arabic text cleaning ──────────────────────────────────────────────────────

def clean_arabic_for_captions(text: str) -> str:
    """
    Strip Arabic tashkeel/diacritics and normalize Unicode for caption display.

    Applies an aggressive character whitelist that keeps only basic Arabic
    letters (U+0621–U+063A, U+0641–U+064A), digits, and common punctuation.

    WARNING: This function is intentionally destructive.  It MUST NOT be
    applied to Quran canonical render_text — the corpus text may contain
    valid characters outside the whitelist (e.g. ٱ U+0671 alef wasla).
    Use it only for raw ASR output or user-facing caption variants.

    The raw transcript files on disk are never touched; this produces a
    display-ready string from an in-memory copy.
    """
    text = str(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\s0-9٠-٩\.\,\!\?\:\;\"\'،؛؟\-]', '', text)
    return text


# ─── Segment text selection ────────────────────────────────────────────────────

def segment_render_text(seg: dict) -> str:
    """
    Return the best render/display text for a segment.

    Selection priority:
      1. seg["quran_guard"]["render_text"]  — corpus-canonical Quran text.
         Present only when quran_guard_server matched this segment (or the
         first segment of a multi-segment window).  Empty string for
         non-first window segments (those are intentionally skipped).
      2. seg["text"]  — Gemini-refined or raw ASR text.

    Callers must NOT apply clean_arabic_for_captions to the result when the
    source is a quran_guard verbose JSON, because the corpus text is already
    correct and cleaning would strip valid Quranic characters.
    """
    qg = seg.get("quran_guard") or {}
    qg_text = str(qg.get("render_text", "")).strip()
    if qg_text:
        return qg_text
    return str(seg.get("text", "")).strip()


# ─── ASS time format ───────────────────────────────────────────────────────────

def sec_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc."""
    total_cs = max(0, int(round(seconds * 100)))
    hours = total_cs // 360000
    total_cs %= 360000
    minutes = total_cs // 6000
    total_cs %= 6000
    secs = total_cs // 100
    cs = total_cs % 100
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def _srt_time_to_sec(time_str: str) -> float:
    """Parse SRT/ASS time string (HH:MM:SS,mmm or HH:MM:SS.mmm) to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0


# ─── SRT parsing ──────────────────────────────────────────────────────────────

def parse_srt_to_segments(srt_text: str) -> List[dict]:
    """
    Parse SRT text content into a list of segment dicts with start/end in seconds.
    Handles missing index lines and both comma and dot decimal separators.
    """
    segments = []
    blocks = re.split(r"\n\s*\n", str(srt_text or "").strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        timecode_match = None
        text_start = None
        for i, line in enumerate(lines):
            m = re.match(
                r"(\d+:\d{2}:\d{2}[,.]\d+)\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d+)",
                line,
            )
            if m:
                timecode_match = m
                text_start = i + 1
                break
        if timecode_match is None or text_start is None:
            continue
        start_s = _srt_time_to_sec(timecode_match.group(1))
        end_s = _srt_time_to_sec(timecode_match.group(2))
        text = " ".join(lines[text_start:]).strip()
        if text:
            segments.append({"start": start_s, "end": end_s, "text": text})
    return segments


# ─── Line breaking ─────────────────────────────────────────────────────────────

def _choose_balanced_break(words: List[str], max_words_per_line: int, max_lines: int) -> List[List[str]]:
    """
    Split a word list into balanced lines for display.

    Tries every legal break point and selects the most balanced split
    (minimises the word-count difference between lines, with a small
    preference for breaking after punctuation).

    If no two-line split is possible within the word limits, falls back to
    chunking by max_words_per_line with soft punctuation preference.
    """
    if not words:
        return []
    if max_lines <= 1 or len(words) <= max_words_per_line:
        return [words]

    candidates = []
    for break_idx in range(2, len(words)):
        left = words[:break_idx]
        right = words[break_idx:]
        if len(left) > max_words_per_line:
            continue
        if len(right) > max_words_per_line:
            continue
        balance_penalty = abs(len(left) - len(right))
        left_punct_bonus = 0 if left[-1].endswith(("،", "؛", ":", ".", "؟", "!")) else 1
        candidates.append((balance_penalty, left_punct_bonus, break_idx))

    if candidates:
        _, _, break_idx = min(candidates)
        return [words[:break_idx], words[break_idx:]]

    # Fallback: chunk by max_words_per_line, preferring punctuation cut points.
    #
    # No-discard policy: silently dropping Quran canonical text is unacceptable.
    # An extra line on screen is recoverable; missing words in a rendered verse
    # are not.  We compute the minimum number of lines needed to hold all words
    # and use that as the effective cap, clamped to a reasonable absolute ceiling
    # (max_lines + 4).  The caller's max_lines is a soft target, not a hard kill.
    needed = (len(words) + max_words_per_line - 1) // max(1, max_words_per_line)
    effective_max = max(max_lines, min(needed, max_lines + 4))
    lines: List[List[str]] = []
    idx = 0
    while idx < len(words) and len(lines) < effective_max:
        remaining = len(words) - idx
        soft_max = min(max_words_per_line, remaining)
        chunk = words[idx:idx + soft_max]
        best_cut = None
        for probe in range(len(chunk), max(1, len(chunk) - 2), -1):
            if chunk[probe - 1].endswith(("،", "؛", ":", ".", "؟", "!")):
                best_cut = probe
                break
        if best_cut:
            chunk = chunk[:best_cut]
        lines.append(chunk)
        idx += len(chunk)

    return [line for line in lines if line]


def split_caption_lines(
    text: str,
    max_words_per_line: int = 4,
    max_lines: int = 2,
    *,
    clean: bool = True,
) -> str:
    """
    Split Arabic text into short balanced lines for display.

    Returns the text with ASS soft line-breaks (\\N) inserted between lines.

    Args:
        text: Input Arabic text — may contain sentence punctuation.
        max_words_per_line: Maximum words allowed on a single line.
        max_lines: Maximum number of lines to produce.
        clean: If True (default), apply clean_arabic_for_captions before
               splitting.  Set False when text is already canonical (e.g.
               quran_guard render_text) to avoid stripping valid Quranic
               characters like ٱ (U+0671 alef wasla).

    Cleaning is controlled explicitly by the caller via the `clean` parameter.
    This function never does hidden cleaning when clean=False.
    """
    if clean:
        text = clean_arabic_for_captions(text)
    if not text:
        return ""

    # Add a space after sentence-break punctuation to help word-splitting.
    text = _SENTENCE_BREAK_RE.sub(r"\1 ", text)
    words = [w for w in text.split() if w]
    if len(words) <= max_words_per_line:
        return " ".join(words)

    lines = _choose_balanced_break(words, max_words_per_line=max_words_per_line, max_lines=max_lines)
    if not lines:
        return text

    joined = [" ".join(line).strip() for line in lines if line]
    joined = [line for line in joined if line]
    if not joined:
        return text
    return r"\N".join(joined)


# ─── ASS file building ─────────────────────────────────────────────────────────

def _build_ass_style_line(cfg: "TextStyleConfig") -> str:
    """
    Build the V4+ Styles line for the ASS header from a TextStyleConfig.

    ASS Format field order:
      Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,
      OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,
      ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,
      Alignment, MarginL, MarginR, MarginV, Encoding
    """
    bold = "-1" if cfg.bold else "0"
    italic = "-1" if cfg.italic else "0"
    return (
        f"Style: Default,"
        f"{cfg.font},{cfg.fontsize},"
        f"{cfg.primary_color},{cfg.secondary_color},"
        f"{cfg.outline_color},{cfg.shadow_color},"
        f"{bold},{italic},0,0,"          # Bold, Italic, Underline, StrikeOut
        f"100,100,"                      # ScaleX, ScaleY
        f"{cfg.spacing:.2f},0,"          # Spacing, Angle
        f"{cfg.border_style},"           # BorderStyle (1=outline+shadow, 3=box)
        f"{cfg.outline_width:.1f},"      # Outline
        f"{cfg.shadow_size:.1f},"        # Shadow
        f"{cfg.alignment},"              # Alignment (5=mid-center, 2=bot-center)
        f"{cfg.margin_l},{cfg.margin_r},{cfg.margin_v},"  # MarginL, MarginR, MarginV
        f"1\n"                           # Encoding
    )


def _build_ass_header(cfg: "TextStyleConfig") -> str:
    """
    Build the complete ASS file header: Script Info + Styles + Events preamble.

    PlayResX/Y are fixed at 1080x1920 (vertical short-form video format).
    WrapStyle=2 disables libass auto-wrapping; line breaks are controlled
    explicitly via \\N in the dialogue text.
    """
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "Timer: 100.0000\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{_build_ass_style_line(cfg)}"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _build_lead_tag(cfg: "TextStyleConfig") -> str:
    r"""
    Build the per-dialogue ASS override tag string.

    The \anN tag here must match cfg.alignment so each Dialogue line is
    self-contained (resilient to style-block inheritance issues):
      \an2  bottom-center  (subtitle mode)
      \an5  middle-center  (center_recitation mode)

    blur, bord, shad are also repeated per-event as overrides to avoid any
    style-inheritance ambiguity in multi-style ASS files.

    When cfg.animate is True, a subtle breath animation is added:
      - fade in / out (fade_in_ms, fade_out_ms)
      - gentle scale pulse: starts at 97%, grows to 103%, settles at 100%
    """
    an = cfg.alignment
    base = (
        rf"{{\an{an}"
        rf"\blur{cfg.blur:.1f}"
        rf"\bord{cfg.outline_width:.1f}"
        rf"\shad{cfg.shadow_size:.1f}"
    )
    if cfg.animate:
        base += (
            rf"\fad({cfg.fade_in_ms},{cfg.fade_out_ms})"
            rf"\fscx97\fscy97"
            rf"\t(0,140,\fscx103\fscy103)"
            rf"\t(140,240,\fscx100\fscy100)"
        )
    return base + "}"


def _escape_ass_text(text: str) -> str:
    r"""
    Escape user-provided text for safe embedding in an ASS Dialogue line.

    Rules:
    - \N (backslash-N) is the ASS soft line-break marker produced by
      split_caption_lines.  It must reach libass unchanged and must NOT
      be treated as a backslash followed by N.
    - Curly braces { } are ASS override-tag delimiters.  Literal { and }
      in display text must be escaped as \{ and \}.

    Strategy: temporarily replace \N with a null-byte placeholder, escape
    braces, then restore \N.
    """
    placeholder = "\x00NL\x00"
    text = text.replace(r"\N", placeholder)
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text.replace(placeholder, r"\N")


# ─── Main ASS generation ───────────────────────────────────────────────────────

def generate_clip_ass(
    segments: List[dict],
    clip_start: float,
    clip_end: float,
    config: Optional["TextStyleConfig"] = None,
    *,
    clean_arabic: Optional[bool] = None,
) -> str:
    """
    Generate an ASS text file for a specific clip time window.

    Supports both style modes via the config object:
      subtitle          — bottom-center, small text, standard subtitle layout
      center_recitation — middle-center, large Arabic display text, cinematic

    Text pipeline per segment (all stages are explicit — no hidden mutations):

      Stage 1  choose render text
               segment_render_text() prefers quran_guard.render_text when present
               (corpus-canonical), falls back to seg["text"] (Gemini-refined or ASR).

      Stage 2  normalise internal newlines
               Literal \\n characters become spaces so they don't split words
               unexpectedly.  The ASS line-break \\N is added later in Stage 4.

      Stage 3  optional display cleaning
               If do_clean is True: clean_arabic_for_captions() is applied.
               For Quran canonical text this MUST be False — the corpus text
               may contain ٱ (U+0671) and other characters outside the whitelist.

      Stage 4  line-break for display
               split_caption_lines() with clean=False (cleaning handled in Stage 3).
               Uses config.max_words_per_line and config.max_lines.

      Stage 5  optional Arabic reshaper + bidi  (default: OFF)
               Enabled only when config.use_reshaper_bidi is True.
               DEFAULT is False because libass uses HarfBuzz for Arabic glyph
               shaping and FriBidi for bidi ordering natively.
               Pre-processing here causes DOUBLE shaping: arabic_reshaper converts
               letters to Unicode presentation forms (U+FE70–U+FEFF), then HarfBuzz
               processes those already-shaped glyphs again, producing disconnected
               letters and corrupted text in the final render.
               Enable ASS_USE_RESHAPER_BIDI=true only if your libass build lacks
               HarfBuzz AND you observe disconnected Arabic glyphs.

      Stage 6  ASS text escaping
               _escape_ass_text() protects \\N line-breaks and escapes literal
               curly braces { } that would otherwise be parsed as override tags.

      Stage 7  per-dialogue style tag
               _build_lead_tag() produces the \\anN + blur/bord/shad + animation
               overrides that are prepended to each Dialogue event text.

    Args:
        segments:    List of segment dicts with start/end/text and optional
                     quran_guard sub-dict.
        clip_start:  Clip start in seconds (absolute source time).
        clip_end:    Clip end in seconds (absolute source time).
        config:      TextStyleConfig instance.  If None, load_text_config() is
                     called lazily.  Always pass an explicit config in production
                     to avoid repeated env reads.
        clean_arabic: If True,  apply clean_arabic_for_captions in Stage 3.
                      If False, skip cleaning (required for Quran canonical text).
                      If None (default), defers to config.clean_for_display.

    Returns:
        Complete ASS file content as a string.
    """
    if config is None:
        from text_config import load_text_config
        config = load_text_config()

    # Determine cleaning behaviour.
    # The renderer always passes clean_arabic=False explicitly when it has
    # quran_guard-sourced text.  This line is the single decision point.
    do_clean: bool = config.clean_for_display if clean_arabic is None else clean_arabic

    header = _build_ass_header(config)
    lead_tag = _build_lead_tag(config)

    dialogue_lines: List[str] = []

    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))

        # Skip segments that do not overlap the clip window.
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Clamp to clip boundaries; compute clip-relative timestamps for the ASS file.
        seg_start = max(seg_start, clip_start)
        seg_end = min(seg_end, clip_end)
        rel_start = seg_start - clip_start
        rel_end = seg_end - clip_start

        # ── Stage 1: Choose render text ──────────────────────────────────────
        raw_text = segment_render_text(seg)

        # ── Stage 2: Normalise internal newlines ─────────────────────────────
        text = raw_text.replace("\n", " ").strip()

        # ── Stage 3: Optional display cleaning ───────────────────────────────
        # For Quran text (do_clean=False): the corpus text is already canonical.
        # clean_arabic_for_captions would strip ٱ (alef wasla) and other
        # valid Quranic characters outside the basic 0621-063A / 0641-064A range.
        if do_clean:
            text = clean_arabic_for_captions(text)

        if not text:
            continue

        # ── Stage 4: Line-break for display ──────────────────────────────────
        # clean=False: Stage 3 handled any needed cleaning already.
        text = split_caption_lines(
            text,
            max_words_per_line=config.max_words_per_line,
            max_lines=config.max_lines,
            clean=False,
        )

        # ── Stage 5: Optional Arabic reshaper + bidi ─────────────────────────
        # Default: OFF — see docstring for full explanation.
        # Imports are lazy so that arabic_reshaper / python-bidi are not required
        # when this feature is disabled (the common case).
        if config.use_reshaper_bidi:
            import arabic_reshaper          # noqa: PLC0415
            from bidi.algorithm import get_display  # noqa: PLC0415
            text = r"\N".join(
                get_display(arabic_reshaper.reshape(line))
                for line in text.split(r"\N")
            )

        # ── Stage 6: ASS text escaping ────────────────────────────────────────
        text = _escape_ass_text(text)

        if not text:
            continue

        # ── Stage 7: Assemble Dialogue event ─────────────────────────────────
        dialogue_lines.append(
            f"Dialogue: 0,"
            f"{sec_to_ass_time(rel_start)},"
            f"{sec_to_ass_time(rel_end)},"
            f"Default,,0,0,0,,{lead_tag}{text}"
        )

    return header + "\n".join(dialogue_lines) + "\n"


# ─── FFmpeg binary resolution ──────────────────────────────────────────────────

def get_ffmpeg() -> str:
    """
    Resolve the best available ffmpeg binary.

    Priority:
    1. FFMPEG_PATH environment variable (explicit override)
    2. ffmpeg-full (Homebrew keg-only, has libass for subtitle filter support)
    3. Standard Homebrew ffmpeg
    4. System PATH fallback
    """
    from_env = os.environ.get("FFMPEG_PATH", "").strip()
    if from_env:
        return from_env

    candidates = [
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return "ffmpeg"


# ─── Atomic file writes ────────────────────────────────────────────────────────

def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to a file atomically (write temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: dict, ensure_ascii: bool = False) -> None:
    """Write a dict as pretty-printed JSON atomically."""
    content = json.dumps(data, ensure_ascii=ensure_ascii, indent=2) + "\n"
    atomic_write_text(path, content)
