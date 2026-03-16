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
    text = unicodedata.normalize("NFC", text)
    text = re.sub(
        r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s0-9٠-٩\.\,\!\?\:\;\"\'،؛؟\-]',
        '',
        text,
    )
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


def _wrap_rtl_lines(text: str) -> str:
    """
    Force each ASS line to be treated as RTL text.

    We wrap each visual line separately so bidi state does not cross \\N.
    """
    RLE = "\u202B"  # Right-to-Left Embedding
    PDF = "\u202C"  # Pop Directional Formatting

    parts = text.split(r"\N")
    wrapped = []
    for part in parts:
        if part:
            wrapped.append(f"{RLE}{part}{PDF}")
        else:
            wrapped.append(part)
    return r"\N".join(wrapped)


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
    Split a word list into balanced display lines with a TRUE hard cap.

    This function never returns more than:
    - max_lines lines
    - max_words_per_line words per line

    Any additional words must be split into a later timed chunk upstream.
    """
    if not words:
        return []

    max_words_per_line = max(1, int(max_words_per_line))
    max_lines = max(1, int(max_lines))
    visible_cap = max_words_per_line * max_lines

    # Hard cap for a single visible event.
    words = list(words[:visible_cap])

    if max_lines <= 1 or len(words) <= max_words_per_line:
        return [words]

    # Best balanced legal split first.
    candidates = []
    for break_idx in range(1, len(words)):
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

    # Strict fallback: pack sequentially within the hard cap.
    lines: List[List[str]] = []
    idx = 0
    while idx < len(words) and len(lines) < max_lines:
        remaining = len(words) - idx
        take = min(max_words_per_line, remaining)
        chunk = words[idx:idx + take]

        # Soft punctuation preference only inside the allowed chunk.
        best_cut = None
        for probe in range(len(chunk), max(1, len(chunk) - 2), -1):
            if chunk[probe - 1].endswith(("،", "؛", ":", ".", "؟", "!")):
                best_cut = probe
                break
        if best_cut:
            chunk = chunk[:best_cut]

        if not chunk:
            chunk = words[idx:idx + take]

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
    """
    an = cfg.alignment
    base = (
        rf"{{\an{an}"
        rf"\blur{cfg.blur:.1f}"
        rf"\bord{cfg.outline_width:.1f}"
        rf"\shad{cfg.shadow_size:.1f}"
    )

    if an == 5:
        y_offset = _env_int_local("ASS_CENTER_Y_OFFSET", 0)
        if y_offset != 0:
            base += rf"\pos(540,{960 + y_offset})"

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
def _env_bool_local(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_float_local(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int_local(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_str_local(key: str, default: str) -> str:
    val = os.environ.get(key, "").strip()
    return val if val else default

def _tokenize_text_words(
    text: str,
    *,
    clean: bool,
) -> List[str]:
    if clean:
        text = clean_arabic_for_captions(text)
    if not text:
        return []

    text = _SENTENCE_BREAK_RE.sub(r"\1 ", text)
    return [w for w in text.split() if w]



def _split_words_into_lines(
    text: str,
    max_words_per_line: int,
    max_lines: int,
    *,
    clean: bool,
) -> List[List[str]]:
    if clean:
        text = clean_arabic_for_captions(text)
    if not text:
        return []

    text = _SENTENCE_BREAK_RE.sub(r"\1 ", text)
    words = [w for w in text.split() if w]
    if not words:
        return []

    return _choose_balanced_break(
        words,
        max_words_per_line=max_words_per_line,
        max_lines=max_lines,
    )


def _flatten_word_lines(lines: List[List[str]]) -> List[str]:
    out: List[str] = []
    for line in lines:
        out.extend(line)
    return out


def _allocate_word_windows(words: List[str], start_sec: float, end_sec: float) -> List[tuple[float, float]]:
    total_duration = max(0.12, end_sec - start_sec)
    if not words:
        return []

    weights: List[float] = []
    for word in words:
        stripped = re.sub(r"[^\u0600-\u06FFA-Za-z0-9]", "", word)
        weights.append(max(1.0, float(len(stripped) or 1)))

    total_weight = sum(weights) or float(len(words))
    cursor = start_sec
    windows: List[tuple[float, float]] = []

    for idx, weight in enumerate(weights):
        piece = total_duration * (weight / total_weight)
        next_cursor = end_sec if idx == len(weights) - 1 else cursor + piece
        windows.append((cursor, next_cursor))
        cursor = next_cursor

    return windows



def _build_progressive_active_word_text(
    words: List[str],
    active_word_index: int,
    *,
    active_color: str,
    active_outline: str,
    active_shadow: str,
    active_blur: float,
    active_scale: float,
) -> str:
    """
    Build progressive reveal text with explicit RTL word reversal.
    
    Uses Unicode Bidi Isolates to prevent the bidi algorithm from 
    reordering across word boundaries.
    """
    # Unicode Bidi control characters
    FSI = "\u2068"  # First Strong Isolate
    PDI = "\u2069"  # Pop Directional Isolate
    
    scale_pct = int(round(active_scale * 100))
    active_tag = (
        rf"{{\1c{active_color}"
        rf"\3c{active_outline}"
        rf"\4c{active_shadow}"
        rf"\blur{active_blur:.1f}"
        rf"\fscx{scale_pct}\fscy{scale_pct}}}"
    )
    reset_tag = r"{\rDefault}"

    visible_words = words[: active_word_index + 1]
    visible_words_reversed = list(reversed(visible_words))

    pieces: List[str] = []

    for idx, word in enumerate(visible_words_reversed):
        token = _escape_ass_text(word)

        if idx > 0:
            pieces.append(" ")

        # Isolate each word to prevent bidi reordering
        if idx == 0:  # Active word (newest, leftmost)
            pieces.append(f"{FSI}{active_tag}{token}{reset_tag}{PDI}")
        else:
            pieces.append(f"{FSI}{token}{PDI}")

    return "".join(pieces)


def _chunk_words(words: List[str], chunk_size: int) -> List[List[str]]:
    chunk_size = max(1, int(chunk_size))
    return [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]


def _chunk_sequence_by_sizes(items, sizes: List[int]):
    out = []
    idx = 0
    for size in sizes:
        size = max(0, int(size))
        if size <= 0:
            continue
        out.append(items[idx:idx + size])
        idx += size
    return out


def _display_chunk_words(
    words: List[str],
    max_words_per_line: int,
    max_lines: int,
    preferred_chunk_words: int | None = None,
) -> List[List[str]]:
    """
    Split a long word list into multiple display chunks.

    Each chunk is guaranteed to fit in one visible event after line fitting.
    """
    if not words:
        return []

    max_words_per_line = max(1, int(max_words_per_line))
    max_lines = max(1, int(max_lines))
    visible_cap = max_words_per_line * max_lines

    target = visible_cap if preferred_chunk_words is None else max(1, int(preferred_chunk_words))
    target = min(target, visible_cap)

    return [words[i:i + target] for i in range(0, len(words), target)]

def _fit_chunk_into_lines(
    words: List[str],
    max_words_per_line: int,
    max_lines: int,
) -> List[List[str]]:
    if not words:
        return []

    if max_lines <= 1 or len(words) <= max_words_per_line:
        return [words]

    return _choose_balanced_break(
        words,
        max_words_per_line=max_words_per_line,
        max_lines=max_lines,
    )

def _build_inline_active_word_text(
    lines: List[List[str]],
    active_word_index: int,
    *,
    active_color: str,
    active_outline: str,
    active_shadow: str,
    active_blur: float,
    active_scale: float,
) -> str:
    """
    Build highlighted Arabic text in normal logical order.
    Do NOT reverse token order.
    """
    scale_pct = int(round(active_scale * 100))
    active_tag = (
        rf"{{\1c{active_color}"
        rf"\3c{active_outline}"
        rf"\4c{active_shadow}"
        rf"\blur{active_blur:.1f}"
        rf"\fscx{scale_pct}\fscy{scale_pct}}}"
    )
    reset_tag = r"{\rDefault}"

    pieces: List[str] = []
    flat_idx = 0

    for line in lines:
        line_buf: List[str] = []

        for pos, word in enumerate(line):
            token = _escape_ass_text(word)

            if pos > 0:
                line_buf.append(" ")

            if flat_idx == active_word_index:
                line_buf.append(f"{active_tag}{token}{reset_tag}")
            else:
                line_buf.append(token)

            flat_idx += 1

        pieces.append(_wrap_rtl_lines("".join(line_buf)))

    return r"\N".join(pieces)




def _build_static_lead_tag(cfg: "TextStyleConfig") -> str:
    base = (
        rf"{{\an{cfg.alignment}"
        rf"\blur{cfg.blur:.1f}"
        rf"\bord{cfg.outline_width:.1f}"
        rf"\shad{cfg.shadow_size:.1f}"
    )

    if cfg.alignment == 5:
        y_offset = _env_int_local("ASS_CENTER_Y_OFFSET", 0)
        if y_offset != 0:
            base += rf"\pos(540,{960 + y_offset})"

    return base + "}"

def _segment_word_windows_from_transcript(
    seg: dict,
    clip_start: float,
    clip_end: float,
    display_words: List[str],
) -> List[tuple[float, float]]:
    raw_words = seg.get("words") or []
    usable = []

    for item in raw_words:
        try:
            w_start = float(item.get("start", 0.0))
            w_end = float(item.get("end", w_start))
        except (TypeError, ValueError):
            continue

        token = str(item.get("word", "")).strip()
        if not token:
            continue

        if w_end <= clip_start or w_start >= clip_end:
            continue

        usable.append(
            {
                "word": token,
                "start": max(w_start, clip_start) - clip_start,
                "end": min(w_end, clip_end) - clip_start,
            }
        )

    if len(usable) != len(display_words):
        return []

    return [(round(x["start"], 3), round(x["end"], 3)) for x in usable]


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

    In normal mode:
      - one Dialogue event per segment

    In center_recitation + ASS_WORD_HIGHLIGHT=true mode:
      - one Dialogue event per estimated word window
      - the full sentence stays visible
      - only the active word is restyled inline
      - no duplicate transparent overlay words are used
    """
    if config is None:
        from text_config import load_text_config
        config = load_text_config()

    do_clean: bool = config.clean_for_display if clean_arabic is None else clean_arabic

    header = _build_ass_header(config)
    lead_tag = _build_lead_tag(config)
    static_lead_tag = _build_static_lead_tag(config)

    highlight_enabled = (
        getattr(config, "mode", "") == "center_recitation"
        and _env_bool_local("ASS_WORD_HIGHLIGHT", True)
    )

    active_color = _env_str_local("ASS_ACTIVE_WORD_COLOR", "&H007878FF")
    active_outline = _env_str_local("ASS_ACTIVE_WORD_OUTLINE_COLOR", "&H00000000")
    active_shadow = _env_str_local("ASS_ACTIVE_WORD_SHADOW_COLOR", "&H660000FF")
    active_scale = _env_float_local("ASS_ACTIVE_WORD_SCALE", 1.06)
    active_blur = _env_float_local("ASS_ACTIVE_WORD_BLUR", 1.4)

    dialogue_lines: List[str] = []

    hard_visible_words = max(1, config.max_words_per_line * config.max_lines)

    highlight_chunk_words = min(
        hard_visible_words,
        max(1, _env_int_local("ASS_HIGHLIGHT_CHUNK_WORDS", hard_visible_words)),
    )
    highlight_gap_sec = max(0.0, _env_float_local("ASS_HIGHLIGHT_GAP_SEC", 0.03))

    # Prevent visible overlap between consecutive dialogue events.
    last_dialogue_end = 0.0

    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))

        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        seg_start = max(seg_start, clip_start)
        seg_end = min(seg_end, clip_end)
        rel_start = seg_start - clip_start
        rel_end = seg_end - clip_start

        raw_text = segment_render_text(seg)
        text = raw_text.replace("\n", " ").strip()

        if do_clean:
            text = clean_arabic_for_captions(text)

        if not text:
            continue


        logical_words = _tokenize_text_words(text, clean=False)
        
        if not logical_words:
            continue

        word_windows = _segment_word_windows_from_transcript(
            seg,
            clip_start,
            clip_end,
            logical_words,
        )
        if not word_windows:
            word_windows = _allocate_word_windows(logical_words, rel_start, rel_end)

        if not word_windows:
            continue

        if highlight_enabled:
            chunk_groups = _chunk_words(logical_words, highlight_chunk_words)
            window_groups = _chunk_sequence_by_sizes(
                word_windows,
                [len(chunk) for chunk in chunk_groups],
            )

            for chunk_words, chunk_windows in zip(chunk_groups, window_groups):
                if not chunk_words or not chunk_windows:
                    continue

                display_chunk_words = chunk_words
                if config.use_reshaper_bidi:
                    import arabic_reshaper
                    from bidi.algorithm import get_display
                    display_chunk_words = [
                        get_display(arabic_reshaper.reshape(word))
                        for word in chunk_words
                    ]

                for local_word_idx, (word_start, word_end) in enumerate(chunk_windows):
                    if local_word_idx >= len(display_chunk_words):
                        break

                    if last_dialogue_end > 0:
                        min_start = last_dialogue_end + highlight_gap_sec
                    else:
                        min_start = 0.0

                    event_start = max(word_start, min_start)
                    event_end = max(word_end, event_start + 0.04)

                    line_text = _build_progressive_active_word_text(
                        display_chunk_words,
                        local_word_idx,
                        active_color=active_color,
                        active_outline=active_outline,
                        active_shadow=active_shadow,
                        active_blur=active_blur,
                        active_scale=active_scale,
                    )

                    dialogue_lines.append(
                        f"Dialogue: 0,"
                        f"{sec_to_ass_time(event_start)},"
                        f"{sec_to_ass_time(event_end)},"
                        f"Default,,0,0,0,,{static_lead_tag}{line_text}"
                    )

                    last_dialogue_end = event_end

        else:
            chunk_groups = _display_chunk_words(
                logical_words,
                max_words_per_line=config.max_words_per_line,
                max_lines=config.max_lines,
                preferred_chunk_words=hard_visible_words,
            )

            window_groups = _chunk_sequence_by_sizes(
                word_windows,
                [len(chunk) for chunk in chunk_groups],
            )

            for chunk_words, chunk_windows in zip(chunk_groups, window_groups):
                if not chunk_words or not chunk_windows:
                    continue

                display_chunk_words = chunk_words
                if config.use_reshaper_bidi:
                    import arabic_reshaper
                    from bidi.algorithm import get_display
                    display_chunk_words = [
                        get_display(arabic_reshaper.reshape(word))
                        for word in chunk_words
                    ]

                chunk_lines = _fit_chunk_into_lines(
                    display_chunk_words,
                    max_words_per_line=config.max_words_per_line,
                    max_lines=config.max_lines,
                )
                line_text = r"\N".join(
                    " ".join(line).strip() for line in chunk_lines if line
                ).strip()
                if not line_text:
                    continue

                event_start = max(chunk_windows[0][0], last_dialogue_end)
                event_end = max(chunk_windows[-1][1], event_start + 0.04)

                dialogue_lines.append(
                    f"Dialogue: 0,"
                    f"{sec_to_ass_time(event_start)},"
                    f"{sec_to_ass_time(event_end)},"
                    f"Default,,0,0,0,,{static_lead_tag}{_wrap_rtl_lines(_escape_ass_text(line_text))}"
                )

                last_dialogue_end = event_end
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
