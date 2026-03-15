"""
Shared helpers for clip-factory MCP servers.

Covers:
- Arabic caption cleaning (tashkeel / diacritic stripping, Unicode normalization)
- Per-clip ASS subtitle generation with stronger styling for Arabic vertical video
- Atomic JSON / text file writes to prevent corruption on crash
- FFmpeg binary resolution (prefers ffmpeg-full which has libass support)
"""

from pathlib import Path
import os
import json
import re
import unicodedata
import tempfile
from typing import List, Optional
import arabic_reshaper
from bidi.algorithm import get_display

_TASHKEEL = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0640"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

_SENTENCE_BREAK_RE = re.compile(r"([،,:;؛.!?؟])")
SUBTITLE_FONT_DEFAULT = os.environ.get("SUBTITLE_FONT", "Geeza Pro")
SUBTITLE_FONTSIZE_DEFAULT = int(os.environ.get("SUBTITLE_FONTSIZE", "60"))
SUBTITLE_MARGIN_V_DEFAULT = int(os.environ.get("SUBTITLE_MARGIN_V", "110"))



def clean_arabic_for_captions(text: str) -> str:
    """
    Strip Arabic tashkeel/diacritics and normalize Unicode for caption display.
    The raw transcript files are never touched; this produces a display-ready string.
    """
    text = str(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\s0-9٠-٩\.\,\!\?\:\;\"\'،؛؟\-]', '', text)
    return text


def segment_render_text(seg: dict) -> str:
    """
    Get the best render/display text for a segment.

    Priority:
    1. quran_guard.render_text if present and non-empty
    2. text
    """
    qg = seg.get("quran_guard") or {}
    qg_text = str(qg.get("render_text", "")).strip()
    if qg_text:
        return qg_text
    return str(seg.get("text", "")).strip()


def merge_refined_and_quran_segments(refined_segments: List[dict], quran_segments: List[dict]) -> List[dict]:
    """
    Merge refined ASR segments with quran-guard metadata when both sources exist.

    If the segment counts do not match, we return the refined segments unchanged.
    """
    if len(refined_segments) != len(quran_segments):
        return refined_segments

    merged = []
    for refined, guarded in zip(refined_segments, quran_segments):
        seg = dict(refined)
        if guarded.get("quran_guard"):
            seg["quran_guard"] = guarded.get("quran_guard")
            render_text = segment_render_text(guarded)
            if render_text:
                seg["text"] = render_text
        merged.append(seg)
    return merged


# ─── ASS subtitle helpers ──────────────────────────────────────────────────────

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


def _choose_balanced_break(words: List[str], max_words_per_line: int, max_lines: int) -> List[List[str]]:
    if not words:
        return []
    if max_lines <= 1 or len(words) <= max_words_per_line:
        return [words]

    # Try all legal break points and pick the most balanced split.
    min_first = max(2, min(max_words_per_line, len(words) - 1))
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

    # If the segment is too long for two ideal lines, keep it readable rather than stuffing
    # everything into the final line.
    per_line = max_words_per_line
    lines: List[List[str]] = []
    idx = 0
    line_idx = 0
    while idx < len(words):
        remaining_words = len(words) - idx
        # Calculate soft_max based on max_words_per_line
        soft_max = min(max_words_per_line, remaining_words)
        
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
        line_idx += 1

    return [line for line in lines if line]


def split_caption_lines(text: str, max_words_per_line: int = 4, max_lines: int = 2) -> str:
    """
    Split Arabic caption text into short balanced lines for vertical-video readability.

    Unlike the old version, this does not dump all remaining words into the final line,
    which was producing very crowded last lines.
    """
    text = clean_arabic_for_captions(text)
    if not text:
        return ""

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




def generate_clip_ass(
    segments: List[dict],
    clip_start: float,
    clip_end: float,
    font: str = SUBTITLE_FONT_DEFAULT,
    fontsize: int = SUBTITLE_FONTSIZE_DEFAULT,
    margin_v: int = SUBTITLE_MARGIN_V_DEFAULT,
    clean_arabic: bool = True,
    animate: bool = True,
    max_words_per_line: int = 4,
    max_lines: int = 2,
) -> str:
    """
    Generate an ASS subtitle file for a specific clip time window.

    Segments are filtered to [clip_start, clip_end] and timestamps are re-offset so
    the clip starts at t=0. The styling is tuned for 1080x1920 short-form video.
    """
    header = (
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
        f"Style: Default,{font},{fontsize},"
        f"&H00FFFFFF,&H0037C8FF,&H00131313,&H6E000000,"
        f"-1,0,0,0,100,100,0.15,0,1,3.8,1.2,2,60,60,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogue_lines: List[str] = []
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
        if clean_arabic:
            text = clean_arabic_for_captions(text)
        text = split_caption_lines(text, max_words_per_line=max_words_per_line, max_lines=max_lines)

        # --- NEW CODE: Fix disconnected and reversed Arabic text ---
        fixed_lines = []
        for line in text.split(r"\N"):
            # Reshape the letters so they connect properly
            reshaped_line = arabic_reshaper.reshape(line)
            # Apply bidirectional algorithm so it reads RTL
            bidi_line = get_display(reshaped_line)
            fixed_lines.append(bidi_line)
        text = r"\N".join(fixed_lines)
        # -----------------------------------------------------------

        text = text.replace("{", r"\{").replace("}", r"\}")
        if not text:
            continue

        lead_tag = r"{\an2\blur0.6\bord3.8\shad1.2}"
        if animate:
            lead_tag = (
                r"{\an2\blur0.6\bord3.8\shad1.2"
                r"\fad(90,80)"
                r"\fscx97\fscy97"
                r"\t(0,140,\fscx103\fscy103)"
                r"\t(140,240,\fscx100\fscy100)}"
            )

        dialogue_lines.append(
            f"Dialogue: 0,{sec_to_ass_time(rel_start)},{sec_to_ass_time(rel_end)},"
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
