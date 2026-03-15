"""
text_config.py — Single source of truth for all text styling in clip-factory.

Two built-in style modes
────────────────────────
  subtitle           Bottom-center standard subtitle.  Small text, classic margins.
  center_recitation  Centered large Arabic display text for Quran/recitation shorts.
                     Middle-screen, dramatic, large font, suitable for TikTok-style
                     Islamic content.

Config is loaded LAZILY — never at module import time.
Call load_text_config() after dotenv is loaded to get a fully-resolved
TextStyleConfig instance.

Environment variables (all optional — all have sensible defaults)
──────────────────────────────────────────────────────────────────
  TEXT_STYLE_MODE           "subtitle" or "center_recitation"   default: center_recitation

  Font
  ────
  SUBTITLE_FONT             Font family for subtitle mode        default: Geeza Pro
  RECITATION_FONT           Font family for recitation mode      default: Geeza Pro
                            (falls back to SUBTITLE_FONT if RECITATION_FONT unset)
  SUBTITLE_FONTSIZE         Font size for subtitle mode          default: 60
  RECITATION_FONTSIZE       Font size for recitation mode        default: 88
                            (falls back to SUBTITLE_FONTSIZE if RECITATION_FONTSIZE unset)
  SUBTITLE_MARGIN_V         Vertical margin for subtitle mode    default: 110
  RECITATION_MARGIN_V       Vertical margin for recitation mode  default: 0

  Colours (ASS format: &HAABBGGRR, little-endian)
  ────────────────────────────────────────────────
  ASS_PRIMARY_COLOR         Text fill colour                     mode-specific default
  ASS_OUTLINE_COLOR         Glyph border colour                  mode-specific default
  ASS_SHADOW_COLOR          Drop shadow / box colour             mode-specific default

  Outline / shadow
  ────────────────
  ASS_OUTLINE_WIDTH         float px                             mode-specific default
  ASS_SHADOW_SIZE           float px                             mode-specific default
  ASS_BLUR                  float gaussian edge blur             mode-specific default

  Margins
  ───────
  ASS_MARGIN_L              left margin px                       mode-specific default
  ASS_MARGIN_R              right margin px                      mode-specific default

  Line breaking
  ─────────────
  ASS_MAX_WORDS_PER_LINE    int                                  mode-specific default
  ASS_MAX_LINES             int                                  mode-specific default

  Animation
  ─────────
  ASS_ANIMATE               "true" / "false"                     mode-specific default
  ASS_FADE_IN_MS            int ms                               mode-specific default
  ASS_FADE_OUT_MS           int ms                               mode-specific default

  Arabic text handling
  ────────────────────
  ASS_USE_RESHAPER_BIDI     "true" / "false"                     default: false

    When false (default): raw UTF-8 Arabic is passed to libass, which uses
    HarfBuzz for glyph shaping and FriBidi for bidi ordering natively.
    This is the correct path for modern libass/ffmpeg-full builds.

    When true: arabic_reshaper + python-bidi pre-process the text before
    embedding in the ASS file.  Enable ONLY if your libass build lacks HarfBuzz
    AND you observe disconnected Arabic letters in the rendered output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ─── Internal env helpers (not exported) ──────────────────────────────────────

def _env_str(key: str, default: str) -> str:
    val = os.environ.get(key, "").strip()
    return val if val else default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


# ─── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class TextStyleConfig:
    """
    Fully resolved text styling config for one render pass.

    Build via load_text_config() — do not construct manually in production
    code unless writing tests or performing explicit style overrides.

    ASS colour format: &HAABBGGRR (little-endian, alpha channel first).
    Examples:
      &H00FFFFFF  = opaque white
      &H00000000  = opaque black
      &H80000000  = 50% transparent black
      &H99000000  = ~60% transparent black
    """

    # Style mode identifier — used for logging and conditional downstream logic.
    mode: str

    # ── Font ──────────────────────────────────────────────────────────────────
    font: str
    fontsize: int

    # ── ASS colours ───────────────────────────────────────────────────────────
    primary_color: str    # text fill
    secondary_color: str  # karaoke highlight (rarely active)
    outline_color: str    # glyph border
    shadow_color: str     # drop shadow or opaque box backing

    # ── Text decoration ────────────────────────────────────────────────────────
    bold: bool
    italic: bool
    spacing: float        # extra letter spacing in pixels

    # ── Outline / shadow ───────────────────────────────────────────────────────
    border_style: int     # 1 = outline+shadow,  3 = opaque box
    outline_width: float
    shadow_size: float
    blur: float           # gaussian edge blur applied per dialogue event

    # ── Positioning ────────────────────────────────────────────────────────────
    # ASS numpad alignment convention:
    #   1=bot-left  2=bot-center  3=bot-right
    #   4=mid-left  5=mid-center  6=mid-right
    #   7=top-left  8=top-center  9=top-right
    alignment: int
    margin_l: int
    margin_r: int
    margin_v: int         # pixel offset from the alignment-anchor edge

    # ── Line breaking ──────────────────────────────────────────────────────────
    max_words_per_line: int
    max_lines: int

    # ── Animation ─────────────────────────────────────────────────────────────
    animate: bool
    fade_in_ms: int
    fade_out_ms: int

    # ── Arabic text handling ───────────────────────────────────────────────────
    # See module docstring for full explanation of when to enable this.
    use_reshaper_bidi: bool

    # ── Display cleaning ──────────────────────────────────────────────────────
    # Whether generate_clip_ass should apply clean_arabic_for_captions by default.
    # Must be False for Quran render_text — the corpus text is already canonical
    # and clean_arabic_for_captions would strip valid Quranic characters (e.g. ٱ).
    # The renderer always passes clean_arabic=False explicitly, so this field
    # acts as a safe default for any other call sites.
    clean_for_display: bool


# ─── Mode presets ──────────────────────────────────────────────────────────────
#
# These are the base values before any env overrides.
# Do NOT edit these for per-deployment customisation — use env vars instead.

_SUBTITLE_PRESET: dict = dict(
    mode="subtitle",
    font="Geeza Pro",
    fontsize=60,
    primary_color="&H00FFFFFF",
    secondary_color="&H000000FF",
    outline_color="&H00131313",
    shadow_color="&H6E000000",
    bold=True,
    italic=False,
    spacing=0.15,
    border_style=1,
    outline_width=3.8,
    shadow_size=1.2,
    blur=0.6,
    alignment=2,           # bottom-center
    margin_l=60,
    margin_r=60,
    margin_v=110,
    max_words_per_line=4,
    max_lines=2,
    animate=True,
    fade_in_ms=90,
    fade_out_ms=80,
    use_reshaper_bidi=False,
    clean_for_display=False,
)

_CENTER_RECITATION_PRESET: dict = dict(
    mode="center_recitation",
    font="Geeza Pro",
    fontsize=88,
    primary_color="&H00FFFFFF",
    secondary_color="&H000000FF",
    outline_color="&H00000000",   # pure black outline — crisp on dark backgrounds
    shadow_color="&H99000000",    # ~60% transparent black — soft backing
    bold=True,
    italic=False,
    spacing=0.0,
    border_style=1,
    outline_width=3.0,
    shadow_size=1.5,
    blur=0.4,
    alignment=5,           # middle-center — cinematic center-screen position
    margin_l=80,
    margin_r=80,
    margin_v=0,            # no vertical offset — true center
    max_words_per_line=5,
    max_lines=2,
    animate=True,
    fade_in_ms=120,
    fade_out_ms=100,
    use_reshaper_bidi=False,
    clean_for_display=False,
)

_PRESETS: dict[str, dict] = {
    "subtitle": _SUBTITLE_PRESET,
    "center_recitation": _CENTER_RECITATION_PRESET,
}

VALID_MODES: frozenset[str] = frozenset(_PRESETS)


# ─── Public loader ─────────────────────────────────────────────────────────────

def load_text_config(mode: str | None = None) -> TextStyleConfig:
    """
    Build a TextStyleConfig by layering environment variables on top of the
    mode preset.

    Call this AFTER dotenv is loaded.  It reads os.environ at call time —
    never at module import time — so the value of .env settings is always
    respected.

    Args:
        mode: "subtitle" or "center_recitation".
              If None, reads TEXT_STYLE_MODE from env (default: "center_recitation").

    Returns:
        A fully resolved TextStyleConfig instance.

    Raises:
        ValueError: if mode is not a recognised value.
    """
    if mode is None:
        mode = os.environ.get("TEXT_STYLE_MODE", "center_recitation").strip().lower()

    if mode not in _PRESETS:
        raise ValueError(
            f"Unknown TEXT_STYLE_MODE {mode!r}. "
            f"Valid values: {sorted(VALID_MODES)}"
        )

    # Fresh copy — never mutate the module-level preset dicts.
    cfg = dict(_PRESETS[mode])

    # ── Font ──────────────────────────────────────────────────────────────────
    if mode == "subtitle":
        cfg["font"] = _env_str("SUBTITLE_FONT", cfg["font"])
        cfg["fontsize"] = _env_int("SUBTITLE_FONTSIZE", cfg["fontsize"])
        cfg["margin_v"] = _env_int("SUBTITLE_MARGIN_V", cfg["margin_v"])
    else:
        # RECITATION_FONT takes priority; SUBTITLE_FONT is a convenience fallback
        # so operators only need one font env var when both modes use the same font.
        cfg["font"] = _env_str(
            "RECITATION_FONT",
            _env_str("SUBTITLE_FONT", cfg["font"]),
        )
        cfg["fontsize"] = _env_int(
            "RECITATION_FONTSIZE",
            _env_int("SUBTITLE_FONTSIZE", cfg["fontsize"]),
        )
        cfg["margin_v"] = _env_int("RECITATION_MARGIN_V", cfg["margin_v"])

    # ── Colours ───────────────────────────────────────────────────────────────
    cfg["primary_color"] = _env_str("ASS_PRIMARY_COLOR", cfg["primary_color"])
    cfg["outline_color"] = _env_str("ASS_OUTLINE_COLOR", cfg["outline_color"])
    cfg["shadow_color"] = _env_str("ASS_SHADOW_COLOR", cfg["shadow_color"])

    # ── Outline / shadow ──────────────────────────────────────────────────────
    cfg["outline_width"] = _env_float("ASS_OUTLINE_WIDTH", cfg["outline_width"])
    cfg["shadow_size"] = _env_float("ASS_SHADOW_SIZE", cfg["shadow_size"])
    cfg["blur"] = _env_float("ASS_BLUR", cfg["blur"])

    # ── Margins ───────────────────────────────────────────────────────────────
    cfg["margin_l"] = _env_int("ASS_MARGIN_L", cfg["margin_l"])
    cfg["margin_r"] = _env_int("ASS_MARGIN_R", cfg["margin_r"])

    # ── Line breaking ──────────────────────────────────────────────────────────
    cfg["max_words_per_line"] = _env_int("ASS_MAX_WORDS_PER_LINE", cfg["max_words_per_line"])
    cfg["max_lines"] = _env_int("ASS_MAX_LINES", cfg["max_lines"])

    # ── Animation ─────────────────────────────────────────────────────────────
    cfg["animate"] = _env_bool("ASS_ANIMATE", cfg["animate"])
    cfg["fade_in_ms"] = _env_int("ASS_FADE_IN_MS", cfg["fade_in_ms"])
    cfg["fade_out_ms"] = _env_int("ASS_FADE_OUT_MS", cfg["fade_out_ms"])

    # ── Arabic rendering ──────────────────────────────────────────────────────
    cfg["use_reshaper_bidi"] = _env_bool("ASS_USE_RESHAPER_BIDI", cfg["use_reshaper_bidi"])

    return TextStyleConfig(**cfg)
