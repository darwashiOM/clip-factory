from pathlib import Path
import os
import json
import re
import math
import unicodedata
from functools import lru_cache
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Optional, List, Tuple, Dict, Set, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import llm_client

# Two-phase bootstrap — see renderer_server_veo_timeline.py for full explanation.
_INITIAL_ROOT = Path(
    os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
).resolve()
load_dotenv(_INITIAL_ROOT / ".env")

ROOT = Path(
    os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
).resolve()
if ROOT != _INITIAL_ROOT and (ROOT / ".env").exists():
    load_dotenv(ROOT / ".env", override=True)

TRANSCRIPTS = ROOT / "transcripts"
DATA_QURAN = ROOT / "data" / "quran"
CORPUS_PATH = DATA_QURAN / "quran_corpus.json"

mcp = FastMCP("clip-factory-quran-guard", json_response=True)

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
PUNCT_RE = re.compile(r"[^\u0600-\u06FF0-9\s]+")

# Common words that are too weak to use as strong anchors by themselves.
_RAW_STOPWORDS = {
    "و", "او", "أو", "ثم", "ف", "ب", "ك", "ل", "في", "من", "عن", "على", "الى", "إلى",
    "ان", "إن", "انا", "أَنَا", "ما", "ماذا", "متى", "كيف", "لم", "لن", "لا", "ليس",
    "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "تلك", "هناك", "هنا", "كل", "كما",
    "قد", "كان", "كانت", "يكون", "تكون", "الذي", "التي", "الذين", "اللاتي", "اللاتي",
    "يا", "اي", "أي", "اذا", "إذا", "بل", "حتى", "بعد", "قبل", "عند", "بين", "مع",
    "له", "لها", "لهم", "عليه", "عليها", "عليهم", "به", "بها", "بهم", "فيه", "فيها",
    "انه", "إنه", "انها", "إنها", "قال", "قالوا", "قلت", "يقول", "تقول", "رب", "ربي",
    "الله", "اللهم",
}
STOPWORDS = set()

_cached_corpus: Optional[dict] = None


def strip_tashkeel(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def normalize_arabic_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("ـ", "")
    text = strip_tashkeel(text)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


for _w in _RAW_STOPWORDS:
    STOPWORDS.add(normalize_arabic_for_match(_w))


def clean_render_arabic_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("ـ", "")
    # Preserve tashkil for display/render text.
    text = re.sub(r"\s+", " ", text).strip()
    return text


_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u066E-\u066F\u0671-\u06D3\u06FA-\u06FC]")


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _token_count(text: str) -> int:
    return len([w for w in str(text or "").split() if w])


@lru_cache(maxsize=4096)
def _diacritize_non_quran_text(text: str) -> str:
    """
    Add tashkil to non-Quran Arabic for display only.

    Safety rules:
    - Never paraphrase
    - Never add/remove words
    - If backend fails, return the original cleaned text
    - If token count changes, return the original cleaned text
    """
    source = clean_render_arabic_text(text)
    if not source:
        return source

    if not _env_bool("ARABIC_DIACRITIZE_NON_QURAN", True):
        return source

    if not _ARABIC_LETTER_RE.search(source):
        return source

    system = (
        "You are a precise Arabic diacritizer. "
        "Add Arabic tashkil to the exact same text. "
        "Do not paraphrase. Do not summarize. Do not explain. "
        "Do not add or remove words. Keep punctuation and word order exactly. "
        "Return JSON only with one key named text."
    )

    prompt = (
        "Diacritize the following Arabic text.\n"
        "Rules:\n"
        "- Preserve the exact same words and order.\n"
        "- Preserve the exact same whitespace-separated token count.\n"
        "- Keep non-Arabic tokens unchanged.\n"
        "- If you are uncertain about a token, keep that token unchanged.\n\n"
        f'Text: "{source}"'
    )

    try:
        llm = llm_client.get_text_llm()
        raw = llm.generate_json(prompt, system=system)
        data = json.loads(raw)
        out = clean_render_arabic_text(data.get("text", ""))

        if not out:
            return source

        if _token_count(out) != _token_count(source):
            return source

        return out
    except Exception:
        if _env_bool("ARABIC_DIACRITIZE_FAIL_OPEN", True):
            return source
        raise


def _tokenize_norm(text: str) -> List[str]:
    return [t for t in normalize_arabic_for_match(text).split() if t]


def _bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def _is_anchor_token(token: str, corpus: dict) -> bool:
    if not token or len(token) < 2:
        return False
    if token in STOPWORDS:
        return False

    df = corpus["token_df"].get(token, 0)
    verse_count = max(1, corpus["verse_count"])
    if (df / verse_count) > 0.08:
        return False
    return True


def _sec_to_srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def _write_srt(cues: List[dict], out_path: Path):
    lines = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_sec_to_srt_time(float(cue['start']))} --> {_sec_to_srt_time(float(cue['end']))}")
        lines.append(str(cue["text"]).strip())
        lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_corpus() -> dict:
    global _cached_corpus
    if _cached_corpus is not None:
        return _cached_corpus

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Missing Quran corpus: {CORPUS_PATH}. Build it first with build_quran_corpus.py")

    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    verses = data.get("verses", [])
    if not verses:
        raise RuntimeError("Quran corpus exists but contains no verses")

    token_df: Counter = Counter()
    token_to_verse_ids: Dict[str, List[int]] = defaultdict(list)
    all_token_to_verse_ids: Dict[str, List[int]] = defaultdict(list)
    bigram_to_verse_ids: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, verse in enumerate(verses):
        simple = str(verse.get("simple", "")).strip()
        uthmani = str(verse.get("uthmani", "")).strip()
        norm_simple = str(verse.get("norm_simple", "")).strip() or normalize_arabic_for_match(simple)

        norm_tokens = [t for t in norm_simple.split() if t]
        simple_tokens = [t for t in simple.split() if t]
        uthmani_tokens = [t for t in uthmani.split() if t]

        verse["norm_simple"] = norm_simple
        verse["_idx"] = idx
        verse["_norm_simple_tokens"] = norm_tokens
        verse["_norm_simple_token_set"] = set(norm_tokens)
        verse["_simple_tokens"] = simple_tokens
        verse["_uthmani_tokens"] = uthmani_tokens
        verse["_token_len"] = len(norm_tokens)

        token_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, tok in enumerate(norm_tokens):
            token_positions[tok].append(pos)
        verse["_token_positions"] = dict(token_positions)

        bigram_positions: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        verse_bigrams = _bigrams(norm_tokens)
        for pos, bg in enumerate(verse_bigrams):
            bigram_positions[bg].append(pos)
        verse["_bigram_positions"] = dict(bigram_positions)
        verse["_norm_bigrams_set"] = set(verse_bigrams)

        unique_all_tokens = set(norm_tokens)
        unique_anchor_tokens = {t for t in unique_all_tokens if len(t) >= 2 and t not in STOPWORDS}

        for tok in unique_all_tokens:
            all_token_to_verse_ids[tok].append(idx)

        for tok in unique_anchor_tokens:
            token_df[tok] += 1
            token_to_verse_ids[tok].append(idx)

        for bg in set(bg for bg in verse_bigrams if bg[0] not in STOPWORDS and bg[1] not in STOPWORDS):
            bigram_to_verse_ids[bg].append(idx)

    verse_count = len(verses)
    token_idf = {}
    for tok, df in token_df.items():
        token_idf[tok] = 1.0 + math.log((verse_count + 1) / (df + 1))

    data["verses"] = verses
    data["verse_count"] = verse_count
    data["token_df"] = dict(token_df)
    data["token_idf"] = token_idf
    data["token_to_verse_ids"] = dict(token_to_verse_ids)
    data["all_token_to_verse_ids"] = dict(all_token_to_verse_ids)
    data["bigram_to_verse_ids"] = dict(bigram_to_verse_ids)

    _cached_corpus = data
    return _cached_corpus


def _load_verbose_transcript(stem: str) -> dict:
    preferred = [
        TRANSCRIPTS / f"{stem}.refined.verbose.json",
        TRANSCRIPTS / f"{stem}.verbose.json",
    ]
    verbose_path = next((p for p in preferred if p.exists()), None)
    if not verbose_path:
        raise FileNotFoundError(f"Missing verbose transcript for stem: {stem}")
    return json.loads(verbose_path.read_text(encoding="utf-8"))


def _load_clip_verbose(clip_stem: str) -> dict:
    preferred = [
        TRANSCRIPTS / f"{clip_stem}.refined.verbose.json",
        TRANSCRIPTS / f"{clip_stem}.verbose.json",
    ]
    verbose_path = next((p for p in preferred if p.exists()), None)
    if not verbose_path:
        raise FileNotFoundError(f"Missing clip verbose transcript: {clip_stem}")
    return json.loads(verbose_path.read_text(encoding="utf-8"))


def _list_verbose_sources(limit: int = 50):
    if not TRANSCRIPTS.exists():
        return []

    files = sorted(
        [p for p in TRANSCRIPTS.glob("*.verbose.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return [
        {
            "stem": p.name.removesuffix(".verbose.json"),
            "path": str(p),
            "size_bytes": p.stat().st_size,
        }
        for p in files[:limit]
    ]


def _word_count(text: str) -> int:
    return len([w for w in normalize_arabic_for_match(text).split() if w])


def _token_counter(tokens: List[str]) -> Counter:
    return Counter(t for t in tokens if t)


def _token_f1_from_tokens(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0

    ca = _token_counter(a_tokens)
    cb = _token_counter(b_tokens)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0

    precision = common / max(1, len(a_tokens))
    recall = common / max(1, len(b_tokens))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _build_source_features(source_norm: str) -> dict:
    corpus = _load_corpus()
    tokens = [t for t in source_norm.split() if t]
    unique_tokens = list(dict.fromkeys(tokens))

    anchor_tokens = [t for t in tokens if _is_anchor_token(t, corpus)]
    unique_anchor_tokens = list(dict.fromkeys(anchor_tokens))
    unique_anchor_tokens_sorted = sorted(
        unique_anchor_tokens,
        key=lambda t: (-corpus["token_idf"].get(t, 0.0), t),
    )

    token_positions: Dict[str, List[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        token_positions[tok].append(idx)

    content_bigrams = [bg for bg in _bigrams(tokens) if _is_anchor_token(bg[0], corpus) and _is_anchor_token(bg[1], corpus)]
    unique_content_bigrams = list(dict.fromkeys(content_bigrams))

    return {
        "source_norm": source_norm,
        "tokens": tokens,
        "token_len": len(tokens),
        "unique_tokens": unique_tokens,
        "anchor_tokens": anchor_tokens,
        "unique_anchor_tokens": unique_anchor_tokens,
        "unique_anchor_tokens_sorted": unique_anchor_tokens_sorted,
        "token_positions": dict(token_positions),
        "content_bigrams": unique_content_bigrams,
        "content_bigrams_set": set(unique_content_bigrams),
    }


def _weighted_token_overlap(unique_source_anchor_tokens: List[str], candidate_token_set: Set[str], corpus: dict) -> Tuple[float, int]:
    if not unique_source_anchor_tokens:
        return 0.0, 0

    total = 0.0
    matched = 0.0
    matched_count = 0

    for tok in unique_source_anchor_tokens:
        w = corpus["token_idf"].get(tok, 1.0)
        total += w
        if tok in candidate_token_set:
            matched += w
            matched_count += 1

    if total <= 0:
        return 0.0, matched_count
    return min(1.0, matched / total), matched_count


def _bigram_overlap(source_bigrams_set: Set[Tuple[str, str]], candidate_bigrams_set: Set[Tuple[str, str]]) -> float:
    if not source_bigrams_set:
        return 0.0
    inter = len(source_bigrams_set & candidate_bigrams_set)
    return inter / max(1, len(source_bigrams_set))

def _safe_sync_segment_words_to_render_text(seg: dict, render_text: str) -> tuple[dict, bool]:
    """
    Safely rewrite seg["words"][i]["word"] to match render_text, but ONLY when
    the corrected text can be mapped 1-to-1 onto the existing timed words.

    Returns:
        (updated_segment, synced_ok)

    Safety rule:
    - If there is any mismatch in non-empty word count, do nothing to seg["words"].
    - We still update seg["text"] so the pipeline keeps working as before.
    """
    seg_out = dict(seg)
    render_text = str(render_text or "").replace("\n", " ").strip()
    seg_out["text"] = render_text

    raw_words = list(seg_out.get("words") or [])
    if not render_text or not raw_words:
        return seg_out, False

    corrected_tokens = [w for w in render_text.split() if w]
    if not corrected_tokens:
        return seg_out, False

    nonempty_positions = []
    for idx, item in enumerate(raw_words):
        token = str((item or {}).get("word", "")).strip()
        if token:
            nonempty_positions.append(idx)

    if len(corrected_tokens) != len(nonempty_positions):
        return seg_out, False

    new_words = []
    cursor = 0
    nonempty_pos_set = set(nonempty_positions)

    for idx, item in enumerate(raw_words):
        item_copy = dict(item or {})
        if idx in nonempty_pos_set:
            item_copy["word"] = corrected_tokens[cursor]
            cursor += 1
        new_words.append(item_copy)

    seg_out["words"] = new_words
    return seg_out, True


def _score_candidate_tokens(source_features: dict, candidate_tokens: List[str]) -> Tuple[float, dict]:
    corpus = _load_corpus()
    source_norm = source_features["source_norm"]
    source_tokens = source_features["tokens"]
    candidate_norm = " ".join(candidate_tokens).strip()
    if not source_tokens or not candidate_tokens or not candidate_norm:
        return 0.0, {
            "token_f1": 0.0,
            "char_ratio": 0.0,
            "weighted_overlap": 0.0,
            "bigram_overlap": 0.0,
            "length_ratio": 0.0,
            "matched_anchor_tokens": 0,
        }

    token_f1 = _token_f1_from_tokens(source_tokens, candidate_tokens)
    char_ratio = SequenceMatcher(None, source_norm, candidate_norm).ratio()

    candidate_token_set = set(candidate_tokens)
    weighted_overlap, matched_anchor_tokens = _weighted_token_overlap(
        source_features["unique_anchor_tokens"],
        candidate_token_set,
        corpus,
    )

    candidate_bigrams_set = set(bg for bg in _bigrams(candidate_tokens) if bg[0] not in STOPWORDS and bg[1] not in STOPWORDS)
    bigram_overlap = _bigram_overlap(source_features["content_bigrams_set"], candidate_bigrams_set)

    source_len = len(source_tokens)
    cand_len = len(candidate_tokens)
    length_ratio = min(source_len, cand_len) / max(1, max(source_len, cand_len))

    contain_bonus = 0.025 if (source_norm in candidate_norm or candidate_norm in source_norm) else 0.0

    score = (
        0.44 * weighted_overlap +
        0.24 * token_f1 +
        0.18 * bigram_overlap +
        0.14 * char_ratio +
        contain_bonus
    )

    if length_ratio < 0.55:
        score *= 0.88

    if matched_anchor_tokens == 0 and len(source_features["unique_anchor_tokens"]) >= 2:
        score *= 0.82

    return min(1.0, max(0.0, score)), {
        "token_f1": round(token_f1, 4),
        "char_ratio": round(char_ratio, 4),
        "weighted_overlap": round(weighted_overlap, 4),
        "bigram_overlap": round(bigram_overlap, 4),
        "length_ratio": round(length_ratio, 4),
        "matched_anchor_tokens": matched_anchor_tokens,
    }


def _retrieve_candidate_verse_ids(source_features: dict, max_candidates: int = 80) -> List[int]:
    corpus = _load_corpus()
    verse_count = corpus["verse_count"]
    scores: Dict[int, float] = defaultdict(float)

    # Strongest signal: rare anchor tokens.
    for tok in source_features["unique_anchor_tokens_sorted"][:10]:
        tok_weight = corpus["token_idf"].get(tok, 1.0)
        for vid in corpus["token_to_verse_ids"].get(tok, []):
            scores[vid] += 1.8 * tok_weight

    # Next signal: anchor bigrams.
    for bg in source_features["content_bigrams"][:8]:
        for vid in corpus["bigram_to_verse_ids"].get(bg, []):
            scores[vid] += 2.4

    # Fallback: all normalized tokens.
    if len(scores) < 20:
        for tok in source_features["unique_tokens"][:12]:
            for vid in corpus["all_token_to_verse_ids"].get(tok, []):
                scores[vid] += 0.35

    if not scores:
        return list(range(verse_count))

    ranked = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            abs(corpus["verses"][kv[0]].get("_token_len", 0) - source_features["token_len"]),
            kv[0],
        ),
    )
    return [vid for vid, _ in ranked[:max_candidates]]


def _anchored_fragment_search(source_features: dict, verse: dict) -> Tuple[float, Optional[Tuple[int, int]], dict]:
    corpus = _load_corpus()
    source_tokens = source_features["tokens"]
    verse_tokens = verse.get("_norm_simple_tokens") or []

    if not source_tokens or not verse_tokens:
        return 0.0, None, {}

    if len(source_tokens) < 3 or len(verse_tokens) < 3:
        return 0.0, None, {}

    if len(source_features["unique_anchor_tokens"]) < 2:
        return 0.0, None, {}

    source_len = len(source_tokens)
    verse_len = len(verse_tokens)
    source_positions = source_features["token_positions"]
    verse_positions = verse.get("_token_positions") or {}
    verse_bigram_positions = verse.get("_bigram_positions") or {}

    anchors: Set[int] = set()

    # Use bigram anchors first.
    for bg in source_features["content_bigrams"][:6]:
        for pos in verse_bigram_positions.get(bg, [])[:4]:
            for delta in (-1, 0, 1):
                start = pos + delta
                if 0 <= start < verse_len:
                    anchors.add(start)

    # Then token-aligned estimated starts.
    for tok in source_features["unique_anchor_tokens_sorted"][:6]:
        src_pos_list = source_positions.get(tok, [])[:3]
        verse_pos_list = verse_positions.get(tok, [])[:5]
        for src_pos in src_pos_list:
            for verse_pos in verse_pos_list:
                est = verse_pos - src_pos
                for delta in (-1, 0, 1):
                    start = est + delta
                    if 0 <= start < verse_len:
                        anchors.add(start)

    if not anchors:
        return 0.0, None, {}

    min_len = max(2, source_len - 1)
    max_len = min(verse_len, source_len + 2)

    best_score = 0.0
    best_span = None
    best_meta: dict = {}

    # Limit evaluated anchor count so this stays fast.
    for start in sorted(anchors)[:40]:
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > verse_len:
                continue

            frag_tokens = verse_tokens[start:end]
            score, meta = _score_candidate_tokens(source_features, frag_tokens)

            if meta.get("matched_anchor_tokens", 0) < 2:
                continue
            if meta.get("weighted_overlap", 0.0) < 0.60:
                continue

            # Slight preference for fragments that do not look wildly offset.
            if start > 0 and end < verse_len:
                score *= 0.995

            if score > best_score:
                best_score = score
                best_span = (start, end)
                best_meta = dict(meta)

    return best_score, best_span, best_meta


@lru_cache(maxsize=8192)
def _best_verse_match_cached(source_norm: str) -> Tuple[Optional[int], float, float, Optional[Tuple[int, int]], str, dict]:
    corpus = _load_corpus()
    if not source_norm:
        return None, 0.0, 0.0, None, "none", {}

    source_features = _build_source_features(source_norm)
    if not source_features["tokens"]:
        return None, 0.0, 0.0, None, "none", {}

    candidate_ids = _retrieve_candidate_verse_ids(source_features, max_candidates=80)

    best_verse_id: Optional[int] = None
    best_score = 0.0
    second_score = 0.0
    best_span: Optional[Tuple[int, int]] = None
    best_mode = "none"
    best_meta: dict = {}

    full_ranked: List[Tuple[float, int, dict]] = []

    for vid in candidate_ids:
        verse = corpus["verses"][vid]
        full_score, full_meta = _score_candidate_tokens(source_features, verse.get("_norm_simple_tokens") or [])
        full_ranked.append((full_score, vid, full_meta))

        if full_score > best_score:
            second_score = best_score
            best_score = full_score
            best_verse_id = vid
            best_span = None
            best_mode = "full"
            best_meta = dict(full_meta)
        elif full_score > second_score:
            second_score = full_score

    # Only attempt fragment search on the best few full-verse candidates.
    for _, vid, full_meta in sorted(full_ranked, key=lambda x: x[0], reverse=True)[:20]:
        verse = corpus["verses"][vid]
        fragment_score, fragment_span, fragment_meta = _anchored_fragment_search(source_features, verse)
        if not fragment_span:
            continue

        fragment_meta = dict(fragment_meta)
        fragment_meta["fragment_strict_threshold"] = 0.60

        # Fragment replacement is intentionally stricter than full-verse replacement,
        # but no longer hard-locked near 0.94.
        if fragment_score < 0.60:
            continue
        if fragment_meta.get("matched_anchor_tokens", 0) < 3:
            continue
        if fragment_meta.get("weighted_overlap", 0.0) < 0.72:
            continue
        if fragment_score <= max((full_meta or {}).get("weighted_overlap", 0.0), 0.0):
            pass

        if fragment_score > (best_score + 0.012):
            second_score = max(second_score, best_score)
            best_score = fragment_score
            best_verse_id = vid
            best_span = fragment_span
            best_mode = "fragment"
            best_meta = fragment_meta
        elif fragment_score > second_score:
            second_score = fragment_score

    if best_verse_id is None:
        return None, 0.0, 0.0, None, "none", {}

    meta_out = dict(best_meta)
    meta_out["candidate_count"] = len(candidate_ids)
    meta_out["source_token_len"] = len(source_features["tokens"])
    meta_out["source_anchor_token_count"] = len(source_features["unique_anchor_tokens"])
    return best_verse_id, best_score, second_score, best_span, best_mode, meta_out


def _best_verse_match(text: str) -> Tuple[Optional[dict], float, float, Optional[Tuple[int, int]], str, dict]:
    corpus = _load_corpus()
    source_norm = normalize_arabic_for_match(text)
    verse_id, best_score, second_score, span, match_mode, meta = _best_verse_match_cached(source_norm)
    if verse_id is None:
        return None, 0.0, 0.0, None, "none", meta
    return corpus["verses"][verse_id], best_score, second_score, span, match_mode, meta


def _choose_render_text(verse: dict, render_script: str, span: Optional[Tuple[int, int]] = None) -> str:
    if render_script == "uthmani":
        full_text = verse.get("uthmani", "")
        tokens = verse.get("_uthmani_tokens") or [t for t in str(full_text).split() if t]
    else:
        full_text = verse.get("simple", "")
        tokens = verse.get("_simple_tokens") or [t for t in str(full_text).split() if t]

    if not span:
        return full_text

    start, end = span
    if 0 <= start < end <= len(tokens):
        fragment = " ".join(tokens[start:end]).strip()
        if fragment:
            return fragment

    return full_text


def _is_acceptable_match(
    best_score: float,
    second_score: float,
    match_mode: str,
    meta: dict,
    min_confidence: float,
    min_margin_over_second: float,
) -> bool:
    margin = best_score - second_score
    if best_score < min_confidence:
        return False
    if margin < min_margin_over_second:
        return False

    weighted_overlap = float(meta.get("weighted_overlap", 0.0))
    matched_anchor_tokens = int(meta.get("matched_anchor_tokens", 0))
    bigram_overlap = float(meta.get("bigram_overlap", 0.0))

    if match_mode == "fragment":
        if best_score < max(0.60, min_confidence):
            return False
        if matched_anchor_tokens < 2:
            return False
        if weighted_overlap < 0.58:
            return False
        if bigram_overlap < 0.22 and matched_anchor_tokens < 3:
            return False
    else:
        if matched_anchor_tokens == 0 and weighted_overlap < 0.52:
            return False
        if matched_anchor_tokens < 2 and weighted_overlap < 0.60:
            return False

    return True


def _guard_segments(
    segments: List[dict],
    min_confidence: float,
    min_margin_over_second: float,
    min_words: int,
    render_script: str,
    max_window_segments: int,
):
    augmented_segments = []
    cues = []
    matches = []

    i = 0
    while i < len(segments):
        best_window = None

        max_size = min(max_window_segments, len(segments) - i)
        for window_size in range(1, max_size + 1):
            window_segments = segments[i:i + window_size]
            window_text = " ".join(str(s.get("text", "")).strip() for s in window_segments).strip()
            if not window_text:
                continue
            if _word_count(window_text) < min_words:
                continue

            verse, best_score, second_score, span, match_mode, meta = _best_verse_match(window_text)
            if verse is None:
                continue

            if not _is_acceptable_match(
                best_score=best_score,
                second_score=second_score,
                match_mode=match_mode,
                meta=meta,
                min_confidence=min_confidence,
                min_margin_over_second=min_margin_over_second,
            ):
                continue

            render_text = _choose_render_text(verse, render_script, span=span)
            if not render_text:
                continue

            candidate = {
                "window_size": window_size,
                "window_segments": window_segments,
                "window_text": window_text,
                "verse": verse,
                "best_score": best_score,
                "second_score": second_score,
                "render_text": render_text,
                "match_mode": match_mode,
                "span": span,
                "meta": meta,
            }

            if best_window is None:
                best_window = candidate
            else:
                # Prefer higher confidence, then stronger overlap, then smaller window.
                current_key = (
                    candidate["best_score"],
                    float(candidate["meta"].get("weighted_overlap", 0.0)),
                    float(candidate["meta"].get("bigram_overlap", 0.0)),
                    -candidate["window_size"],
                )
                best_key = (
                    best_window["best_score"],
                    float(best_window["meta"].get("weighted_overlap", 0.0)),
                    float(best_window["meta"].get("bigram_overlap", 0.0)),
                    -best_window["window_size"],
                )
                if current_key > best_key:
                    best_window = candidate

        if best_window:
            win = best_window
            verse = win["verse"]
            window_start = float(win["window_segments"][0]["start"])
            window_end = float(win["window_segments"][-1]["end"])
            render_text = win["render_text"]
            matched_span = list(win["span"]) if win["span"] else None
            canonical_uthmani = _choose_render_text(verse, "uthmani", span=win["span"])

            cues.append(
                {
                    "start": window_start,
                    "end": window_end,
                    "text": render_text,
                    "is_quran_match": True,
                    "verse_key": verse.get("verse_key", ""),
                    "match_confidence": round(win["best_score"], 4),
                    "match_mode": win["match_mode"],
                }
            )

            matches.append(
                {
                    "start": window_start,
                    "end": window_end,
                    "verse_key": verse.get("verse_key", ""),
                    "match_confidence": round(win["best_score"], 4),
                    "second_score": round(win["second_score"], 4),
                    "margin_over_second": round(win["best_score"] - win["second_score"], 4),
                    "raw_window_text": win["window_text"],
                    "canonical_render_text": render_text,
                    "canonical_uthmani_text": canonical_uthmani,
                    "match_mode": win["match_mode"],
                    "matched_span": matched_span,
                    "evidence": {
                        "weighted_overlap": win["meta"].get("weighted_overlap"),
                        "bigram_overlap": win["meta"].get("bigram_overlap"),
                        "token_f1": win["meta"].get("token_f1"),
                        "char_ratio": win["meta"].get("char_ratio"),
                        "matched_anchor_tokens": win["meta"].get("matched_anchor_tokens"),
                        "candidate_count": win["meta"].get("candidate_count"),
                    },
                }
            )

            for idx_in_window, seg in enumerate(win["window_segments"]):
                seg_copy = dict(seg)

                # ── Timing fix for multi-segment windows ──────────────────────
                # The renderer iterates augmented_segments and creates one
                # Dialogue line per non-empty text segment.  Segments 1..N-1
                # carry empty text and are skipped.  Without this fix, segment 0
                # keeps its original end time, causing the verse text to vanish
                # too early (e.g. a window covering 10–16 s would only render
                # 10–12 s if segment 0 originally ended at 12 s).
                #
                # Fix: extend segment 0's end to the full window end so the
                # renderer produces a Dialogue line spanning the entire window.
                # The SRT cue output above already has the correct timing; this
                # aligns the verbose JSON to match it.
                if idx_in_window == 0 and win["window_size"] > 1:
                    seg_copy["end"] = window_end

                display_text = render_text if idx_in_window == 0 else ""

                if idx_in_window == 0 and display_text:
                    seg_copy, timed_words_synced = _safe_sync_segment_words_to_render_text(seg_copy, display_text)
                else:
                    seg_copy["text"] = ""
                    timed_words_synced = False

                seg_copy["quran_guard"] = {
                    "is_quran_match": True,
                    "verse_key": verse.get("verse_key", ""),
                    "match_confidence": round(win["best_score"], 4),
                    "canonical_render_text": render_text,
                    "canonical_uthmani_text": canonical_uthmani,
                    "absorbed_into_previous_window_segment": idx_in_window > 0,
                    "render_text": display_text,
                    "window_start": window_start,
                    "window_end": window_end,
                    "match_mode": win["match_mode"],
                    "matched_span": matched_span,
                    "timed_words_synced": timed_words_synced,
                }
                augmented_segments.append(seg_copy)

            i += win["window_size"]
            continue

        seg = dict(segments[i])
        raw_text = str(seg.get("text", "")).strip()
        render_text = _diacritize_non_quran_text(raw_text)

        seg, timed_words_synced = _safe_sync_segment_words_to_render_text(seg, render_text)

        seg["quran_guard"] = {
            "is_quran_match": False,
            "render_text": render_text,
            "timed_words_synced": timed_words_synced,
            "render_source": "diacritized_non_quran",
        }
        augmented_segments.append(seg)

        if render_text:
            cues.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": render_text,
                    "is_quran_match": False,
                    "verse_key": "",
                    "match_confidence": None,
                    "match_mode": "none",
                }
            )

        i += 1

    return augmented_segments, cues, matches


@mcp.tool()
def quran_guard_healthcheck() -> dict:
    corpus_exists = CORPUS_PATH.exists()
    verse_count = 0
    if corpus_exists:
        try:
            verse_count = int(json.loads(CORPUS_PATH.read_text(encoding="utf-8")).get("verse_count", 0))
        except Exception:
            verse_count = 0

    return {
        "ok": True,
        "root": str(ROOT),
        "transcripts_exists": TRANSCRIPTS.exists(),
        "corpus_path": str(CORPUS_PATH),
        "corpus_exists": corpus_exists,
        "verse_count": verse_count,
    }


@mcp.tool()
def list_quran_guard_sources(limit: int = 50) -> dict:
    return {
        "folder": str(TRANSCRIPTS),
        "sources": _list_verbose_sources(limit),
    }


@mcp.tool()
def match_text_against_quran(text: str, render_script: str = "uthmani") -> dict:
    if render_script not in {"simple", "uthmani"}:
        raise ValueError("render_script must be 'simple' or 'uthmani'")

    verse, best_score, second_score, span, match_mode, meta = _best_verse_match(text)
    if not verse:
        return {
            "ok": True,
            "input_text": text,
            "normalized_text": normalize_arabic_for_match(text),
            "matched": False,
            "score": 0.0,
        }

    return {
        "ok": True,
        "input_text": text,
        "normalized_text": normalize_arabic_for_match(text),
        "matched": True,
        "best_score": round(best_score, 4),
        "second_score": round(second_score, 4),
        "margin_over_second": round(best_score - second_score, 4),
        "verse_key": verse.get("verse_key", ""),
        "match_mode": match_mode,
        "matched_span": list(span) if span else None,
        "canonical_render_text": _choose_render_text(verse, render_script, span=span),
        "canonical_uthmani_text": _choose_render_text(verse, "uthmani", span=span),
        "evidence": meta,
    }


@mcp.tool()
def fix_quran_in_transcript(
    stem: str,
    min_confidence: float = 0.60,
    min_margin_over_second: float = 0.02,
    min_words: int = 4,
    render_script: str = "uthmani",
    max_window_segments: int = 3,
) -> dict:
    """
    Create Quran-corrected render artifacts from transcripts/<stem>.verbose.json

    Output files:
    - transcripts/<stem>.quran_guard.verbose.json
    - transcripts/<stem>.quran_guard.srt
    - transcripts/<stem>.quran_guard.summary.json

    Raw ASR transcript is NOT modified.
    """
    if render_script not in {"simple", "uthmani"}:
        raise ValueError("render_script must be 'simple' or 'uthmani'")

    data = _load_verbose_transcript(stem)
    segments = data.get("segments") or []
    if not segments:
        raise RuntimeError(f"No segments found for stem {stem}")

    augmented_segments, cues, matches = _guard_segments(
        segments=segments,
        min_confidence=min_confidence,
        min_margin_over_second=min_margin_over_second,
        min_words=min_words,
        render_script=render_script,
        max_window_segments=max_window_segments,
    )

    guarded_verbose = dict(data)
    guarded_verbose["segments"] = augmented_segments
    guarded_verbose["text"] = " ".join(
        str(seg.get("text", "")).strip()
        for seg in augmented_segments
        if str(seg.get("text", "")).strip()
    )
    guarded_verbose["quran_guard"] = {
        "enabled": True,
        "render_script": render_script,
        "match_count": len(matches),
        "min_confidence": min_confidence,
        "min_margin_over_second": min_margin_over_second,
        "min_words": min_words,
        "max_window_segments": max_window_segments,
        "supports_partial_verse_matching": True,
        "matcher_version": "fast-indexed-v2",
    }

    verbose_out = TRANSCRIPTS / f"{stem}.quran_guard.verbose.json"
    srt_out = TRANSCRIPTS / f"{stem}.quran_guard.srt"
    summary_out = TRANSCRIPTS / f"{stem}.quran_guard.summary.json"

    _write_json(verbose_out, guarded_verbose)
    _write_srt(cues, srt_out)
    _write_json(
        summary_out,
        {
            "stem": stem,
            "render_script": render_script,
            "match_count": len(matches),
            "matches": matches,
            "output_files": {
                "verbose_json": str(verbose_out),
                "srt": str(srt_out),
                "summary_json": str(summary_out),
            },
        },
    )

    return {
        "ok": True,
        "stem": stem,
        "match_count": len(matches),
        "output_files": {
            "verbose_json": str(verbose_out),
            "srt": str(srt_out),
            "summary_json": str(summary_out),
        },
        "matches": matches[:20],
    }


@mcp.tool()
def fix_quran_in_clip_candidate(
    stem: str,
    clip_number: int,
    min_confidence: float = 0.60,
    min_margin_over_second: float = 0.02,
    min_words: int = 4,
    render_script: str = "uthmani",
    max_window_segments: int = 3,
) -> dict:
    """
    Create Quran-corrected render artifacts for one selected clip candidate.
    Expected source files:
    - transcripts/<stem>__clipNN.refined.verbose.json
    or fallback:
    - transcripts/<stem>__clipNN.verbose.json
    """
    if render_script not in {"simple", "uthmani"}:
        raise ValueError("render_script must be 'simple' or 'uthmani'")

    clip_stem = f"{stem}__clip{clip_number:02d}"
    data = _load_clip_verbose(clip_stem)
    segments = data.get("segments") or []
    if not segments:
        raise RuntimeError(f"No segments found for clip stem {clip_stem}")

    augmented_segments, cues, matches = _guard_segments(
        segments=segments,
        min_confidence=min_confidence,
        min_margin_over_second=min_margin_over_second,
        min_words=min_words,
        render_script=render_script,
        max_window_segments=max_window_segments,
    )

    guarded_verbose = dict(data)
    guarded_verbose["segments"] = augmented_segments
    guarded_verbose["text"] = " ".join(
        str(seg.get("text", "")).strip()
        for seg in augmented_segments
        if str(seg.get("text", "")).strip()
    )
    guarded_verbose["quran_guard_summary"] = {
        "source_stem": stem,
        "clip_number": clip_number,
        "clip_stem": clip_stem,
        "render_script": render_script,
        "match_count": len(matches),
        "matches": matches,
        "supports_partial_verse_matching": True,
        "matcher_version": "fast-indexed-v2",
    }

    verbose_out = TRANSCRIPTS / f"{clip_stem}.quran_guard.verbose.json"
    srt_out = TRANSCRIPTS / f"{clip_stem}.quran_guard.srt"
    summary_out = TRANSCRIPTS / f"{clip_stem}.quran_guard.summary.json"

    _write_json(verbose_out, guarded_verbose)
    _write_srt(cues, srt_out)
    _write_json(
        summary_out,
        {
            "source_stem": stem,
            "clip_number": clip_number,
            "clip_stem": clip_stem,
            "render_script": render_script,
            "match_count": len(matches),
            "matches": matches,
            "output_files": {
                "verbose_json": str(verbose_out),
                "srt": str(srt_out),
                "summary_json": str(summary_out),
            },
        },
    )

    return {
        "ok": True,
        "source_stem": stem,
        "clip_number": clip_number,
        "clip_stem": clip_stem,
        "match_count": len(matches),
        "output_files": {
            "verbose_json": str(verbose_out),
            "srt": str(srt_out),
            "summary_json": str(summary_out),
        },
        "matches": matches[:20],
    }


if __name__ == "__main__":
    mcp.run()
