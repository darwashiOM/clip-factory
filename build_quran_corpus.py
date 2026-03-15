from pathlib import Path
import json
import re
import argparse
import unicodedata


ARABIC_DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]'
)

PUNCT_RE = re.compile(r'[^\u0600-\u06FF0-9\s]+')


def strip_tashkeel(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub('', text)


def normalize_arabic_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
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


def parse_tanzil_file(path: Path) -> dict:
    """
    Expected common format:
    surah|ayah|text
    Also supports tab-delimited.
    Skips blank lines and comment/header lines that start with #.
    """
    out = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "|" in line:
            parts = line.split("|", 2)
        elif "\t" in line:
            parts = line.split("\t", 2)
        else:
            raise ValueError(f"Unsupported line format in {path}: {line[:80]}")

        if len(parts) != 3:
            raise ValueError(f"Bad line in {path}: {line[:80]}")

        surah, ayah, text = parts
        key = f"{int(surah)}:{int(ayah)}"
        out[key] = text.strip()

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", required=True, help="Path to Tanzil simple/clean text with ayah numbers")
    parser.add_argument("--uthmani", required=True, help="Path to Tanzil uthmani text with ayah numbers")
    parser.add_argument("--out", required=True, help="Output corpus json path")
    args = parser.parse_args()

    simple_path = Path(args.simple).expanduser().resolve()
    uthmani_path = Path(args.uthmani).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    simple = parse_tanzil_file(simple_path)
    uthmani = parse_tanzil_file(uthmani_path)

    keys = sorted(set(simple.keys()) & set(uthmani.keys()), key=lambda k: tuple(map(int, k.split(":"))))

    verses = []
    for key in keys:
        surah, ayah = map(int, key.split(":"))
        simple_text = simple[key]
        uthmani_text = uthmani[key]

        verses.append(
            {
                "verse_key": key,
                "surah": surah,
                "ayah": ayah,
                "simple": simple_text,
                "uthmani": uthmani_text,
                "norm_simple": normalize_arabic_for_match(simple_text),
                "norm_uthmani": normalize_arabic_for_match(uthmani_text),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "source": "tanzil",
                "verse_count": len(verses),
                "verses": verses,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote Quran corpus to: {out_path}")
    print(f"Verse count: {len(verses)}")


if __name__ == "__main__":
    main()
