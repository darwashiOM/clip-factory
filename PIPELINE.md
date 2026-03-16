# clip-factory pipeline

Stock-footage-based short-form video pipeline for Arabic recitation clips.

```
incoming/*.mp4
    │
    ▼  step 1 — transcribe
transcripts/<stem>.verbose.json
    │
    ▼  step 2 — quran_guard  (optional, enriches with canonical Quran text)
transcripts/<stem>.quran_guard.verbose.json
    │
    ▼  step 3 — clip_finder  (LLM or heuristic)
clips/<stem>__clips.json
    │
    ▼  step 4 — transcript_refiner  (LLM ASR correction, per-clip)
clips/<stem>__clip01.verbose.json
    │
    ▼  step 5 — stock_fetch  (Pexels → Pixabay)
broll/<stem>__clip01__ai01__stock.mp4  (symlinks into broll/stock/ cache)
    │
    ▼  step 6 — render
final/<stem>__clip01__*.mp4
```

---

## Minimum required env vars

Add these to `.env` before running anything.

```ini
# Path to this directory
CLIP_FACTORY_ROOT=/Users/you/clip-factory

# Whisper transcription
OPENAI_API_KEY=sk-...

# Clip-finder + transcript-refiner (DeepSeek is ~40× cheaper than GPT-4o)
TEXT_MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...

# Stock footage (get free keys at pexels.com/api and pixabay.com/api)
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
```

Copy `.env.example` → `.env` for the full list of tuneable options.

---

## Step-by-step commands

All commands run from the project root: `cd /Users/you/clip-factory`

### 0. Place your file

```bash
# Drop your video into:
ls incoming/
# Expected: tbfpe_qimmw.mp4  (or any stem you choose)
```

### 1. Transcribe

```bash
python3 -c "
import sys, os
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from transcribe_server import transcribe_file, list_transcribable_files
# List what's available:
import json; print(json.dumps(list_transcribable_files(), indent=2))
"
```

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from transcribe_server import transcribe_file
result = transcribe_file('tbfpe_qimmw.mp4', language='ar')
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Outputs written to `transcripts/`:
- `<stem>.verbose.json` — full Whisper output with word-level timestamps
- `<stem>.srt` — plain subtitles
- `<stem>.captions.srt` — cleaned/stripped subtitles
- `<stem>.txt` — plain text
- `<stem>.chunks.json` — chunking manifest

### 2. Quran guard

Matches ASR segments against the Quran corpus and replaces imprecise transcriptions
with canonical corpus text. Skip this step if the audio is not Quran recitation.

**Prerequisite:** build the corpus once (only needed on first install):
```bash
python3 build_quran_corpus.py
```

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from quran_guard_server import fix_quran_in_transcript
result = fix_quran_in_transcript('tbfpe_qimmw', min_confidence=0.92)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Outputs written to `transcripts/`:
- `<stem>.quran_guard.verbose.json` — enriched segments with `render_text` fields
- `<stem>.quran_guard.srt`
- `<stem>.quran_guard.summary.json`

### 3. Find clips

Finds self-contained 18–40 s clip windows, builds a visual plan (speaker timings +
scenic insert slots), and writes a candidate JSON.

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from clip_finder_server_veo_timeline import find_clips_from_stem
result = find_clips_from_stem(
    stem='tbfpe_qimmw',
    max_clips=3,
    min_seconds=18,
    max_seconds=40,
    generation_count=18,
)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Key response fields:
- `clip_count` — how many candidates were kept
- `clips[].editorial_score` — ranking score (higher = better)
- `clips[].start_sec / end_sec` — timestamps in the source video
- `clips[].visual_plan` — list of `original` / `stock_video` beats

Output: `clips/<stem>__clips.json`

### 4. Refine transcript (per clip)

LLM corrects ASR errors for the exact segment window of the chosen clip.

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from transcript_refiner_server import refine_clip_candidate
result = refine_clip_candidate(stem='tbfpe_qimmw', clip_number=1)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Output: `clips/<stem>__clip01.verbose.json`

Skip this step (zero LLM cost) by setting `TRANSCRIPT_REFINER_USE_LLM=false` in `.env`.

### 5. Fetch stock footage

Downloads scenic inserts for all `stock_video` beats in the clip's visual plan.

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from stock_fetcher_server import fetch_stock_for_candidate
result = fetch_stock_for_candidate(stem='tbfpe_qimmw', clip_number=1)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Downloads are cached in `broll/stock/` keyed by `{provider}__{id}.mp4`.
Per-clip symlinks are created in `broll/`:
```
broll/tbfpe_qimmw__clip01__ai01__stock.mp4 → broll/stock/pexels__12345.mp4
```

### 6. Render

Assembles the final vertical short with alternating speaker + scenic footage and
burned subtitles.

```bash
python3 -c "
import sys, os, json
sys.path.insert(0, 'mcp'); os.chdir('mcp')
from renderer_server_veo_timeline import render_clip_from_candidate
result = render_clip_from_candidate(
    stem='tbfpe_qimmw',
    clip_number=1,
    preset='dark-soft-recitation',
    burn_subtitles=True,
    auto_broll=True,
)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

Output: `final/<stem>__clip01__<preset>.mp4`

---

## Full pipeline in one command

Run steps 3–6 (clip_finder → refine → stock_fetch → render) against an already-
transcribed stem:

```bash
# Best clip, auto-selected:
python3 run_pipeline.py tbfpe_qimmw

# Specific clip number:
python3 run_pipeline.py tbfpe_qimmw --clip 2

# More candidates, skip expensive refiner:
python3 run_pipeline.py tbfpe_qimmw --max-clips 5 --skip-refine

# Re-download stock footage (ignore cache):
python3 run_pipeline.py tbfpe_qimmw --overwrite
```

Quick render of a pre-selected clip (skips clip_finder entirely):
```bash
python3 scripts/render_one.py tbfpe_qimmw 1 dark-soft-recitation
```

---

## Smoke-test checklist

Run these checks after a full pipeline pass to confirm everything worked.

### Transcript artifacts
```bash
ls -lh transcripts/tbfpe_qimmw.*
# Expect: .verbose.json  .srt  .captions.srt  .txt  .chunks.json
# Optional: .quran_guard.verbose.json  .quran_guard.srt  .quran_guard.summary.json
```

### Clip candidates
```bash
ls -lh clips/tbfpe_qimmw*
# Expect: tbfpe_qimmw__clips.json  tbfpe_qimmw__clip01.verbose.json

python3 -c "
import json
data = json.load(open('clips/tbfpe_qimmw__clips.json'))
for c in data.get('clips', []):
    print(f\"  clip {c['clip_number']}: {c['start_sec']:.1f}s-{c['end_sec']:.1f}s  score={c['editorial_score']:.3f}  {c['title']}\")
"
```

### Stock assets downloaded
```bash
ls -lh broll/tbfpe_qimmw__clip01__ai*.mp4
# Expect: one symlink per scenic beat, e.g.:
#   broll/tbfpe_qimmw__clip01__ai01__stock.mp4 -> broll/stock/pexels__12345.mp4

ls -lh broll/stock/
# Cache directory — files are re-used across stems and clips
```

### Final render exists
```bash
ls -lh final/tbfpe_qimmw__clip01*
# Expect: tbfpe_qimmw__clip01__dark-soft-recitation.mp4

# Check file is non-empty and playable:
python3 -c "
import subprocess, json
r = subprocess.run(
    ['ffprobe','-v','error','-show_entries','format=duration,size',
     '-of','json','final/tbfpe_qimmw__clip01__dark-soft-recitation.mp4'],
    capture_output=True, text=True)
print(json.loads(r.stdout)['format'])
"
```

### Subtitles burned
```bash
# Extract a frame at 5 s and open in Preview to confirm text is visible:
ffprobe -v error -show_entries format=duration \
  final/tbfpe_qimmw__clip01__dark-soft-recitation.mp4

/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg -y \
  -ss 5 -i final/tbfpe_qimmw__clip01__dark-soft-recitation.mp4 \
  -frames:v 1 /tmp/frame_5s.png && open /tmp/frame_5s.png
```

### Speaker shown first (SPEAKER_HOLD_SECS check)
```bash
# Frame at 1 s should be the speaker, not a scenic insert.
/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg -y \
  -ss 1 -i final/tbfpe_qimmw__clip01__dark-soft-recitation.mp4 \
  -frames:v 1 /tmp/frame_1s.png && open /tmp/frame_1s.png
```

### Scenic cutaway appears
```bash
# Frame after SPEAKER_HOLD_SECS (default 3 s) should be scenic footage.
/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg -y \
  -ss 8 -i final/tbfpe_qimmw__clip01__dark-soft-recitation.mp4 \
  -frames:v 1 /tmp/frame_8s.png && open /tmp/frame_8s.png
```

---

## Preset reference

| Preset | Style | Notes |
|--------|-------|-------|
| `dark-soft-recitation` | Center Arabic, dark background | Default for Quran shorts |
| `subtitle` | Bottom bar | Standard subtitle look |
| `center_recitation` | Center Arabic | Same as dark-soft but lighter shadow |

Change `TEXT_STYLE_MODE` in `.env` to switch between `center_recitation` and
`subtitle` globally. Use `RECITATION_FONTSIZE`, `ASS_PRIMARY_COLOR`, etc. for
fine-grained per-deployment tuning without touching code.

---

## Cost reference (approximate)

| Step | Provider | Cost for 1 clip |
|------|----------|-----------------|
| Transcribe (Whisper) | OpenAI | ~$0.006 / minute |
| Clip finder (LLM) | DeepSeek | ~$0.002–0.010 |
| Transcript refiner (LLM) | DeepSeek | ~$0.001–0.005 |
| Stock footage | Pexels/Pixabay | Free (rate-limited) |
| Render | Local ffmpeg | Free |

Set `CLIP_FINDER_USE_LLM=false` + `TRANSCRIPT_REFINER_USE_LLM=false` in `.env`
for zero-cost heuristic mode (no DeepSeek key needed).

---

## Directory layout

```
clip-factory/
├── incoming/          Drop source videos here (mp4, mov, m4a, mp3, …)
├── transcripts/       Whisper output + quran_guard artifacts
├── clips/             Clip candidates + refined per-clip transcripts
├── broll/
│   ├── stock/         Cached downloaded stock footage (provider__id.mp4)
│   └── <stem>__clip*__ai*__stock.mp4   Per-clip symlinks for renderer
├── final/             Finished rendered shorts
├── mcp/               MCP server modules (all logic lives here)
├── run_pipeline.py    Runs steps 3–6 for a given stem
├── scripts/
│   └── render_one.py  Direct render of a pre-selected clip
├── .env               Your local config (never commit this)
└── .env.example       Template for .env
```
