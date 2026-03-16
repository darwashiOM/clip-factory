#!/usr/bin/env python3
"""
clip-factory pipeline: transcribe → quran_guard → clip_finder → stock_fetch → render

Cost-conscious defaults (set in .env):
  TEXT_MODEL_PROVIDER=deepseek          Primary text provider (DeepSeek Chat)
  CLIP_FINDER_USE_LLM=true              Set false for zero-cost heuristic mode
  TRANSCRIPT_REFINER_USE_LLM=true       Set false to skip ASR correction
  SCENE_DIRECTOR_USE_LLM=false          Off by default — uses deterministic planning
  ASSET_GUARD_USE_VISION=false          Off by default — technical checks only

Minimum keys for the default flow:
  DEEPSEEK_API_KEY                      clip_finder + transcript_refiner
  PEXELS_API_KEY / PIXABAY_API_KEY      stock_fetch
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MCP_DIR = ROOT / "mcp"
sys.path.insert(0, str(MCP_DIR))
sys.path.insert(0, str(ROOT))

os.chdir(MCP_DIR)

from bootstrap import resolve_root_and_load_env

ROOT = resolve_root_and_load_env()

import json


def run_clip_selection(stem: str, max_clips: int = 3):
    from clip_finder_server_veo_timeline import find_clips_from_stem
    print(f"[1/4] Running clip selection for: {stem}")
    result = find_clips_from_stem(
        stem=stem,
        max_clips=max_clips,
        min_seconds=18,
        max_seconds=40,
        generation_count=18,
    )
    print(f"  Generated {result['generated_candidate_count']} candidates, kept {result['clip_count']}")
    for i, c in enumerate(result['clips'], 1):
        print(f"  Clip {i}: [{c['start_sec']:.1f}s - {c['end_sec']:.1f}s] ({c['duration_sec']:.1f}s) score={c['editorial_score']:.3f} | {c['title']}")
    return result


def run_refine(stem: str, clip_number: int):
    from transcript_refiner_server import refine_clip_candidate
    print(f"\n[2/4] Refining clip {clip_number} for: {stem}")
    result = refine_clip_candidate(stem=stem, clip_number=clip_number)
    print(f"  Corrected {result['corrected_segment_count']} segments")
    print(f"  Boundary suggestion: {result['boundary_suggestion']}")
    return result


def run_stock_fetch(stem: str, clip_number: int):
    """Fetch and cache real stock scenic inserts for the clip's visual plan."""
    from stock_fetcher_server import fetch_stock_for_candidate
    print(f"\n[3/4] Fetching stock scenic footage for clip {clip_number} of: {stem}")
    result = fetch_stock_for_candidate(stem=stem, clip_number=clip_number)
    fetched = result.get("fetched_count", 0)
    results = result.get("results", [])
    for r in results:
        status = r.get("status", "?")
        slot = r.get("asset_slot", "?")
        if status == "downloaded":
            print(f"  Slot {slot}: downloaded from {r.get('provider','?')} ({r.get('duration','?')}s) — {r.get('saved_file','')}")
        elif status == "existing":
            print(f"  Slot {slot}: reused cached file — {r.get('saved_file','')}")
        elif status == "no_results":
            print(f"  Slot {slot}: no results for query '{r.get('query','?')}' (errors: {r.get('errors',[])})")
        else:
            print(f"  Slot {slot}: {status} — {r.get('error','')}")
    if not result.get("ok"):
        print(f"  Stock fetch skipped: {result.get('message','')}")
    else:
        print(f"  {fetched} stock asset(s) ready")
    return result


def run_render(stem: str, clip_number: int):
    from renderer_server_veo_timeline import render_clip_from_candidate
    print(f"\n[4/4] Rendering clip {clip_number} for: {stem}")
    result = render_clip_from_candidate(
        stem=stem,
        clip_number=clip_number,
        preset="dark-soft-recitation",
        burn_subtitles=True,
        auto_broll=True,
    )
    if result.get("ok"):
        print(f"  Output: {result['output_file']}")
    else:
        print(f"  ERROR: {result.get('error')}")
        print(f"  STDERR: {result.get('stderr_tail', '')[-500:]}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="clip-factory stock-footage pipeline: clip_finder → refine → stock_fetch → render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites (run once per file, before this script):
  python3 -c "
import sys, os; sys.path.insert(0, 'mcp'); os.chdir('mcp')
from transcribe_server import transcribe_file
print(transcribe_file('YOUR_FILE.mp4', language='ar'))
"
  python3 -c "
import sys, os; sys.path.insert(0, 'mcp'); os.chdir('mcp')
from quran_guard_server import fix_quran_in_transcript
print(fix_quran_in_transcript('YOUR_STEM'))
"

Examples:
  python3 run_pipeline.py tbfpe_qimmw
  python3 run_pipeline.py tbfpe_qimmw --max-clips 5
  python3 run_pipeline.py tbfpe_qimmw --clip 2
  python3 run_pipeline.py tbfpe_qimmw --skip-refine
""",
    )
    parser.add_argument("stem", nargs="?", default="tbfpe_qimmw",
                        help="File stem (no extension) from incoming/ — e.g. tbfpe_qimmw")
    parser.add_argument("--max-clips", type=int, default=3,
                        help="Number of clip candidates to generate (default: 3)")
    parser.add_argument("--clip", type=int, default=0,
                        help="Force a specific clip number instead of auto-selecting the best (1-based)")
    parser.add_argument("--skip-refine", action="store_true",
                        help="Skip the transcript-refiner LLM step (saves cost)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download stock assets even if already cached")
    args = parser.parse_args()

    STEM = args.stem

    selection = run_clip_selection(STEM, max_clips=args.max_clips)

    if not selection.get("clips"):
        print("No clips found. Exiting.")
        sys.exit(1)

    if args.clip:
        clips = selection["clips"]
        if args.clip > len(clips):
            print(f"--clip {args.clip} out of range: only {len(clips)} clips found.")
            sys.exit(1)
        best_clip_number = args.clip
        best = clips[args.clip - 1]
    else:
        # Best clip is first (sorted by editorial_score desc)
        best_clip_number = 1
        best = selection["clips"][0]

    print(f"\nSelected clip {best_clip_number}: '{best['title']}' [{best['start_sec']:.1f}s-{best['end_sec']:.1f}s] score={best['editorial_score']:.3f}")
    print(f"  Why: {best['why_it_works']}")

    if not args.skip_refine:
        run_refine(STEM, best_clip_number)

    run_stock_fetch(STEM, best_clip_number)
    render_result = run_render(STEM, best_clip_number)

    print("\n=== DONE ===")
    if render_result.get("ok"):
        print(f"Final clip: {render_result['output_file']}")
    else:
        print("Render failed.")
        sys.exit(1)
