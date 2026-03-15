#!/usr/bin/env python3
"""
Single-run pipeline: clip selection -> refine -> render for tbfpe_qimmw.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MCP_DIR = ROOT / "mcp"
sys.path.insert(0, str(MCP_DIR))
sys.path.insert(0, str(ROOT))

os.chdir(MCP_DIR)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import json


def run_clip_selection(stem: str, max_clips: int = 3):
    from clip_finder_server import find_clips_from_stem
    print(f"[1/3] Running clip selection for: {stem}")
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
    print(f"\n[2/3] Refining clip {clip_number} for: {stem}")
    result = refine_clip_candidate(stem=stem, clip_number=clip_number)
    print(f"  Corrected {result['corrected_segment_count']} segments")
    print(f"  Boundary suggestion: {result['boundary_suggestion']}")
    return result


def run_render(stem: str, clip_number: int):
    from renderer_server import render_clip_from_candidate
    print(f"\n[3/3] Rendering clip {clip_number} for: {stem}")
    result = render_clip_from_candidate(
        stem=stem,
        clip_number=clip_number,
        preset="golden-islamic",
        burn_subtitles=True,
        auto_broll=False,
    )
    if result.get("ok"):
        print(f"  Output: {result['output_file']}")
    else:
        print(f"  ERROR: {result.get('error')}")
        print(f"  STDERR: {result.get('stderr_tail', '')[-500:]}")
    return result


if __name__ == "__main__":
    STEM = "tbfpe_qimmw"

    selection = run_clip_selection(STEM, max_clips=3)

    if not selection.get("clips"):
        print("No clips found. Exiting.")
        sys.exit(1)

    # Best clip is first (sorted by editorial_score desc)
    best_clip_number = 1
    best = selection["clips"][0]
    print(f"\nSelected clip 1: '{best['title']}' [{best['start_sec']:.1f}s-{best['end_sec']:.1f}s] score={best['editorial_score']:.3f}")
    print(f"  Why: {best['why_it_works']}")

    refine_result = run_refine(STEM, best_clip_number)
    render_result = run_render(STEM, best_clip_number)

    print("\n=== DONE ===")
    if render_result.get("ok"):
        print(f"Final clip: {render_result['output_file']}")
    else:
        print("Render failed.")
        sys.exit(1)
