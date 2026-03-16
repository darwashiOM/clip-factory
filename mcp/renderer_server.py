"""
Legacy compatibility wrapper for the current timeline renderer.

Do not add logic here.
Use renderer_server_veo_timeline.py as the real implementation.
"""

from renderer_server_veo_timeline import (
    mcp,
    list_filter_presets,
    list_broll_files,
    list_final_renders,
    suggest_broll_for_clip,
    render_clip_from_candidate,
    batch_render_from_candidates,
)

if __name__ == "__main__":
    mcp.run()