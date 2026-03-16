"""
Legacy compatibility wrapper for the current timeline clip finder.

Do not add logic here.
Use clip_finder_server_veo_timeline.py as the real implementation.
"""

from clip_finder_server_veo_timeline import (
    mcp,
    list_clip_sources,
    find_clips_from_stem,
    list_saved_clip_plans,
)

if __name__ == "__main__":
    mcp.run()