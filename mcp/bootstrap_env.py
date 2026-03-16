# Compatibility shim — bootstrap_env.py and bootstrap.py are the same module.
# Import from bootstrap.py directly; this file exists only for legacy callers.
from bootstrap import resolve_root_and_load_env

__all__ = ["resolve_root_and_load_env"]
