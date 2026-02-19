"""Wrapper to run OligoAI2 ablation study from the justfile.

Uses importlib.util to import from 05_oligoai2 (digit prefix prevents
normal python -m invocation). Same pattern as evaluate_model.py.
"""

import importlib.util
import sys
import types
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_oai_dir = _root / "analyses" / "05_oligoai2"

# Register the package so imports inside ablation.py resolve
_pkg = types.ModuleType("_oai_abl")
_pkg.__path__ = [str(_oai_dir)]
_pkg.__package__ = "_oai_abl"
sys.modules["_oai_abl"] = _pkg

spec = importlib.util.spec_from_file_location(
    "_oai_abl.ablation", _oai_dir / "ablation.py", submodule_search_locations=[],
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "_oai_abl"
sys.modules["_oai_abl.ablation"] = mod
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.main()
