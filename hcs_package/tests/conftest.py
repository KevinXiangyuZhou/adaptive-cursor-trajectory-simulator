"""Ensure tests import THIS repo's hcs_package, not any installed copy
(the environment may carry an editable install from another checkout)."""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

for mod in [m for m in list(sys.modules) if m.split(".")[0] == "hcs_package"]:
    del sys.modules[mod]
