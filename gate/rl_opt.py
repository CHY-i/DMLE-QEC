"""
Alias for gate-level optimization (GateNoiseToDEM + PlanarNet).

Same as :mod:`gate.pl_opt` — kept for scripts that invoke ``gate/rl_opt.py``.

    python gate/rl_opt.py --distance 3 --rounds 3 --epochs 200
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "pl_opt.py"), run_name="__main__")
