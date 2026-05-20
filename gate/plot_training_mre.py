"""
Plot mean relative error (MRE) vs epoch from gate/pl_opt training logs.

Usage::

    python gate/plot_training_mre.py
    python gate/plot_training_mre.py --out gate/log/rep/sim/rep_d3r3_mre.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "gate" / "log" / "rep" / "sim"

EPOCH_RE = re.compile(r"^\[(?:init|checkpoint)\] epoch=(\d+)")
GATE_MRE_RE = re.compile(r"^\s+gate vs true:.*mean_rel=([\d.e+-]+)")
DEM_MRE_RE = re.compile(r"^\s+dem vs true:.*mean_rel=([\d.e+-]+)")


def parse_training_log(path: Path) -> dict[str, list[float]]:
    epochs: list[int] = []
    gate_mre: list[float] = []
    dem_mre: list[float] = []
    cur_epoch: int | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        m_ep = EPOCH_RE.match(line)
        if m_ep:
            cur_epoch = int(m_ep.group(1))
            continue
        if cur_epoch is None:
            continue
        m_g = GATE_MRE_RE.match(line)
        if m_g:
            epochs.append(cur_epoch)
            gate_mre.append(float(m_g.group(1)))
            continue
        m_d = DEM_MRE_RE.match(line)
        if m_d:
            dem_mre.append(float(m_d.group(1)))

    if len(epochs) != len(gate_mre) or len(epochs) != len(dem_mre):
        raise ValueError(f"{path}: mismatched epoch/gate/dem counts")
    return {"epoch": [float(e) for e in epochs], "gate_mre": gate_mre, "dem_mre": dem_mre}


def plot_d3r3_comparison(
    *,
    elementary_log: Path,
    time_shared_log: Path,
    out_path: Path,
) -> None:
    elem = parse_training_log(elementary_log)
    tsh = parse_training_log(time_shared_log)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        elem["epoch"],
        elem["gate_mre"],
        linestyle="-",
        linewidth=2,
        color="#1f77b4",
        label="gate MRE (elementary)",
    )
    ax.plot(
        elem["epoch"],
        elem["dem_mre"],
        linestyle="--",
        linewidth=2,
        color="#1f77b4",
        label="DEM MRE (elementary)",
    )
    ax.plot(
        tsh["epoch"],
        tsh["gate_mre"],
        linestyle="-",
        linewidth=2,
        color="#d62728",
        label="gate MRE (time_shared)",
    )
    ax.plot(
        tsh["epoch"],
        tsh["dem_mre"],
        linestyle="--",
        linewidth=2,
        color="#d62728",
        label="DEM MRE (time_shared)",
    )
    ax.set_yscale('log')
    ax.set_xlabel("epoch")
    ax.set_ylabel("MRE (mean_rel vs true)")
    ax.set_title("rep d3r3 — gate / DEM error vs training epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MRE curves from gate training logs.")
    ap.add_argument(
        "--elementary-log",
        type=Path,
        default=LOG_DIR / "rep_d3r3_p0.001_elementary.txt",
    )
    ap.add_argument(
        "--time-shared-log",
        type=Path,
        default=LOG_DIR / "rep_d3r3_p0.001_time_shared.txt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=LOG_DIR / "rep_d3r3_p0.001_mre_comparison.png",
    )
    args = ap.parse_args()

    plot_d3r3_comparison(
        elementary_log=args.elementary_log,
        time_shared_log=args.time_shared_log,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
