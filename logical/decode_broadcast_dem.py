"""
Decode experimental syndromes with Planar + ``MWPM_dem`` on two DEMs:

1. **Baseline** — init DEM / rates (same as :func:`logical.rep_pl_opt.initialize_from_sim_dem`).
2. **Broadcast** — DEM from ``data/logical_qubit/dem/{circuit_type}_reset{reset}/``
   ``dmle_r{round:02d}_{circuit_type}_reset{round}.txt`` (``logical/broadcast.py`` output).

Uses the same minibatched decoding as :func:`logical.rep_pl_opt.decoding_benchmark`.

This file loads ``circuits`` / ``raw_data_reader`` via ``importlib`` (lazy, after ``parse_args``)
so ``--help`` works without ``h5py``. Loading experimental HDF5 still requires ``h5py`` like
``raw_data_reader``.

Run::

    python logical/decode_broadcast_dem.py --round 7 --circuit-type phase_flip --reset 0
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import stim
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if not hasattr(np, "bool8"):
    np.bool8 = bool

from src.decoder import Planar, MWPM_dem  # noqa: E402
from src.utils import get_error_rates, rep_dem  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_circuits_mod = None
_raw_reader_mod = None


def _circuits():
    """Lazy-load ``circuits.py`` (no HDF5)."""
    global _circuits_mod
    if _circuits_mod is None:
        _circuits_mod = _load_module(
            "_rep_decode_circuits",
            REPO_ROOT / "logical" / "repetition_code_exp_data" / "circuits.py",
        )
    return _circuits_mod


def _raw_reader():
    """Lazy-load ``raw_data_reader.py`` (requires ``h5py`` for HDF5 data)."""
    global _raw_reader_mod
    if _raw_reader_mod is None:
        _raw_reader_mod = _load_module(
            "_rep_decode_raw_reader",
            REPO_ROOT / "logical" / "repetition_code_exp_data" / "raw_data_reader.py",
        )
    return _raw_reader_mod


# --- copied from logical/rep_pl_opt.py (keep in sync intentionally) ---

LER_REPORT_DECIMALS = 6


def _ler_round_for_report(x: float, *, ndigits: int = LER_REPORT_DECIMALS) -> float:
    if x != x:
        return x
    return round(float(x), ndigits)


def measurements_to_dets_obs(
    measurements: np.ndarray,
    *,
    qubits: np.ndarray,
    ini_state: np.ndarray | list,
    cycle: int,
    circuit_type: str,
    reset: bool,
) -> tuple[np.ndarray, np.ndarray]:
    circuit = _circuits().build_stim_circuit(
        qubits=qubits,
        ini_state=ini_state,
        cycle=cycle,
        circuit_type=circuit_type,
        reset=reset,
        add_noise=False,
    )
    converter = circuit.compile_m2d_converter()
    dets, obs = converter.convert(
        measurements=np.asarray(measurements, dtype=np.bool_),
        separate_observables=True,
    )
    return np.asarray(dets, dtype=np.uint8), np.asarray(obs, dtype=np.uint8).reshape(-1)


def _dataset_number_from_hdf5_path(filepath: Path) -> int:
    stem = filepath.stem
    num_part = ""
    for char in stem:
        if char.isdigit():
            num_part += char
        else:
            break
    if not num_part:
        raise ValueError(f"No dataset number in filename: {filepath}")
    return int(num_part)


def repetition_code_hdf5_glob_pattern(cycle: int, circuit_type: str, reset: int) -> str:
    ct = circuit_type.replace("_", " ")
    reset_str = "True" if bool(reset) else "False"
    return f"* - * Repetition code {ct} reset={reset_str} cycle={int(cycle)}.hdf5"


def resolve_repetition_hdf5_path(
    raw_dir: Path,
    cycle: int,
    circuit_type: str,
    reset: int,
) -> Path:
    pat = repetition_code_hdf5_glob_pattern(cycle, circuit_type, reset)
    matches = sorted(raw_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(
            f"No HDF5 matching {pat!r} under {raw_dir}. "
            "Check --round, --circuit-type, --reset, and --raw-dir, or pass --hdf5."
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            "Multiple HDF5 files match the pattern; narrow parameters or use --hdf5:\n  "
            + "\n  ".join(m.name for m in matches)
        )
    return matches[0]


def _dets_obs_one_hdf5(filepath: Path, raw_dir: Path) -> tuple[int, str, bool, np.ndarray, np.ndarray]:
    dl = _raw_reader().DataLab(data_dir=raw_dir)
    dl.load_dataset(_dataset_number_from_hdf5_path(filepath), noisy=False)
    if "shape" not in dl.parameters:
        dl.parameters["shape"] = [
            round(dl.data.shape[0] / dl.parameters["stats"]),
            dl.parameters["stats"],
        ]
    cycle = int(dl.parameters["cycle"])
    circuit_type = str(dl.parameters["circuit_type"])
    reset = bool(dl.parameters["reset_after_measure"])
    ini_state_idx, _, state = dl.get_data(dep_index=np.arange(dl.data.shape[1] - 2))
    qubits = np.arange(len(dl.parameters["qubits"]))
    ini_states_param = dl.parameters["ini_state"]

    dets_parts: list[np.ndarray] = []
    obs_parts: list[np.ndarray] = []
    for i in range(len(ini_state_idx)):
        idx = int(round(ini_state_idx[i]))
        ini_state = ini_states_param[idx]
        measurements = state[i]
        dets, obs = measurements_to_dets_obs(
            measurements,
            qubits=qubits,
            ini_state=ini_state,
            cycle=cycle,
            circuit_type=circuit_type,
            reset=reset,
        )
        dets_parts.append(dets)
        obs_parts.append(obs)

    return cycle, circuit_type, reset, np.vstack(dets_parts), np.concatenate(obs_parts)


def load_dets_obs_from_repetition_hdf5(
    hdf5_path: Path,
    *,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, dets, obs = _dets_obs_one_hdf5(hdf5_path, raw_dir)
    return dets, obs


def initialize_from_sim_dem(
    *,
    distance: int,
    round: int,
    circuit_type: str,
    reset: int,
    device: str,
    dtype: torch.dtype,
    max_face_length: int = 4,
) -> tuple[torch.Tensor, object, stim.DetectorErrorModel]:
    q_num = 2 * distance - 1
    qubits = np.arange(q_num)
    ini_state = np.zeros(distance, dtype=int)
    circuit_type_pd = circuit_type.replace("_", " ")
    circuit = _circuits().build_stim_circuit(
        qubits=qubits,
        ini_state=ini_state,
        cycle=round,
        circuit_type=circuit_type_pd,
        reset=bool(reset),
        add_noise=True,
    )
    dem = circuit.detector_error_model()
    er = torch.tensor(get_error_rates(dem), dtype=dtype, device=device)
    init_er = er
    print("circuit generated", flush=True)
    print(
        "[init] Building rep_dem (dual graph from DEM); this can take minutes on large experiments…",
        flush=True,
    )
    rep = rep_dem(dem, max_face_length=max_face_length)
    print("abstract code generated", flush=True)
    return init_er, rep, dem


def decoding_benchmark(
    *,
    dets: np.ndarray,
    obs: np.ndarray,
    rep: object,
    dem: stim.DetectorErrorModel,
    error_rates: torch.Tensor,
    mini_batch: int,
    max_shots: int | None = None,
    device: str = "cpu",
) -> dict:
    n = dets.shape[0]
    if max_shots is not None and n > max_shots:
        dets = dets[:max_shots]
        obs = obs[:max_shots]

    n = int(dets.shape[0])
    mb = max(1, int(mini_batch))

    pl = Planar(rep, dev=device)
    mw = MWPM_dem(dem)
    er_np = error_rates.detach().cpu().numpy()

    total_eq_pl = 0.0
    total_eq_mw = 0.0
    for s in range(0, n, mb):
        e = min(s + mb, n)
        nb = e - s
        ler_b = pl.logical_error_rate(dets[s:e], obs[s:e], error_rates=error_rates)
        total_eq_pl += (1.0 - float(ler_b.detach().cpu().item())) * nb
        ler_m = mw.logical_error_rate(dets[s:e], obs[s:e], error_rates=er_np)
        total_eq_mw += (1.0 - float(ler_m)) * nb
    ler_pl = 1.0 - total_eq_pl / n if n else float("nan")
    ler_mw = 1.0 - total_eq_mw / n if n else float("nan")

    return {
        "shots": int(dets.shape[0]),
        "planar_ler": _ler_round_for_report(ler_pl),
        "mwpm_dem_ler": _ler_round_for_report(ler_mw),
    }


def load_dem_from_txt(path: Path) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Planar + MWPM_dem: init DEM vs broadcast DEM txt.")
    ap.add_argument("--round", type=int, required=True, help="HDF5 cycle / DEM round (must match data and DEM file)")
    ap.add_argument("--circuit-type", type=str, required=True, choices=["phase_flip", "bit_flip"])
    ap.add_argument("--reset", type=int, required=True, choices=[0, 1])
    ap.add_argument("--distance", type=int, default=9)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--mini-batch", type=int, default=1000)
    ap.add_argument("--max-face-length", type=int, default=4)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--raw-dir", type=str, default=None)
    ap.add_argument("--hdf5", type=str, default=None)
    ap.add_argument("--dem-txt", type=str, default=None)
    ap.add_argument("--dem-root", type=str, default=None)
    args = ap.parse_args()

    if args.max_face_length < 3 or args.max_face_length > 6:
        ap.error("--max-face-length must be between 3 and 6 (inclusive)")

    dtype = torch.float64
    raw_dir = Path(args.raw_dir) if args.raw_dir else _raw_reader().default_raw_data_dir()
    if args.hdf5:
        hdf5_path = Path(args.hdf5).resolve()
        raw_dir = hdf5_path.parent
    else:
        hdf5_path = resolve_repetition_hdf5_path(
            raw_dir,
            args.round,
            args.circuit_type,
            args.reset,
        )

    if args.dem_txt:
        dem_bc_path = Path(args.dem_txt).resolve()
    else:
        root = Path(args.dem_root).resolve() if args.dem_root else REPO_ROOT / "data" / "logical_qubit" / "dem"
        sub = f"{args.circuit_type}_reset{int(args.reset)}"
        stem = f"dmle_r{int(args.round):02d}_{args.circuit_type}_reset{int(args.reset)}"
        dem_bc_path = root / sub / f"{stem}.txt"

    if not dem_bc_path.is_file():
        raise FileNotFoundError(
            f"Broadcast DEM not found: {dem_bc_path}\n"
            "Run logical/broadcast.py or pass --dem-txt."
        )

    print(f"[data] HDF5: {hdf5_path}", flush=True)
    dets, obs = load_dets_obs_from_repetition_hdf5(hdf5_path, raw_dir=raw_dir)
    obs = np.asarray(obs, dtype=np.uint8).reshape(-1)
    print(f"[data] dets {dets.shape}, obs {obs.shape}", flush=True)

    print(f"[dem] broadcast txt: {dem_bc_path}", flush=True)
    dem_bc = load_dem_from_txt(dem_bc_path)
    if dem_bc.num_detectors != dets.shape[1]:
        raise ValueError(
            f"Broadcast DEM num_detectors={dem_bc.num_detectors} != dets width {dets.shape[1]}."
        )

    er_bc = torch.tensor(get_error_rates(dem_bc), dtype=dtype, device=args.device)
    print(f"[rep] rep_dem (broadcast DEM, L={args.max_face_length})…", flush=True)
    rep_bc = rep_dem(dem_bc, max_face_length=args.max_face_length)
    print("[rep] done.", flush=True)

    print("[dem] init (default-noise circuit, same as rep_pl_opt.initialize_from_sim_dem)…", flush=True)
    init_er, rep_init, dem_init = initialize_from_sim_dem(
        distance=args.distance,
        round=args.round,
        circuit_type=args.circuit_type,
        reset=args.reset,
        device=args.device,
        dtype=dtype,
        max_face_length=args.max_face_length,
    )
    if dem_init.num_detectors != dets.shape[1]:
        raise ValueError(
            f"Init DEM num_detectors={dem_init.num_detectors} != dets width {dets.shape[1]}."
        )

    common_kw = dict(
        dets=dets,
        obs=obs,
        mini_batch=args.mini_batch,
        max_shots=args.max_shots,
        device=args.device,
    )

    print("[decode] baseline (init DEM + init rates) …", flush=True)
    bm_init = decoding_benchmark(
        **common_kw,
        rep=rep_init,
        dem=dem_init,
        error_rates=init_er,
    )

    print("[decode] broadcast DEM + rates from txt …", flush=True)
    bm_bc = decoding_benchmark(
        **common_kw,
        rep=rep_bc,
        dem=dem_bc,
        error_rates=er_bc,
    )

    print("", flush=True)
    print(f"shots={bm_init['shots']}", flush=True)
    print(
        f"  baseline (init):      planar_ler={bm_init['planar_ler']}  mwpm_dem_ler={bm_init['mwpm_dem_ler']}",
        flush=True,
    )
    print(
        f"  broadcast ({dem_bc_path.name}): planar_ler={bm_bc['planar_ler']}  mwpm_dem_ler={bm_bc['mwpm_dem_ler']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
