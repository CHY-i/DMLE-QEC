import pathlib
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatching
import tqdm
from numpy.typing import NDArray
from scipy.optimize import least_squares

from . import circuits
from . import plot_helper as ph
from .raw_data_reader import DataLab
from .result_cache import ResultCache

CACHE_PATH = pathlib.Path(__file__).parent.parent.joinpath("data/processed")
result_cache = ResultCache(CACHE_PATH)


def fit_cycle_raise(x, y):
    def fit_func(p, x):
        return 0.5 - 0.5 * (1 - 2 * p[0]) ** x

    def error(p):
        return y - fit_func(p, x)

    bounds = [[0], [0.5]]
    p0 = [max(y) / max(x)]

    result_opt = least_squares(error, p0, bounds=bounds)

    def fit_func_opt(x):
        return fit_func(result_opt.x, x)

    return result_opt, fit_func_opt


def fit_Lambda(x, y):
    def fit_func(p, x):
        return p[0] / p[1] ** ((x + 1) / 2)

    def error(p):
        return y - fit_func(p, x)

    bounds = [[0, 0.1], [np.inf, np.inf]]
    p0 = [y[0], (y[1] / y[0]) ** (2 / (x[1] - x[0]))]

    result_opt = least_squares(error, p0, bounds=bounds)

    def fit_func_opt(x):
        return fit_func(result_opt.x, x)

    return result_opt, fit_func_opt


def detection_event_fraction(dataset: int, do_plot=True, collect=False):
    dl = DataLab()
    dl.load_dataset(dataset)
    if "shape" not in dl.parameters:
        dl.parameters["shape"] = [
            round(dl.data.shape[0] / dl.parameters["stats"]),
            dl.parameters["stats"],
        ]
    cycle = dl.parameters["cycle"]
    circuit_type = dl.parameters["circuit_type"]
    reset = dl.parameters["reset_after_measure"]
    ini_state = dl.parameters["ini_state"]

    ini_state_idx, _, state = dl.get_data(dep_index=np.arange(dl.data.shape[1] - 2))
    qubits = np.arange(len(dl.parameters["qubits"]))
    cycles = np.arange(cycle + 1)
    ini_state_to_def = {}
    for _dep_idx, _idx in tqdm.tqdm(enumerate(ini_state_idx), total=len(ini_state_idx)):
        _idx = round(_idx)
        _ini_state = ini_state[_idx]
        circuit = circuits.build_stim_circuit(
            qubits=qubits,
            ini_state=_ini_state,
            cycle=cycle,
            circuit_type=circuit_type,
            reset=reset,
        )
        measurements = state[_dep_idx]

        converter = circuit.compile_m2d_converter()
        detection_events, _ = converter.convert(
            measurements=measurements.astype(np.bool_), separate_observables=True
        )
        if circuit_type == "bit flip":
            key = "".join(["01"[s] for s in _ini_state])
        elif circuit_type == "phase flip":
            key = "".join(["+-"[s] for s in _ini_state])
        ini_state_to_def["|" + key + ">"] = np.mean(detection_events, axis=0)

    ini_state_to_def_df = pd.DataFrame(ini_state_to_def)
    ini_state_to_def_mean = ini_state_to_def_df.mean(axis=1)
    ini_state_to_def_std = ini_state_to_def_df.std(axis=1)
    ini_state_to_def_mean = np.reshape(ini_state_to_def_mean, [cycle + 1, -1])
    ini_state_to_def_std = np.reshape(ini_state_to_def_std, [cycle + 1, -1])

    if do_plot:
        rc_context = "small"
        atsp = ph.AutoSubplot(
            ax_size=[2.5, 2],
            ax_num=len(ini_state_to_def),
            max_rows=4,
            max_cols=5,
            rc_context=rc_context,
        )
        for k, v in ini_state_to_def.items():
            v = np.reshape(v, [cycle + 1, -1])
            atsp.add_subplot()
            with plt.rc_context(ph.RC_CONTEXT[rc_context]):
                for idx in range(v.shape[1]):
                    plt.plot(cycles, v[:, idx], ".-", alpha=0.1, color="k")
                plt.plot(cycles, np.mean(v, axis=1), ".-", alpha=1, color="k")
                plt.title(k)
        atsp.xlabel("Cycle")
        atsp.ylabel("Detection event fraction")
        atsp.ylim([0, 0.5])
        atsp.suptitle(dl.dataset_name)
        atsp.tight_layout()

        atsp = ph.AutoSubplot(
            ax_size=[2.5, 2],
            ax_num=ini_state_to_def_mean.shape[1],
            max_rows=4,
            max_cols=5,
            rc_context=rc_context,
        )
        for idx in range(ini_state_to_def_mean.shape[1]):
            atsp.add_subplot()
            with plt.rc_context(ph.RC_CONTEXT[rc_context]):
                plt.errorbar(
                    cycles,
                    ini_state_to_def_mean[:, idx],
                    ini_state_to_def_std[:, idx],
                    marker="o",
                    markersize=1,
                    capsize=0,
                    color="k",
                )
                plt.title(dl.parameters["qubits"][2 * idx + 1])
        atsp.xlabel("Cycle")
        atsp.ylabel("Detection event fraction")
        atsp.ylim([0, 0.5])
        atsp.tight_layout()

        plt.figure(figsize=[4, 3])
        for idx in range(ini_state_to_def_mean.shape[1]):
            plt.errorbar(
                cycles,
                ini_state_to_def_mean[:, idx],
                ini_state_to_def_std[:, idx],
                marker="o",
                markersize=1,
                capsize=0,
                color="k",
                alpha=0.1,
            )
        plt.errorbar(
            cycles,
            np.mean(ini_state_to_def_mean, axis=1),
            np.std(ini_state_to_def_mean, axis=1),
            marker="o",
            markersize=3,
            capsize=0,
            color="k",
        )

        plt.xlabel("Cycle")
        plt.ylabel("Detection event fraction")
        plt.ylim(0, 0.5)
        plt.title(f"{dl.dataset_num}")
        plt.tight_layout()

    if collect:
        return cycles, ini_state_to_def_mean


def _measurements_to_logical_error(
    measurements: NDArray,
    cycle: int,
    qubits: Iterable[int],
    ini_state: Iterable[int],
    circuit_type: str,
    reset: bool,
    decoder: str = "pymatching",
    **kwargs,
):
    """
    :params measurements: shape:[shots, record_idx]
    """
    circuit = circuits.build_stim_circuit(
        qubits=qubits,
        ini_state=ini_state,
        cycle=cycle,
        circuit_type=circuit_type,
        reset=reset,
        **kwargs,
    )
    converter = circuit.compile_m2d_converter()
    detection_events, observable_flips = converter.convert(
        measurements=measurements.astype(np.bool_), separate_observables=True
    )
    dem = circuit.detector_error_model()

    if decoder == "pymatching":
        matcher = pymatching.Matching.from_detector_error_model(dem)
        predictions = matcher.decode_batch(detection_events)
    elif decoder == "mwpm_dem":
        from src.decoder import MWPM_dem

        predictions = MWPM_dem(dem).decode(detection_events)
    else:
        raise ValueError(f"Unknown decoder {decoder!r}; use 'pymatching' or 'mwpm_dem'.")

    logical_error_corrected = 1 - np.mean(observable_flips == predictions)
    logical_error = np.mean(observable_flips)
    return logical_error_corrected, logical_error


def _logical_error(
    measurements: NDArray,
    cycle: int,
    qubits: Iterable[int],
    ini_state: Iterable[int],
    circuit_type: str,
    reset: bool,
    subsampling: bool = True,
    decoder: str = "pymatching",
    **kwargs,
):
    """
    Returns:
        subsampling_results:
        Tuple with 3 elements
        1. ini_state_str
            e.g.: '|00100>'
        2. ini_state_to_logical_error_corrected
            e.g.: if subsamplig,
                {'|00100>': 0.001, '|001__>': 0.009,
                '|_010_>': 0.011, '|__100>': 0.012}
                else, {'|00100>': 0.001}
        3. ini_state_to_logical_error
            same data structure with ini_state_to_logical_error_corrected
    """
    ini_state_to_logical_error_corrected = {}
    ini_state_to_logical_error = {}
    if circuit_type == "bit flip":
        state_str = "01"
    elif circuit_type == "phase flip":
        state_str = "+-"
    q_num = len(qubits)
    distance = round((q_num + 1) / 2)
    if subsampling:
        start_distance = 3
    else:
        start_distance = distance
    for d in range(start_distance, distance + 1, 2):  # Iterate different distance
        _q_num = d * 2 - 1
        for start_idx in range(0, distance - d + 1):
            _qubits = [qubits[_idx] for _idx in range(start_idx, start_idx + _q_num)]
            _ini_state = []
            ini_state_str = ["_"] * distance
            for _idx in range(d):
                s = ini_state[_idx + start_idx]
                _ini_state.append(s)
                ini_state_str[_idx + start_idx] = state_str[s]
            ini_state_str = "|" + "".join(ini_state_str) + ">"
            _measurement_idx = start_idx + np.hstack(
                [np.arange(d - 1) + _cycle * (distance - 1) for _cycle in range(cycle)]
                + [np.arange(d) + cycle * (distance - 1)]
            )
            logical_error_corrected, logical_error = _measurements_to_logical_error(
                measurements=measurements[:, _measurement_idx],
                cycle=cycle,
                qubits=_qubits,
                ini_state=_ini_state,
                circuit_type=circuit_type,
                reset=reset,
                decoder=decoder,
                **kwargs,
            )
            ini_state_to_logical_error_corrected[ini_state_str] = (
                logical_error_corrected
            )
            ini_state_to_logical_error[ini_state_str] = logical_error
    return (
        ini_state_str,
        ini_state_to_logical_error_corrected,
        ini_state_to_logical_error,
    )


def convert_ini_state_str(ini_state_str: str):
    """
    |010__>  ->  |***__>
    |_010_>  ->  |_***_>
    """
    for target in "+-01":
        ini_state_str = ini_state_str.replace(target, "*")
    return ini_state_str


@result_cache(key_params=["dataset", "decoder"])
def subsampling_logical_error(dataset: int, decoder: str = "pymatching", **kwargs):
    """
    Returns:
        subsampling_results:
        Tuple with 2 elements
        1. cycle
            int
        2. ini_state_to_logical_error_corrected
            e.g.: {'|\*\*\*\*\*>': [0.001, ...],
                    '|\*\*\*\_\_>': [0.009, ...],
                    '|\_\*\*\*\_>': [0.011, ...],
                    '|\_\_\*\*\*>': [0.012, ...]}}
    """
    dl = DataLab()
    dl.load_dataset(dataset)
    if "shape" not in dl.parameters:
        dl.parameters["shape"] = [
            round(dl.data.shape[0] / dl.parameters["stats"]),
            dl.parameters["stats"],
        ]
    cycle = dl.parameters["cycle"]
    circuit_type = dl.parameters["circuit_type"]
    reset = dl.parameters["reset_after_measure"]
    ini_state = dl.parameters["ini_state"]

    ini_state_idx, _, state = dl.get_data(dep_index=np.arange(dl.data.shape[1] - 2))
    qubits = np.arange(len(dl.parameters["qubits"]))

    ini_state_to_logical_error_corrected = {}
    for _dep_idx, _idx in enumerate(ini_state_idx):
        _idx = round(_idx)
        (
            _,
            _ini_state_to_logical_error_corrected,
            _,
        ) = _logical_error(
            state[_dep_idx],
            cycle,
            qubits,
            ini_state[_idx],
            circuit_type=circuit_type,
            reset=reset,
            subsampling=True,
            decoder=decoder,
            **kwargs,
        )
        for ini_state_str, error in _ini_state_to_logical_error_corrected.items():
            ini_state_to_logical_error_corrected.setdefault(
                convert_ini_state_str(ini_state_str), []
            ).append(error)
    return cycle, ini_state_to_logical_error_corrected


def subsampling_logical_error_per_cycle(datasets, decoder: str = "pymatching", **kwargs):
    cycles = []
    ini_state_to_logical_error = {}  # {'|*****>': [0.001, ...], '|***__>':[0.01, ....], ...}
    for dataset in datasets:
        _cycle, _ini_state_to_logical_error = subsampling_logical_error(
            dataset, decoder=decoder, **kwargs
        )
        cycles.append(_cycle)
        for ini_state_str, errors in _ini_state_to_logical_error.items():
            ini_state_to_logical_error.setdefault(ini_state_str, []).append(
                np.mean(errors)
            )
    distance_to_logical_error = {}
    for ini_state_str, errors in ini_state_to_logical_error.items():
        distance = ini_state_str.count("*")
        distance_to_logical_error.setdefault(distance, []).append(errors)

    distance_to_logical_error_per_cycle = {}
    distance_to_fit_func = {}
    for distance, errors in distance_to_logical_error.items():
        result_opt, fit_func_opt = fit_cycle_raise(cycles, np.mean(errors, axis=0))
        distance_to_logical_error_per_cycle[distance] = result_opt.x[0]
        distance_to_fit_func[distance] = fit_func_opt
    # Lambda fit
    x = []
    y = []
    for _x, _y in distance_to_logical_error_per_cycle.items():
        x.append(_x)
        y.append(_y)
    x = np.array(x)
    y = np.array(y)
    result_opt_Lambda, fit_func_opt_Lambda = fit_Lambda(x[x > 3], y[x > 3])
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = fit_func_opt_Lambda(x_fit)

    cycles_fit = np.linspace(min(cycles), max(cycles), 100)
    fig = plt.figure(figsize=[8, 4])
    fig.add_subplot(1, 3, (1, 2))
    for ini_state_str, errors in ini_state_to_logical_error.items():
        ini_state_str: str
        distance = ini_state_str.count("*")
        plt.plot(
            cycles,
            errors,
            marker="o",
            ls="",
            color=f"C{round((distance - 3) / 2)}",
            markersize=2,
        )
    for distance, errors in distance_to_logical_error.items():
        plt.errorbar(
            cycles,
            np.mean(errors, axis=0),
            np.std(errors, axis=0),
            marker="o",
            markerfacecolor="None",
            ls="",
            color=f"C{round((distance - 3) / 2)}",
        )
        plt.plot(
            cycles_fit,
            distance_to_fit_func[distance](cycles_fit),
            ls="-",
            color=f"C{round((distance - 3) / 2)}",
            label=f"d={distance}, "
            + r"$\epsilon_L$"
            + f"={distance_to_logical_error_per_cycle[distance]:.2e}",
        )
    plt.xlabel("Cycle")
    plt.ylabel("Logical error")
    plt.legend()
    plt.ylim(0, 0.5)

    fig.add_subplot(1, 3, 3)
    plt.plot(x, y, marker="o", ls="", c="C0")
    plt.plot(
        x_fit,
        y_fit,
        ls="--",
        c="C0",
        label=r"$\Lambda$=" + f"{result_opt_Lambda.x[1]:.1f}",
    )
    plt.xlabel("Code distance")
    plt.ylabel(r"$\epsilon_L$")
    plt.yscale("log")
    plt.legend()

    fig.suptitle(f"{min(datasets)}-{max(datasets)}")
    fig.tight_layout()

    fig.savefig('phase_flip_reset0.png', dpi=300, bbox_inches="tight")
    
    raw_ler_by_distance = {}
    for distance, errors in distance_to_logical_error.items():
        # 对不同初始态的错误率求平均，得到这个码距在各个 cycle 下的 LER 数组
        raw_ler_by_distance[distance] = np.mean(errors, axis=0)
        
    # 修改 return，把 cycles 和 原始数据 也一起返回
    return cycles, raw_ler_by_distance, distance_to_logical_error_per_cycle, result_opt_Lambda.x[1]