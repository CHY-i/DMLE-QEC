from typing import Dict, Iterable, Union

import stim


def build_stim_circuit(
    qubits: Iterable[int],
    ini_state: Iterable[int],
    cycle: int = 1,
    circuit_type: str = "phase flip",
    reset=True,
    add_noise=True,
    sq_error: Union[float, Dict[int, float]] = 0.0013,
    cz_error: Union[float, Dict[int, float]] = 0.0072,
    measure_error: Union[float, Dict[int, float]] = 0.02,
    idle_z_error: Union[float, Dict[int, float]] = 0.0,
    idle_dep_error: Union[float, Dict[int, float]] = 0.036,
) -> stim.Circuit:
    assert circuit_type == "phase flip" or circuit_type == "bit flip"
    assert len(qubits) % 2 == 1
    data_qubits = list(qubits[::2])
    measure_qubits = list(qubits[1::2])
    data_qubit_num = len(data_qubits)
    measure_qubit_num = len(measure_qubits)
    assert len(ini_state) == len(data_qubits)

    circuit = stim.Circuit()
    # prepare initial state
    targets_0 = []
    targets_1 = []
    for data_qubit, s in zip(data_qubits, ini_state):
        if s == 0:
            targets_0.append(data_qubit)
        elif s == 1:
            targets_1.append(data_qubit)
        else:
            raise ValueError(s)
    if circuit_type == "phase flip":
        circuit.append("SQRT_Y_DAG", targets_1)
        if add_noise:
            circuit.append("DEPOLARIZE1", targets_1, sq_error)

        circuit.append("SQRT_Y", targets_0)
        if add_noise:
            circuit.append("DEPOLARIZE1", targets_0, sq_error)
    else:
        circuit.append("X", targets_1)
        if add_noise:
            circuit.append("DEPOLARIZE1", targets_1, sq_error)

    # cycle
    for cycle_idx in range(cycle):
        circuit.append("TICK")
        if circuit_type == "phase flip":
            circuit.append("H", data_qubits + measure_qubits)
            if add_noise:
                circuit.append("DEPOLARIZE1", data_qubits + measure_qubits, sq_error)
        else:
            circuit.append("H", measure_qubits)
            if add_noise:
                circuit.append("DEPOLARIZE1", measure_qubits, sq_error)

        circuit.append("TICK")
        targets = []
        for measure_qubit, data_qubit in zip(measure_qubits, data_qubits[:-1]):
            targets.extend([measure_qubit, data_qubit])
        circuit.append("CZ", targets)
        if add_noise:
            circuit.append("DEPOLARIZE2", targets, cz_error)

        circuit.append("TICK")
        targets = []
        for measure_qubit, data_qubit in zip(measure_qubits, data_qubits[1:]):
            targets.extend([measure_qubit, data_qubit])
        circuit.append("CZ", targets)
        if add_noise:
            circuit.append("DEPOLARIZE2", targets, cz_error)

        if cycle_idx == cycle - 1:  # last cycle
            circuit.append("TICK")
            if circuit_type == "phase flip":
                circuit.append("H", measure_qubits)
                if add_noise:
                    circuit.append("DEPOLARIZE1", measure_qubits, sq_error)
            else:
                circuit.append("H", measure_qubits)
                circuit.append("Y", data_qubits)
                if add_noise:
                    circuit.append("DEPOLARIZE1", measure_qubits, sq_error)
                    circuit.append("DEPOLARIZE1", data_qubits, sq_error)
            circuit.append("TICK")
            if add_noise:
                circuit.append("X_ERROR", measure_qubits + data_qubits, measure_error)
            circuit.append("M", measure_qubits)
        else:
            circuit.append("TICK")
            if circuit_type == "phase flip":
                circuit.append("H", data_qubits + measure_qubits)
                if add_noise:
                    circuit.append(
                        "DEPOLARIZE1", data_qubits + measure_qubits, sq_error
                    )
            else:
                circuit.append("H", measure_qubits)
                circuit.append("Y", data_qubits)
                if add_noise:
                    circuit.append("DEPOLARIZE1", measure_qubits, sq_error)
                    circuit.append("DEPOLARIZE1", data_qubits, sq_error)
            circuit.append("TICK")
            if add_noise:
                circuit.append("X_ERROR", measure_qubits, measure_error)
            if reset:
                circuit.append("MR", measure_qubits)
            else:
                circuit.append("M", measure_qubits)
            if add_noise:
                circuit.append("Z_ERROR", data_qubits, idle_z_error)
                circuit.append("DEPOLARIZE1", data_qubits, idle_dep_error)

        if reset:
            if cycle_idx == 0:  # first cycle
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR", [stim.target_rec(-measure_qubit_num + idx)]
                    )  # error event detectors
            else:  # subsequent rounds
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-measure_qubit_num + idx),
                            stim.target_rec(-2 * measure_qubit_num + idx),
                        ],
                    )  # error event detectors
        else:
            if cycle_idx == 0 or cycle_idx == 1:  # first cycle & second cycle
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR", [stim.target_rec(-measure_qubit_num + idx)]
                    )  # error event detectors
            else:  # subsequent rounds
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-measure_qubit_num + idx),
                            stim.target_rec(-3 * measure_qubit_num + idx),
                        ],
                    )  # error event detectors

    circuit.append("M", data_qubits)
    for idx in range(len(measure_qubits)):
        if reset:
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-data_qubit_num + idx),
                    stim.target_rec(-data_qubit_num + idx + 1),
                    stim.target_rec(-data_qubit_num - measure_qubit_num + idx),
                ],
            )  # error event detectors
        else:
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-data_qubit_num + idx),
                    stim.target_rec(-data_qubit_num + idx + 1),
                    stim.target_rec(-data_qubit_num - measure_qubit_num + idx),
                    stim.target_rec(-data_qubit_num - 2 * measure_qubit_num + idx),
                ],
            )  # error event detectors

    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

    return circuit
