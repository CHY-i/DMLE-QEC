import stim
import numpy as np
import torch
import re
import correlation
from pymatching import Matching
from typing import List


def rep_code_noreset(distance, rounds, depolarize_rate = 0.01, initial_state = 'random', sg_error=[], m_error=[], tg_error = [], simulation = True):
    # distance=5
    # rounds=5
    # depolarize_rate = 0.01
    # simulation = True
    # distance += 1
    qubits = []
    for i in range(2*(distance-1)+1):
        qubits.append(i)
    data=qubits[::2]
    ancilla=qubits[1::2]
    
    if simulation == True:
        initialize_error = [depolarize_rate for _ in range(len(qubits))]
        sg_error=[depolarize_rate for _ in range(len(qubits))]
        m_error=[depolarize_rate for _ in range(len(qubits))]
        def cz_error(pos1, pos2,error_list = [depolarize_rate for _ in range(len(qubits)-1)]):
            assert abs(pos1 - pos2) == 1
            cz_dict = {}
            for i in range(len(error_list)):
                cz_dict[frozenset([i,i+1])] = error_list[i]
            return cz_dict[frozenset([pos1,pos2])]
    else:
        initialize_error = [0.001 for _ in range(len(qubits))]
        def cz_error(pos1, pos2,error_list = tg_error):
            assert abs(pos1 - pos2) == 1
            cz_dict = {}
            for i in range(len(error_list)):
                cz_dict[frozenset([i,i+1])] = error_list[i]
            return cz_dict[frozenset([pos1,pos2])]


    # begin of repeat rounds
    circuit_repeat = stim.Circuit()
    for q in data:
        circuit_repeat.append('depolarize1', q, sg_error[q])
    circuit_repeat.append('H',ancilla)
    for q in qubits:
        circuit_repeat.append('depolarize1',q, sg_error[q])
    circuit_repeat.append("tick")

    qubits_left = qubits
    for m in ancilla:
        circuit_repeat.append('cz',[m,m-1])
        qubits_left = [item for item in qubits_left if item not in [m,m-1]]
    for m in ancilla:
        circuit_repeat.append('depolarize2',[m,m-1],cz_error(m,m-1))
    circuit_repeat.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    circuit_repeat.append('tick')

    qubits_left = qubits
    for m in ancilla:
        circuit_repeat.append('cz',[m,m+1])
        qubits_left = [item for item in qubits_left if item not in [m,m+1]]
    for m in ancilla:
        circuit_repeat.append('depolarize2',[m,m+1],cz_error(m,m+1))
    circuit_repeat.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    circuit_repeat.append('tick')

    circuit_repeat.append('H',ancilla)
    for q in qubits:
        circuit_repeat.append('depolarize1',q, sg_error[q])
    circuit_repeat.append("tick")

    for m in ancilla:
        circuit_repeat.append('x_error',m,m_error[m])
    circuit_repeat.append('M',ancilla)
    circuit_repeat.append('tick')

    detector_repeat_circuit_str = ""
    for k , m in enumerate(ancilla):
        detector_repeat_circuit_str += f"DETECTOR({m}, 0) rec[{-1-k}] rec[{-1-k-(distance-1)*2}]\n"
    detector_repeat_circuit_str += "SHIFT_COORDS(0,1)"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)
    circuit_repeat += detector_repeat_circuit

    # begin of full circuit
    full_circuit = stim.Circuit()
    full_circuit.append('R',qubits)
    full_circuit.append('tick')
    full_circuit.append('M',ancilla)
    for q in qubits:
        full_circuit.append("x_error", q, initialize_error[q])
    full_circuit.append('tick')
    
    if initial_state == '1':
        full_circuit.append('X', data)
    elif initial_state == 'random':
        for i, d in enumerate(data):
            full_circuit.append("CX", [stim.target_sweep_bit(i), d])

    full_circuit.append('H',ancilla)
    for q in qubits:
        full_circuit.append('depolarize1',q, sg_error[q])
    full_circuit.append("tick")

    qubits_left = qubits
    for m in ancilla:
        full_circuit.append('cz',[m,m-1])
        qubits_left = [item for item in qubits_left if item not in [m,m-1]]
    for m in ancilla:
        full_circuit.append('depolarize2',[m,m-1],cz_error(m,m-1))
    full_circuit.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    full_circuit.append('tick')

    qubits_left = qubits
    for m in ancilla:
        full_circuit.append('cz',[m,m+1])
        qubits_left = [item for item in qubits_left if item not in [m,m+1]]
    for m in ancilla:
        full_circuit.append('depolarize2',[m,m+1],cz_error(m,m+1))
    full_circuit.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    full_circuit.append('tick')

    full_circuit.append('H',ancilla)
    for q in qubits:
        full_circuit.append('depolarize1',q, sg_error[q])
    full_circuit.append("tick")

    for m in ancilla:
        full_circuit.append('x_error',m,m_error[m])
    full_circuit.append('M',ancilla)
    full_circuit.append('tick')

    detector_repeat_circuit_str = ""
    for k , m in enumerate(ancilla):
        detector_repeat_circuit_str += f"DETECTOR({m}, 0) rec[{-1-k}]\n"
    detector_repeat_circuit_str += "SHIFT_COORDS(0,1)"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)
    full_circuit += detector_repeat_circuit

    full_circuit += circuit_repeat*(rounds-1)

    for q in data:
        full_circuit.append('x_error',q,m_error[q])
    full_circuit.append('M',data)
    detector_circuit_str = ""
    for k , m in enumerate(ancilla):
        detector_circuit_str += f"DETECTOR({m}, 0) rec[{-1-k}] rec[{-2-k}] rec[{-2-k-(distance-1)*2}] rec[{-2-k-(distance-1)}]\n"
    detector_circuit = stim.Circuit(detector_circuit_str)
    full_circuit += detector_circuit
    full_circuit.append_operation("observable_include",[stim.target_rec(-1)],0)

    return full_circuit

def rep_code(distance, rounds, depolarize_rate = 0.01, initial_state = 0, sg_error=[], m_error=[], tg_error = [], simulation = True):
    # distance=5
    # rounds=5
    # depolarize_rate = 0.01
    # simulation = True
    # distance += 1
    qubits = []
    for i in range(2*(distance-1)+1):
        qubits.append(i)
    data=qubits[::2]
    ancilla=qubits[1::2]
    
    if simulation == True:
        sg_error=[depolarize_rate for _ in range(len(qubits))]
        m_error=[depolarize_rate for _ in range(len(qubits))]
        def cz_error(pos1, pos2,error_list = [depolarize_rate for _ in range(len(qubits)-1)]):
            assert abs(pos1 - pos2) == 1
            cz_dict = {}
            for i in range(len(error_list)):
                cz_dict[frozenset([i,i+1])] = error_list[i]
            return cz_dict[frozenset([pos1,pos2])]
    else:
        def cz_error(pos1, pos2,error_list = tg_error):
            assert abs(pos1 - pos2) == 1
            cz_dict = {}
            for i in range(len(error_list)):
                cz_dict[frozenset([i,i+1])] = error_list[i]
            return cz_dict[frozenset([pos1,pos2])]


    # begin of repeat rounds
    circuit_repeat = stim.Circuit()
    circuit_repeat.append('R',ancilla)
    for m in ancilla:
        circuit_repeat.append('x_error',m,m_error[m])
    for q in data:
        circuit_repeat.append('depolarize1', q, sg_error[q])
    circuit_repeat.append('tick')

    circuit_repeat.append('H',ancilla)
    for q in qubits:
        circuit_repeat.append('depolarize1',q, sg_error[q])
    circuit_repeat.append("tick")

    qubits_left = qubits
    for m in ancilla:
        circuit_repeat.append('cz',[m,m-1])
        qubits_left = [item for item in qubits_left if item not in [m,m-1]]
    for m in ancilla:
        circuit_repeat.append('depolarize2',[m,m-1],cz_error(m,m-1))
    circuit_repeat.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    circuit_repeat.append('tick')

    qubits_left = qubits
    for m in ancilla:
        circuit_repeat.append('cz',[m,m+1])
        qubits_left = [item for item in qubits_left if item not in [m,m+1]]
    for m in ancilla:
        circuit_repeat.append('depolarize2',[m,m+1],cz_error(m,m+1))
    circuit_repeat.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    circuit_repeat.append('tick')

    circuit_repeat.append('H',ancilla)
    for q in qubits:
        circuit_repeat.append('depolarize1',q, sg_error[q])
    circuit_repeat.append("tick")

    for m in ancilla:
        circuit_repeat.append('x_error',m,m_error[m])
    circuit_repeat.append('M',ancilla)
    circuit_repeat.append('tick')

    detector_repeat_circuit_str = ""
    for k , m in enumerate(ancilla):
        detector_repeat_circuit_str += f"DETECTOR({m}, 0) rec[{-1-k}] rec[{-1-k-(distance-1)}]\n"
    detector_repeat_circuit_str += "SHIFT_COORDS(0,1)"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)
    circuit_repeat += detector_repeat_circuit

    # begin of full circuit
    full_circuit = stim.Circuit()
    # full_circuit.append('R',qubits)
    if initial_state == 1:
        full_circuit.append('X', data)

    # full_circuit.append('M',ancilla)

    for q in data:
        full_circuit.append("x_error", q, sg_error[q])
    full_circuit.append('tick')

    full_circuit.append('H',ancilla)
    for q in qubits:
        full_circuit.append('depolarize1',q, sg_error[q])
    full_circuit.append("tick")

    qubits_left = qubits
    for m in ancilla:
        full_circuit.append('cz',[m,m-1])
        qubits_left = [item for item in qubits_left if item not in [m,m-1]]
    for m in ancilla:
        full_circuit.append('depolarize2',[m,m-1],cz_error(m,m-1))
    full_circuit.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    full_circuit.append('tick')

    qubits_left = qubits
    for m in ancilla:
        full_circuit.append('cz',[m,m+1])
        qubits_left = [item for item in qubits_left if item not in [m,m+1]]
    for m in ancilla:
        full_circuit.append('depolarize2',[m,m+1],cz_error(m,m+1))
    full_circuit.append('depolarize1',qubits_left,sg_error[qubits_left[0]])
    full_circuit.append('tick')

    full_circuit.append('H',ancilla)
    for q in qubits:
        full_circuit.append('depolarize1',q, sg_error[q])
    full_circuit.append("tick")

    for m in ancilla:
        full_circuit.append('x_error',m,m_error[m])
    full_circuit.append('M',ancilla)
    full_circuit.append('tick')

    detector_repeat_circuit_str = ""
    for k , m in enumerate(ancilla):
        detector_repeat_circuit_str += f"DETECTOR({m}, 0) rec[{-1-k}]\n"
    detector_repeat_circuit_str += "SHIFT_COORDS(0,1)"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)
    full_circuit += detector_repeat_circuit

    full_circuit += circuit_repeat*(rounds-1)

    for q in data:
        full_circuit.append('x_error',q,m_error[q])
    full_circuit.append('M',data)
    detector_circuit_str = ""
    for k , m in enumerate(ancilla):
        detector_circuit_str += f"DETECTOR({m}, 0) rec[{-1-k}] rec[{-2-k}] rec[{-2-k-(distance-1)}]\n"
    detector_circuit = stim.Circuit(detector_circuit_str)
    full_circuit += detector_circuit
    full_circuit.append_operation("observable_include",[stim.target_rec(-1)],0)

    return full_circuit

def matching_dem(detector_error_model, detection_events, observable_flips):
    num_shots = detection_events.shape[0]
    matcher = Matching.from_detector_error_model(detector_error_model)
    predictions = matcher.decode_batch(detection_events)
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors/num_shots

def matching_pcm(pcm, lx, error_rates, syndrome):
    # print(error_rates)
    weights = np.log((1-np.array(error_rates))/np.array(error_rates))
    decoder = Matching(pcm, weights=weights)
    recover = decoder.decode(syndrome)
    L = (lx@recover.T)%2
    return L

def experiment_data_qlab(distance, rounds, num_shots = None, initial_state='random', analyze=False, idea = False, old = False):
    if initial_state != 'random':
        if old == True:
            filepath = f"experiment_data/old/d{distance}r{rounds}l{initial_state}.npz"
        else:
            filepath = f"experiment_data/new/d{distance}r{rounds}l{initial_state}.npz"
            # filepath = "experiment_data/new/D3_rep/code_T1_Q21Q16Q10Q17Q23_1.npz"
        data = np.load(filepath)
        sg_error = data['sg_error']
        cz_error = data['cz_error']
        m_error = data['m_error']
        # initial_bitstring = data['initial_bitstring']
        data = data['bitstrings']
        print('num_experiment_shots = ', data.shape[0])
        if num_shots is not None:
            num_rows = data.shape[0]
            if num_shots > num_rows:
                raise ValueError("num_shots cannot be greater than the number of trials in the arrays")
            indices = np.random.permutation(num_rows)[:num_shots]
            # Select the rows using the random indices
            data = data[indices, :]
    else:
        from os.path import dirname, abspath
        filepath = abspath(dirname(__file__))+f'/experiment_data/random_initial/d{distance}r{rounds}.npz' #f"/experiment_data/random_initial/d{distance}r{rounds}.npz"
        data = np.load(filepath)
        sg_error = data['sg_error']
        cz_error = data['cz_error']
        m_error = data['m_error']
        initial_bitstring = data['initial_bitstrings']
        data = data['bitstrings']
        print('num_experiment_trails = ', data.shape[0])
        assert initial_bitstring.shape[0] == data.shape[0]
        if num_shots is not None:
            num_rows = data.shape[0]
            if num_shots > num_rows:
                print('num_experiment_trails = ', data.shape[0])
                raise ValueError("num_shots cannot be greater than the number of trials in the arrays")
            indices = np.random.permutation(num_rows)[:num_shots]
            data = data[indices, :]
            initial_bitstring = initial_bitstring[indices, :]
    circuit = rep_code_noreset(distance=distance,
                   rounds=rounds,
                    simulation=False,
                    initial_state=initial_state,
                    sg_error=sg_error,
                    tg_error=cz_error,
                    m_error=m_error)
    # circuit = rep_code_noreset(distance=distance,
    #                rounds=rounds,
    #                 simulation=True,
    #                 initial_state=initial_state,)
    circuit: stim.Circuit
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    num_ems = dem.num_errors
    num_dets = dem.num_detectors
    assert data.shape[1] == (distance-1)*rounds + distance, "experiment data num_m do not match"
    premeasurement = np.zeros((data.shape[0], int(distance-1) ), dtype=data.dtype)
    total_m = np.concatenate((premeasurement, data), axis=1)
    bool_m = total_m.astype(np.bool_)
    if idea == True:
        dets, obs = circuit.compile_detector_sampler(seed = None).sample(500000, separate_observables=True) 
    else:
        if initial_state == 'random':
            bool_i = initial_bitstring.astype(np.bool_)
            dets, obs = circuit.compile_m2d_converter().convert(measurements = bool_m,sweep_bits=bool_i, separate_observables=True)
        else: dets, obs = circuit.compile_m2d_converter().convert(measurements = bool_m, separate_observables=True)
    er_estimate = []
    tanner_graph = correlation.TannerGraph(dem)
    hyperedges = tanner_graph.hyperedges
    if analyze == True:

        detectors = []
        boundary = []
        for i in range((distance-1)*(rounds+1)):
            detectors.append(i)
        up=detectors[::(distance-1)]
        upb = [frozenset({i}) for i in up]
        down=detectors[(distance-2)::(distance-1)]
        downb = [frozenset({i}) for i in down]
        countb = 0

        result = correlation.cal_2nd_order_correlations(dets, hyperedges=hyperedges)
        for hyperedge in hyperedges:
            if hyperedge in upb:
                sum = 0
                count = 0
                i = next(iter(hyperedge))
                for j in range(i, i + distance-2):
                    value = result.get(frozenset({j, j+1}))
                    sum += value
                    count += 1
                avg = sum / count
                error_p = avg
                boundary.append(avg)
                # print('up')
                # print(result.get(hyperedge))
                # print(error_p)
                # print('---------')
            elif hyperedge in downb:
                error_p = boundary[countb]
                countb += 1
                # print('down')
                # print(result.get(hyperedge))
                # print(error_p)
                # print('---------')
            else:
                error_p = result.get(hyperedge)
            if error_p <= 0:   
                print("negative error occurred: ", error_p)
                error_p = 0.08
            # if error_p > 0.1: 
            #     print(error_p)
            #     error_p = 0.1

            er_estimate.append(error_p)
    else:
        for i in range(num_ems):
            er_estimate.append(dem[i].args_copy()[0])
    # order= re_order(distance,rounds,dem,False)
    # print(order)
    # er = []
    # for i in range(num_ems):
        # er.append(er_estimate[order[i]])
    er = er_estimate

    assert len(er) == num_ems
    # assert edges.shape == (num_dets, num_dets)
    # assert bdy.shape == (num_dets, )
    # np.testing.assert_equal(edges, edges.T)
    return dets, obs, er, dem

class Sycamore():
    def __init__(self,
                 distance,
                 rounds,
                 file_path_syndrome,
                 file_path_logical_flip) -> None:
        self.distance = distance
        self.rounds = rounds
        self.file_path_syndrome = file_path_syndrome
        self.file_path_logical_flip = file_path_logical_flip
        self.bits_per_shot = (distance - 1) * (rounds)

    def parse_b8(self, data: bytes, bits_per_shot: int) -> List[List[bool]]:
        shots = []
        bytes_per_shot = (bits_per_shot + 7) // 8
        for offset in range(0, len(data), bytes_per_shot):
            shot = []
            for k in range(bits_per_shot):
                byte = data[offset + k // 8]
                bit = (byte >> (k % 8)) % 2 == 1
                shot.append(bit)
            shots.append(shot)
        return shots

    def parse_01(self, data: str) -> List[List[bool]]:
        shots = []
        for line in data.split('\n'):
            if not line:
                continue
            shot = []
            for c in line:
                assert c in '01'
                shot.append(c == '1')
            shots.append(shot)
        return shots

    def logical_flip(self) -> List[List[bool]]:
        with open(self.file_path_logical_flip, 'rb') as f:
            data = f.read()
        shots = self.parse_b8(data, 1)
        return shots
    
    def syndromes(self) -> List[List[bool]]:
        with open(self.file_path_syndrome, 'rb') as f:
            data = f.read()
        shots = self.parse_b8(data, self.bits_per_shot)
        return shots


def experiment_data_sycamore(distance = 29, rounds = 1001, num_shots = 50000):
    file_path_syndrome = '/data/fengdongyang/workspace/planar-decoder/experiment_data/sample_00/detection_events.b8'
    file_path_dem = '/data/fengdongyang/workspace/planar-decoder/experiment_data/sample_00/decoding_results/MWPM_decoder_with_RL_optimized_prior/error_model.dem'
    file_path_logical_flip = '/data/fengdongyang/workspace/planar-decoder/experiment_data/sample_00/obs_flips_actual.b8'
    sycamore = Sycamore(distance=distance,
             rounds = rounds,
            #  file_path_dem=file_path_dem,
             file_path_logical_flip=file_path_logical_flip,
             file_path_syndrome=file_path_syndrome,
             )
    samples = sycamore.syndromes() # about 7 minuet
    logical_samples = sycamore.logical_flip() # about 30 second
    return samples, logical_samples


