import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import stim
from src import (TensorNetwork,
                 GroupTN,
                 get_error_rates,
                 PCM,
                 generate_compactified_pcm_from_seperated_dem,
                 subsample_d3_pcms,
                 MWPM_dem,
                 MWPM_graph,
                 TensorNetworkDecoder,
                 )
                 #BeliefMatching_dem)

def load_google_data(b8_path, obvs_path, circuit_path):
    """
    Load Google detection event data from .b8 file.
    
    Args:
        b8_path: Path to .b8 file containing detection events
        num_detectors: Number of detectors per shot
        max_shots: Maximum number of shots to load. If None, loads all shots.
                   If specified, loads only the first max_shots shots.
    
    Returns:
        torch.Tensor of shape (num_shots, num_detectors)
    """
    from tqec import NoiseModel
    # stim.read_shot_data_file returns boolean numpy array (shots, num_detectors)
    # bit_packed=False returns uint8 array of 0s and 1s
    circuit_from_file = stim.Circuit.from_file(circuit_path)
    noise_model = NoiseModel.si1000(p=0.003)
    circuit = noise_model.noisy_circuit(circuit_from_file)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    num_detectors = dem.num_detectors
    print(f"Reading data from {b8_path}...")
    data = stim.read_shot_data_file(path=b8_path, format="b8", num_detectors=num_detectors, bit_packed=False)
    total_shots = len(data)
    print(f"  Total shots in file: {total_shots:,}")
    obvs = stim.read_shot_data_file(path=obvs_path, format="b8", num_detectors=1, bit_packed=False).flatten()
    
    
    return torch.from_numpy(data.astype(np.float64)), torch.from_numpy(obvs.astype(np.float64)), dem
    

def decoding_d3_decomposed(d=3, r='05', sample='00', coor='4_5', dev='cuda:0', task=0):
    dets, obvs, _ = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                    )
    
    dem = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/simple.dem')
    pcm, l = PCM(dem)

    dem_cor = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/correlations.dem')
    dem_rl_cm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_correlated_matching.dem')
    dem_rl_cm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_correlated_matching.dem')
    dem_rl_bm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_belief_matching.dem')
    dem_rl_bm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_belief_matching.dem')
    
    er_sim = get_error_rates(dem)
    er_cor = get_error_rates(dem_cor)
    er_rl_cm_sh = get_error_rates(dem_rl_cm_shared)
    er_rl_cm_iso = get_error_rates(dem_rl_cm_iso)
    er_rl_bm_sh = get_error_rates(dem_rl_bm_shared)
    er_rl_bm_iso = get_error_rates(dem_rl_bm_iso)

    opt_ers = torch.load(f'data/sycamore/decompose_d{d}r{r}_s{sample}_q{coor}_task{task}.pt')

    tn_contract_path_file = f'path/sycamore/d{d}r{r}.pkl'
    tn = TensorNetwork(pcm=pcm, l=l.flatten(), dtype=torch.float64, dev=dev, decoding=True)
    tn.load_path(filename=tn_contract_path_file)

    decoder = TensorNetworkDecoder(model=tn, dev=dev)

    ler_sim = decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=torch.from_numpy(er_sim).to(dev))
    ler_cor = decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=torch.from_numpy(er_cor).to(dev))
    ler_rl_cm_sh = decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=torch.from_numpy(er_rl_cm_sh).to(dev))
    ler_rl_cm_iso= decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=torch.from_numpy(er_rl_cm_iso).to(dev))
    ler_rl_bm_sh = decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=torch.from_numpy(er_rl_bm_sh).to(dev))
    ler_rl_bm_iso= decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=torch.from_numpy(er_rl_bm_iso).to(dev))


    print(f'TN Decoder LER with simple error rates: {ler_sim}')
    print(f'TN Decoder LER with correlated error rates: {ler_cor}')
    print(f'TN Decoder LER with rl_shared_correlated_matching error rates: {ler_rl_cm_sh}')
    print(f'TN Decoder LER with rl_isolated_correlated_matching error rates: {ler_rl_cm_iso}')
    print(f'TN Decoder LER with rl_shared_belief_matching error rates: {ler_rl_bm_sh}')
    print(f'TN Decoder LER with rl_isolated_belief_matching error rates: {ler_rl_bm_iso}')

    log_path = f'log/sc_tn/sycamore/decompose_d{d}r{r}s{sample}_q{coor}_task{task}.txt'
    log_file = open(log_path,'a')
    log_file.write(f'TN Decoder LER with simple error rates: {ler_sim}\n')
    log_file.write(f'TN Decoder LER with correlated error rates: {ler_cor}\n')
    log_file.write(f'TN Decoder LER with rl_shared_correlated_matching error rates: {ler_rl_cm_sh}\n')
    log_file.write(f'TN Decoder LER with rl_isolated_correlated_matching error rates: {ler_rl_cm_iso}\n')
    log_file.write(f'TN Decoder LER with rl_shared_belief_matching error rates: {ler_rl_bm_sh}\n')
    log_file.write(f'TN Decoder LER with rl_isolated_belief_matching error rates: {ler_rl_bm_iso}\n')   
    log_file.flush()
    

    for i in range(len(opt_ers)):
        ler_opt = decoder.logical_error_rate(dets.to(dev), obvs.to(dev), error_rates=opt_ers[i].to(dev))
        print(f'TN Decoder LER with optimized error rates {i} : {ler_opt}')
        log_file.write(f'TN Decoder LER with optimized error rates {i} : {ler_opt}\n')
        log_file.flush()

def decoding_d5_decomposed(d=5, r=5, sample=0, coor='6_5', batch_size=25, dev='cuda:0', task=0):

    sample = f"{sample:02d}"
    r = f"{r:02d}"

    dets, obvs, _ = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                    )
    
    # dem = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/simple.dem')
    

    dem_cor = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/correlations.dem')
    pcm, l = PCM(dem_cor)
    dem_rl_cm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_correlated_matching.dem')
    # dem_rl_cm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_correlated_matching.dem')
    dem_rl_bm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_belief_matching.dem')
    # dem_rl_bm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_belief_matching.dem')
    
    dem_plnet = stim.DetectorErrorModel.from_file(f'data/sycamore/pl/s{sample}/dem_s{sample}_r{r}_PL.txt')

    dem_tn = stim.DetectorErrorModel.from_file(f'data/sycamore/TN/s{sample}/dem_s{sample}_r{r}_TN.txt')

    # er_sim = get_error_rates(dem)
    er_cor = get_error_rates(dem_cor)
    er_rl_cm_sh = get_error_rates(dem_rl_cm_shared)
    # er_rl_cm_iso = get_error_rates(dem_rl_cm_iso)
    er_rl_bm_sh = get_error_rates(dem_rl_bm_shared)
    # er_rl_bm_iso = get_error_rates(dem_rl_bm_iso)

    er_plnet = get_error_rates(dem_plnet)#torch.load('data/sycamore/decompose_d5r05_s32_q6_5_task0.pt')[-1].numpy()#
    
    er_tn = get_error_rates(dem_tn)


    tn_contract_tree_file = f"path/sycamore/d5_r{r}.json"
    tn = TensorNetwork(pcm=pcm, l=l.flatten(), dtype=torch.float64, dev=dev, decoding=True)
    tn.load_tree(tn_contract_tree_file)
    decoder = TensorNetworkDecoder(model=tn, dev=dev)
    nb = dets.shape[0] // batch_size

    ler_cor=0
    ler_rl_cm_sh=0
    ler_rl_bm_sh=0
    ler_plnet=0
    ler_tn=0

   

    import time
    t = 0
    for i in range(nb):
        torch.cuda.synchronize(dev)
        print('batch idx:', i)
        t0 = time.perf_counter()
        ler_tn += decoder.logical_error_rate(dets[i*batch_size:(i+1)*batch_size].to(dev), obvs[i*batch_size:(i+1)*batch_size].to(dev), error_rates=torch.from_numpy(er_tn).to(dev).to(torch.float64))
        ler_plnet += decoder.logical_error_rate(dets[i*batch_size:(i+1)*batch_size].to(dev), obvs[i*batch_size:(i+1)*batch_size].to(dev), error_rates=torch.from_numpy(er_plnet).to(dev).to(torch.float64))
        
        ler_cor += decoder.logical_error_rate(dets[i*batch_size:(i+1)*batch_size].to(dev), obvs[i*batch_size:(i+1)*batch_size].to(dev), error_rates=torch.from_numpy(er_cor).to(dev).to(torch.float64))
        ler_rl_cm_sh += decoder.logical_error_rate(dets[i*batch_size:(i+1)*batch_size].to(dev), obvs[i*batch_size:(i+1)*batch_size].to(dev), error_rates=torch.from_numpy(er_rl_cm_sh).to(dev).to(torch.float64))
        ler_rl_bm_sh += decoder.logical_error_rate(dets[i*batch_size:(i+1)*batch_size].to(dev), obvs[i*batch_size:(i+1)*batch_size].to(dev), error_rates=torch.from_numpy(er_rl_bm_sh).to(dev).to(torch.float64))
        
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter()

        print(f'ler_tn:{ler_tn/(i+1)}, ler_pl:{ler_plnet/(i+1)}, ler_cor:{ler_cor/(i+1)}')
        
        print(t1-t0)
        t+=t1-t0

    print(t/nb)

    ler_tn=ler_tn/nb
    ler_plnet=ler_plnet/nb
    ler_cor=ler_cor/nb
    ler_rl_bm_sh=ler_rl_bm_sh/nb
    ler_rl_cm_sh=ler_rl_cm_sh/nb
    

    print(f'TN Decoder LER with correlated error rates: {ler_cor}')
    print(f'TN Decoder LER with rl_share_correlated_matching error rates: {ler_rl_cm_sh}')
    print(f'TN Decoder LER with rl_share_belief_matching error rates: {ler_rl_bm_sh}')
    print(f'TN Decoder LER with planarnet error rates: {ler_plnet}') 
    print(f'TN Decoder LER with tn error rates: {ler_tn}') 

    log_path = f'log/sc_tn/sycamore/decompose_d{d}r{r}s{sample}_q{coor}_task{task}.txt'
    log_file = open(log_path,'a')

    log_file.write(f'TN Decoder LER with correlated error rates: {ler_cor}\n')
    log_file.write(f'TN Decoder LER with rl_share_correlated_matching error rates: {ler_rl_cm_sh}\n')
    log_file.write(f'TN Decoder LER with rl_share_belief_matching error rates: {ler_rl_bm_sh}\n')
    log_file.write(f'TN Decoder LER with planarnet error rates: {ler_plnet}\n') 
    log_file.write(f'TN Decoder LER with tn error rates: {ler_tn}') 

    log_file.flush()
    
    log_file.close()
    # print(f'TN Decoder LER with simple error rates: {ler_sim}')
    # print(f'TN Decoder LER with rl_shared_correlated_matching error rates: {ler_rl_cm_sh}')
    # print(f'TN Decoder LER with rl_isolated_correlated_matching error rates: {ler_rl_cm_iso}')
    # print(f'TN Decoder LER with rl_shared_belief_matching error rates: {ler_rl_bm_sh}')

    # log_file.write(f'TN Decoder LER with simple error rates: {ler_sim}\n')
    # log_file.write(f'TN Decoder LER with rl_shared_correlated_matching error rates: {ler_rl_cm_sh}\n')
    # log_file.write(f'TN Decoder LER with rl_isolated_correlated_matching error rates: {ler_rl_cm_iso}\n')
    # log_file.write(f'TN Decoder LER with rl_shared_belief_matching error rates: {ler_rl_bm_sh}\n')
    

  

def nll(d=5, r='25', sample='00', coor='6_5', batch_size=25, dev='cuda:0', task=0):
    dets, obvs, _ = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                    )

    dem_cor = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/correlations.dem')
    pcm, l = PCM(dem_cor)
    dem_rl_bm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_belief_matching.dem')
    dem_plnet = stim.DetectorErrorModel.from_file(f'data/sycamore/dem_s{sample}_r{r}.txt')

    er_cor = get_error_rates(dem_cor)
    er_rl_bm_iso = get_error_rates(dem_rl_bm_iso)
    er_plnet = get_error_rates(dem_plnet)


    tn_contract_tree_file = f"path/sycamore/d5_r{r}.json"
    tn = TensorNetwork(pcm=pcm, dtype=torch.float64, dev=dev)
    tn.load_tree(tn_contract_tree_file)
    nb = dets.shape[0] // batch_size


    nll_cor=0
    nll_rl_bm_iso=0
    nll_plnet=0
    for i in range(nb):
        with torch.no_grad():
            nll_plnet += tn.forward(dets[i*batch_size:(i+1)*batch_size].to(dev), priors_logits = torch.logit(torch.from_numpy(er_plnet).to(dev).to(torch.float64))).detach().cpu().item()
            nll_cor += tn.forward(dets[i*batch_size:(i+1)*batch_size].to(dev), priors_logits = torch.logit(torch.from_numpy(er_cor).to(dev).to(torch.float64))).detach().cpu().item()
            nll_rl_bm_iso += tn.forward(dets[i*batch_size:(i+1)*batch_size].to(dev), priors_logits = torch.logit(torch.from_numpy(er_rl_bm_iso).to(dev).to(torch.float64))).detach().cpu().item()
            print(f'i:{i} nll_pl:{nll_plnet/(i+1)}, nll_cor:{nll_cor/(i+1)}, nll_rl_bm_iso:{nll_rl_bm_iso/(i+1)}')


    nll_plnet=nll_plnet/nb
    nll_cor=nll_cor/nb
    nll_rl_bm_iso=nll_rl_bm_iso/nb
   

    print(f'Nll with correlated error rates: {nll_cor}\n')
    print(f'Nll with rl_isolated_belief_matching error rates: {nll_rl_bm_iso}\n')
    print(f'Nll with planarnet error rates: {nll_plnet}\n')

    log_path = f'log/sc_tn/sycamore/decompose_d{d}r{r}s{sample}_q{coor}_task{task}.txt'
    log_file = open(log_path,'a')
 

    log_file.write(f'Nll with correlated error rates: {nll_cor}\n')
    log_file.write(f'Nll with rl_isolated_belief_matching error rates: {nll_rl_bm_iso}\n')
    log_file.write(f'Nll with planarnet error rates: {nll_plnet}\n')

    log_file.flush()

    log_file.close

    



def bm_worker_ler(dem, dets_chunk, obvs_chunk, error_rates):
        # 在子进程中导入并实例化，避开序列化限制
        from src import BeliefMatching_dem
        bm = BeliefMatching_dem(dem, max_iter=10)
        # 计算该 chunk 的 LER
        return bm.logical_error_rate(dets_chunk, obvs_chunk, error_rates=error_rates)

def decoding_belief_matching(d=5, r=5, sample=0, coor='6_5' , nprocesses=50, task=0):

    sample = f"{sample:02d}"
    r = f"{r:02d}"

    dets, obvs, _ = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                    )
    
    # dem = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/simple.dem')
    

    dem_cor = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/correlations.dem')
    pcm, l = PCM(dem_cor)
    # dem_rl_cm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_correlated_matching.dem')
    # dem_rl_cm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_correlated_matching.dem')
    # dem_rl_bm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_belief_matching.dem')
    # dem_rl_bm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_belief_matching.dem')
    
    dem_plnet = stim.DetectorErrorModel.from_file(f'data/sycamore/pl/s{sample}/dem_s{sample}_r{r}_PL.txt')
    dem_tn = stim.DetectorErrorModel.from_file(f'data/sycamore/TN/s{sample}/dem_s{sample}_r{r}_TN.txt')

    # er_sim = get_error_rates(dem)
    er_cor = get_error_rates(dem_cor)
    # er_rl_cm_sh = get_error_rates(dem_rl_cm_shared)
    # er_rl_cm_iso = get_error_rates(dem_rl_cm_iso)
    # er_rl_bm_sh = get_error_rates(dem_rl_bm_shared)
    # er_rl_bm_iso = get_error_rates(dem_rl_bm_iso)

    er_plnet = get_error_rates(dem_plnet)
    er_tn = get_error_rates(dem_tn)
    
    import multiprocessing as mp
    def multiprocessing_decode(dem, dets, obvs, error_rates, nprocesses=nprocesses):
        # 2. 使用 array_split 自动处理无法整除的情况，比手动切片更稳健
        dets_chunks = np.array_split(dets, nprocesses)
        obvs_chunks = np.array_split(obvs, nprocesses)
        
        # 3. 构造参数列表，传递 dem 而不是 ilp 实例
        args = [(dem, dets_chunks[i], obvs_chunks[i], error_rates) for i in range(nprocesses)]
        
        # 4. 使用 Pool 进行并行计算
        with mp.Pool(nprocesses) as p:
            # 使用 starmap 调用顶层 worker 函数
            results = p.starmap(bm_worker_ler, args)
        
        # 5. 加权平均计算总 LER (防止各 chunk 大小不一时结果不准)
        total_samples = dets.shape[0]
        weighted_ler = sum(res * len(chunk) for res, chunk in zip(results, dets_chunks)) / total_samples
        
        return weighted_ler
    

    bm_obvs_correlation=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_correlations/belief_matching.b8',
                                                        format="b8", num_detectors=1, bit_packed=False).flatten()
    bm_obvs_rl_shared=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_shared_belief_matching/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
    bm_obvs_rl_shared_cm=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_shared_correlated_matching/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()

    obvs_actual=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()

    ler_cor = (bm_obvs_correlation != obvs_actual).sum().item()/len(obvs_actual)
    ler_rl_bm_sh = (bm_obvs_rl_shared != obvs_actual).sum().item()/len(obvs_actual)
    ler_rl_cm_sh = (bm_obvs_rl_shared_cm != obvs_actual).sum().item()/len(obvs_actual)


    print(f'BM Decoder LER with correlated error rates: {ler_cor}')
    print(f'BM Decoder LER with rl_share_correlated_matching error rates: {ler_rl_cm_sh}')
    print(f'BM Decoder LER with rl_share_belief_matching error rates: {ler_rl_bm_sh}')

    ler_plnet =  multiprocessing_decode(dem_cor, dets, obvs, error_rates=er_plnet)
    print(f'BM Decoder LER with planarnet error rates: {ler_plnet}') 
    ler_tn = multiprocessing_decode(dem_cor, dets, obvs, error_rates=er_tn)
    print(f'BM Decoder LER with tn error rates: {ler_tn}') 

    log_path = f'log/sc_tn/sycamore/decompose_d{d}r{r}s{sample}_q{coor}_task{task}.txt'
    log_file = open(log_path,'a')
    log_file.write('\n')
    log_file.write(f'BM Decoder LER with correlated error rates: {ler_cor}\n')
    log_file.write(f'BM Decoder LER with rl_share_correlated_matching error rates: {ler_rl_cm_sh}\n')
    log_file.write(f'BM Decoder LER with rl_share_belief_matching error rates: {ler_rl_bm_sh}\n')
    log_file.write(f'BM Decoder LER with planarnet error rates: {ler_plnet}\n') 
    log_file.write(f'BM Decoder LER with tn error rates: {ler_tn}') 
   
   
def decoding_correlated_matching(d=5, r=5, sample=0, coor='6_5' , task=0):

    sample = f"{sample:02d}"
    r = f"{r:02d}"

    dets, obvs, _ = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                    )
    
    # dem = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/simple.dem')
    

    dem_cor = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/correlations.dem')
    
    mwpm = MWPM_dem(dem_cor, enable_correlations=True)

    dem_rl_cm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_correlated_matching.dem')
    # dem_rl_cm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_correlated_matching.dem')
    dem_rl_bm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_belief_matching.dem')
    # dem_rl_bm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_belief_matching.dem')
    
    dem_plnet = stim.DetectorErrorModel.from_file(f'data/sycamore/pl/s{sample}/dem_s{sample}_r{r}_PL.txt')
    dem_tn = stim.DetectorErrorModel.from_file(f'data/sycamore/TN/s{sample}/dem_s{sample}_r{r}_TN.txt')

    # er_sim = get_error_rates(dem)
    er_cor = get_error_rates(dem_cor)
    er_rl_cm_sh = get_error_rates(dem_rl_cm_shared)
    # er_rl_cm_iso = get_error_rates(dem_rl_cm_iso)
    er_rl_bm_sh = get_error_rates(dem_rl_bm_shared)
    # er_rl_bm_iso = get_error_rates(dem_rl_bm_iso)

    er_plnet = get_error_rates(dem_plnet)
    er_tn = get_error_rates(dem_tn)
    

    cm_obvs_correlation=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_correlations/belief_matching.b8',
                                                        format="b8", num_detectors=1, bit_packed=False).flatten()
    cm_obvs_rl_shared_bm=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_shared_belief_matching/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
    cm_obvs_rl_shared_cm=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_shared_correlated_matching/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()

    obvs_actual=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()

    ler_cor = (cm_obvs_correlation != obvs_actual).sum().item()/len(obvs_actual)
    ler_rl_bm_sh = (cm_obvs_rl_shared_bm != obvs_actual).sum().item()/len(obvs_actual)
    ler_rl_cm_sh = (cm_obvs_rl_shared_cm != obvs_actual).sum().item()/len(obvs_actual)


    print(f'CM Decoder LER with correlated error rates (read): {ler_cor}')
    print(f'CM Decoder LER with rl_share_correlated_matching error rates (read): {ler_rl_cm_sh}')
    print(f'CM Decoder LER with rl_share_belief_matching error rates (read): {ler_rl_bm_sh}')

    ler_cor_r =  mwpm.logical_error_rate(dets, obvs, error_rates=er_cor)
    ler_rl_cm_sh_r = mwpm.logical_error_rate(dets, obvs, error_rates=er_rl_cm_sh)
    ler_rl_bm_sh_r = mwpm.logical_error_rate(dets, obvs, error_rates=er_rl_bm_sh)

    print(f'CM Decoder LER with correlated error rates (recal): {ler_cor_r}')
    print(f'CM Decoder LER with rl_share_correlated_matching error rates (recal): {ler_rl_cm_sh_r}')
    print(f'CM Decoder LER with rl_share_belief_matching error rates (recal): {ler_rl_bm_sh_r}')
    

    ler_plnet =  mwpm.logical_error_rate(dets, obvs, error_rates=er_plnet)
    print(f'CM Decoder LER with planarnet error rates: {ler_plnet}') 
    ler_tn = mwpm.logical_error_rate(dets, obvs, error_rates=er_tn)
    print(f'CM Decoder LER with tn error rates: {ler_tn}') 

    log_path = f'log/sc_tn/sycamore/decompose_d{d}r{r}s{sample}_q{coor}_task{task}.txt'
    log_file = open(log_path,'a')
    log_file.write('\n')
    log_file.write(f'CM Decoder LER with correlated error rates (read): {ler_cor} \n')
    log_file.write(f'CM Decoder LER with rl_share_correlated_matching error rates (read): {ler_rl_cm_sh} \n')
    log_file.write(f'CM Decoder LER with rl_share_belief_matching error rates (read): {ler_rl_bm_sh} \n')
    log_file.write(f'CM Decoder LER with correlated error rates (recal): {ler_cor_r} \n')
    log_file.write(f'CM Decoder LER with rl_share_correlated_matching error rates (recal): {ler_rl_cm_sh_r} \n')
    log_file.write(f'CM Decoder LER with rl_share_belief_matching error rates (recal): {ler_rl_bm_sh_r} \n')
    
    log_file.write(f'CM Decoder LER with planarnet error rates: {ler_plnet}\n') 
    log_file.write(f'CM Decoder LER with tn error rates: {ler_tn}') 


def main(sample, dev='cuda:0', decoder='tn', nprocesses=50):
    for r in [9, 13, 17, 21, 25]:  
        if decoder == 'tn':
            if r == 9 or r == 13:
                batch_size=100
            elif r == 17:
                batch_size=30
            else:
                batch_size=25
            
            decoding_d5_decomposed(batch_size=batch_size, r=r, dev=dev, sample=sample, task=1)
        elif decoder == 'bm':
            decoding_belief_matching(r=r, sample=sample, nprocesses=nprocesses, task=1)
    
        elif decoder == 'cm':
            decoding_correlated_matching(r=r, sample=sample, task=1)
    
    


if __name__ == "__main__":
    # decoding_ilp(coor='8_5', task=0, nprocesses=20)
    # decoding_fdecomposed(dev='cuda:0', coor='8_5', task=0)

    import fire
    fire.Fire({
        'decoding': decoding_d5_decomposed,
        'nll': nll,
        'main': main,
               })
    
    # decoding_d5_decomposed(batch_size=25, r='17', dev='cuda:7', sample='10', task=0)
    # nll(batch_size=25, r='17', dev='cuda:7', sample='10', task=0)