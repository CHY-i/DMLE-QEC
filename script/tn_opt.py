import sys
import os
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import stim
from src import (TensorNetwork,
                 get_error_rates,
                 PCM,
                 MWPM_dem,
                 BeliefMatching_dem)


def generate_data(d, r, num_shots, cir_type, error_prob=0.001):
    if cir_type == 'simulation':
        circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z",
                                            distance=d,
                                            rounds=r,
                                            after_clifford_depolarization=error_prob,
                                            before_measure_flip_probability=error_prob,
                                            after_reset_flip_probability=error_prob,
                                            )
    if cir_type == 'google':
        from tqec import NoiseModel
        circuit_file = stim.Circuit.from_file(f"data/sycamore_surface_code_d3_d5/circuits/d{d}r{r}_circuit_ideal.stim")
        noise_model = NoiseModel.si1000(error_prob)
        circuit = noise_model.noisy_circuit(circuit_file)

    dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
    er = get_error_rates(dem)
    sampler = dem.compile_sampler(seed=75328)
    dets, obvs, _ = sampler.sample(shots=num_shots)
    dets = torch.from_numpy(dets*1.)
    obvs = torch.from_numpy(obvs*1.)
    pcm, l = PCM(dem)
    
    return dets, obvs, pcm, er, dem


def optimizing(d=3, r=3, error_prob=0.001, num_shots=1000000,
               epochs=1000, lr=0.001, batch_size=10000, mini_batch=1000, dev='cuda:0', dtype=torch.float64, nprint=20, task=0):

    dets, obvs, pcm, er, dem = generate_data(d=d, r=r, num_shots=num_shots, error_prob=error_prob, cir_type='simulation')
    edge_degrees = pcm.sum(0)
    regular_edges = np.where(edge_degrees > 2)[0]
    
    dataset = TensorDataset(dets)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    mwpm = MWPM_dem(dem, enable_correlations=False)

    er = torch.from_numpy(er).to(dtype)
    pertub = torch.rand_like(er).to(dtype)
    init_er = (er + (2*torch.bernoulli(torch.ones(len(er)).to(dtype)/2)-1.)*er*pertub)
    priors_logits = torch.logit(init_er)

    tn = TensorNetwork(pcm=pcm, priors_logits=priors_logits, dtype=dtype, dev=dev)

    tn_contract_path_file = f'path/simulation/si1000_d{d}r{r}.pkl'

    if not os.path.exists(tn_contract_path_file):
        path = tn.find_contraction_path(batch_size=10, max_time=120)
        tn.save_path(path, filename=tn_contract_path_file)

    tn.load_path(filename=tn_contract_path_file)

    log_path = f'log/sc_tn/simulation/si1000_d{d}r{r}er{error_prob}.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path,'a')
    log_file.write(f'# Training Parameters : Num Shots:{dets.shape[0]} epochs:{epochs} batch:{batch_size} lr:{lr}\n')
    log_file.write(f'initial logical error rate :{mwpm.logical_error_rate(dets, obvs, init_er.numpy())} \n')
    log_file.write(f'true logical error rate :{mwpm.logical_error_rate(dets, obvs, er.numpy())} \n')
    log_file.flush()
    
    optimizer = torch.optim.Adam(tn.parameters(), lr=lr)

    loss_list = []
    priors_list = []
    for epoch in range(1, epochs+1):
        losses = []
        for j, syndrome_data in enumerate(dataloader):
            l1_reg = torch.sum(torch.sigmoid(tn.priors_logits[regular_edges]))
   
            loss = tn.forward(syndrome_data[0])
            total_loss = loss + l1_reg*0.001
            optimizer.zero_grad()
            total_loss.backward()      
            optimizer.step()

            losses.append(loss.detach().cpu().item())

        if epoch >=10 and abs(np.array(losses).mean() - loss_list[-1])/loss_list[-1] < 1e-12 :
            print(f'loss converge : {np.array(losses).mean()} at epoch {epoch}')
            break

        loss_list.append(np.array(losses).mean())

        if epoch % nprint == 0 or epoch <=20:
            oer = torch.sigmoid(tn.priors_logits.detach().cpu())
            priors_list.append(oer)
            print('epoch:', epoch, 'non-zero grad ratio:', (torch.count_nonzero(tn.priors_logits.grad)/len(tn.priors_logits.grad)).item())
            print('loss :', np.array(losses).mean())
    
            log_file.write(f'epoch : {epoch} loss : {np.array(losses).mean()} mre : {(abs(oer-er)/er).mean()} training logical error rate :{mwpm.logical_error_rate(dets, obvs, oer.numpy())} \n')
            log_file.flush()
        
    log_file.close()
    torch.save(priors_list, f=f'data/simulation/d{d}r{r}er{error_prob}_task{task}.pt')


def load_google_data(b8_path, obvs_path, circuit_path):
    from tqec import NoiseModel
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
    
    return torch.from_numpy(data.astype(np.float64)), obvs, dem


def optimizing_google_data(method='decompose' ,d=5, r='05', sample=20, coor='6_5', 
               epochs=100, lr=0.01, batch_size=15000, minibatch=50, dev='cuda:0', dtype=torch.float64, nprint=10, task=0, read_data=True):
    sample = f"{sample:02d}"

    # 设置并创建 Checkpoint 目录
    ckpt_dir = f'data/checks/sample_{sample}'
    os.makedirs(ckpt_dir, exist_ok=True)

    if method == 'decompose':
        dets, obvs, _ = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                    )
        dem = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/simple.dem')
        tn_contract_path_file = f'path/sycamore/d{d}r{r}.pkl'
        tn_contract_tree_file = f"path/sycamore/d{d}_r{r}.json"
    elif method == 'fdecompose':
        dets, obvs, dem = load_google_data(b8_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/detection_events.b8',
                                    obvs_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                    circuit_path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/circuit_ideal.stim',
                                   )
        tn_contract_path_file = f'path/sycamore/fd_d{d}r{r}.pkl'
    
    pcm = PCM(dem)[0]
    dataset = TensorDataset(dets)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    er_sim = get_error_rates(dem)
    
    if read_data:
        dem_cor = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/correlations.dem')
        dem_rl_cm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_correlated_matching.dem')
        dem_rl_cm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_correlated_matching.dem')
        dem_rl_bm_shared = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_shared_belief_matching.dem')
        dem_rl_bm_iso = stim.DetectorErrorModel.from_file(f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/dems/rl_isolated_belief_matching.dem')
        
        er_cor = get_error_rates(dem_cor)
        er_rl_cm_sh = get_error_rates(dem_rl_cm_shared)
        er_rl_cm_iso = get_error_rates(dem_rl_cm_iso)
        er_rl_bm_sh = get_error_rates(dem_rl_bm_shared)
        er_rl_bm_iso = get_error_rates(dem_rl_bm_iso)

        cm_obvs_correlation=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_correlations/correlated_matching.b8',
                                                        format="b8", num_detectors=1, bit_packed=False).flatten()
        cm_obvs_simple=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_simple/correlated_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
        cm_obvs_rl_shared=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_shared_correlated_matching/correlated_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
        cm_obvs_rl_iso=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_isolated_correlated_matching/correlated_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten() 

        bm_obvs_correlation=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_correlations/belief_matching.b8',
                                                        format="b8", num_detectors=1, bit_packed=False).flatten()
        bm_obvs_simple=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_simple/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
        bm_obvs_rl_shared=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_shared_belief_matching/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
        bm_obvs_rl_iso=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_predicted/dem_rl_isolated_belief_matching/belief_matching.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten() 
        
        obvs_actual=stim.read_shot_data_file(path=f'data/sycamore_surface_code_d3_d5/sample_{sample}/d{d}_at_q{coor}/X/r{r}/obs_flips_actual.b8',
                                                    format="b8", num_detectors=1, bit_packed=False).flatten()
        
        ler_correlation_bm = (bm_obvs_correlation != obvs_actual).sum().item()/len(obvs_actual)
        ler_simple_bm = (bm_obvs_simple != obvs_actual).sum().item()/len(obvs_actual)
        ler_rl_bm_shared = (bm_obvs_rl_shared != obvs_actual).sum().item()/len(obvs_actual)
        ler_rl_bm_isolated = (bm_obvs_rl_iso != obvs_actual).sum().item()/len(obvs_actual)

        ler_correlation_cm = (cm_obvs_correlation != obvs_actual).sum().item()/len(obvs_actual)
        ler_simple_cm = (cm_obvs_simple != obvs_actual).sum().item()/len(obvs_actual)
        ler_rl_cm_shared = (cm_obvs_rl_shared != obvs_actual).sum().item()/len(obvs_actual)
        ler_rl_cm_isolated = (cm_obvs_rl_iso != obvs_actual).sum().item()/len(obvs_actual)


    log_path = f'log/sc_tn/sycamore/{method}_d{d}r{r}s{sample}_q{coor}_task{task}.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path,'a')
    
    if read_data:
        if method == 'decompose':
            mwpm = MWPM_dem(dem, enable_correlations=True)
            bm = BeliefMatching_dem(dem, max_iter=10)
    
    er = torch.from_numpy(er_sim).to(dtype)
    pertub = torch.rand_like(er).to(dtype)
    init_er = er + (2*torch.bernoulli(torch.ones(len(er)).to(dtype)/2)-1.)*er*pertub
    priors_logits = torch.logit(init_er)

    tn = TensorNetwork(pcm=pcm, priors_logits=priors_logits, dtype=dtype, dev=dev)
    
    if d == 3:
        tn.load_path(filename=tn_contract_path_file)
    elif d==5:
        tn.load_tree(tn_contract_tree_file)
    
    optimizer = torch.optim.Adam(tn.parameters(), lr=lr)

    loss_list = []
    er_list = []
    nb = batch_size//minibatch
    for epoch in range(1, epochs+1):
        losses = []
        forward_time_list = []
        backward_time_list = []

        for j, syndrome_data in enumerate(dataloader):
            if nb>1:
                inputs = syndrome_data[0].reshape(nb, minibatch, syndrome_data[0].size(1))
                loss = 0
                optimizer.zero_grad()
                for k in range(nb):
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    loss_k = tn.forward(inputs[k])/nb
                    torch.cuda.synchronize(dev)
                    t1 = time.perf_counter()
                    loss_k.backward()
                    torch.cuda.synchronize(dev)
                    t2 = time.perf_counter()
                    loss += loss_k.detach().item()
                    
                    # 记录 forward 和 backward 时间
                    forward_time_list.append(t1 - t0)
                    backward_time_list.append(t2 - t1)

                    print(f'epoch {epoch} batch idx {j} mini batch idx {k} \n ft : {t1-t0} bt : {t2-t1}')
                optimizer.step()
                losses.append(loss)

            else:
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                loss = tn.forward(syndrome_data[0])
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter()
                
                optimizer.zero_grad()
                loss.backward()      
                
                torch.cuda.synchronize(dev)
                t2 = time.perf_counter()
                
                # 记录 forward 和 backward 时间
                forward_time_list.append(t1 - t0)
                backward_time_list.append(t2 - t1)
                
                optimizer.step()
                losses.append(loss.detach().cpu().item())
       
            
        if epoch >=10 and abs(np.array(losses).mean() - loss_list[-1])/loss_list[-1] < 1e-12 :
            print(f'loss converge : {np.array(losses).mean()} at epoch {epoch}')
            break

        loss_list.append(np.array(losses).mean())

        oer = torch.sigmoid(tn.priors_logits.detach().cpu())
        er_list.append(oer)

        # 每次 epoch 结束，自动保存 Checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': tn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.array(losses).mean(),
            'oer': oer
        }, ckpt_path)

        print('epoch:', epoch, 'non-zero grad ratio:', (torch.count_nonzero(tn.priors_logits.grad)/len(tn.priors_logits.grad)).item())
        print('loss :', np.array(losses).mean())
        
        # 安全打印时间均值 (防止列表为空时报错)
        mean_ft = np.array(forward_time_list).mean() if forward_time_list else 0.0
        mean_bt = np.array(backward_time_list).mean() if backward_time_list else 0.0
        print('forward time:', mean_ft)
        print('backward time:', mean_bt)

        if epoch<= 20 or epoch % nprint == 0 :
            if method == 'decompose':
                if epoch == 100 :
                    log_file.write(f'epoch : {epoch} loss : {np.array(losses).mean()} logical error rate (cm):{mwpm.logical_error_rate(dets, obvs, oer.numpy())} logical error rate (bm):{bm.logical_error_rate(dets, obvs, oer.numpy())} \n')
                else:
                    log_file.write(f'epoch : {epoch} loss : {np.array(losses).mean()} \n')
            elif method == 'fdecompose':
                log_file.write(f'epoch : {epoch} loss : {np.array(losses).mean()} \n')
            log_file.flush()

        
    log_file.close()
    
    # 最终完整保存
    os.makedirs('data/sycamore/', exist_ok=True)
    torch.save(er_list, f=f'data/sycamore/{method}_d{d}r{r}_s{sample}_q{coor}_task{task}.pt')


if __name__ == "__main__":
    import fire
    fire.Fire({
        'training': optimizing_google_data,
               })