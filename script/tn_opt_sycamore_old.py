import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import stim
from src import (TensorNetwork,
                 get_error_rates,
                 PCM,
                 MWPM_dem,
                 BeliefMatching_dem,
                 TensorNetworkDecoder)

def get_or_create_contraction_path(tn, path_file, minibatch=50, max_time=120):
    """
    检查指定的 contraction path 文件是否存在。
    如果不存在，则调用 tn 的方法寻找路径，满足复杂度限制才保存。
    """
    if not os.path.exists(path_file):
        print(f"  --> Path file '{path_file}' 不存在. 开始使用 cotengra 寻找最佳收缩路径...")
        os.makedirs(os.path.dirname(path_file), exist_ok=True)
        
        # 寻找路径
        path = tn.find_contraction_path(batch_size=minibatch, max_time=max_time)
        
        # 判断返回的 path 是否有效 (即 space complexity 是否 < 30)
        if path is not None:
            tn.save_path(path, filename=path_file)
            print(f"  --> 成功找到并保存健康的收缩路径！")
        else:
            # 如果找不到符合要求的路径，直接抛出异常中断程序
            raise RuntimeError(
                "Cotengra 未能在规定时间内找到 Space Complexity < 30 的路径。\n"
                "建议尝试以下方案：\n"
                "1. 增大 max_time 让它搜索更久。\n"
                "2. 减小 minibatch 的大小。\n"
                "3. 确保已经安装了 optuna 以提升搜索质量。"
            )
    else:
        print(f"  --> Path file '{path_file}' 已存在，将直接加载。")


# 3_5 5_3 5_7 7_5
def train_sycamore_old(basis='X', r='03', center='3_5', epochs=100, lr=0.01, batch_size=10000, minibatch=100, nprint=10, dev='cuda:3'):
    """
    极简版 TN 训练函数 (附带 LER 验证与日志记录)
    :param basis: 'X' 或 'Z'
    :param nprint: 每隔多少 epoch 打印/记录一次日志
    """
    print(f"========== 开始训练 Basis: {basis} ==========")
    
    # 1. 相对路径设置
    data_dir = f'data/sycamore_old/surface_code_b{basis}_d3_r{r}_center_{center}'
    ckpt_dir = f'data/checks_old/d3/r{r}_basis_{basis}_c{center}'
    log_dir  = f'log/sc_tn/sycamore_old/d3/r{r}_basis_{basis}_c{center}'
    # os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化日志文件
    log_path = f'{log_dir}/training_log.txt'
    log_file = open(log_path, 'a')

    # 2. 读取 DEM 并获取 detector 数量
    dem = stim.DetectorErrorModel.from_file(f'{data_dir}/circuit_detector_error_model.dem')
    num_detectors = dem.num_detectors
    er_sim = get_error_rates(dem)
    
    # 3. 直接使用 stim 读取数据
    b8_path = f'{data_dir}/detection_events.b8'
    obvs_path = f'{data_dir}/obs_flips_actual.01'

    tn_obvs_path = f'{data_dir}/obs_flips_predicted_by_tensor_network_contraction.01'
    bm_obvs_path = f'{data_dir}/obs_flips_predicted_by_belief_matching.01'
    cm_obvs_path = f'{data_dir}/obs_flips_predicted_by_correlated_matching.01'
    
    dets_np = stim.read_shot_data_file(path=b8_path, format="b8", num_detectors=num_detectors, bit_packed=False)
    total_shots = len(dets_np)
    print(f"  Total shots in file: {total_shots:,}")
    
    
    dets = torch.from_numpy(dets_np.astype(np.float64))
    obvs = stim.read_shot_data_file(path=obvs_path, format="01", num_detectors=1, bit_packed=False).flatten()
    
    obvs_tn = stim.read_shot_data_file(path=tn_obvs_path, format="01", num_detectors=1, bit_packed=False).flatten()
    obvs_bm = stim.read_shot_data_file(path=bm_obvs_path, format="01", num_detectors=1, bit_packed=False).flatten()
    obvs_cm = stim.read_shot_data_file(path=cm_obvs_path, format="01", num_detectors=1, bit_packed=False).flatten()

    ler_tn_google = np.sum(obvs != obvs_tn) / obvs.shape[0]
    ler_bm_google = np.sum(obvs != obvs_bm) / obvs.shape[0]
    ler_cm_google = np.sum(obvs != obvs_cm) / obvs.shape[0]

    log_file.write(f'Read from data --- ler_tn : {ler_tn_google:.8f} , ler_cm : {ler_cm_google:.8f} , ler_bm : {ler_bm_google:.8f} \n')
    log_file.flush()

    # 5. 初始化数据集
    pcm, l = PCM(dem)
    dataset = TensorDataset(dets)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    # 6. 初始化错误率与张量网络
    
    init_er = torch.from_numpy(er_sim).to(torch.float64)
    priors_logits = torch.logit(init_er)

    tn = TensorNetwork(pcm=pcm, priors_logits=priors_logits, dtype=torch.float64, dev=dev)
    tn_decoding = TensorNetwork(pcm=pcm, l=l.flatten(), dtype=torch.float64, dev=dev, decoding=True)
    
    path_file = f"path/sycamore_old/d3_r{r}_{basis}.pkl"
    get_or_create_contraction_path(tn, path_file, minibatch=minibatch, max_time=120)
    tn.load_path(path_file)
    tn_decoding.load_path(path_file)

    decoder = TensorNetworkDecoder(model=tn_decoding, dev=dev)
    # mwpm = MWPM_dem(dem, enable_correlations=True)
    # bm = BeliefMatching_dem(dem, max_iter=10)

    # ler_cm = mwpm.logical_error_rate(dets_np, obvs, er_sim)
    # ler_bm = bm.logical_error_rate(dets_np, obvs, er_sim)
    
    with torch.no_grad():
        if int(r) < 19:
            ler_tn = decoder.logical_error_rate(
                torch.from_numpy(dets_np).to(dev).to(torch.float64), 
                torch.from_numpy(obvs).to(dev).to(torch.float64), 
                torch.from_numpy(er_sim).to(dev).to(torch.float64)
                                                                    )
        else:
            total_shots = dets_np.shape[0]
            total_errors = 0.0
            er_sim_tensor = torch.from_numpy(er_sim).to(dev).to(torch.float64)
            
            for i in range(0, total_shots, minibatch):
                end_i = min(i + minibatch, total_shots)
                current_batch_size = end_i - i
                
                batch_dets = torch.from_numpy(dets_np[i:end_i]).to(dev).to(torch.float64)
                batch_obvs = torch.from_numpy(obvs[i:end_i]).to(dev).to(torch.float64)
                batch_ler = decoder.logical_error_rate(batch_dets, batch_obvs, er_sim_tensor)
                
                total_errors += batch_ler * current_batch_size
            ler_tn = total_errors / total_shots
    log_file.write(f'Decoding from dem --- ler_tn : {ler_tn:.8f}\n') # , ler_cm : {ler_cm:.8f} , ler_bm : {ler_bm:.8f} 
    log_file.flush()
    
    optimizer = torch.optim.Adam(tn.parameters(), lr=lr)

    # 7. 开始训练循环
    loss_list = []
    er_list = []
    nb = batch_size // minibatch
    
    for epoch in range(1, epochs + 1):
        losses = []
        
        for j, syndrome_data in enumerate(dataloader):
            inputs = syndrome_data[0]
            
            if nb > 1:
                inputs = inputs.reshape(nb, minibatch, inputs.size(1))
                loss_accum = 0
                optimizer.zero_grad()
                
                for k in range(nb):
                    loss_k = tn.forward(inputs[k]) / nb
                    loss_k.backward()
                    loss_accum += loss_k.detach().item()
                    
                optimizer.step()
                losses.append(loss_accum)
            else:
                optimizer.zero_grad()
                loss = tn.forward(inputs)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())
                
        # 计算当前 Epoch 的平均 Loss
        avg_loss = np.mean(losses)
        loss_list.append(avg_loss)
        
        oer = torch.sigmoid(tn.priors_logits.detach().cpu())
        er_list.append(oer)

        # 打印控制台日志
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

        # 早停机制 (Early Stopping)
        if epoch >= 10 and abs(avg_loss - loss_list[-2]) / loss_list[-2] < 1e-12:
            print(f"Loss converged at epoch {epoch}!")
            # 如果触发早停，强制将当前 epoch 设为验证节点，保证输出 LER
            epochs = epoch 

        # =============== 核心测试与日志写入逻辑 ===============
        if epoch <= 20 or epoch % nprint == 0 or epoch == epochs:
            oer_np = oer.numpy()
            
            # 💡 修复 2：获取当前最新训练出的错误率，准备传给解码器
            current_er_tensor = oer.to(dev).to(torch.float64)
            
            # 💡 修复 1：加上 torch.no_grad()，这是防止验证期爆显存的关键！
            with torch.no_grad():
                if int(r) < 19:
                    ler_tn = decoder.logical_error_rate(
                        torch.from_numpy(dets_np).to(dev).to(torch.float64), 
                        torch.from_numpy(obvs).to(dev).to(torch.float64), 
                        current_er_tensor  # <--- 注意这里换成了 current_er_tensor
                    )
                else:
                    total_shots = dets_np.shape[0]
                    total_errors = 0.0
                    
                    for i in range(0, total_shots, minibatch):
                        end_i = min(i + minibatch, total_shots)
                        current_batch_size = end_i - i
                        
                        batch_dets = torch.from_numpy(dets_np[i:end_i]).to(dev).to(torch.float64)
                        batch_obvs = torch.from_numpy(obvs[i:end_i]).to(dev).to(torch.float64)
                        
                        batch_ler = decoder.logical_error_rate(batch_dets, batch_obvs, current_er_tensor)
                        
                        total_errors += batch_ler * current_batch_size
                        
                        # 💡 修复 3：手动删除 batch 变量，确保当前 batch 显存在进入下一个循环前被立刻释放
                        del batch_dets, batch_obvs
                        # 如果你的显存真的处于极限边缘，可以把下面这句解开注释：
                        # torch.cuda.empty_cache() 
                        
                    ler_tn = total_errors / total_shots
                    
            log_msg = f'epoch : {epoch} loss : {avg_loss:.8f} \n' #logical error rate (tn):{ler_tn:.8f} \n' 
            log_file.write(log_msg)
            log_file.flush()
        # ====================================================

        # 保存 Checkpoint
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': tn.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': avg_loss,
        #     'oer': oer
        # }, os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pt'))

        # 如果早停触发，跳出循环
        if epoch == epochs and len(loss_list) > 1 and abs(avg_loss - loss_list[-2]) / loss_list[-2] < 1e-12:
            break

    # 关闭日志文件
    log_file.close()

    # 8. 训练结束，保存最终结果
    save_dir = 'data/sycamore_old/processed_results'
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = f'{save_dir}/d3_r{r}_b{basis}_c{center}.pt'
    torch.save(er_list, final_save_path)
    print(f"训练完成！结果已保存至: {final_save_path}\n")

if __name__ == "__main__":
    import fire
    fire.Fire({
        'training': train_sycamore_old,
               })