"""
Train d=7 surface code DEM using GroupTN (5 subsampled d=5 TNs).

Workflow:
  1. Generate contraction trees (one-time):
     python script/generate_d7_subsample_trees.py --r 1
  2. Train:
     python script/train_d7_subsample.py --r 1 --epochs 500 --lr 0.01 --dev cuda:0
"""

import sys
import os
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import stim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import GroupTN, get_error_rates, PCM, subsample_d5_pcms, MWPM_dem


def _parse_devices(dev='cuda:0', devices=None):
    if devices is None:
        return [str(dev)]
    if isinstance(devices, str):
        parsed = [item.strip() for item in devices.split(',') if item.strip()]
        return parsed or [str(dev)]
    return [str(item) for item in devices]


def generate_data(d, r, error_prob=0.001, num_shots=500000, seed=75328):
    """Generate simulation data for d=7 surface code."""
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=d,
        rounds=r,
        after_clifford_depolarization=error_prob,
        before_round_data_depolarization=error_prob,
        before_measure_flip_probability=error_prob,
        after_reset_flip_probability=error_prob,
    )
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    er = get_error_rates(dem)
    sampler = dem.compile_sampler(seed=seed)
    dets, obvs, _ = sampler.sample(shots=num_shots)
    dets = torch.from_numpy(dets.astype(np.float64))
    obvs = torch.from_numpy(obvs.astype(np.float64))
    pcm, l = PCM(dem)
    return dets, obvs, pcm, er, dem


def sample_dem_data(dem, num_shots=10000, seed=75329):
    """Sample a fixed evaluation set from a detector error model."""
    sampler = dem.compile_sampler(seed=seed)
    dets, obvs, _ = sampler.sample(shots=num_shots)
    dets = torch.from_numpy(dets.astype(np.float64))
    obvs = torch.from_numpy(obvs.astype(np.float64))
    return dets, obvs


def add_perturbation(er, perturbation_strength=0.3):
    """Add random perturbation to ground truth error rates."""
    er_t = torch.from_numpy(er).to(torch.float64)
    pertub = torch.rand_like(er_t) * perturbation_strength
    direction = 2 * torch.bernoulli(torch.ones(len(er_t)) / 2) - 1
    init_er = er_t + direction * er_t * pertub
    init_er = torch.clamp(init_er, 1e-10, 1.0 - 1e-10)
    return er_t, init_er


def _lr_for_epoch(epoch, initial_lr, final_lr=None, decay_epochs=0):
    """Linearly decay LR from initial_lr to final_lr over decay_epochs epochs."""
    if final_lr is None or decay_epochs is None or decay_epochs <= 1:
        return initial_lr
    if epoch >= decay_epochs:
        return final_lr
    progress = (epoch - 1) / (decay_epochs - 1)
    return initial_lr + progress * (final_lr - initial_lr)


def _set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def _configure_torch_threads(torch_threads=None, torch_interop_threads=None):
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    if torch_interop_threads is not None:
        torch.set_num_interop_threads(int(torch_interop_threads))


def train(r=1, error_prob=0.001, num_shots=500000, epochs=500, lr=0.01,
          batch_size=10000, mini_batch=1000, dev='cuda:0', devices=None,
          perturbation_strength=0.3, seed=75328, decode_interval=10,
          keep_top_k=5, stop_grad_partial=False, partial_only_grad=True,
          path_dir=None, manual_sync_grads=False, sequential_patch_backward=False,
          eval_size=10000, eval_seed=None, lr_final=None, lr_decay_epochs=0,
          torch_threads=None, torch_interop_threads=None):
    """Train GroupTN on d=7 subsampled surface code.

    Prerequisite: run generate_d7_subsample_trees.py --r {r} first to generate
    contraction trees in path/d7r{r}/.
    """
    d = 7
    dtype = torch.float64
    dtype_name = str(dtype).replace('torch.', '')
    run_name = f'd{d}r{r}_subsample'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _configure_torch_threads(torch_threads, torch_interop_threads)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    eval_seed = seed + 1 if eval_seed is None else int(eval_seed)
    lr_decay_epochs = int(lr_decay_epochs)
    device_list = _parse_devices(dev=dev, devices=devices)
    primary_dev = device_list[0]
    path_dir = path_dir or f'path/d{d}r{r}'

    log_dir = f'log/sc_tn/simulation/{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = f'data/simulation/{run_name}/{timestamp}'
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Generate data
    print(f"Generating d={d} r={r} data ({num_shots} shots)...")
    dets, obvs, pcm, er, dem = generate_data(d, r, error_prob, num_shots, seed=seed)
    print(f"  PCM shape: {pcm.shape}")
    print(f"  Ground truth error rates: {er[:5]}...")

    eval_size = int(eval_size)
    if eval_size <= 0:
        raise ValueError("eval_size must be positive.")
    print(f"Generating fixed evaluation data ({eval_size} shots, seed={eval_seed})...")
    eval_dets, eval_obvs = sample_dem_data(dem, num_shots=eval_size, seed=eval_seed)

    # 2. Add perturbation
    er_gt, init_er = add_perturbation(er, perturbation_strength)
    mre_init = (torch.abs(init_er - er_gt) / er_gt).mean().item()
    print(f"  Initial MRE (Mean Relative Error): {mre_init:.4f}")

    # 3. Subsample
    print(f"Subsampling d={d} into d=5 sub-codes...")
    sub_pcms, sub_dets, sub_errors, sub_full_masks = subsample_d5_pcms(
        d,
        r,
        print_info=True,
        return_edge_masks=True,
    )
    missing_trees = [
        f'{path_dir}/subsample_tree_{i}.json'
        for i in range(len(sub_pcms))
        if not os.path.exists(f'{path_dir}/subsample_tree_{i}.json')
    ]
    if missing_trees:
        raise FileNotFoundError(
            "Missing contraction tree files. Generate them first with "
            f"`python script/generate_d7_subsample_trees.py --source simulation --r {r}`. "
            f"Missing: {missing_trees}"
        )

    # 4. Setup data loader
    dataset = TensorDataset(dets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer_steps_per_epoch = len(dataloader)

    # 5. Initialize GroupTN (loads pre-generated contraction trees)
    print(f"\nInitializing GroupTN with {len(sub_pcms)} sub-TNs...")
    model = GroupTN(d=d, r=r, sub_pcms=sub_pcms, sub_dets=sub_dets,
                    sub_errors=sub_errors, init_priors=init_er,
                    dev=primary_dev, devices=device_list, dtype=dtype,
                    sub_full_masks=sub_full_masks,
                    stop_grad_partial=stop_grad_partial,
                    partial_only_grad=partial_only_grad,
                    manual_sync_grads=manual_sync_grads,
                    path_dir=path_dir)

    # 6. Setup optimizer and logging
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    decoder_name = 'MWPM_dem'
    enable_correlations = False
    mwpm = MWPM_dem(dem, enable_correlations=False)

    print(f"Evaluating initial/true LER on fixed eval set ({eval_size} shots)...")
    initial_ler = mwpm.logical_error_rate(eval_dets, eval_obvs, init_er.numpy())
    true_ler = mwpm.logical_error_rate(eval_dets, eval_obvs, er_gt.numpy())
    print(
        f"  Initial Eval LER ({decoder_name}, enable_correlations={enable_correlations}, "
        f"eval_size={eval_size}): {initial_ler:.6f}"
    )
    print(
        f"  True Eval LER ({decoder_name}, enable_correlations={enable_correlations}, "
        f"eval_size={eval_size}): {true_ler:.6f}"
    )

    log_path = f'{log_dir}/{run_name}_{timestamp}.log'
    print(f"Detailed run log: {log_path}")
    log_file = open(log_path, 'w')
    log_file.write(f'# {run_name} training\n')
    log_file.write(f'# timestamp: {timestamp}\n')
    log_file.write(f'# parameters:\n')
    params = {
        'run_name': run_name,
        'timestamp': timestamp,
        'code_task': 'surface_code:rotated_memory_z',
        'd': d,
        'r': r,
        'error_prob': error_prob,
        'after_clifford_depolarization': error_prob,
        'before_round_data_depolarization': error_prob,
        'before_measure_flip_probability': error_prob,
        'after_reset_flip_probability': error_prob,
        'num_shots': num_shots,
        'samples_per_epoch': int(dets.shape[0]),
        'epochs': epochs,
        'lr': lr,
        'lr_final': lr_final,
        'lr_decay_epochs': lr_decay_epochs,
        'batch_size': batch_size,
        'mini_batch': mini_batch,
        'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
        'dev': primary_dev,
        'devices': device_list,
        'dtype': dtype_name,
        'perturbation_strength': perturbation_strength,
        'seed': seed,
        'eval_size': eval_size,
        'eval_seed': eval_seed,
        'eval_source': 'stim DEM sampler, independent fixed eval set',
        'decode_interval': decode_interval,
        'keep_top_k': keep_top_k,
        'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'decoder': decoder_name,
        'enable_correlations': enable_correlations,
        'stop_grad_partial': stop_grad_partial,
        'partial_only_grad': partial_only_grad,
        'manual_sync_grads': manual_sync_grads,
        'sequential_patch_backward': sequential_patch_backward,
        'partial_gradient_stats': getattr(model, 'partial_gradient_stats', None),
        'path_dir': path_dir,
        'log_dir': log_dir,
        'checkpoint_dir': ckpt_dir,
    }
    for key, value in params.items():
        log_file.write(f'#   {key}: {value}\n')
    log_file.write(f'# Initial MRE: {mre_init:.6f}\n')
    log_file.write(f'# Initial Eval LER ({decoder_name}, enable_correlations={enable_correlations}, eval_size={eval_size}): {initial_ler:.6f}\n')
    log_file.write(f'# True Eval LER ({decoder_name}, enable_correlations={enable_correlations}, eval_size={eval_size}):    {true_ler:.6f}\n')
    log_file.flush()

    # 7. Training loop
    loss_list = []
    best_checkpoints = []
    last_epoch = 0

    def _checkpoint_payload(epoch, loss, ler, mre, oer, *, checkpoint_kind, lr_value):
        return {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_schedule_state': {
                'kind': 'linear_decay_then_constant' if lr_final is not None and lr_decay_epochs > 1 else 'constant',
                'initial_lr': lr,
                'final_lr': lr_final,
                'decay_epochs': lr_decay_epochs,
                'current_lr': lr_value,
            },
            'loss': loss,
            'mre': mre,
            'ler': ler,
            'er_gt': er_gt,
            'oer': oer,
            'decoder': decoder_name,
            'enable_correlations': enable_correlations,
            'checkpoint_kind': checkpoint_kind,
            'params': params,
        }

    for epoch in range(1, epochs + 1):
        current_lr = _lr_for_epoch(epoch, lr, final_lr=lr_final, decay_epochs=lr_decay_epochs)
        _set_optimizer_lr(optimizer, current_lr)
        last_epoch = epoch
        losses = []
        epoch_start = time.perf_counter()
        num_batches = len(dataloader)
        for j, (syndrome_batch,) in enumerate(dataloader, start=1):
            syndrome_batch = syndrome_batch.to(primary_dev).to(dtype)

            if mini_batch and mini_batch < syndrome_batch.size(0):
                inputs = syndrome_batch.split(mini_batch, dim=0)
                optimizer.zero_grad()
                batch_loss = 0
                for input_chunk in inputs:
                    loss_scale = 1.0 / len(inputs)
                    if sequential_patch_backward:
                        loss_k = model.sequential_sub_loss_and_grad(
                            input_chunk, loss_scale=loss_scale
                        )
                    elif manual_sync_grads:
                        loss_k = model.manual_sync_loss_and_grad(
                            input_chunk, loss_scale=loss_scale
                        )
                    else:
                        loss_k = model(input_chunk) * loss_scale
                        loss_k.backward()
                    batch_loss += loss_k.detach().item()
                optimizer.step()
            else:
                optimizer.zero_grad()
                if sequential_patch_backward:
                    loss = model.sequential_sub_loss_and_grad(syndrome_batch)
                elif manual_sync_grads:
                    loss = model.manual_sync_loss_and_grad(syndrome_batch)
                else:
                    loss = model(syndrome_batch)
                    loss.backward()
                optimizer.step()
                batch_loss = loss.detach().item()

            losses.append(batch_loss)
            batch_oer = torch.sigmoid(model.priors_logits.detach().cpu())
            batch_mre = (torch.abs(batch_oer - er_gt) / er_gt).mean().item()
            elapsed_hours = (time.perf_counter() - epoch_start) / 3600.0
            print(
                f'epoch {epoch:4d} batch {j:4d}/{num_batches}: '
                f'loss={batch_loss:.6f}, MRE={batch_mre:.6f}, lr={current_lr:.6g}, elapsed={elapsed_hours:.2f}h',
                flush=True,
            )
            log_file.write(
                f'epoch {epoch} batch {j}/{num_batches}: '
                f'loss={batch_loss:.6f}, MRE={batch_mre:.6f}, lr={current_lr:.6g}, elapsed={elapsed_hours:.2f}h\n'
            )
            log_file.flush()

        avg_loss = np.mean(losses)
        loss_list.append(avg_loss)

        oer = torch.sigmoid(model.priors_logits.detach().cpu())
        mre = (torch.abs(oer - er_gt) / er_gt).mean().item()

        # Logging: print every epoch
        print(f'epoch {epoch:4d}: loss={avg_loss:.6f}, MRE={mre:.6f}, lr={current_lr:.6g}')
        log_file.write(f'epoch {epoch}: loss={avg_loss:.6f}, MRE={mre:.6f}, lr={current_lr:.6g}\n')
        log_file.flush()

        if decode_interval and epoch % decode_interval == 0:
            current_ler = mwpm.logical_error_rate(eval_dets, eval_obvs, oer.numpy())
            separator = '-' * 45
            print(separator)
            print(f'epoch {epoch:4d}: {decoder_name}(eval_size={eval_size}, enable_correlations={enable_correlations}) LER={current_ler:.6f}')
            print(separator)
            log_file.write(separator + '\n')
            log_file.write(f'epoch {epoch}: {decoder_name}(eval_size={eval_size}, enable_correlations={enable_correlations}) LER={current_ler:.6f}\n')
            log_file.write(separator + '\n')
            log_file.flush()

            ckpt_path = (
                f'{ckpt_dir}/{run_name}_{timestamp}_'
                f'epoch{epoch}_ler{current_ler:.6f}.pt'
            )
            torch.save(
                _checkpoint_payload(
                    epoch,
                    avg_loss,
                    current_ler,
                    mre,
                    oer,
                    checkpoint_kind='eval',
                    lr_value=current_lr,
                ),
                ckpt_path,
            )
            best_checkpoints.append((current_ler, epoch, ckpt_path))
            best_checkpoints.sort(key=lambda item: (item[0], item[1]))
            while len(best_checkpoints) > keep_top_k:
                _, _, stale_path = best_checkpoints.pop()
                if os.path.exists(stale_path):
                    os.remove(stale_path)

        # Convergence check
        if epoch >= 10 and abs(avg_loss - loss_list[-2]) / abs(loss_list[-2]) < 1e-12:
            print(f'Loss converged at epoch {epoch}: {avg_loss:.6f}')
            log_file.write(f'Loss converged at epoch {epoch}: {avg_loss:.6f}\n')
            log_file.flush()
            break

    log_file.close()

    # 8. Final results
    final_er = torch.sigmoid(model.priors_logits.detach().cpu())
    final_mre = (torch.abs(final_er - er_gt) / er_gt).mean().item()
    final_ler = mwpm.logical_error_rate(eval_dets, eval_obvs, final_er.numpy())
    final_loss = loss_list[-1] if loss_list else None
    final_lr = _lr_for_epoch(last_epoch, lr, final_lr=lr_final, decay_epochs=lr_decay_epochs)
    final_ckpt_path = (
        f'{ckpt_dir}/{run_name}_{timestamp}_'
        f'final_epoch{last_epoch}_ler{final_ler:.6f}.pt'
    )
    torch.save(
        _checkpoint_payload(
            last_epoch,
            final_loss,
            final_ler,
            final_mre,
            final_er,
            checkpoint_kind='final',
            lr_value=final_lr,
        ),
        final_ckpt_path,
    )
    print(f'\n=== Final Results ===')
    print(f'Initial MRE: {mre_init:.6f}')
    print(f'Final MRE:   {final_mre:.6f}')
    print(f'Initial Eval LER: {initial_ler:.6f}')
    print(f'True Eval LER:    {true_ler:.6f}')
    print(f'Final Eval LER:   {final_ler:.6f}')
    print(f'Improvement: {(1 - final_mre / mre_init) * 100:.1f}%')
    log_file = open(log_path, 'a')
    log_file.write('# Final results:\n')
    log_file.write(f'#   Initial MRE: {mre_init:.6f}\n')
    log_file.write(f'#   Final MRE: {final_mre:.6f}\n')
    log_file.write(f'#   Initial Eval LER: {initial_ler:.6f}\n')
    log_file.write(f'#   True Eval LER: {true_ler:.6f}\n')
    log_file.write(f'#   Final Eval LER: {final_ler:.6f}\n')
    log_file.write(f'#   Final checkpoint: {os.path.basename(final_ckpt_path)}\n')
    log_file.write(f'#   Improvement: {(1 - final_mre / mre_init) * 100:.1f}%\n')
    if best_checkpoints:
        log_file.write('# Best checkpoints by LER:\n')
        for ler, epoch, ckpt_path in best_checkpoints:
            log_file.write(f'#   epoch {epoch}: LER={ler:.6f}, file={os.path.basename(ckpt_path)}\n')
    log_file.close()

    return final_mre


if __name__ == '__main__':
    import fire
    fire.Fire(train)
