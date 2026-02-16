import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from os.path import abspath, dirname
sys.path.append(abspath(dirname(dirname(__file__)))+'/src')
import stim
from src import (rep_cir,  
                 PlanarNet, 
                 get_error_rates)

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def cleanup():
    dist.destroy_process_group()


def generate_data(d, r, num_shots, cir_type, error_prob=0.001):
    #['simulation', 'baqis', '...']
    if cir_type == 'simulation':
        circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                            distance=d,
                                            rounds=r,
                                            after_clifford_depolarization=error_prob,
                                            before_measure_flip_probability=error_prob,
                                            after_reset_flip_probability=error_prob,
                                            )
        dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        er = get_error_rates(dem)
        sampler = dem.compile_sampler(seed=75328)
        dets, obvs, _ = sampler.sample(shots=num_shots)

    elif cir_type == 'baqis':
        from correlation import experiment_data_qlab
        dets, obvs, er, dem = experiment_data_qlab(distance=d, rounds=r, num_shots=num_shots, initial_state='random', analyze=True, idea = False, old = False)
    dets = torch.from_numpy(dets*1.)
    
    er = torch.tensor(er).to(torch.float64)
    return  dets, dem, er


def initialzation(d=5, ds=3, r=3, 
                    cir_type='simulation', error_prob=0.001, num_shots=100000):
    
    dets, dem, er= generate_data(d=d, r=r, num_shots=num_shots, cir_type=cir_type, error_prob=error_prob)
    dataset = TensorDataset(dets)

    if d==ds:
        print('Normal initialzation')

        if cir_type=='simulation':
            pertub = torch.rand_like(er)
            init_er = (er + (2*torch.bernoulli(torch.ones(len(er)).to(torch.float64)/2)-1.)*er*pertub) 
            rep = rep_cir(d, r)
            rep.reorder(dem)
        else:
            init_er = er
            rep = rep_cir(d, r)

        return er, init_er, rep, dataset
    
    elif d>ds:
        print('Subsample initialzation')
        from src import subsamples
        
        subsamples = subsamples(ds=ds, d=d, r=r, dem=dem)

        circuit_sub = stim.Circuit.generated(code_task="repetition_code:memory",
                                                distance=ds,
                                                rounds=r,
                                                after_clifford_depolarization=error_prob,
                                                before_measure_flip_probability=error_prob,
                                                after_reset_flip_probability=error_prob,
                                                )
        dem_sub = circuit_sub.detector_error_model(decompose_errors=False, flatten_loops=True)

        if cir_type=='simulation':
            
            pertub = torch.rand_like(er)
            init_er = (er+ (2*torch.bernoulli(torch.ones(len(er)).to(torch.float64)/2)-1.)*er*pertub)
            rep = rep_cir(ds, r)
            rep.reorder(dem_sub)
        else:
            init_er = er
            rep = rep_cir(ds, r)
    
        return er, init_er, rep, dataset, subsamples

def optimizing(rank, gpus, 
            init_er, rep, dataset, log_path, 
            epochs=100, batch_size=100000, mini_batch=10000, lr=0.001,
            lnprint=20, enprint=50):
   
    if log_path:
        log_file = open(log_path,'a')
    
    world_size = len(gpus) 
        
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl' 
    torch.cuda.set_device(gpus[rank])
    dev = f'cuda:{gpus[rank]}'
    dtype = init_er.dtype
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method='env://')

    planar = PlanarNet(abstract_code=rep, init_priors=init_er, dev=dev)
    opt = DDP(planar, device_ids=[gpus[rank]])

    optim = torch.optim.AdamW(opt.parameters(), lr=lr, weight_decay=0.01)

    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler)

    nb = int(batch_size/mini_batch)
    for epoch in range(1, 1+epochs):
        sampler.set_epoch(epoch)
        loss_list = []
        for _, det in enumerate(dataloader):
            det = det[0]
            det = det.reshape(nb, mini_batch, -1)

            optim.zero_grad()
            loss_item = 0
            for i in range(nb):
                x = det[i].to(dev).to(dtype)
                loss = opt.module.forward(x) / nb
                loss.backward()
                with torch.no_grad():
                    global_loss = loss.clone()  
                    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM) 
                    global_loss = global_loss/world_size
                    loss_item += global_loss.detach().cpu().item()
    
            optim.step()
            loss_list.append(loss_item)

        if rank == 0:
            if lnprint!=None and epoch%lnprint == 0:
                print('epoch:{}, nll:{}'.format(epoch, np.array(loss_list).mean()))

                log_file.write('epoch:{}, nll:{} \n '.format(epoch, np.array(loss_list).mean()))
                log_file.flush()

        if enprint!=None and epoch%enprint == 0:
                
            opt_er = torch.sigmoid(opt.module.para)
            log_file.write('optimized error rate : \n {} \n'.format(repr(opt_er.detach().cpu().numpy())))
            log_file.flush()

        # dist.barrier()
    if rank == 0:
        if epochs % enprint != 0:
            opt_er = torch.sigmoid(opt.module.para)
            log_file.write('optimized error rate : \n {} \n'.format(repr(opt_er.detach().cpu().numpy())))
            log_file.close()
    else:
        log_file.close()

def subsample_optimizing(rank, gpus, 
                        init_er, rep, dataset, subsamples, 
                        log_path, sub_list=[0, 1, 2],
                        epochs=100, batch_size=100000, mini_batch=10000, lr=0.001,
                        lnprint=20, enprint=50):
    
    if log_path:
        log_file = open(log_path,'a')

    world_size = len(gpus) 
        
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl' 
    torch.cuda.set_device(gpus[rank])
    dev = f'cuda:{gpus[rank]}'
    dtype = init_er.dtype
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method='env://')

    
    
    planar = PlanarNet(abstract_code=rep, init_priors=init_er, dev=dev)
    opt = DDP(planar, device_ids=[gpus[rank]])

    optim = torch.optim.AdamW(opt.parameters(), lr=lr, weight_decay=0.01)

    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler)

    nb = int(batch_size/mini_batch)
    

    for epoch in range(1, 1+epochs):
        sampler.set_epoch(epoch)
        loss_list = []
        for _, det in enumerate(dataloader):
            det = det[0]

            det = det.reshape(nb, mini_batch, -1)

            optim.zero_grad()
            loss_item = 0
            for i in range(nb):
                x = det[i].to(dev).to(dtype)
                loss = opt.module.subsample_forward(x, subsamples=subsamples, sub_list=sub_list) / nb
                loss.backward()
                with torch.no_grad():
                    global_loss = loss.clone()  
                    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM) 
                    global_loss = global_loss/world_size
                    loss_item += global_loss.detach().cpu().item()
    
            optim.step()
            loss_list.append(loss_item)

        if rank == 0:
            if lnprint!=None and epoch%lnprint == 0:
                print('epoch:{}, nll:{}'.format(epoch, np.array(loss_list).mean()))

                log_file.write('epoch:{}, nll:{} \n '.format(epoch, np.array(loss_list).mean()))
                log_file.flush()

        if enprint!=None and epoch%enprint == 0:
                
            opt_er = torch.sigmoid(opt.module.para)
            log_file.write('optimized error rate : \n {} \n'.format(repr(opt_er.detach().cpu().numpy())))
            log_file.flush()

        # dist.barrier()
    if rank == 0:
        if epochs % enprint != 0:
            opt_er = torch.sigmoid(opt.module.para)
            log_file.write('optimized error rate : \n {} \n'.format(repr(opt_er.detach().cpu().numpy())))
            log_file.close()
    else:
        log_file.close()

    cleanup()
    
def main(gpus=[3], 
        d=11, ds=11, r=11, sub_list=[0, 1, 2],#[0, 2 ,4, 6, 8], #
        cir_type='simulation', error_prob=0.001, num_shots=500000,
        epochs=500, batch_size=12500, mini_batch=1250, lr=0.001,
        lnprint=1, enprint=100):
    
    
    
    if d==ds:
        log_path = 'log/'+cir_type+'/d{}r{}er{}.txt'.format(d, r, error_prob)
        er, init_er, rep, dataset = initialzation(d=d, ds=ds, r=r, cir_type=cir_type, error_prob=error_prob, num_shots=num_shots)
        
        log_file = open(log_path,'a')
        log_file.write(f'# Code Parameters : d:{d} r:{r} \n')
        log_file.write(f'# Training Parameters : Num Shots:{num_shots} epochs:{epochs} batch_size:{batch_size} lr:{lr}\n')

        if cir_type == 'simulation':
            log_file.write('true error rate : \n {} \n '.format(repr(er.detach().cpu().numpy())))
            log_file.write('initial error rate : \n {} \n '.format(repr(init_er.detach().cpu().numpy())))
            log_file.flush()
        elif cir_type == 'baqis':
            log_file.write('initial error rate : \n {} \n '.format(repr(er.detach().cpu().numpy())))
            log_file.flush()


        mp.spawn(optimizing, args=(gpus, init_er, rep, dataset, log_path,
                                        epochs, batch_size, mini_batch, lr,
                                        lnprint, enprint), nprocs=len(gpus), join=True)
        

    elif d>ds:
        log_path = 'log/'+cir_type+'/sub_d{}ds{}r{}er{}.txt'.format(d, ds, r, error_prob)
        log_file = open(log_path,'a')
        log_file.write(f'# Code Parameters : d:{d} ds:{ds} r:{r} sub_list {sub_list}\n')
        log_file.write(f'# Training Parameters : Num Shots:{num_shots} epochs:{epochs} batch_size:{batch_size} lr:{lr}\n')
        log_file.flush()

        er, init_er, rep, dataset, subsamples = initialzation(d=d, ds=ds, r=r, cir_type=cir_type, error_prob=error_prob, num_shots=num_shots)
        if cir_type == 'simulation':
            log_file.write('true error rate : \n {} \n '.format(repr(er.detach().cpu().numpy())))
            log_file.write('initial error rate : \n {} \n '.format(repr(init_er.detach().cpu().numpy())))
            log_file.flush()
        elif cir_type == 'baqis':
            log_file.write('true error rate : \n {} \n '.format(repr(er.detach().cpu().numpy())))
            log_file.flush()

        mp.spawn(subsample_optimizing, args=(gpus, init_er, rep, dataset, subsamples, log_path, sub_list,
                                        epochs, batch_size, mini_batch, lr,
                                        lnprint, enprint), nprocs=len(gpus), join=True)

            
    
    
    log_file.close()




if __name__ == '__main__':
    
    import fire
    
    fire.Fire({
        'main': main}
        )