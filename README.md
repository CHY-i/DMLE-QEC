# Overview

This repository provides a framework for Differentiable Maximum Likelihood Estimation (MLE) of noise parameters in Quantum Error Correction (QEC).

# Quick Start: PlanarNet Optimization

## 1.Initialization

```python
import stim
from src import rep_cir, PlanarNet, get_error_rates
from torch.utils.data import dataset

d=3
r=3
error_prob=0.001
dtype=torch.float64

# Define simulation circuit
circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                            distance=d,
                                            rounds=r,
                                            after_clifford_depolarization=error_prob,
                                            before_measure_flip_probability=error_prob,
                                            after_reset_flip_probability=error_prob,
                                            )
dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
# Define computational tools
rep = rep_cir(d, r)
rep.reorder(dem)
# Loading error rates from DEM
er = torch.from_numpy(get_error_rates(dem)).to(dtype)
# Draw samples
sampler = dem.compile_sampler(seed=75328)
dets, _, _ = sampler.sample(shots=num_shots)
dataset = TensorDataset(dets)
# Random pertubation
pertub = torch.rand_like(er)
init_er = (er + (2*torch.bernoulli(torch.ones(len(er)).to(torch.float64)/2)-1.)*er*pertub) 

```

## 2.Optimization


```python
import torch
from torch.utils.data import DataLoader

def pl_optimizing(init_er, rep, dataset, dev='cuda:0', 
                           epochs=100, batch_size=100000, mini_batch=10000, lr=0.001):
    
    # Initialize the Planar Network
    planar = PlanarNet(abstract_code=rep, init_priors=init_er, dev=dev)
    planar.to(dev)
    
    # Setup Optimizer and standard DataLoader
    optim = torch.optim.AdamW(planar.parameters(), lr=lr, weight_decay=0.01)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dtype = init_er.dtype
    nb = int(batch_size / mini_batch)
    
    print(f"Starting optimization on {dev}...")
    
    for epoch in range(1, epochs + 1):
        loss_list = []
        
        for det, in dataloader:
            # Reshape into mini-batches
            det = det.reshape(nb, mini_batch, -1)
            
            optim.zero_grad()
            loss_item = 0
            
            # Process mini-batches to manage memory
            for i in range(nb):
                x = det[i].to(dev).to(dtype)
                
                loss = planar(x) / nb 
                loss.backward()
                
                loss_item += loss.detach().cpu().item()
            
            optim.step()
            loss_list.append(loss_item)
            
        if epoch % 20 == 0:
            import numpy as np
            print(f'Epoch: {epoch} | NLL Loss: {np.mean(loss_list):.4f}')
            
    # Extract final optimized error rates
    opt_er = torch.sigmoid(planar.para)
    print("\nOptimization Complete.")
    
    return opt_er

# Run the optimization
optimized_error_rates = pl_optimizing(
    init_er=init_er, 
    rep=rep, 
    dataset=dataset, 
    epochs=100
)

```

# Quick Start: TensorNetwork Optimization

## 1.Initialization

```python
import stim
from src import TensorNetwork, get_error_rates, PCM
from torch.utils.data import dataset

d=3
r=3
error_prob=0.001

# Define simulation circuit
circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z",
                                            distance=d,
                                            rounds=r,
                                            after_clifford_depolarization=error_prob,
                                            before_measure_flip_probability=error_prob,
                                            after_reset_flip_probability=error_prob,
                                            )
dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
# Define computational tools
pcm, l= PCM(dem)
# Loading error rates from DEM
er = torch.from_numpy(get_error_rates(dem)).to(dtype)
# Draw samples
sampler = dem.compile_sampler(seed=75328)
dets, _, _ = sampler.sample(shots=num_shots)
dataset = TensorDataset(dets)
# Random pertubation
pertub = torch.rand_like(er)
init_er = (er + (2*torch.bernoulli(torch.ones(len(er)).to(torch.float64)/2)-1.)*er*pertub) 

```

## 2. Optimization

```python
import torch
from torch.utils.data import DataLoader

def tn_optimizing(dataset, pcm, init_er,
               epochs=100, lr=0.01, batch_size=10000, dev='cuda:0', dtype=torch.float64, nprint=20):

    
    priors_logits = torch.logit(init_er)
    # Define optimizer
    tn = TensorNetwork(pcm=pcm, priors_logits=priors_logits, dtype=dtype, dev=dev)
    tn_contract_path_file = f'path/simulation/d{d}r{r}.pkl'

    # If contraction path not exists
    import os
    if not os.path.exists(tn_contract_path_file):
        path = tn.find_contraction_path(batch_size=10, max_time=120)
        tn.save_path(path, filename=tn_contract_path_file)

    tn.load_path(filename=tn_contract_path_file)

    # Setup Optimizer and standard DataLoader
    optimizer = torch.optim.Adam(tn.parameters(), lr=lr)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    # Training
    for epoch in range(1, epochs+1):
        losses = []
        for j, syndrome_data in enumerate(dataloader):
   
            loss = tn.forward(syndrome_data[0])

            optimizer.zero_grad()
            loss.backward()      
            optimizer.step()

            losses.append(loss.detach().cpu().item())

        if epoch % nprint == 0 or epoch <=20:
            oer = torch.sigmoid(tn.priors_logits.detach().cpu())
            
            print('epoch:', epoch)
            print('loss :', np.array(losses).mean(), 'mre :' {(abs(oer-er)/er).mean()})

    oer = torch.sigmoid(tn.priors_logits.detach().cpu())

    return oer

# Run the optimization
optimized_error_rates = tn_optimizing(
    dataset=dataset, 
    pcm=pcm,
    init_er=init_er,
    epochs=100
)
```