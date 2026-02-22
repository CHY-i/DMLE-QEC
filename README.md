# Overview

This repository provides a framework for Differentiable Maximum Likelihood Estimation (MLE) of noise parameters in Quantum Error Correction (QEC).

# Quick Start: PlanarNet Optimization

## 1.Initialization

<pre>
import stim
from src import rep_cir, PlanarNet, get_error_rates

d=3
r=3
error_prob=0.001

# define simulation circuit
circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                            distance=d,
                                            rounds=r,
                                            after_clifford_depolarization=error_prob,
                                            before_measure_flip_probability=error_prob,
                                            after_reset_flip_probability=error_prob,
                                            )
dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
# define computational tools
rep = rep_cir(d, r)
rep.reorder(dem)
# loading error rates from DEM
er = get_error_rates(dem)
# draw samples
sampler = dem.compile_sampler(seed=75328)
dets, _, _ = sampler.sample(shots=num_shots)
dataset = TensorDataset(dets)
# random pertubation
pertub = torch.rand_like(er)
init_er = (er + (2*torch.bernoulli(torch.ones(len(er)).to(torch.float64)/2)-1.)*er*pertub) 

</pre>

## 2.Optimization

<pre>

def optimizing(init_er, rep, dataset, dev='cuda:0', 
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
        
        for det, in dataloader: # Unpack the tuple from TensorDataset
            # Reshape into mini-batches
            det = det.reshape(nb, mini_batch, -1)
            
            optim.zero_grad()
            loss_item = 0
            
            # Process mini-batches to manage memory
            for i in range(nb):
                x = det[i].to(dev).to(dtype)
                
                # Forward pass
                loss = planar(x) / nb 
                loss.backward()
                
                loss_item += loss.detach().cpu().item()
            
            optim.step()
            loss_list.append(loss_item)
            
        # Logging
        if epoch % 20 == 0:
            import numpy as np
            print(f'Epoch: {epoch} | NLL Loss: {np.mean(loss_list):.4f}')
            
    # Extract final optimized error rates
    opt_er = torch.sigmoid(planar.para)
    print("\nOptimization Complete.")
    
    return opt_er

# 2. Run the optimization
optimized_error_rates = optimizing(
    init_er=init_er, 
    rep=rep, 
    dataset=dataset, 
    dev=device,
    epochs=100
)

</pre>

