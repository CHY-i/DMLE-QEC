import sys
import torch
import numpy as np
from os.path import abspath, dirname
sys.path.append(abspath(dirname(dirname(__file__)))+'/src')
import stim
from src import (rep_cir,  
                 PlanarNet, 
                 get_error_rates)


    
    
def main(d, r, error_prob, dev, dtype):
    if (d-1)*(r+1)>=15:
        print('The code is too large for exact optimization!')
        
    else:
        import warnings
        warnings.filterwarnings("ignore", message="Casting complex values to real")
    
        #circuit
        circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                                distance=d,
                                                rounds=r,
                                                after_clifford_depolarization=error_prob,
                                                before_measure_flip_probability=error_prob,
                                                after_reset_flip_probability=error_prob,
                                                )

        # detector error model
        dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        # define the DEM-code
        rep = rep_cir(d, r)
        rep.reorder(dem)

        er = get_error_rates(dem)
        er = torch.tensor(er).to(dev).to(dtype)
        pertub = torch.rand_like(er)
        init_er = (er + (2*torch.bernoulli(torch.ones(len(er)).to(dtype)/2)-1.)*er*pertub) 

        pln = PlanarNet(rep, init_er, dev=dev)

        x_exact = np.array(
        [list(map(int, bin(x)[2:].zfill(rep.hx.shape[0]))) for x in range(2**(rep.hx.shape[0]))]
        ).astype(bool)
        x_exact = torch.from_numpy(x_exact*1.0).to(dtype=dtype)

        with torch.no_grad():
            operators = x_exact @ pln.pebz % 2

            # operators_l = ((x_exact @ pln.pebz)+rep.lz) % 2
            # '''log probabilities and probabilities of each configuration'''
            # logp_0, logp_1 = pln.logp0(operators, er), pln.logp0(operators_l, er)
            # p_0, p_1 = torch.exp(logp_0), torch.exp(logp_1)
            # '''p(s) = p(s, l=0) + p(s, l=1)'''
            # with torch.no_grad():
            #     p = p_0.detach() + p_1.detach()
            # print('normalization: {:5f}'.format(p.sum().item()))
            # logp = torch.log(p_0+p_1)
            # '''nll = - sum p*log(p)'''
            # nll_exact = - (p*logp).sum()
            # print('exact NLL:', nll_exact.item())

            logpl = pln.logp(operators, er)
            p_exact = torch.exp(pln.logp(operators, er))
            nll1 = -(p_exact*logpl).sum()
            print('exact Nll (L):', nll1.item())

        
            

        er_his = pln.test(p_exact, x_exact, 20)

        np.savez('log/exact/d{}r{}p{}.npz'.format(d, r, error_prob), true_er=er, init_er=init_er, er_his=er_his)

        er_opt = torch.sigmoid(pln.para.detach().cpu())


        print('Mean of Relative Errors :', (abs(er_opt-er)/er).mean())
if __name__ == "__main__":
    d = 3 # distance
    r = 5 # rounds
    error_prob = 0.001 # probability of errors generation

    dev = 'cpu'
    dtype=torch.float64

    main(d, r, error_prob, dev, dtype)