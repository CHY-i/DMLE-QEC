from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import opt_einsum as oe
from pymatching import Matching
from .utils import construct_kac_ward_solution, torch_rep_cir_log_coset_p, rep_cir, get_error_rates, generate_compactified_pcm_from_seperated_dem


class PlanarNet(nn.Module):
    """Planar repetition-code NLL; optional external DEM priors (e.g. from GateNoiseToDEM)."""

    def __init__(
        self,
        abstract_code,
        init_priors,
        *,
        dev: str = "cpu",
        param_mode: str = "logit",
        log_scale: float = 1.0,
        learn_priors: bool = True,
    ) -> None:
        super().__init__()
        if abstract_code is None or init_priors is None:
            raise ValueError("abstract_code and init_priors are required")

        self.dev = dev
        self.param_mode = param_mode
        self.log_scale = log_scale
        self.learn_priors = learn_priors
        self.dtype = init_priors.dtype
        self.num_priors = int(init_priors.numel())

        self.generators = np.concatenate([abstract_code.hz, abstract_code.lz.reshape(1, -1)], axis=0)
        self.kwz, self.edges_dict_z = construct_kac_ward_solution(self.generators)
        self.pebz = torch.from_numpy(abstract_code.pebz).to(self.dtype).to(self.dev)
        self.lz = torch.from_numpy(abstract_code.lz).to(self.dtype).to(self.dev)

        init_priors = init_priors.detach().to(self.dtype).to(self.dev)
        if param_mode == "logit":
            init_param = torch.log(init_priors / (1.0 - init_priors))
        elif param_mode == "log_prior":
            init_priors_clamped = torch.clamp(init_priors, 1e-10, 1.0 - 1e-10)
            init_param = torch.log(init_priors_clamped) / log_scale
        else:
            raise ValueError(f"Unknown param_mode: {param_mode}. Must be 'logit' or 'log_prior'")

        if learn_priors:
            self.para = nn.Parameter(init_param.clone())
        else:
            self.register_buffer("para", init_param.clone())

    def get_priors(self) -> torch.Tensor:
        """Learnable DEM error probabilities (only when learn_priors=True)."""
        if not self.learn_priors:
            raise RuntimeError("learn_priors=False: use forward(..., priors=external_dem)")
        if self.param_mode == "logit":
            priors = torch.sigmoid(self.para) + 1e-20
        elif self.param_mode == "log_prior":
            priors = torch.exp(self.para * self.log_scale) + 1e-20
            priors = torch.clamp(priors, 1e-20, 1.0 - 1e-20)
        else:
            raise ValueError(f"Unknown param_mode: {self.param_mode}")
        return priors

    def _resolve_priors(self, priors: torch.Tensor | None) -> torch.Tensor:
        if priors is None:
            return self.get_priors()
        if priors.ndim != 1:
            raise ValueError(f"priors must be 1-D, got shape {tuple(priors.shape)}")
        if priors.numel() != self.num_priors:
            raise ValueError(
                f"priors length {priors.numel()} != num_priors {self.num_priors} "
                "(must match rep_cir.reorder'd DEM / g2dem.num_dem)"
            )
        return priors.to(device=self.pebz.device, dtype=self.dtype).clamp(1e-20, 1.0 - 1e-20)

    def logp(self, operator: torch.Tensor, error_rates: torch.Tensor) -> torch.Tensor:
        return torch_rep_cir_log_coset_p(operator, self.kwz, self.edges_dict_z, error_rates=error_rates)

    def forward(self, det: torch.Tensor, priors: torch.Tensor | None = None) -> torch.Tensor:
        """
        Negative log-likelihood of detection events.

        When learn_priors=False, pass differentiable DEM from g2dem: planar(det, priors=g2d()).
        """
        error_rates = self._resolve_priors(priors)
        with torch.no_grad():
            operator = (det * 1.0) @ self.pebz % 2
        logp = self.logp(operator, error_rates)
        return -logp.mean()

    def cal_eloss(self, p_exact: torch.Tensor, x_exact: torch.Tensor, priors: torch.Tensor | None = None):
        error_rates = self._resolve_priors(priors)
        with torch.no_grad():
            operator = x_exact @ self.pebz % 2
        logp = self.logp(operator, error_rates)
        return -(p_exact * logp).sum()

    def test(self, p_exact, x_exact, nprint=None):
        if not self.learn_priors:
            raise RuntimeError("test() requires learn_priors=True")
        optim = torch.optim.AdamW([self.para], lr=0.01)
        er_his = []
        loss_his = []
        epochs = 500
        for epoch in range(1, 1 + epochs):
            loss = self.cal_eloss(p_exact, x_exact)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if nprint is not None and epoch % nprint == 0:
                print(
                    f"epoch:{epoch}, nll:{loss.item()}, grad_mean:{self.para.grad.mean().item()}"
                )
                loss_his.append(loss.detach().cpu().item())
                with torch.no_grad():
                    er_his.append(self.get_priors().detach().cpu())
        return er_his



def xor_tensor(degree, dtype):
    grids = torch.meshgrid(*[torch.arange(2) for _ in range(degree)], indexing='ij')
    total_sum = sum(grids)
    xor_tensor = (total_sum % 2 == 1).to(dtype)
    return xor_tensor

def prob_tensor(degree, p):
    shape = [2]*degree
    copy_ten = torch.zeros(shape, dtype=p.dtype, device=p.device)
    copy_ten[(0,)*degree] = 1.-p
    copy_ten[(1,)*degree] = p
    return copy_ten



def hadamard_tensor(degree, dtype):
    val = torch.tensor([[1, 1], [1, -1]], dtype=dtype) / (2 ** 0.5)
    return val

def prob_tensor(degree, p, connect_to_l=False):
    shape = [2]*degree
    copy_ten = torch.zeros(shape, dtype=p.dtype, device=p.device)
    copy_ten[(0,)*degree] = 1.-p
    if connect_to_l:
        copy_ten[(1,)*degree] = -p
    else:
        copy_ten[(1,)*degree] = p
    return copy_ten


class TensorNetwork(nn.Module):
    def __init__(
        self, 
        pcm, 
        l=None, 
        priors_logits=None, 
        dev='cpu', 
        dtype=torch.float32, 
        decoding=False,
        learn_priors=True  # [新增] 决定本层参数是否可学习
    ):
        super().__init__()
        self.pcm = pcm
        self.l = l
        self.dev = dev
        self.dtype = dtype
        self.decoding = decoding
        self.learn_priors = learn_priors # [新增]
        self.n_check, self.n_bit = pcm.shape
        
        # [修改] 仿照 PlanarNet 处理参数：可学习时用 Parameter，否则用 buffer
        if priors_logits is None:
            init_logits = torch.randn(self.n_bit, dtype=dtype, device=dev)
        else:
            init_logits = priors_logits.detach().to(dev).to(dtype)
            
        if self.learn_priors:
            self.priors_logits = nn.Parameter(init_logits.clone())
        else:
            self.register_buffer("priors_logits", init_logits.clone())
            
        self.path = None
        self.tree = None

        self.generate_xor_tensors()
        self.generate_equation()

    # [新增] 仿照 PlanarNet 解析优先概率
    def _resolve_priors(self, priors: torch.Tensor | None) -> torch.Tensor:
        """
        处理内部或外部传入的先验概率。
        如果是外部传入（如 GateNoiseToDEM 输出），假定其已经是概率空间 [0, 1] 内的值。
        如果是内部参数，则走 sigmoid 激活。
        """
        if priors is not None:
            # 使用外部传入的底层概率（不需要再过 sigmoid）
            return priors.to(device=self.dev, dtype=self.dtype).clamp(1e-20, 1.0 - 1e-20)
        
        if not self.learn_priors:
            raise RuntimeError("learn_priors=False: you must provide external priors (probs) during forward()")
            
        # 使用内部学习的对数几率转概率
        return torch.sigmoid(self.priors_logits).clamp(1e-20, 1.0 - 1e-20)

    def generate_xor_tensors(self):
        degree_xor = self.pcm.sum(axis=1)
        self.xor_list = []
        
        H_base = hadamard_tensor(0, self.dtype).to(self.dev)
        self.register_buffer('shared_H', H_base)
        
        for i in range(self.n_check):
            deg = int(degree_xor[i]) + 1 
            correction_factor = torch.tensor(2.0 ** (deg / 2.0 - 1.0), dtype=self.dtype, device=self.dev)
            scaled_H = self.shared_H * correction_factor
            name = f'hadamard_scaled_{i}'
            self.register_buffer(name, scaled_H)
            for _ in range(deg - 1):
                self.xor_list.append(self.shared_H)
            self.xor_list.append(getattr(self, name))

    def generate_equation(self):
        rows, cols = torch.where(torch.tensor(self.pcm)==1)
        self.edge_map = {} 
        symbol_counter = 0
        check_conn_symbols = [[] for _ in range(self.n_check)]
        bit_conn_symbols = [[] for _ in range(self.n_bit)]
        
        for r, c in zip(rows.tolist(), cols.tolist()):
            sym = oe.get_symbol(symbol_counter)
            self.edge_map[(r, c)] = sym
            check_conn_symbols[r].append(sym)
            bit_conn_symbols[c].append(sym)
            symbol_counter += 1

        syndrome_symbols = []
        for i in range(self.n_check):
            sym = oe.get_symbol(symbol_counter)
            check_conn_symbols[i].append(sym) 
            syndrome_symbols.append(sym)
            symbol_counter += 1
            
        lhs_terms = []
        for i in range(self.n_check):
            edges = check_conn_symbols[i]
            hyper_sym = oe.get_symbol(symbol_counter)
            symbol_counter += 1
            for edge_sym in edges:
                lhs_terms.append(edge_sym + hyper_sym)

        for syms in bit_conn_symbols:
            lhs_terms.append("".join(syms))
            
        for sym in syndrome_symbols:
            lhs_terms.append('...'+sym)
            
        lhs = ",".join(lhs_terms)
        rhs = "..." 
        
        self.eq_str = f"{lhs}->{rhs}"
        
    def generate_prob_tensors(self, probs):
        tensor_list = []
        degree_prob = self.pcm.sum(axis=0)
        for j in range(self.n_bit):
            degree = int(degree_prob[j])
            p = probs[j]
            if self.decoding and self.l[j] == 1:
                pt = prob_tensor(degree, p, connect_to_l=True).to(self.dev).to(self.dtype)
            else:
                pt = prob_tensor(degree, p).to(self.dev).to(self.dtype)
            tensor_list.append(pt)
        return tensor_list
    
    # [修改] 接口参数改为 priors
    def decoding_forward(self, syndromes, priors=None):
        path = self.path
        tree = self.tree
        is_batched = syndromes.ndim == 2
        if not is_batched:
            syndromes = syndromes.unsqueeze(0)
        syndrome_onehot = torch.nn.functional.one_hot(syndromes.long(), num_classes=2).to(dtype=self.dtype, device=self.dev) 
        syndrome_vectors = list(syndrome_onehot.unbind(dim=1))  
        
        # [修改] 使用 _resolve_priors 统一解析概率
        probs = self._resolve_priors(priors)
        
        scaled_prob_tensors = []
        log_scale_factor = 0.0 
        raw_prob_tensors = self.generate_prob_tensors(probs)
        
        for t in raw_prob_tensors:
            max_val = t.abs().max().detach()
            if max_val < 1e-12:
                max_val = torch.tensor(1.0, device=self.dev, dtype=self.dtype)
            scaled_t = t / max_val
            scaled_prob_tensors.append(scaled_t)
            log_scale_factor = log_scale_factor + torch.log(max_val)

        operands = self.xor_list + scaled_prob_tensors + syndrome_vectors
        
        optimize_arg = path if path is not None else 'auto'
        # Contraction
        if tree is not None:
            from .utils import contract
            p0_minus_p1 = contract(tree['tree'], operands)
        else:
            p0_minus_p1 = oe.contract(self.eq_str, *operands, optimize=optimize_arg)
        return (1-p0_minus_p1.sign())/2

    # [修改] 接口参数从 priors_logits 改为 priors
    def forward(self, syndromes, priors=None):
        path = self.path
        tree = self.tree
        is_batched = syndromes.ndim == 2
        if not is_batched:
            syndromes = syndromes.unsqueeze(0)   

        syndrome_onehot = torch.nn.functional.one_hot(syndromes.long(), num_classes=2).to(dtype=self.dtype, device=self.dev)
        syndrome_vectors = list(syndrome_onehot.unbind(dim=1))
        
        # [修改] 使用 _resolve_priors 统一解析概率
        probs = self._resolve_priors(priors)
        
        scaled_prob_tensors = []
        log_scale_factor = 0.0 
        raw_prob_tensors = self.generate_prob_tensors(probs)
        
        for t in raw_prob_tensors:
            max_val = t.abs().max().detach()
            if max_val < 1e-12:
                max_val = torch.tensor(1.0, device=self.dev, dtype=self.dtype)
            scaled_t = t / max_val
            scaled_prob_tensors.append(scaled_t)
            log_scale_factor = log_scale_factor + torch.log(max_val)

        operands = self.xor_list + scaled_prob_tensors + syndrome_vectors
        optimize_arg = path if path is not None else 'auto'
        # Contraction
        if tree is not None:
            from .utils import contract
            result_normalized = contract(tree['tree'], operands)
        else:
            result_normalized = oe.contract(self.eq_str, *operands, optimize=optimize_arg)
        eps = 1e-30
        
        log_likelihood = torch.log(result_normalized + eps) + log_scale_factor
            
        return - log_likelihood.squeeze(0).mean()
    

    def find_contraction_path(self, batch_size=50, max_time=600):
        # 保持不变...
        import cotengra as ctg
        shapes = []

        # 1. XOR Tensors
        for _ in self.xor_list:
            shapes.append((2, 2))

        # 2. Probability Tensors
        degree_prob = self.pcm.sum(axis=0)
        for j in range(self.n_bit):
            deg = int(degree_prob[j])
            shapes.append((2,) * deg)

        # 3. Syndrome Vectors
        for _ in range(self.n_check):
            shapes.append((batch_size, 2))

        print(f"Searching for contraction path using Cotengra (max_time={max_time}s)...")

        opt = ctg.HyperOptimizer(
            max_time=max_time, 
            max_repeats=128,
            minimize='size',
            progbar=True,
            parallel=120,
        )

        path, path_info = oe.contract_path(self.eq_str, *shapes, optimize=opt, shapes=True)
        
        import math
        flops_log10 = math.log10(path_info.opt_cost)
        print(f"1. FLOPs (log10):    {flops_log10:.2f}")
        space_complexity = math.log2(path_info.largest_intermediate)
        print(f"2. Space Complexity (log2): {space_complexity:.2f}")
        max_tensor_gb = (path_info.largest_intermediate * 8) / (1024**3)
        print(f"3. Peak Memory:            {max_tensor_gb:.2f} GB")
        
        if space_complexity >= 30:
            print(f"❌ 警告: 找到的路径 Space Complexity ({space_complexity:.2f}) 达到或超过 30！")
            print("这会导致极高的显存占用，拒绝采用该路径。")
            return None
        
        return path

    def save_path(self, path, filename="best_contraction_path.pkl"):
        import pickle  
        try:
            with open(filename, 'wb') as f:
                pickle.dump(path, f)
            print(f"Path successfully saved to: {filename}")
        except Exception as e:
            print(f"Error saving path: {e}")

    def load_path(self, filename="best_contraction_path.pkl"):
        import pickle
        import os
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Path file '{filename}' not found. Please run find_contraction_path first.")
        
        try:
            with open(filename, 'rb') as f:
                path = pickle.load(f)
            print(f"Path successfully loaded from: {filename}")
            self.path = path
        except Exception as e:
            print(f"Error loading path: {e}")
            self.path = None

    def load_tree(self, filename):
        import json
        import os
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Tree file '{filename}' not found.")
        
        try:
            with open(filename, 'rb') as f:
                tree = json.load(f)
            print(f"Tree successfully loaded from: {filename}")
            self.tree = tree
        except Exception as e:
            print(f"Error loading path: {e}")
            self.tree = None

            
    
class GroupTN(nn.Module):
    def __init__(self, d, r, sub_pcms, sub_dets, sub_errors, init_priors, dev='cpu', dtype=torch.float32):
        super().__init__()
        self.d, self.r = d, r
        self.sub_pcms = sub_pcms
        self.n_sub = len(sub_pcms)
        self.sub_dets = sub_dets
        self.sub_errors = sub_errors
        self.dev=dev
        self.dtype=dtype

        self.priors_logits = nn.Parameter(torch.logit(init_priors))

        self.construct_tns()
    
    def construct_tns(self):
        self.tns = nn.ModuleList()

        for i in range(self.n_sub):
            pcmi = self.sub_pcms[i]
            for j in range(i):
                if pcmi.shape == self.sub_pcms[j].shape:
                    if (pcmi - self.sub_pcms[j]).all()==0:
                        self.tns.append(self.tns[j])
                        break
            else:
                tn = TensorNetwork(pcm=pcmi, dev=self.dev, dtype=self.dtype)
                tn.load_path(filename=f'path/d{self.d}r{self.r}/subsample_path_{i}.pkl')
                self.tns.append(tn)
                
    def forward(self, syndromes):
        
        # loss  = 
                # + self.tns[1].forward(syndromes[:, self.sub_dets[1]])
        loss = torch.cat([self.tns[i].forward(
            syndromes[:, self.sub_dets[i]], 
            priors_logits=self.priors_logits[self.sub_errors[i]]
            ).unsqueeze(0)
            for i in range(self.n_sub)
            ]).mean()
        
        return loss





class MatchingNet(nn.Module):
    def __init__(self, dem, init_priors=None, dev='cpu', dtype=torch.float32):
        super().__init__()
        '''The 'decompose_errors' must be True'''
        self.dtype=dtype
        self.dev=dev
        self.pcm, self.edges_mapping = generate_compactified_pcm_from_seperated_dem(dem)

        if init_priors == None:
            init_priors = torch.from_numpy(get_error_rates(dem)).to(dev).to(dtype)
            self.negative_priors_logits = nn.Parameter(-torch.logit(init_priors).to(dev).to(dtype))
        else:
            self.negative_priors_logits = nn.Parameter(-torch.logit(init_priors).to(dev).to(dtype))
    
    def forward(self, syndromes):
        probs = torch.sigmoid(-self.negative_priors_logits)

        probs_list = []
        for edge_info in self.edges_mapping:
            # idx = edge_info['new_edge_id']
            source = edge_info['source_hyperedge_indices']
            if len(source) > 1:
                prob_edge = 0.5 * (1 - torch.prod(1 - 2 * probs[source]))
            else:
                prob_edge = probs[source]
            probs_list.append(prob_edge.squeeze())
        # print(probs_list)
        probs_compactified = torch.stack(probs_list)
        
        # log_one_minus_p = torch.log(1.-probs_compactified).sum(0)
        
        decoder = Matching(self.pcm, 
                           error_probabilities=probs_compactified.detach().cpu().numpy())

        if isinstance(syndromes, torch.Tensor):
            syndromes_np = syndromes.detach().cpu().numpy().astype(np.uint8)
        else:
            syndromes_np = syndromes
        
        error_configs = decoder.decode_batch(syndromes_np)
        error_configs = torch.from_numpy(error_configs).to(self.dtype).to(self.dev)
        log_operators_probs = torch.log(probs_compactified*error_configs + (1-probs_compactified)*(1-error_configs)).sum(1)
        # print(error_configs.shape)
        # print(log_operators_probs.shape)
        return -log_operators_probs.mean(0)


if __name__ == '__main__':
    import stim
    import warnings
    warnings.filterwarnings("ignore", message="Casting complex values to real")
    
    d = 3 # distance
    r = 5 # rounds
    error_prob = 0.001 # probability of errors generation

    dev = 'cpu'
    dtype=torch.float64
    task_check = 'exact_probability' # ['exact_probability', 'gradient_check', 'logical_error_rate']

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

       
        

    pln.test(p_exact, x_exact, 100)

    er_opt = torch.sigmoid(pln.para.detach().cpu())


    print('Mean of Relative Errors :', (abs(er_opt-er)/er).mean())


    

