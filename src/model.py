import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import opt_einsum as oe
from pymatching import Matching
from .utils import construct_kac_ward_solution, torch_rep_cir_log_coset_p, rep_cir, get_error_rates, generate_compactified_pcm_from_seperated_dem


class PlanarNet(nn.Module):
    def __init__(self, abstract_code=None, init_priors=None, dev='cpu', rep_configs=None,
                 param_mode='logit', log_scale=1.0):
        """
        Initialize PlanarNet with either a single abstract_code or multiple rep_configs.
        
        Args:
            abstract_code: Single rep_dem object (for backward compatibility)
            init_priors: Initial error rate priors tensor (size matches original DEM)
            dev: Device ('cpu' or 'cuda')
            rep_configs: Dictionary mapping config_idx to rep_dem objects (for multiple configs)
            param_mode: Parameterization mode
                - 'logit': Use logit = log(p/(1-p)) with sigmoid (default, backward compatible)
                - 'log_prior': Use log(p) with exp (better for small probabilities)
            log_scale: Scaling factor for log_prior mode (default 1.0)
                - When log_prior is used, parameter = log(p) / log_scale
                - This helps optimize parameters in a more reasonable range
                - Typical values: 0.1 to 1.0 (smaller = more scaling)
        """
        super().__init__()
        self.dev = dev
        self.param_mode = param_mode
        self.log_scale = log_scale
        
        # Support both single abstract_code (backward compatibility) and multiple rep_configs
        if rep_configs is not None:
            # Multiple configs mode: store rep_configs and cache for each
            self.rep_configs = rep_configs
            self.dtype = init_priors.dtype
            
            # Initialize parameter based on param_mode
            if param_mode == 'logit':
                init_param = torch.log(init_priors/(1.-init_priors))
            elif param_mode == 'log_prior':
                init_priors_clamped = torch.clamp(init_priors, 1e-10, 1.0 - 1e-10)
                init_param = torch.log(init_priors_clamped) / log_scale
            else:
                raise ValueError(f"Unknown param_mode: {param_mode}. Must be 'logit' or 'log_prior'")
            self.para = nn.Parameter(init_param.detach().to(self.dtype).to(self.dev))
            
            # Cache for each rep_dem (will be computed on first use in surface_subsample_forward_test)
            self.rep_cache = {}
        else:
            # Single abstract_code mode (backward compatibility)
            if abstract_code is None or init_priors is None:
                raise ValueError("Either abstract_code and init_priors, or rep_configs must be provided")
            
            self.dtype = init_priors.dtype
            self.init_priors = init_priors
            
            self.generators = np.concatenate([abstract_code.hz, abstract_code.lz.reshape(1, -1)], axis=0)
            self.kwz, self.edges_dict_z = construct_kac_ward_solution(self.generators)

            # Initialize parameter based on param_mode
            if param_mode == 'logit':
                init_param = torch.log(init_priors/(1.-init_priors))
            elif param_mode == 'log_prior':
                init_priors_clamped = torch.clamp(init_priors, 1e-10, 1.0 - 1e-10)
                init_param = torch.log(init_priors_clamped) / log_scale
            else:
                raise ValueError(f"Unknown param_mode: {param_mode}. Must be 'logit' or 'log_prior'")
            self.para = nn.Parameter(init_param.detach().to(self.dtype).to(self.dev))

            self.pebz = torch.from_numpy(abstract_code.pebz).to(self.dtype).to(self.dev)
            self.lz = torch.from_numpy(abstract_code.lz).to(self.dtype).to(self.dev)
  
        pass
    
    def get_priors(self):
        """
        Convert parameters to probabilities based on param_mode.
        Returns probabilities clamped to [1e-20, 1-1e-20] for numerical stability.
        """
        if self.param_mode == 'logit':
            # Original: sigmoid(logit)
            priors = torch.sigmoid(self.para) + 1e-20
        elif self.param_mode == 'log_prior':
            # New: exp(log_prior * log_scale)
            # Since para = log(p) / log_scale, we have: p = exp(para * log_scale)
            priors = torch.exp(self.para * self.log_scale) + 1e-20
            # Clamp to ensure probabilities are in [0, 1]
            priors = torch.clamp(priors, 1e-20, 1.0 - 1e-20)
        else:
            raise ValueError(f"Unknown param_mode: {self.param_mode}")
        return priors
    
    # def logp_check(self, operator, error_rates):
    #             return torch_rep_cir_log_coset_p(operator, self.kwz0, self.edges_dict_z0, error_rates=error_rates)
    
    def logp(self, operator, error_rates):
        return torch_rep_cir_log_coset_p(operator, self.kwz, self.edges_dict_z, error_rates=error_rates)
    
    def forward(self, det):
        priors = self.get_priors()
        with torch.no_grad():
            x = det*1.
            operator = x @ self.pebz % 2
        logp = self.logp(operator, priors)
        loss = - logp.mean()
        return loss
    
    def subsample_forward(self, det, subsamples, sub_list):

        loss = 0
        for sub_idx in sub_list:
            sub = subsamples[sub_idx]
            det_sub = det[:, sub['det']]
            
            priors = self.get_priors() 

            priors_sub = priors[sub['em']]
            for e in sub['merge_e']:
                new_value = (1 - priors_sub[e[0]]) * priors[e[2]] + priors_sub[e[0]] * (1 - priors[e[2]])
                priors_sub = torch.cat([
                    priors_sub[:e[0]],
                    new_value.unsqueeze(0),
                    priors_sub[e[0]+1:]
                ])

            with torch.no_grad():
                x = det_sub
                operator = x @ self.pebz % 2

            logp = self.logp(operator, priors_sub)
            loss = loss + - logp.mean()
        
        loss = loss/len(sub_list)
        return loss
    
    def surface_subsample_forward(self, det, subsamples, sub_list, sub_idx_to_rep_mapping):
        """
        Forward pass for surface code subsampling.
        Processes all configurations simultaneously, using rep_dem objects from sub_idx_to_rep_mapping.
        
        Args:
            det: Detector data tensor
            subsamples: List of subsample mappings
            sub_list: List of subsample indices to process (all configurations are processed)
            sub_idx_to_rep_mapping: Dict mapping sub_idx to rep_dem objects.
                                   Must map each sub_idx to a rep_dem instance.
        """
        from .utils import construct_kac_ward_solution
        
        if sub_idx_to_rep_mapping is None or len(sub_idx_to_rep_mapping) == 0:
            raise ValueError("sub_idx_to_rep_mapping must be provided and non-empty")
        
        loss = 0
        
        # Cache for computed pebz, kwz, edges_dict_z for each rep_dem object
        # Use id(rep) as key to cache based on object identity
        rep_cache = {}
        
        for sub_idx in sub_list:
            sub = subsamples[sub_idx]  # Mapping from mapping_nomerge
            
            # Select relevant detectors for this subsample pattern
            det_sub = det[:, sub['det']]
            
            # Get original priors (parameters)
            # priors = self.para
            priors = self.get_priors()
            
            # Construct subsampled priors vector using merge_e mapping
            # merge_e format: (merged_idx, [original_idx1, original_idx2, ...])
            # We need to build a tensor where the k-th element is the merged probability 
            # for the k-th error in the subsample DEM (odd error probability)
            merged_probs_list = []
            
            # Iterate through merge_e in order (sorted by merged_idx)
            for merged_idx, orig_indices in sub['merge_e']:
                # Extract probabilities for this group
                # print(merged_idx, orig_indices)
                group_probs = priors[orig_indices]
                
                if len(group_probs) == 0:
                    merged_p = torch.tensor(0.0, dtype=self.dtype, device=self.dev)
                elif len(group_probs) == 1:
                    merged_p = group_probs[0]
                else:
                    # Calculate odd error probability: p_odd = 0.5 * (1 - prod(1 - 2*p))
                    # Let q = 1 - 2p. Then q_new = q1 * q2 * ... * qn
                    # p_odd = (1 - q_new) / 2
                    q = 1.0 - 2.0 * group_probs
                    q_prod = torch.prod(q)
                    merged_p = 0.5 * (1.0 - q_prod)
                
                merged_probs_list.append(merged_p)
                # Print merged probabilities as a pure python error rate list for inspection
            # merged_probs_numpy = [p.item() for p in merged_probs_list]
            # print(f"Merged error rates{sub_idx}:", merged_probs_numpy)
            if not merged_probs_list:
                # Handle case with no errors (should not happen in valid subsamples)
                continue
            
            priors_sub = torch.stack(merged_probs_list)
            
            # Get rep_dem object for this sub_idx
            rep_obj = sub_idx_to_rep_mapping.get(sub_idx)
            if rep_obj is None:
                raise ValueError(f"No rep_dem object found for sub_idx {sub_idx} in sub_idx_to_rep_mapping")
            
            # Get or compute cached values for this rep_dem object
            rep_id = id(rep_obj)
            if rep_id not in rep_cache:
                # Compute and cache pebz, kwz, edges_dict_z for this rep_dem
                generators = np.concatenate([rep_obj.hz, rep_obj.lz.reshape(1, -1)], axis=0)
                kwz, edges_dict_z = construct_kac_ward_solution(generators)
                pebz_tensor = torch.from_numpy(rep_obj.pebz).to(self.dtype).to(self.dev)
                rep_cache[rep_id] = {
                    'pebz': pebz_tensor,
                    'kwz': kwz,
                    'edges_dict_z': edges_dict_z
                }
            cached = rep_cache[rep_id]
            pebz = cached['pebz']
            kwz = cached['kwz']
            edges_dict_z = cached['edges_dict_z']
            
            # Calculate loss using the merged priors
            with torch.no_grad():
                x = det_sub
                operator = x @ pebz % 2
            
            # Use the appropriate logp calculation
            logp = torch_rep_cir_log_coset_p(operator, kwz, edges_dict_z, error_rates=priors_sub)
            loss = loss + - logp.mean()
        
        if len(sub_list) > 0:
            loss = loss / len(sub_list)
        
        return loss


    def cal_eloss(self, p_exact, x_exact):
        priors = self.get_priors()
        with torch.no_grad():
            operator = x_exact @ self.pebz % 2
        logp = self.logp(operator, priors)
        loss = - (p_exact*logp).sum()
        return loss
    
    def test(self, p_exact, x_exact, nprint=None):
        optim = torch.optim.AdamW([self.para], lr=0.01)
        er_his = []
        loss_his = []
        # mean_re_his = []
        epochs = 500
        for epoch in range(1, 1+epochs):
            # print(self.para)
            loss = self.cal_eloss(p_exact, x_exact)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # if epoch <= 100 and epoch%5 == 0:
            #     er_his.append(torch.sigmoid(self.para.detach().cpu()))
            if nprint!=None and epoch%nprint == 0:
                print('epoch:{}, nll:{}, grad_mean:{}'.format(epoch, loss.item(), self.para.grad.mean().item()))
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
    def __init__(self, pcm, l=None, priors_logits=None, dev='cpu', dtype=torch.float32, decoding=False):
        super().__init__()
        self.pcm = pcm
        self.l=l
        self.dev = dev
        self.dtype = dtype
        self.decoding=decoding
        self.n_check, self.n_bit = pcm.shape
        if priors_logits is None:
            self.priors_logits = nn.Parameter(torch.randn(pcm.shape[1]))
        else:
            self.priors_logits = nn.Parameter(priors_logits.to(dev).to(dtype))
        self.path = None
        self.tree = None

        self.generate_xor_tensors()

        self.generate_equation()

        pass

    def generate_xor_tensors(self):
        """
        生成 Check Node Tensors。
        策略：利用 Hadamard 分解 (Hyper Tensor)，并将归一化常数融合进张量值中。
        """
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
        # 保持不变
        tensor_list = []
        degree_prob = self.pcm.sum(axis=0)
        for j in range(self.n_bit):
            degree = int(degree_prob[j])
            p = probs[j]
            if self.decoding and self.l[j] ==1 :
                pt = prob_tensor(degree, p, connect_to_l=True).to(self.dev).to(self.dtype)
            else:
                pt = prob_tensor(degree, p).to(self.dev).to(self.dtype)
            tensor_list.append(pt)
        return tensor_list
    
    def decoding_forward(self, syndromes, probs=None):
        path = self.path
        tree = self.tree
        is_batched = syndromes.ndim == 2
        if not is_batched:
            syndromes = syndromes.unsqueeze(0)
        syndrome_onehot = torch.nn.functional.one_hot(syndromes.long(), num_classes=2).to(dtype=self.dtype, device=self.dev) 
        syndrome_vectors = list(syndrome_onehot.unbind(dim=1))  
        
        
        if probs == None:   
            probs = torch.sigmoid(self.priors_logits)
        else:
            None
        # print(probs)
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

    def forward(self, syndromes, priors_logits=None):
        path = self.path
        tree = self.tree
        is_batched = syndromes.ndim == 2
        if not is_batched:
            syndromes = syndromes.unsqueeze(0)   

        syndrome_onehot = torch.nn.functional.one_hot(syndromes.long(), num_classes=2).to(dtype=self.dtype, device=self.dev)
        syndrome_vectors = list(syndrome_onehot.unbind(dim=1))
        if priors_logits == None:   
            probs = torch.sigmoid(self.priors_logits)
        else:
            # self.priors_logits=None
            probs = torch.sigmoid(priors_logits)
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
        """
        Args:
            batch_size (int): 假设的 Batch Size
            max_time (int): cotengra 搜索路径的最长时间（秒）。时间越长，路径越好。
        """
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

        # --- 配置 Cotengra ---
        # minimize: 'flops' (计算最快) 或 'write' (总写入量) 或 'size' (峰值显存最小)
        # 对于显存不足的情况，建议优先尝试 'flops'，如果爆显存则改为 'size'
        opt = ctg.HyperOptimizer(
            max_time=max_time, 
            max_repeats=64,    # 尝试多少次搜索
            minimize='size',    # 优化目标
            progbar=True,        # 显示进度条
            parallel=20
        )

        # --- 获取路径 ---
        # 注意：这里把 optimize 参数设为 opt 对象
        path, path_info = oe.contract_path(self.eq_str, *shapes, optimize=opt, shapes=True)
        
        import math
        flops_log10 = math.log10(path_info.opt_cost)
        print(f"1. FLOPs (log10):    {flops_log10:.2f}")
        space_complexity = math.log2(path_info.largest_intermediate)
        print(f"2. Space Complexity (log2): {space_complexity:.2f}")
        max_tensor_gb = (path_info.largest_intermediate * 8) / (1024**3)
        print(f"3. Peak Memory:            {max_tensor_gb:.2f} GB")
        
        # Cotengra 还可以可视化路径树（如果安装了 graphviz）
        # opt.get_tree().plot_ring() 
        
        return path

    def save_path(self, path, filename="best_contraction_path.pkl"):
        """
        将缩并路径序列化并保存到文件。
        """
        import pickle  

        try:
            with open(filename, 'wb') as f:
                pickle.dump(path, f)
            print(f"Path successfully saved to: {filename}")
        except Exception as e:
            print(f"Error saving path: {e}")

    # --- 新增：加载路径 ---
    def load_path(self, filename="best_contraction_path.pkl"):
        """
        从文件加载缩并路径。
        """
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
        """
        从文件加载缩并路径。
        """
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


    

