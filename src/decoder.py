
import numpy as np
import torch
from pymatching import Matching
from beliefmatching import BeliefMatching
from .utils import torch_rep_cir_log_coset_p, construct_kac_ward_solution, update_dem
from .model import TensorNetwork


class Planar:
    def __init__(self, abstract_code, dev='cpu') -> None:
        '''pcm : hx,
           logical_check : lx,
           summation_var : hz,
           logical_er : lz
        '''
        self.dev = dev
        self.hx, self.hz, self.lx, self.lz = abstract_code.hx, abstract_code.hz, abstract_code.lx, abstract_code.lz
        self.n = self.hx.shape[1] 
        self.pebz = abstract_code.pebz

        self.kwz, self.edges_dict_z = construct_kac_ward_solution(self.hz)
        pass
    
    def logp(self, operator, error_rates):
        return torch_rep_cir_log_coset_p(operator, self.kwz, self.edges_dict_z, error_rates=error_rates)
        

    def decode(self, syndrome, error_rates):
        if isinstance(syndrome, torch.Tensor):
            syndrome = syndrome.detach().cpu().numpy().astype(bool).squeeze()
        elif isinstance(syndrome, list):
            syndrome = np.array(syndrome).astype(bool).squeeze()
        elif isinstance(syndrome, np.ndarray):
            syndrome = syndrome.astype(bool).squeeze()

        if isinstance(error_rates, torch.Tensor):
            error_rates = error_rates.to(torch.float64).to(self.dev)
        elif isinstance(error_rates, np.ndarray):
            error_rates = torch.from_numpy(error_rates).to(torch.float64).to(self.dev)
        elif isinstance(error_rates, list):
            error_rates = torch.tensor(error_rates, dtype=torch.float64, device=self.dev)


        pez =  torch.tensor((syndrome @ self.pebz % 2), dtype=torch.float64, device=self.dev)#, dtype=torch.float64
        pez_l =  torch.tensor((syndrome @ self.pebz % 2 + self.lz) % 2, dtype=torch.float64, device=self.dev)
        log_coest_probs_z = torch.cat([self.logp(pez, error_rates=error_rates).unsqueeze(dim=1),
                                        self.logp(pez_l, error_rates=error_rates).unsqueeze(dim=1)
                                        ], dim=1)
                    
        idx_z = torch.argmax(log_coest_probs_z, dim=1)
        logical_flip = ((pez @ torch.from_numpy(self.lx).to(torch.float64).to(self.dev)) % 2 + idx_z) % 2

        return logical_flip.to(torch.bool)

    def logical_error_rate(self, syndrome, logical_ideal, error_rates):

        if isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.to(torch.bool).to(self.dev).squeeze()
        elif isinstance(logical_ideal, np.ndarray):
            logical_ideal = torch.from_numpy(logical_ideal).to(torch.bool).to(self.dev).squeeze()
        elif isinstance(logical_ideal, list):
            logical_ideal = torch.tensor(logical_ideal, device=self.dev).to(torch.bool).squeeze()

        ns = syndrome.shape[0]
        logical_flip = self.decode(syndrome, error_rates)
        # print(logical_flip.size(), logical_ideal.size())
        return 1-torch.eq(logical_flip, logical_ideal).sum()/ns

class MWPM:
    def __init__(self, abstract_code)-> None:
        '''pcm : hx,
           logical_check : lx,
        '''

        self.hx, self.lx= abstract_code.hx, abstract_code.lx
      
        self.n = self.hx.shape[1] 
        # self.pebz = abstract_code.pebz
        
    def decode(self, syndrome, error_rates=None, weights=None):
        if isinstance(syndrome, torch.Tensor):
            syndrome = syndrome.detach().cpu().numpy().astype(bool).squeeze()
        elif isinstance(syndrome, np.ndarray):
            syndrome = syndrome.astype(bool).squeeze()
        elif isinstance(syndrome, list):
            syndrome = np.array(syndrome).astype(bool).squeeze()

        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                error_rates = np.array(error_rates)
            weights = np.log(np.array((1-error_rates)/error_rates))

        elif weights is not None and error_rates is None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                weights = np.array(weights)
        else:
            print('Must input error rates or weights !!!')

        decoder = Matching(self.hx, weights=weights)
        recover = decoder.decode_batch(syndrome)
        logical_flip = ((self.lx@recover.T)%2).squeeze()
        return logical_flip
    
    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        # print(logical_ideal.type)
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        ns = syndrome.shape[0]
        if weights is None and error_rates is not None:
            logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates).squeeze().astype(bool)
        elif weights is not None and error_rates is None:
            logical_flip = self.decode(syndrome=syndrome, weights=weights).squeeze().astype(bool)
        else:
            print('Must input error rates or weights !!!')

        # print(logical_flip, logical_ideal)
        ler = 1-np.equal(logical_flip, logical_ideal).sum()/ns
        return ler

class MWPM_dem:
    def __init__(self, dem, enable_correlations=False):
        from pymatching import Matching
        self.dem=dem
        self.enable_correlations = enable_correlations
        
    def decode(self, syndrome, error_rates=None, weights=None, enable_correlations=None):


        if isinstance(syndrome, torch.Tensor):
            syndrome = syndrome.detach().cpu().numpy().astype(bool).squeeze()
        elif isinstance(syndrome, np.ndarray):
            syndrome = syndrome.astype(bool).squeeze()
        elif isinstance(syndrome, list):
            syndrome = np.array(syndrome).astype(bool).squeeze()
        
        if syndrome.ndim == 1:
            syndrome = syndrome[np.newaxis, :]

        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                error_rates = np.array(error_rates)
            new_dem = update_dem(dem=self.dem, ers=error_rates)

        elif weights is not None and error_rates is None:
            if isinstance(weights, torch.Tensor):
                error_rates = torch.sigmoid(-weights.detach()).cpu().numpy()
            new_dem = update_dem(dem=self.dem, ers=error_rates)

        else:
            new_dem = self.dem
        
        # Use enable_correlations if specified, otherwise use instance default
        use_correlations = enable_correlations if enable_correlations is not None else self.enable_correlations
        
        matcher = Matching.from_detector_error_model(new_dem, enable_correlations=use_correlations)

        logical_flip = matcher.decode_batch(syndrome, enable_correlations=use_correlations).squeeze()
        return logical_flip
    
    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        # print(logical_ideal.type)
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        ns = syndrome.shape[0] if syndrome.ndim > 1 else 1
        
        if weights is None and error_rates is not None:
            logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates)
        elif weights is not None and error_rates is None:
            logical_flip = self.decode(syndrome=syndrome, weights=weights)
        else:
            logical_flip = self.decode(syndrome=syndrome)

        # print(logical_flip, logical_ideal)
        ler = 1-np.equal(logical_flip, logical_ideal).sum()/ns
        return ler


class MWPM_graph:
    def __init__(self, dem, enable_correlations=False):
        # 初始化 Matching 对象
        self.enable_correlations = enable_correlations
        self.matcher = Matching.from_detector_error_model(dem, enable_correlations=enable_correlations)
    
    def set_edges_weights(self, error_rates=None, weights=None):
        """
        直接修改 Matching 图中边的权重，无需重建。
        """
        # --- 1. 数据预处理 (保持你的逻辑) ---
        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                error_rates = np.array(error_rates)
            
            # 避免 log(0)
            epsilon = 1e-15
            error_rates = np.clip(error_rates, epsilon, 1 - epsilon)
            weights = np.log((1 - error_rates) / error_rates)
            
        elif weights is not None and error_rates is None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            elif isinstance(weights, list):
                weights = np.array(weights)
            # 反推 error_rates (可选，PyMatching 主要用 weight)
            error_rates = 1 / (1 + np.exp(weights))
        else:
            raise ValueError('Must input error rates or weights !!!')

        # --- 2. 关键：获取现有边，覆盖更新 ---
        # 注意：这里假设输入的 weights 数组长度和顺序与 matcher.edges() 完全一致
        current_edges = self.matcher.edges()
        
        if len(weights) != len(current_edges):
            print(f"警告: 权重数量 ({len(weights)}) 与边数量 ({len(current_edges)}) 不匹配！")
        
        for i, (u, v, attr) in enumerate(current_edges):
            if i >= len(weights):
                break
                
            new_weight = weights[i]
            new_prob = error_rates[i]
            existing_fault_ids = attr.get('fault_ids', set())
            
            # --- 关键修改开始 ---
            if v is None:
                # 这是一个边界边 (Boundary Edge)
                self.matcher.add_boundary_edge(
                    u,
                    fault_ids=existing_fault_ids,
                    weight=new_weight,
                    error_probability=new_prob,
                    merge_strategy="replace"
                )
            else:
                # 这是一个普通边 (Internal Edge)
                self.matcher.add_edge(
                    u, 
                    v, 
                    fault_ids=existing_fault_ids, 
                    weight=new_weight, 
                    error_probability=new_prob, 
                    merge_strategy="replace"
                )

        
    def decode(self, syndrome, error_rates=None, weights=None):

        if isinstance(syndrome, torch.Tensor):
            syndrome = syndrome.detach().cpu().numpy().astype(bool).squeeze()
        elif isinstance(syndrome, np.ndarray):
            syndrome = syndrome.astype(bool).squeeze()
        elif isinstance(syndrome, list):
            syndrome = np.array(syndrome).astype(bool).squeeze()

        if weights is None and error_rates is not None:
            self.set_edges_weights(error_rates=error_rates)
        elif weights is not None and error_rates is None:
            self.set_edges_weights(weights=weights)
        else:
            None
            
        logical_flip = self.matcher.decode_batch(syndrome, enable_correlations=self.enable_correlations).squeeze()
        return logical_flip
    
    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        # print(logical_ideal.type)
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        ns = syndrome.shape[0]
        if weights is None and error_rates is not None:
            logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates).squeeze().astype(bool)
        elif weights is not None and error_rates is None:
            logical_flip = self.decode(syndrome=syndrome, weights=weights).squeeze().astype(bool)
        else:
            logical_flip = self.decode(syndrome=syndrome).squeeze().astype(bool)
        ler = 1-np.equal(logical_flip, logical_ideal).sum()/ns
        return ler


class BeliefMatching_dem:
    def __init__(self, dem, max_iter=10):
        """
        Args:
            dem: Stim 的 DetectorErrorModel
            max_iter: BP 迭代的最大次数
            bp_method: BP 算法类型 ('min_sum' 或 'product_sum')
            enable_correlations: 虽然 Belief Matching 内部处理逻辑不同，但保留接口一致性
        """
        self.dem = dem
        self.max_iter = max_iter
        self.decoder = BeliefMatching(dem, max_bp_iters=max_iter)

        
    def decode(self, syndrome, error_rates=None, weights=None):
        # 1. 预处理 syndrome 格式
        if isinstance(syndrome, torch.Tensor):
            syndrome = syndrome.detach().cpu().numpy().astype(np.uint8).squeeze()
        elif isinstance(syndrome, np.ndarray):
            syndrome = syndrome.astype(np.uint8).squeeze()
        elif isinstance(syndrome, list):
            syndrome = np.array(syndrome).astype(np.uint8).squeeze()
        
        if syndrome.ndim == 1:
            syndrome = syndrome[np.newaxis, :]

        # 2. 获取更新后的 DEM 或错误概率
        # 注意：BeliefFind 需要从 DEM 提取检查矩阵 (H) 和 错误概率 (p)
        
        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            dem = update_dem(dem=self.dem, ers=error_rates)
        elif weights is not None and error_rates is None:
            if isinstance(weights, torch.Tensor):
                # 转换 log-odds (weights) 回 概率 (error_rates)
                error_rates = torch.sigmoid(-weights.detach()).cpu().numpy()
            dem = update_dem(dem=self.dem, ers=error_rates)
        else:
            dem = self.dem
        self.decoder = BeliefMatching(dem, max_bp_iters=self.max_iter)

        logical_flips = self.decoder.decode_batch(syndrome)

        return logical_flips.squeeze()

    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        ns = syndrome.shape[0] if syndrome.ndim > 1 else 1
        
        logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates, weights=weights)

        ler = 1 - np.equal(logical_flip, logical_ideal).sum() / ns
        return ler



class TensorNetworkDecoder:
    def __init__(self, model: TensorNetwork, dev='cpu'):
        self.model = model
        self.dev = dev

    def decode(self, syndrome, error_rates):
        if isinstance(syndrome, np.ndarray):
            syndrome = torch.from_numpy(syndrome).to(torch.long).to(self.dev)
        elif isinstance(syndrome, list):
            syndrome = torch.tensor(syndrome, dtype=torch.long, device=self.dev)

        
        if isinstance(error_rates, np.ndarray):
            error_rates = torch.from_numpy(error_rates).to(torch.float64).to(self.dev)
        elif isinstance(error_rates, list):
            error_rates = torch.tensor(error_rates, dtype=torch.float64, device=self.dev)

        logical_flip= self.model.decoding_forward(syndrome, probs=error_rates)
        return logical_flip

    def logical_error_rate(self, syndrome, logical_ideal, error_rates):
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = torch.from_numpy(logical_ideal).to(torch.bool).to(self.dev)
        elif isinstance(logical_ideal, list):
            logical_ideal = torch.tensor(logical_ideal, dtype=torch.bool, device=self.dev)

        ns = syndrome.shape[0]
        logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates)
        return 1 - torch.eq(logical_flip, logical_ideal).sum().item() / ns


if __name__ == "__main__":
    import stim
    from utils import exact_coset_prob, get_error_rates, rep_cir
    d = 3 # distance
    r = 1 # rounds
    error_prob = 0.1 # probability of errors generation
    
    dev = 'cpu'

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

    # print('PCM(Hx) size:\n', rep.hx.shape)
    # print('Hz size:\n', rep.hz.shape)

    
    


    pl = Planar(rep, dev=dev)

    if task_check == 'exact_probability':#
        def numerical_gradient(f, x, eps=1e-5):
            """计算数值梯度（中心差分）"""
            grad = torch.zeros_like(x)
            for i in range(x.numel()):
                x_plus = x.clone().flatten()
                x_minus = x.clone().flatten()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
            return grad.reshape(x.shape)

        
        import warnings
        warnings.filterwarnings("ignore", message="Casting complex values to real")

        error_rates = torch.tensor(er, dtype=torch.float64, requires_grad=True, device=dev)
        pl = Planar(rep, dev=dev)
        x = np.array(
            [list(map(int, bin(x)[2:].zfill(rep.hx.shape[0]))) for x in range(2**rep.hx.shape[0])]
        ).astype(bool)
        # print(x.shape, rep.pebz.shape, rep.hz.shape)
        pe = torch.tensor(x @ rep.pebz % 2, dtype=torch.float64)
        pel =  torch.tensor((x @ rep.pebz % 2 + rep.lz) % 2, dtype=torch.float64)
        p0, p1 = torch.exp(pl.logp(pe, error_rates)), torch.exp(pl.logp(pel, error_rates))
        # p0.sum().backward()
        p = p0.sum() + p1.sum()
        print('sum of probabilities:', p.item())
        
        logp0, logp1 = torch.log(p0), torch.log(p1)
        p_0, p_1 = p0.detach(), p1.detach()
        nll = - (p_0*logp0).sum() + - (p_1*logp1).sum()
        print('exact NLL:', nll.item())
        nll.backward()
        print('mean grad of prior:', error_rates.grad.mean().item())
    elif task_check == 'gradient_check':
        from utils import numerical_gradient
        pe = torch.ones((rep.hx.shape[1],)).to(torch.float64)
        error_rates = torch.tensor(er, dtype=torch.float64, requires_grad=True, device=dev)
        logp_t = pl.logp(pe, error_rates)
        logp_t.backward()
        for i in range(rep.hx.shape[1]):
            index = i
            grad_numerical = numerical_gradient(pl.logp, pe, error_rates, index, eps=1e-8)
            print('--------------------------------------------------------------------------------')
            print('index:', index)
            print('numerical gradient:', grad_numerical.detach().item())
            print('grad of prior:', error_rates.grad[index].item())
            print('difference:', abs(grad_numerical.detach().item() - error_rates.grad[index].item()))
        
        # trace_grad_fn(p0.grad_fn, target_index=0)
        # import torchviz
        # from torchviz import make_dot
        # make_dot(nll, params={'x': error_rates}).render("graph", format="png")
        
    elif task_check == 'logical_error_rate':
        seed = 0
        num_shots = 1000
        sampler = dem.compile_sampler(seed=seed)
        det, obv, _ = sampler.sample(shots=num_shots) 

        pl = Planar(rep, dev=dev)
        mw = MWPM(rep, dev=dev)
        Ler_pl = pl.logical_error_rate(det, obv, er)
        print('LER planar:{:7f}'.format(Ler_pl.item()))
        Ler_mw = mw.logical_error_rate(det, obv, er)
        print('LER MWPM:{:7f}'.format(Ler_mw.item()))
