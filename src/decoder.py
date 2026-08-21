
import numpy as np
import torch
from pymatching import Matching
from .utils import update_dem
from .model import TensorNetwork


def _syndrome_to_numpy_batch(syndrome, dtype):
    """
    Return syndrome with shape (n_shots, n_detectors) for batch decoders.

    Avoid ``squeeze()`` on the last axis: ``(n, 1)`` must not become ``(n,)``,
    otherwise a follow-up ``reshape(1, -1)`` wrongly treats shots as detectors.
    """
    if isinstance(syndrome, torch.Tensor):
        s = syndrome.detach().cpu().numpy()
    elif isinstance(syndrome, list):
        s = np.asarray(syndrome)
    else:
        s = np.asarray(syndrome)
    s = np.asarray(s, dtype=dtype, order="C")
    if s.ndim == 0:
        return s.reshape(1, 1)
    if s.ndim == 1:
        return s.reshape(1, -1)
    if s.ndim == 2:
        return s
    raise ValueError(f"syndrome must be 0/1/2-D, got shape {tuple(s.shape)}")


def _align_logical_bits(pred, ideal):
    """Compare logical bits element-wise; flatten (n,1) vs (n,) pitfalls."""
    p = np.asarray(pred, dtype=bool).reshape(-1)
    t = np.asarray(ideal, dtype=bool).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"logical pred shape {p.shape} != ideal {t.shape}")
    return p, t


def _predictions_as_decode_batch(pred):
    """Normalize to (n_shots, n_obs) like PyMatching ``decode_batch``."""
    p = np.asarray(pred, dtype=bool)
    if p.ndim == 1:
        return p[:, np.newaxis]
    return p


def _weights_to_error_rates(weights):
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    elif isinstance(weights, list):
        weights = np.array(weights)
    return 1 / (1 + np.exp(weights))


class MWPM:
    def __init__(self, abstract_code)-> None:
        '''pcm : hx,
           logical_check : lx,
        '''

        self.hx, self.lx= abstract_code.hx, abstract_code.lx

        self.n = self.hx.shape[1]
        # self.pebz = abstract_code.pebz

    def decode(self, syndrome, error_rates=None, weights=None):
        syndrome = _syndrome_to_numpy_batch(syndrome, bool)

        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                error_rates = np.array(error_rates)
            weights = np.log(np.array((1-error_rates)/error_rates))

        elif weights is not None and error_rates is None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            elif isinstance(weights, list):
                weights = np.array(weights)
        else:
            print('Must input error rates or weights !!!')

        decoder = Matching(self.hx, weights=weights)
        recover = decoder.decode_batch(syndrome)
        logical_flip = ((self.lx @ recover.T) % 2).squeeze()
        return logical_flip

    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        # print(logical_ideal.type)
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        ns = _syndrome_to_numpy_batch(syndrome, bool).shape[0]
        if weights is None and error_rates is not None:
            logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates).astype(bool)
        elif weights is not None and error_rates is None:
            logical_flip = self.decode(syndrome=syndrome, weights=weights).astype(bool)
        else:
            print('Must input error rates or weights !!!')

        logical_flip, logical_ideal = _align_logical_bits(logical_flip, logical_ideal)
        ler = 1 - np.equal(logical_flip, logical_ideal).mean()
        return ler

class MWPM_dem:
    def __init__(self, dem, enable_correlations=False):
        from pymatching import Matching
        self.dem=dem
        self.enable_correlations = enable_correlations

    def decode(self, syndrome, error_rates=None, weights=None, enable_correlations=None):
        syndrome = _syndrome_to_numpy_batch(syndrome, bool)

        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                error_rates = np.array(error_rates)
            new_dem = update_dem(dem=self.dem, ers=error_rates)

        elif weights is not None and error_rates is None:
            error_rates = _weights_to_error_rates(weights)
            new_dem = update_dem(dem=self.dem, ers=error_rates)

        else:
            new_dem = self.dem

        # Use enable_correlations if specified, otherwise use instance default
        use_correlations = enable_correlations if enable_correlations is not None else self.enable_correlations

        matcher = Matching.from_detector_error_model(new_dem, enable_correlations=use_correlations)

        logical_flip = _predictions_as_decode_batch(
            matcher.decode_batch(syndrome, enable_correlations=use_correlations)
        )
        return logical_flip

    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):

        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        if weights is None and error_rates is not None:
            logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates)
        elif weights is not None and error_rates is None:
            logical_flip = self.decode(syndrome=syndrome, weights=weights)
        else:
            logical_flip = self.decode(syndrome=syndrome)

        logical_flip, logical_ideal = _align_logical_bits(logical_flip, logical_ideal)
        ler = 1 - np.equal(logical_flip, logical_ideal).mean()
        return ler


class MWPM_graph:
    def __init__(self, dem, enable_correlations=False):

        self.enable_correlations = enable_correlations
        self.matcher = Matching.from_detector_error_model(dem, enable_correlations=enable_correlations)

    def set_edges_weights(self, error_rates=None, weights=None):

        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            elif isinstance(error_rates, list):
                error_rates = np.array(error_rates)

            epsilon = 1e-15
            error_rates = np.clip(error_rates, epsilon, 1 - epsilon)
            weights = np.log((1 - error_rates) / error_rates)

        elif weights is not None and error_rates is None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            elif isinstance(weights, list):
                weights = np.array(weights)

            error_rates = 1 / (1 + np.exp(weights))
        else:
            raise ValueError('Must input error rates or weights !!!')


        current_edges = self.matcher.edges()


        for i, (u, v, attr) in enumerate(current_edges):
            if i >= len(weights):
                break

            new_weight = weights[i]
            new_prob = error_rates[i]
            existing_fault_ids = attr.get('fault_ids', set())


            if v is None:

                self.matcher.add_boundary_edge(
                    u,
                    fault_ids=existing_fault_ids,
                    weight=new_weight,
                    error_probability=new_prob,
                    merge_strategy="replace"
                )
            else:

                self.matcher.add_edge(
                    u,
                    v,
                    fault_ids=existing_fault_ids,
                    weight=new_weight,
                    error_probability=new_prob,
                    merge_strategy="replace"
                )


    def decode(self, syndrome, error_rates=None, weights=None):
        syndrome = _syndrome_to_numpy_batch(syndrome, bool)

        if weights is None and error_rates is not None:
            self.set_edges_weights(error_rates=error_rates)
        elif weights is not None and error_rates is None:
            self.set_edges_weights(weights=weights)
        else:
            None

        logical_flip = _predictions_as_decode_batch(
            self.matcher.decode_batch(syndrome, enable_correlations=self.enable_correlations)
        )
        return logical_flip

    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        # print(logical_ideal.type)
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        ns = _syndrome_to_numpy_batch(syndrome, bool).shape[0]
        if weights is None and error_rates is not None:
            logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates).astype(bool)
        elif weights is not None and error_rates is None:
            logical_flip = self.decode(syndrome=syndrome, weights=weights).astype(bool)
        else:
            logical_flip = self.decode(syndrome=syndrome).astype(bool)
        logical_flip, logical_ideal = _align_logical_bits(logical_flip, logical_ideal)
        ler = 1 - np.equal(logical_flip, logical_ideal).mean()
        return ler


class BeliefMatching_dem:
    def __init__(self, dem, max_iter=10):
        try:
            from beliefmatching import BeliefMatching
        except ImportError as exc:
            raise ImportError("BeliefMatching_dem requires the optional 'beliefmatching' package.")
        self._belief_matching_cls = BeliefMatching

        self.dem = dem
        self.max_iter = max_iter
        self.decoder = self._belief_matching_cls(dem, max_bp_iters=max_iter)


    def decode(self, syndrome, error_rates=None, weights=None):
        syndrome = _syndrome_to_numpy_batch(syndrome, np.uint8)

        if weights is None and error_rates is not None:
            if isinstance(error_rates, torch.Tensor):
                error_rates = error_rates.detach().cpu().numpy()
            dem = update_dem(dem=self.dem, ers=error_rates)
        elif weights is not None and error_rates is None:
            error_rates = _weights_to_error_rates(weights)
            dem = update_dem(dem=self.dem, ers=error_rates)
        else:
            dem = self.dem
        self.decoder = self._belief_matching_cls(dem, max_bp_iters=self.max_iter)

        logical_flips = self.decoder.decode_batch(syndrome)

        return _predictions_as_decode_batch(logical_flips)

    def logical_error_rate(self, syndrome, logical_ideal, error_rates=None, weights=None):
        if isinstance(logical_ideal, np.ndarray):
            logical_ideal = logical_ideal.squeeze().astype(bool)
        elif isinstance(logical_ideal, list):
            logical_ideal = np.array(logical_ideal).squeeze().astype(bool)
        elif isinstance(logical_ideal, torch.Tensor):
            logical_ideal = logical_ideal.detach().cpu().numpy().squeeze().astype(bool)

        logical_flip = self.decode(syndrome=syndrome, error_rates=error_rates, weights=weights)

        logical_flip, logical_ideal = _align_logical_bits(logical_flip, logical_ideal)
        ler = 1 - np.equal(logical_flip, logical_ideal).mean()
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
