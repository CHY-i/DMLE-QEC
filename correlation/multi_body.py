from typing import Iterable, Optional, Tuple
import numpy as np
from correlation.result import CorrelationResult
from correlation.utils import HyperEdge
import torch


class FourBody:
    """Four-body correlation analysis class for quantum error correction.
    
    This class encapsulates all the multi-body correlation calculation functions
    with max_order=4 as the default value.
    """
    
    def __init__(self, detection_events: np.ndarray, device: str = "cuda:0"):
        """Initialize the FourBody class.
        
        Args:
            detection_events: Detection events in (0,1) format, shape (num_shots, num_dets)
            device: Device to use for computation ("cuda:0", "cuda:1", etc. or "cpu")
        """
        self.detection_events = detection_events
        self.device = device
        self.max_order = 4  # Fixed to 4 for four-body analysis
        
        # Transform and cache the sigma tensor once to avoid repeated conversions
        num_shots, num_dets = detection_events.shape
        sigma_events = 1 - 2 * detection_events  # Transform (0,1) to (-1,1)
        self.sigma_tensor = torch.from_numpy(sigma_events.astype(np.float64)).to(device)
        self.num_shots = num_shots
        
        # Calculate multi-point expectations once during initialization
        self.M_list = self._cal_multi_body_expectations_tensor()
        
        # Calculate f functions
        self.f_list = self._cal_f_functions()
        
        # Calculate w functions
        self.w_list = self._cal_w_functions()
        
        # Calculate p functions
        self.p_list = self._cal_p_functions()
    
    def _transform_detection_events(self) -> np.ndarray:
        """Transform detection events from (0,1) to (-1,1) format.
        
        Note: This method is kept for compatibility but the transformation
        is now done once during initialization and cached.
        """
        return 1 - 2 * self.detection_events
    
    def _cal_multi_body_expectations_tensor(self) -> list:
        """Calculate multi-point expectation values using torch tensors.
        
        Returns:
            list: List of torch tensors for orders 1 to max_order
            - result[0]: M_i (single body)
            - result[1]: M_ij (two body) 
            - result[2]: M_ijk (three body)
            - result[3]: M_ijkl (four body)
        """
        sigma_tensor = self.sigma_tensor
        num_shots = self.num_shots
        
        results = []
        
        if self.max_order >= 1:
            results.append(torch.einsum('ai->i', sigma_tensor) / num_shots)
        
        if self.max_order >= 2:
            results.append(torch.einsum('ai,aj->ij', sigma_tensor, sigma_tensor) / num_shots)
        
        if self.max_order >= 3:
            results.append(torch.einsum('ai,aj,ak->ijk', sigma_tensor, sigma_tensor, sigma_tensor) / num_shots)
        
        if self.max_order >= 4:
            results.append(torch.einsum('ai,aj,ak,al->ijkl', sigma_tensor, sigma_tensor, sigma_tensor, sigma_tensor) / num_shots)
        
        return results
    
    def _cal_f_functions(self) -> list:
        """Calculate f functions from M expectations using vectorized operations.
        
        Returns:
            List of f tensors [f_i, f_ij, f_ijk, f_ijkl]
        """
        M_i = self.M_list[0]
        M_ij = self.M_list[1]
        f_i = M_i
        # f_ij = M_i * M_j / M_ij (vectorized)
        # Create outer product of M_i with itself
        f_ij = torch.outer(M_i, M_i) / M_ij
        # Set diagonal to 1 (since we don't want f_ii)
        f_ij.fill_diagonal_(1)
        f_list = [f_i, f_ij]
        
        if self.max_order >= 3:
            M_ijk = self.M_list[2]
            numerator_3d = torch.einsum('ijk,i,j,k->ijk', M_ijk, M_i, M_i, M_i)
            denominator_3d = torch.einsum('ij,jk,ik->ijk', M_ij, M_ij, M_ij)
            f_ijk = numerator_3d / denominator_3d
            # Set elements where indices are equal to 1
            for i in range(f_ijk.shape[0]):
                f_ijk[i, i, i] = 1
            f_list.append(f_ijk)
        
        if self.max_order >= 4:
            M_ijkl = self.M_list[3]
            f_ijkl = torch.zeros_like(M_ijkl)
            for i in range(M_ijkl.shape[0]):
                for j in range(M_ijkl.shape[1]):
                    for k in range(M_ijkl.shape[2]):
                        for l in range(M_ijkl.shape[3]):
                            if len(set([i, j, k, l])) == 4:  # All indices are different
                                # Calculate ∏M_i: M_i * M_j * M_k * M_l
                                prod_M_i = M_i[i] * M_i[j] * M_i[k] * M_i[l]
                                
                                # Calculate ∏M_ijk: all combinations of M_ijk
                                prod_M_ijk = (M_ijk[i, j, k] * M_ijk[i, j, l] * 
                                            M_ijk[i, k, l] * M_ijk[j, k, l])
                                
                                # Calculate ∏M_ij: all pairs of M_ij
                                prod_M_ij = (M_ij[i, j] * M_ij[i, k] * M_ij[i, l] * 
                                            M_ij[j, k] * M_ij[j, l] * M_ij[k, l])
                                
                                # f_ijkl = ∏M_ijk * ∏M_i / (M_ijkl * ∏M_ij)
                                numerator = prod_M_ijk * prod_M_i
                                denominator = M_ijkl[i, j, k, l] * prod_M_ij
                                f_ijkl[i, j, k, l] = numerator / denominator
            f_list.append(f_ijkl)
        
        return f_list
    
    def _cal_w_functions(self) -> list:
        """Calculate w functions from f functions following Note.md formulas exactly.
        
        Returns:
            List of w tensors [w_i, w_ij, w_ijk, w_ijkl]
        """
        f_i, f_ij, f_ijk, f_ijkl = self.f_list
        
        # w_ijkl = sqrt[4](f_ijkl) - exact formula
        w_ijkl = torch.pow(f_ijkl, 1/8)
        
        # w_ijk = sqrt[4](f_ijk) / ∏_m w_ijkm - Note.md formula (连等号最后一个等式)
        # We need to compute ∏_m w_ijkm for all m != i,j,k
        # w_ijkm = sqrt[4](f_ijkm), where f_ijkm is approximated by f_ijkl[i,j,k,m]
        w_ijk = torch.zeros_like(f_ijk)
        
        for i in range(f_ijk.shape[0]):
            for j in range(f_ijk.shape[1]):
                for k in range(f_ijk.shape[2]):
                    if i != j and j != k and i != k:  # All indices different
                        # Calculate numerator: sqrt[4](f_ijk)
                        numerator = torch.pow(f_ijk[i, j, k], 1/4)
                        
                        # Calculate denominator: ∏_m w_ijkm where m != i,j,k
                        prod_w_ijkm = 1.0
                        for m in range(f_ijkl.shape[3]):
                            if m != i and m != j and m != k:
                                # w_ijkm = sqrt[4](f_ijkm), using f_ijkl[i,j,k,m] as approximation for f_ijkm
                                prod_w_ijkm *= w_ijkl[i, j, k, m]
                        
                        # w_ijk = sqrt[4](f_ijk) / ∏_m w_ijkm - Note.md formula
                        if prod_w_ijkm != 0:
                            w_ijk[i, j, k] = numerator / prod_w_ijkm
                        else:
                            w_ijk[i, j, k] = numerator
        
        # w_ij = sqrt(f_ij) / (∏_m w_ijm * ∏_{m,n} w_ijmn)
        # We need to compute ∏_m w_ijm and ∏_{m,n} w_ijmn
        w_ij = torch.zeros_like(f_ij)
        
        for i in range(f_ij.shape[0]):
            for j in range(f_ij.shape[1]):
                if i != j:
                    # Calculate ∏_m w_ijm where m != i,j
                    prod_w_ijm = 1.0
                    for m in range(w_ijk.shape[2]):
                        if m != i and m != j:
                            # Use w_ijk[i,j,m] as approximation for w_ijm
                            prod_w_ijm *= w_ijk[i, j, m]
                    
                    # Calculate ∏_{m,n} w_ijmn where m,n != i,j
                    prod_w_ijmn = 1.0
                    for m in range(w_ijkl.shape[2]):
                        if m == i or m == j:
                            continue
                        for n in range(m+1, w_ijkl.shape[3]):
                            if n == i or n == j:
                                continue
                            # Use w_ijkl[i,j,m,n] as approximation for w_ijmn
                            prod_w_ijmn *= w_ijkl[i, j, m, n]
                    
                    # w_ij = sqrt(f_ij) / (∏_m w_ijm * ∏_{m,n} w_ijmn)
                    denominator = prod_w_ijm * prod_w_ijmn
                    if denominator != 0:
                        w_ij[i, j] = torch.sqrt(f_ij[i, j]) / denominator
                    else:
                        w_ij[i, j] = torch.sqrt(f_ij[i, j])
        
        # w_i = M_i / (∏_j w_ij * ∏_{j,k} w_ijk * ∏_{j,k,l} w_ijkl)
        w_i = torch.zeros_like(f_i)
        
        for i in range(f_i.shape[0]):
            # Calculate product of w_ij for all j != i
            prod_w_ij = 1.0
            for j in range(w_ij.shape[1]):
                if i != j:
                    prod_w_ij *= w_ij[i, j]
            
            # Calculate product of w_ijk for all j,k != i
            prod_w_ijk = 1.0
            # 依旧重复计数
            for j in range(w_ijk.shape[1]):
                for k in range(j+1, w_ijk.shape[2]):
                    if i != j and i != k:
                        prod_w_ijk *= w_ijk[i, j, k]
            
            # Calculate product of w_ijkl for all j,k,l != i
            prod_w_ijkl = 1.0
            for j in range(w_ijkl.shape[1]):
                for k in range(j+1, w_ijkl.shape[2]):
                    for l in range(k+1, w_ijkl.shape[3]):
                        if i not in [j, k, l]:  # All indices are different
                            prod_w_ijkl *= w_ijkl[i, j, k, l]
            
            # w_i = M_i / (prod_w_ij * prod_w_ijk * prod_w_ijkl)
            denominator = prod_w_ij * prod_w_ijk * prod_w_ijkl
            if denominator != 0:
                w_i[i] = f_i[i] / denominator
            else:
                w_i[i] = f_i[i]  # If denominator is 0, just use M_i
        
        # Set diagonal elements to 1 (same as f functions)
        w_ij.fill_diagonal_(1)
        for i in range(w_ijk.shape[0]):
            w_ijk[i, i, :] = 1
            w_ijk[i, :, i] = 1
            w_ijk[:, i, i] = 1
        for i in range(w_ijkl.shape[0]):
            w_ijkl[i, i, :, :] = 1
            w_ijkl[i, :, i, :] = 1
            w_ijkl[i, :, :, i] = 1
            w_ijkl[:, i, i, :] = 1
            w_ijkl[:, i, :, i] = 1
            w_ijkl[:, :, i, i] = 1
        
        return [w_i, w_ij, w_ijk, w_ijkl]
    
    def _cal_p_functions(self) -> list:
        """Calculate p functions from w functions using w = 1-2p.
        
        Returns:
            List of p tensors [p_i, p_ij, p_ijk, p_ijkl]
        """
        w_i, w_ij, w_ijk, w_ijkl = self.w_list
        
        # p = (1 - w) / 2
        p_i = (1 - w_i) / 2
        p_ij = (1 - w_ij) / 2
        p_ijk = (1 - w_ijk) / 2
        p_ijkl = (1 - w_ijkl) / 2
        
        # Set diagonal elements to 0 (same as w functions)
        # Note: The diagonal elements are already handled in w functions
        
        return [p_i, p_ij, p_ijk, p_ijkl]
    
    def cal_multi_body_correlations(
            self,
            detector_mask: Optional[np.ndarray] = None,
            hyperedges: Iterable[HyperEdge] = ()
    ) -> CorrelationResult:
        """Calculate the multi-body correlation analytically following Note.md formulas.
        
        This is the main function of the class.
        
        Args:
            detector_mask: A boolean mask of the shape (num_dets_per_shot, ). If True, the
                corresponding detection events will be excluded.
            hyperedges: The hyperedges to be calculated. If None, all hyperedges up to max_order
                will be calculated.
        
        Returns:
            The correlation result containing multi-body correlation probabilities
        """
        if detector_mask is not None:
            detection_events = self.detection_events[:, ~detector_mask]
        else:
            detection_events = self.detection_events
        
        num_dets = detection_events.shape[1]
        if any(i >= num_dets for h in hyperedges for i in h):
            raise ValueError("Hyperedge index out of range.")
        
        # Generate hyperedges if not provided
        if not hyperedges:
            hyperedges = []
            for order in range(1, self.max_order + 1):
                if order == 1:
                    # Single body: {i}
                    hyperedges.extend([frozenset([i]) for i in range(num_dets)])
                elif order == 2:
                    # Two body: {i, j}
                    hyperedges.extend([frozenset([i, j]) for i in range(num_dets) for j in range(i)])
                elif order == 3:
                    # Three body: {i, j, k}
                    hyperedges.extend([frozenset([i, j, k]) for i in range(num_dets) 
                                     for j in range(i) for k in range(j)])
                elif order == 4:
                    # Four body: {i, j, k, l}
                    hyperedges.extend([frozenset([i, j, k, l]) for i in range(num_dets) 
                                     for j in range(i) for k in range(j) for l in range(k)])
        
        # Organize results by hyperedge order
        p_i, p_ij, p_ijk, p_ijkl = self.p_list
        
        # Create result dictionary
        result_data = {}
        
        for hyperedge in hyperedges:
            hyperedge_list = list(hyperedge)
            hyperedge_list.sort()  # Ensure consistent ordering
            
            if len(hyperedge_list) == 1:
                # Single body: p_i
                i = hyperedge_list[0]
                result_data[hyperedge] = p_i[i].cpu().item()
            elif len(hyperedge_list) == 2:
                # Two body: p_ij
                i, j = hyperedge_list
                result_data[hyperedge] = p_ij[i, j].cpu().item()
            elif len(hyperedge_list) == 3:
                # Three body: p_ijk
                i, j, k = hyperedge_list
                result_data[hyperedge] = p_ijk[i, j, k].cpu().item()
            elif len(hyperedge_list) == 4:
                # Four body: p_ijkl
                i, j, k, l = hyperedge_list
                result_data[hyperedge] = p_ijkl[i, j, k, l].cpu().item()
            else:
                raise ValueError(f"Unsupported hyperedge order: {len(hyperedge_list)}")
        
        return CorrelationResult(result_data)
    
    def get_p_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the raw p tensors for direct access.
        
        Returns:
            Tuple of (p_i, p_ij, p_ijk, p_ijkl) tensors
        """
        return tuple(self.p_list)
    
    def get_w_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the raw w tensors for direct access.
        
        Returns:
            Tuple of (w_i, w_ij, w_ijk, w_ijkl) tensors
        """
        return tuple(self.w_list)
    
    def get_f_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the raw f tensors for direct access.
        
        Returns:
            Tuple of (f_i, f_ij, f_ijk, f_ijkl) tensors
        """
        return tuple(self.f_list)
    
    def get_M_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the raw M tensors for direct access.
        
        Returns:
            Tuple of (M_i, M_ij, M_ijk, M_ijkl) tensors
        """
        return tuple(self.M_list)