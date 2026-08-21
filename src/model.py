from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import opt_einsum as oe
from .utils import get_error_rates


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
        self.tree = None

        self.generate_xor_tensors()

        self.generate_equation()

        pass

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
        tree = self.tree
        is_batched = syndromes.ndim == 2
        if not is_batched:
            syndromes = syndromes.unsqueeze(0)
        syndrome_onehot = torch.nn.functional.one_hot(syndromes.long(), num_classes=2).to(dtype=self.dtype, device=self.dev)
        syndrome_vectors = list(syndrome_onehot.unbind(dim=1))


        if probs is None:
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

        # Contraction
        if tree is not None:
            from .utils import contract
            p0_minus_p1 = contract(tree['tree'], operands)
        else:
            p0_minus_p1 = oe.contract(self.eq_str, *operands, optimize='auto')
        return (1-p0_minus_p1.sign())/2

    def forward(self, syndromes, priors_logits=None):
        tree = self.tree
        is_batched = syndromes.ndim == 2
        if not is_batched:
            syndromes = syndromes.unsqueeze(0)

        syndrome_onehot = torch.nn.functional.one_hot(syndromes.long(), num_classes=2).to(dtype=self.dtype, device=self.dev)
        syndrome_vectors = list(syndrome_onehot.unbind(dim=1))
        if priors_logits is None:
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
        # Contraction
        if tree is not None:
            from .utils import contract
            result_normalized = contract(tree['tree'], operands)
        else:
            # Use oe.contract with memory_limit to enable slicing
            result_normalized = oe.contract(self.eq_str, *operands,
                                           optimize='auto',
                                           memory_limit=int(4e9))  # 4GB limit
        eps = 1e-30

        log_likelihood = torch.log(result_normalized + eps) + log_scale_factor

        return - log_likelihood.squeeze(0).mean()
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
            print(f"Error loading tree: {e}")
            self.tree = None


class GroupTN(nn.Module):
    def __init__(self, d, r, sub_pcms, sub_dets, sub_errors, init_priors,
                 dev='cpu', devices=None, dtype=torch.float32, use_tree=True, path_dir=None,
                 parallel_subs=True, manual_sync_grads=False, sub_full_masks=None,
                 stop_grad_partial=False, partial_only_grad=True):
        super().__init__()
        self.d, self.r = d, r
        self.sub_pcms = sub_pcms
        self.n_sub = len(sub_pcms)
        self.sub_dets = sub_dets
        self.sub_errors = sub_errors
        self.stop_grad_partial = bool(stop_grad_partial)
        self.partial_only_grad = bool(partial_only_grad)
        self.sub_full_masks = self._normalize_sub_full_masks(sub_full_masks)
        self.sub_grad_masks = self._build_sub_grad_masks()
        if devices is None:
            devices = [dev]
        self.devices = [str(device) for device in devices]
        if len(self.devices) == 0:
            raise ValueError("GroupTN requires at least one device.")
        self.primary_dev = self.devices[0]
        self.dev = self.primary_dev
        self.dtype = dtype
        self.use_tree = use_tree
        self.parallel_subs = parallel_subs and len(self.devices) > 1
        self.manual_sync_grads = manual_sync_grads and len(self.devices) > 1

        # Default path directory for Julia contraction trees
        if path_dir is None:
            self.path_dir = f'path/d{self.d}r{self.r}'
        else:
            self.path_dir = path_dir

        self.priors_logits = nn.Parameter(torch.logit(init_priors).to(self.primary_dev).to(self.dtype))

        self.use_tree = use_tree
        self._cached_priors_by_device = {}
        self._cached_priors_version = None
        self.construct_tns()

    def _normalize_sub_full_masks(self, sub_full_masks):
        if sub_full_masks is None:
            return None
        if len(sub_full_masks) != self.n_sub:
            raise ValueError(
                f"sub_full_masks length {len(sub_full_masks)} does not match "
                f"number of sub-PCMs {self.n_sub}."
            )

        masks = []
        for i, (mask, errors) in enumerate(zip(sub_full_masks, self.sub_errors)):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape[0] != len(errors):
                raise ValueError(
                    f"sub_full_masks[{i}] length {mask.shape[0]} does not match "
                    f"sub_errors[{i}] length {len(errors)}."
                )
            masks.append(mask)
        return masks

    def _build_sub_grad_masks(self):
        if not self.stop_grad_partial or self.sub_full_masks is None:
            return None

        full_covered_errors = set()
        all_touched_errors = set()
        for errors, full_mask in zip(self.sub_errors, self.sub_full_masks):
            errors = np.asarray(errors, dtype=int)
            all_touched_errors.update(errors.tolist())
            full_covered_errors.update(errors[full_mask].tolist())

        grad_masks = []
        stopped_partial_occurrences = 0
        partial_only_occurrences = 0
        for errors, full_mask in zip(self.sub_errors, self.sub_full_masks):
            errors = np.asarray(errors, dtype=int)
            grad_mask = np.array(full_mask, dtype=bool, copy=True)
            if self.partial_only_grad:
                partial_only = np.array(
                    [error not in full_covered_errors for error in errors],
                    dtype=bool,
                )
                partial_only_occurrences += int(np.count_nonzero(partial_only & ~full_mask))
                grad_mask |= partial_only
            stopped_partial_occurrences += int(np.count_nonzero(~grad_mask))
            grad_masks.append(torch.as_tensor(grad_mask, dtype=torch.bool))

        self.partial_gradient_stats = {
            "stop_grad_partial": True,
            "partial_only_grad": self.partial_only_grad,
            "touched_errors": len(all_touched_errors),
            "full_covered_errors": len(full_covered_errors),
            "partial_only_errors": len(all_touched_errors - full_covered_errors),
            "stopped_partial_occurrences": stopped_partial_occurrences,
            "partial_only_occurrences_with_grad": partial_only_occurrences,
        }
        return grad_masks

    def _refresh_priors_cache(self):
        version = self.priors_logits._version
        if self._cached_priors_version == version and self._cached_priors_by_device:
            return self._cached_priors_by_device

        priors_by_device = {}
        unique_devices = sorted(set(self.tn_devices))
        for dev in unique_devices:
            if dev == self.primary_dev:
                priors_by_device[dev] = self.priors_logits
            else:
                priors_by_device[dev] = self.priors_logits.to(dev, non_blocking=True)
        self._cached_priors_by_device = priors_by_device
        self._cached_priors_version = version
        return priors_by_device

    def _forward_single_sub(self, i, syndromes_by_device, priors_by_device):
        tn_dev = self.tn_devices[i]
        device_syndromes = syndromes_by_device[tn_dev]
        sub_syndromes = device_syndromes[:, self.sub_dets[i]]
        sub_priors_logits = priors_by_device[tn_dev][self.sub_errors[i]]
        if self.sub_grad_masks is not None:
            grad_mask = self.sub_grad_masks[i].to(tn_dev, non_blocking=True)
            sub_priors_logits = torch.where(
                grad_mask,
                sub_priors_logits,
                sub_priors_logits.detach(),
            )
        return self.tns[i].forward(
            sub_syndromes,
            priors_logits=sub_priors_logits
        )

    def construct_tns(self):
        self.tns = nn.ModuleList()
        self.tn_devices = []

        for i in range(self.n_sub):
            pcmi = self.sub_pcms[i]
            for j in range(i):
                if np.array_equal(pcmi, self.sub_pcms[j]):
                    self.tns.append(self.tns[j])
                    self.tn_devices.append(self.tn_devices[j])
                    break
            else:
                tn_dev = self.devices[i % len(self.devices)]
                tn = TensorNetwork(pcm=pcmi, dev=tn_dev, dtype=self.dtype)
                if not self.use_tree:
                    raise ValueError("GroupTN requires Julia JSON contraction trees; use_tree=False is unsupported.")
                tree_file = f'{self.path_dir}/subsample_tree_{i}.json'
                tn.load_tree(tree_file)
                self.tns.append(tn)
                self.tn_devices.append(tn_dev)

    def _prepare_syndromes_by_device(self, syndromes):
        unique_devices = sorted(set(self.tn_devices))
        syndromes_by_device = {}
        for dev in unique_devices:
            if str(syndromes.device) == dev:
                syndromes_by_device[dev] = syndromes
            else:
                syndromes_by_device[dev] = syndromes.to(dev, non_blocking=True)
        return syndromes_by_device

    def _parallel_map_sub_losses(self, syndromes_by_device, priors_by_device):
        if self.parallel_subs:
            with ThreadPoolExecutor(max_workers=self.n_sub) as executor:
                futures = [
                    executor.submit(self._forward_single_sub, i, syndromes_by_device, priors_by_device)
                    for i in range(self.n_sub)
                ]
                return [future.result() for future in futures]
        return [
            self._forward_single_sub(i, syndromes_by_device, priors_by_device)
            for i in range(self.n_sub)
        ]

    def forward(self, syndromes):
        syndromes_by_device = self._prepare_syndromes_by_device(syndromes)
        priors_by_device = self._refresh_priors_cache()
        raw_losses = self._parallel_map_sub_losses(syndromes_by_device, priors_by_device)

        loss_terms = []
        for loss_i in raw_losses:
            if str(loss_i.device) != self.primary_dev:
                loss_i = loss_i.to(self.primary_dev)
            loss_terms.append(loss_i)
        loss = torch.stack(loss_terms).mean()

        return loss

    def manual_sync_loss_and_grad(self, syndromes, loss_scale=1.0):
        if not self.manual_sync_grads:
            loss = self.forward(syndromes) * loss_scale
            loss.backward()
            return loss.detach()

        syndromes_by_device = self._prepare_syndromes_by_device(syndromes)
        unique_devices = sorted(set(self.tn_devices))
        priors_by_device = {}
        for dev in unique_devices:
            priors_by_device[dev] = (
                self.priors_logits.detach().to(dev, non_blocking=True).requires_grad_(True)
            )

        raw_losses = self._parallel_map_sub_losses(syndromes_by_device, priors_by_device)
        per_device_losses = {dev: [] for dev in unique_devices}
        total_loss_value = 0.0
        for i, loss_i in enumerate(raw_losses):
            per_device_losses[self.tn_devices[i]].append(loss_i / self.n_sub)
            total_loss_value += loss_i.detach().to(self.primary_dev).item() / self.n_sub

        total_grad = torch.zeros_like(self.priors_logits, device=self.primary_dev)
        for dev, loss_terms in per_device_losses.items():
            if not loss_terms:
                continue
            local_loss = torch.stack(loss_terms).sum() * loss_scale
            local_loss.backward()
            local_grad = priors_by_device[dev].grad
            if local_grad is not None:
                total_grad.add_(local_grad.to(self.primary_dev, non_blocking=True))

        if self.priors_logits.grad is None:
            self.priors_logits.grad = total_grad
        else:
            self.priors_logits.grad.copy_(total_grad)

        return torch.tensor(total_loss_value * loss_scale, device=self.primary_dev, dtype=self.dtype)

    def sequential_sub_loss_and_grad(self, syndromes, loss_scale=1.0):
        """Backward each sub-TN loss immediately while preserving the mean-loss objective.

        This is useful on a single large GPU: it avoids keeping all sub-TN
        autograd graphs alive at the same time. The optimizer step should still
        be called once after all sub losses have accumulated.
        """
        syndromes_by_device = self._prepare_syndromes_by_device(syndromes)
        priors_by_device = self._refresh_priors_cache()
        total_loss_value = 0.0

        for i in range(self.n_sub):
            loss_i = self._forward_single_sub(i, syndromes_by_device, priors_by_device)
            scaled_loss = loss_i * (loss_scale / self.n_sub)
            scaled_loss.backward()
            total_loss_value += scaled_loss.detach().to(self.primary_dev).item()

        return torch.tensor(total_loss_value, device=self.primary_dev, dtype=self.dtype)
