import networkx as nx
import numpy as np
from math import atan2, pi, log
import scipy
from scipy.sparse import lil_matrix
import torch
import stim
from ldpc.mod2 import row_echelon, row_basis, rank

class rep_dem():
    def __init__(self, dem):
        self.dem = dem
        self.hx, self.lx = PCM(dem)
        self.hz, self.lz = build_dual_matrix(self.hx, self.lx, seperate_dual_l=True)
        self.pebz = get_pseudo_inverse(self.hx)

        assert ((self.hx @ self.hz.T)%2).sum() == 0 
        assert ((self.hx @ self.lz.reshape(-1).T)%2).sum() == 0
        assert ((self.lx.reshape(-1) @ self.lz.reshape(-1))%2) == 1
        assert (self.pebz @ self.hx.T % 2 - np.eye(self.hx.shape[0], self.hx.shape[0])).all() == 0

def build_dual_matrix(pcm, l , seperate_dual_l=False):

    pcm_l = np.concatenate([pcm, l.reshape(1, -1)], axis=0)
    compact_var = pcm_l.sum(0)%2
    compact_pcm_l = np.concatenate([compact_var.reshape(1, -1), pcm_l], axis=0)
    # print(compact_pcm_l)
    rows, num_edges = compact_pcm_l.shape

    G_primal = nx.Graph()
    G_primal.add_nodes_from(range(int(rows)))
    

    node_pair_to_edge_idx = {}
    
    for c in range(num_edges):
        connected_nodes = np.where(compact_pcm_l[:, c] == 1)[0]

        if len(connected_nodes) == 2:
            u, v = int(connected_nodes[0]), int(connected_nodes[1])
            G_primal.add_edge(u, v, id=c)
            
            key = tuple(sorted((u, v)))
            node_pair_to_edge_idx[key] = c
            
        elif len(connected_nodes) == 1:

            pass

    cycles_nodes = nx.minimum_cycle_basis(G_primal)
    
    # Filter to only keep simple cycles (cycles where all consecutive nodes are directly connected)
    # Non-simple cycles are cycles in the cycle space but not actual faces in the graph
    simple_cycles_basis = []
    for cycle in cycles_nodes:
        is_simple = True
        for k in range(len(cycle)):
            u = cycle[k]
            v = cycle[(k + 1) % len(cycle)]
            if not G_primal.has_edge(u, v):
                is_simple = False
                break
        if is_simple:
            simple_cycles_basis.append(cycle)
    
    # Find all simple cycles and select linearly independent ones
    # Expected number of faces from Euler's formula: |E| - |V| + 1
    expected_faces = G_primal.number_of_edges() - G_primal.number_of_nodes() + 1
    
    # Get all simple cycles, but only consider cycles of length 3 and 4 (triangles and quadrilaterals)
    # For the graphs we handle, only 3-edge and 4-edge faces exist
    all_simple_cycles = list(nx.simple_cycles(G_primal))
    
    # Filter to only keep cycles of length 3 or 4
    valid_cycles = [c for c in all_simple_cycles if len(c) == 3 or len(c) == 4]
    
    # Separate cycles by length
    cycles_by_length = {}
    for cycle in valid_cycles:
        length = len(cycle)
        if length not in cycles_by_length:
            cycles_by_length[length] = []
        cycles_by_length[length].append(cycle)
    
    # Build prioritized list: first length 3 cycles, then length 4 cycles
    # Within each length, prioritize those in the basis
    prioritized_cycles = []
    basis_cycle_set = set(tuple(sorted(c)) for c in simple_cycles_basis)
    
    # First add length 3 cycles (triangles) - basis first, then others
    if 3 in cycles_by_length:
        for cycle in cycles_by_length[3]:
            if tuple(sorted(cycle)) in basis_cycle_set:
                prioritized_cycles.append(cycle)
        for cycle in cycles_by_length[3]:
            if tuple(sorted(cycle)) not in basis_cycle_set:
                prioritized_cycles.append(cycle)
    
    # Then add length 4 cycles (quadrilaterals) - basis first, then others
    if 4 in cycles_by_length:
        for cycle in cycles_by_length[4]:
            if tuple(sorted(cycle)) in basis_cycle_set:
                prioritized_cycles.append(cycle)
        for cycle in cycles_by_length[4]:
            if tuple(sorted(cycle)) not in basis_cycle_set:
                prioritized_cycles.append(cycle)
    
    all_simple_cycles = prioritized_cycles
    
    # Represent cycles as edge vectors (mod 2) and find linearly independent set
    cycle_vectors = []
    cycle_list = []
    
    # Helper function to represent cycle as edge vector
    def cycle_to_vector(cycle):
        edge_vector = np.zeros(num_edges, dtype=int)
        for k in range(len(cycle)):
            u = cycle[k]
            v = cycle[(k + 1) % len(cycle)]
            key = tuple(sorted((u, v)))
            if key in node_pair_to_edge_idx:
                e_idx = node_pair_to_edge_idx[key]
                edge_vector[e_idx] = 1
        return edge_vector
    
    # Helper function to check linear independence
    def is_linearly_independent(edge_vector, existing_vectors):
        if len(existing_vectors) == 0:
            return True
        current_matrix = np.array(existing_vectors, dtype=int) % 2
        aug_matrix = np.vstack([current_matrix, edge_vector.reshape(1, -1)]) % 2
        rank_current = np.linalg.matrix_rank(current_matrix)
        rank_aug = np.linalg.matrix_rank(aug_matrix)
        return rank_aug > rank_current
    
    # Helper function to check if a cycle can be expressed as XOR of smaller cycles
    def is_elementary_face(cycle, smaller_cycles_vectors):
        """Check if cycle is an elementary face (cannot be expressed as XOR of smaller cycles)
        
        Optimized version: Only checks pairs of cycles that share edges with the candidate cycle.
        This reduces complexity from O(n^2) to O(k^2) where k is the number of cycles sharing edges.
        """
        if len(cycle) == 3:
            return True  # Triangles are always elementary
        if len(smaller_cycles_vectors) == 0:
            return True
        
        cycle_vector = cycle_to_vector(cycle)
        cycle_edges = set(np.where(cycle_vector > 0)[0])
        
        # Optimization: Only check cycles that share at least one edge with the candidate
        # This dramatically reduces the number of pairs to check
        candidate_cycles = []
        candidate_indices = []
        for idx, vec in enumerate(smaller_cycles_vectors):
            vec_edges = set(np.where(vec > 0)[0])
            if cycle_edges.intersection(vec_edges):  # Share at least one edge
                candidate_cycles.append(vec)
                candidate_indices.append(idx)
        
        # If no cycles share edges, it's definitely elementary
        if len(candidate_cycles) == 0:
            return True
        
        # Check pairs of candidate cycles (much smaller set)
        candidate_matrix = np.array(candidate_cycles, dtype=int) % 2
        num_candidates = len(candidate_cycles)
        
        # Check all pairs of candidate cycles
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                xor_result = (candidate_matrix[i] + candidate_matrix[j]) % 2
                if np.array_equal(cycle_vector, xor_result):
                    return False  # Found as XOR of two smaller cycles
        
        # Also check if it's in the linear span (for more complex combinations)
        # Use all smaller cycles for this check to catch any linear combination
        smaller_matrix = np.array(smaller_cycles_vectors, dtype=int) % 2
        aug_matrix = np.vstack([smaller_matrix, cycle_vector.reshape(1, -1)]) % 2
        rank_smaller = np.linalg.matrix_rank(smaller_matrix)
        rank_aug = np.linalg.matrix_rank(aug_matrix)
        # If rank doesn't increase, it's in the span (not elementary)
        return rank_aug > rank_smaller
    
    # First, add all length-3 cycles (triangles are always elementary faces)
    length3_cycles = [c for c in prioritized_cycles if len(c) == 3]
    for cycle in length3_cycles:
        edge_vector = cycle_to_vector(cycle)
        if is_linearly_independent(edge_vector, cycle_vectors):
            cycle_vectors.append(edge_vector)
            cycle_list.append(cycle)
    
    # Store length-3 cycle vectors for checking length-4 cycles
    length3_vectors = cycle_vectors.copy()
    
    # Then add length-4 cycles that are elementary faces
    # Important: Check against ALL already selected cycles (both length-3 and length-4)
    # to ensure we don't add composite faces that can be formed by XORing existing cycles
    length4_cycles = [c for c in prioritized_cycles if len(c) == 4]
    for cycle in length4_cycles:
        if len(cycle_list) >= expected_faces:
            break
        if cycle in cycle_list:
            continue  # Already added
        # Check if it's an elementary face (cannot be expressed as XOR of ANY smaller cycles)
        # Use all cycle_vectors (including already added length-4 cycles) for checking
        if is_elementary_face(cycle, cycle_vectors):
            edge_vector = cycle_to_vector(cycle)
            if is_linearly_independent(edge_vector, cycle_vectors):
                cycle_vectors.append(edge_vector)
                cycle_list.append(cycle)
    
    simple_cycles = cycle_list
    num_faces = len(simple_cycles)


    edge_to_faces = {i: [] for i in range(num_edges)}
    
    for face_id, cycle in enumerate(simple_cycles):

        for k in range(len(cycle)):
            u = cycle[k]
            v = cycle[(k + 1) % len(cycle)]
            key = tuple(sorted((u, v)))
            
            if key in node_pair_to_edge_idx:
                e_idx = node_pair_to_edge_idx[key]
                edge_to_faces[e_idx].append(face_id)

    dual_matrix = np.zeros((num_faces, num_edges), dtype=int)
    for e_idx, faces in edge_to_faces.items():
        for f_id in faces:
            dual_matrix[f_id, e_idx] = 1
    dual_l = dual_matrix.sum(0)%2
    dual_l[:len(dual_l)//2] = 0  
    if seperate_dual_l:
        return dual_matrix, dual_l
    else:
        return np.concatenate([dual_matrix, dual_l.reshape(1, -1)], axis=0)


def get_pseudo_inverse(H):
   
    H = np.array(H, dtype=int)
    m, n = H.shape
    

    temp_H = H.copy()
    pivots = []
    row = 0
    
    for col in range(n):
        if row >= m:
            break
            

        pivot_candidates = np.where(temp_H[row:, col] == 1)[0]
        
        if len(pivot_candidates) == 0:
            continue 
            

        pivot_row = row + pivot_candidates[0]
        temp_H[[row, pivot_row]] = temp_H[[pivot_row, row]]
        
        pivots.append(col)
        

        idx = np.where(temp_H[:, col] == 1)[0]
        idx = idx[idx != row] 
        temp_H[idx] ^= temp_H[row]
        
        row += 1


    if len(pivots) < m:
        raise ValueError(f"Matrix is not full row rank. Rank={len(pivots)}, Rows={m}")


    A = H[:, pivots]
    

    aug = np.hstack([A, np.eye(m, dtype=int)])

    for i in range(m):

        if aug[i, i] == 0:
            swap_candidates = np.where(aug[i+1:, i] == 1)[0]
            if len(swap_candidates) == 0:
                raise ValueError("Singular submatrix encountered.")
            swap_row = i + 1 + swap_candidates[0]
            aug[[i, swap_row]] = aug[[swap_row, i]]
        

        idx = np.where(aug[:, i] == 1)[0]
        idx = idx[idx != i] 
        aug[idx] ^= aug[i] 
        

    A_inv = aug[:, m:]
    
    H_inv = np.zeros((n, m), dtype=int)
    H_inv[pivots, :] = A_inv
    
    return H_inv.T






class rep_cir():
    def __init__(self, d, r):
        self.d = d # number of data qubits
        self.r = r # number of rounds
        self.n = 3*(d-1)*r+d
        self.m = (d-1)*(r+1)
        self.hx, self.hz, self.lx, self.lz = self.gen_rep_cir() # hx: parity check; hz: dual of hx; lx: logical check; lz: logical error.
        self.pure_error_basis()

    def gen_rep_cir(self):
        r = self.r
        d = self.d - 1 # number of ancilla qubits
        l=d+1
        n = 3*(l-1)*r+l
        mx = (l-1)*(r+1)
        mz = n-mx-1
        assert mz == (2*l-2)*r
        Hz=np.zeros((mz, n))
        Hx_Lx=np.zeros((mx+1, n))
        Lz=np.zeros(n)

        self.E = []

        self.boundary_nodes = []
        for i in range(mx):
            if (i+1)%d==0:
                self.boundary_nodes.append(i)
                self.E.append([i])
                if int(i/d) != r:
                    self.E.append([i, i+d-1])
                    self.E.append([i, i+d])

            else:
                self.E.append([i, i+1])
                if i%d==0: 
                    if int(i/d) != r:
                        self.E.append([i, i+d])
                    self.E.append([-1, i])
                    self.boundary_nodes.append(i)
                
                elif i%d !=0 and int(i/d) != r:
                    self.E.append([i, i+d-1])
                    self.E.append([i, i+d])

        self.single_error = []
        for i in range(n):
            Hx_Lx[self.E[i], i] = 1
            if len(self.E[i]) == 1:
                self.single_error.append(i)
            elif -1 in self.E[i]:
                self.single_error.append(i)
        Hx = Hx_Lx[:-1]
        Lx = Hx_Lx[-1]       
        Lz[-l:]=1
        # print(E)

        for i in range(mz):
            a, b = int(i/2), i%2
            if b ==  0:
                if i%(2*l-2) == 0:

                    if int(i/(2*l-2)) == r-1:
                        Hz[i, (i+a+1, i+a+2, i+a+3*d+1)] = 1
                        # print(i, i+a+1, i+a+2, i+a+3*d+1)
                    else:
                        Hz[i, (i+a+1, i+a+2, i+a+3*d+2)] = 1
                        # print(i, i+a+1, i+a+2, i+a+3*d+2)
                else:
                    if i > (r-1)*(2*l-2)+2:
                        # print(i, int((mz-i)/2))
                        Hz[i, (i+a+1, i+a+2, n-1-int((mz-i)/2))] = 1
                        # print(i, i+a+1, i+a+2, n-1-int((mz-i)/2))
                    else:
                        Hz[i, (i+a+1, i+a+2, i+a+3*(d-1))] = 1
                        # print(i, i+a+1, i+a+2, i+a+3*(d-1))
            else:
                if i%(2*l-2) == 1:
                    Hz[i, (3*a, 3*a+1, 3*a+4)] = 1
                    # print(i, 3*a, 3*a+1, 3*a+4)
                elif (i+1)%(2*l-2) == 0:
                    if i == mz-1:
                        Hz[i, (3*a, 3*a+2, n-1)] = 1
                        # print(i, 3*a, 3*a+2, n-1)
                    else:
                        Hz[i, (3*a, 3*a+2, 3*a+3*d)] = 1
                        # print(i, 3*a, 3*a+2, 3*a+3*d)
                else:
                    Hz[i, (3*a, 3*a+2, 3*a+4)] = 1
                    # print(i, 3*a, 3*a+2, 3*a+4)


        assert ((Hx@Hz.T)%2).sum() == 0
        assert ((Lx@Hz.T)%2).sum() == 0
        assert ((Lz@Hx.T)%2).sum() == 0
        assert ((Lz@Lx.T)%2).sum() == 1
        return  Hx, Hz, Lx, Lz

    def pure_error_basis(self):
        n = self.n
        dm = self.d - 1
        self.pebz = np.zeros((dm*(self.r+1), n))
        self.pebz[self.boundary_nodes, self.single_error] = 1
        
        for i in range(self.m):
            if i not in self.boundary_nodes:
                dis = abs(i - np.array(self.boundary_nodes))
                mdis = np.min(dis)
                eidx = np.argmin(dis)
                didx = self.boundary_nodes[eidx]
                self.pebz[i, self.single_error[eidx]] = 1

                for j in range(mdis):
                    # print('mdis:', mdis)
                    if i > didx:
                        # print(didx+j, didx+j+1)
                        self.pebz[i, np.where(self.hx[[didx+j,didx+j+1]].sum(0)==2)[0]] = 1
                        
                    else:
                        # print(didx-j, didx-j-1)
                        self.pebz[i, np.where(self.hx[[didx-j,didx-j-1]].sum(0)==2)[0]] = 1
        
        assert (self.hx @ self.pebz.T % 2 - np.eye(self.m, self.m)).all() == 0
 

    def reorder(self, dem, rotated_graph=True):
        d=self.d-1
        if rotated_graph == False:
            E1 = []
            for i, e in enumerate(dem[:dem.num_errors]):
                Dec = e.targets_copy()
                edge1 = []
                for j in range(len(Dec)):
                    D = str(Dec[j])
                    if D.startswith('D'):
                        idx = int(D[1:])
                        edge1.append(idx)
                    if D.startswith('L'):
                        idx = int(D[1:])
                        edge1.append(-1)
                edge1.sort()
                # order.append(E.index(edge1))
                E1.append(edge1) 
        elif rotated_graph == True:
            dorder = [(d-i%d-1)+int(i/d)*d for i in range(dem.num_detectors)]
            E1 = []
            detectors = []
            for i, e in enumerate(dem[:dem.num_errors]):
                Dec = e.targets_copy()
                edge1 = []
                for j in range(len(Dec)):
                    D = str(Dec[j])
                    if D.startswith('D'):
                        idx = int(D[1:])
                        row = int(idx/d)
                        cul = d-idx%d-1
                        nidx = d*row+cul
                        
                        edge1.append(nidx)
                        if nidx not in detectors:
                            detectors.append(nidx)
                    if D.startswith('L'):
                        idx = int(D[1:])
                        edge1.append(-1)
                edge1.sort()
      
                E1.append(edge1)
 
        eorder = []
        for i in range(self.n):
            index = self.E.index(E1[i])
            eorder.append(index)

        
        if rotated_graph == False:
            self.hx = self.hx[:, eorder]
            self.pebz = self.pebz[:, eorder]
            self.hz = self.hz[:, eorder]
            self.lx = self.hx[:, eorder]
            self.lz = self.hz[:, eorder]

        elif rotated_graph == True:
            self.hx = self.hx[:, eorder]
            self.hx = self.hx[dorder, :]
            self.pebz = self.pebz[:, eorder]
            self.pebz = self.pebz[dorder, :]
            assert (self.hx @ self.pebz.T % 2 - np.eye(self.m, self.m)).all() == 0

            self.hz = self.hz[:, eorder]
            self.lx = self.lx[eorder]
            self.lz = self.lz[eorder]

            assert ((self.hx @ self.hz.T)%2).sum() == 0
            assert ((self.lx @ self.hz.T)%2).sum() == 0
            assert ((self.lz @ self.hx.T)%2).sum() == 0
            assert ((self.lx @ self.lz.T)%2).sum() == 1
            return eorder, dorder



def get_error_rates(dem):
    num_ems = dem.num_errors   
    er = []
    for i in range(num_ems):
        er.append(dem[i].args_copy()[0])
    return np.array(er)

def get_weights(dem):
    num_ems = dem.num_errors   
    er = []
    for i in range(num_ems):
        er.append(dem[i].args_copy()[0])
    return np.array(np.log((1-np.array(er))/np.array(er)))
   

class KacWardSolution:
    def __init__(self, adj_matrix) -> None:
        adj = scipy.sparse.csr_matrix(adj_matrix)
        graph = nx.from_scipy_sparse_array(adj)
        is_planar, _ = nx.check_planarity(graph)
        if not is_planar:
            raise ValueError("The graph is not planar, the KacWard solution does not apply.")
        pos = nx.planar_layout(graph)
        self.n, self.m = graph.number_of_nodes(), graph.number_of_edges()
        di_edge_list = sum([[(i,j), (j,i)] for (i,j) in graph.edges()], start=[])
        self.di_edge_dict = {edge:idx for idx, edge in enumerate(di_edge_list)}
        self.nonbacktrack = scipy.sparse.lil_matrix(
            (2*self.m, 2*self.m), dtype=np.complex128
        ) # the non-backtracking matrix of size 2m x 2m
        for i, j in di_edge_list:
            for k in graph.neighbors(j):
                if k != i:
                    self.nonbacktrack[self.di_edge_dict[(i,j)], self.di_edge_dict[(j,k)]] = \
                        np.exp(1J/2. * compute_angle(pos[i], pos[j], pos[k]))
        pass

    def torch_logZ(self, weights):
        # Compute diagonal matrix D (sparse in PyTorch)
        dtype = weights.dtype
        if dtype == torch.float64:
            cdtype = torch.complex128
        else:
            cdtype = torch.complex64
        D_diag = torch.tanh(weights)
        D = torch.diag_embed(D_diag).to(cdtype)
        logZ = 0.5 * torch.log(torch.cosh(weights)).sum(1)
        logZ = logZ + self.n * torch.log(torch.tensor(2.0, dtype=dtype, device=weights.device))
        nonbacktrack = torch.tensor(self.nonbacktrack.toarray(), dtype=cdtype, device=weights.device)

        # Compute A matrix
        A = torch.eye(2*self.m, dtype=cdtype, device=weights.device) - torch.einsum('ij,bjk->bik', nonbacktrack, D)
        logdet = torch.log(torch.det(A)+1e-20)  
        logdet = logdet.real
        logZ = logZ + 0.5 * logdet
        return logZ
  
  
def logcosh(x):
    return np.abs(x) + np.log(1+np.exp(-2.0*np.abs(x))) - np.log(2.0)
    
def compute_angle(i,j,k):
    """ return angle difference between (ij) and (jk) (or, in other words, k to j and j to i), giving the 2D coordinates of the points.
    """ 
    k2j = j-k
    j2i = i-j
    return (atan2(k2j[1],k2j[0]) - atan2(j2i[1],j2i[0]) + pi)%(2.0*pi) - pi

def minmax(x, y):
    return min(x, y), max(x, y)

def construct_kac_ward_solution(generate_matrix):
    n = generate_matrix.shape[0]
    edges_dict = {}
    for m in range(generate_matrix.shape[1]):
        edge = generate_matrix[:, m].nonzero()[0]
        assert len(edge) <= 2, print('non planar graph')
        if len(edge) == 1:
            if (edge[0], n) not in edges_dict.keys():
                edges_dict[(edge[0], n)] = [m]
            else:
                edges_dict[(edge[0], n)].append(m)
        else:
            assert edge[0] < edge[1]
            if (edge[0], edge[1]) not in edges_dict.keys():
                edges_dict[(edge[0], edge[1])] = [m]
            else:
                edges_dict[(edge[0], edge[1])].append(m)
    g = nx.from_edgelist(edges_dict.keys())
    kwsolution = KacWardSolution(nx.adjacency_matrix(g, nodelist=np.arange(n+1)))
    return kwsolution, edges_dict

def torch_rep_cir_log_coset_p(operator, solution:KacWardSolution, edges_dict, error_rates):
    if len(operator.shape) < 2:
        operator = operator.unsqueeze(0)

    weight = 0.5*torch.log((1-error_rates)/error_rates)
    weights = weight*(1-2*operator)
    
    weights_nonbacktrack = torch.stack([
        torch.sum(weights[:, edges_dict[minmax(*edge)]], dim=1) for edge in solution.di_edge_dict
    ], dim=1) 

    logZ = solution.torch_logZ(weights_nonbacktrack)

    constants = 0.5* torch.log(error_rates*(1-error_rates)).sum() - log(2)
    logZ += constants
    return logZ

def exact_coset_prob(summation_var, operator, error_rates):
    if operator.dtype != torch.long:
        operator = operator.long()

    m = summation_var.shape[0]
    bitstrings = torch.tensor(
        [list(map(int, bin(x)[2:].zfill(m))) for x in range(2**m)]
    , dtype=torch.long)

    operators = (bitstrings @ torch.tensor(summation_var).T + operator) % 2

    prob_matrix = torch.stack([1-error_rates, error_rates], axis=1)
    prob_matrix_expanded = prob_matrix.unsqueeze(0)  # [1, n, 2]
    

    operators_expanded = operators.unsqueeze(-1)  # [b, n, 1]
    selected_probs = torch.gather(prob_matrix_expanded, dim=2, index=operators_expanded)
    coset_prob = selected_probs.squeeze(-1).prod(dim=1).sum(0)
    return coset_prob


def numerical_gradient(logp, pe, er, index, eps=1e-5):
    x_plus = er.clone()
    x_minus = er.clone()
    x_plus[index] += eps
    x_minus[index] -= eps
    return (logp(pe, x_plus) - logp(pe, x_minus)) / (2 * eps)


def PCM(dem):
    
    pcm = np.zeros([dem.num_detectors, dem.num_errors])
    l = np.zeros([dem.num_observables, dem.num_errors])
    
   
    for i, e in enumerate(dem[:dem.num_errors]):
        Dec = e.targets_copy()

        for j in range(len(Dec)):
            D = str(Dec[j])
            if D.startswith('D'):
                idx = int(D[1:])
                pcm[idx, i] = 1.#e.args_copy()[0]

            elif D.startswith('L'):
                idx = int(D[1:])
                l[idx, i] = 1
            
    non_zero_rows = np.where(pcm.sum(axis=1) != 0)[0]
    pcm = pcm[non_zero_rows, :] 
    return  pcm, l

def update_dem(dem, ers):
    new_dem = stim.DetectorErrorModel()
    for i, instruction in enumerate(dem):
    # print(instruction.type)
        if instruction.type == "error":  
            args = instruction.args_copy()
            targets = instruction.targets_copy()
            new_p = ers[i]  
            new_dem.append(stim.DemInstruction(
                "error",  
                args=[new_p],
                targets=targets
            ))
        else:
            new_dem.append(instruction)
    return new_dem

def subsamples(ds, d, r, dem):
    '''
    det : detector-idices in original rep
    em : error-indices in original rep (after merge)
    er : error rates of sub-rep (after merge)
    merge_e : indices of merged em in sub-rep, indices of two original ems in original rep
    '''
    n = dem.num_errors
    ns = d-ds+1
    subsample = {}.fromkeys(range(ns))
    
    for i in range(ns):
        subsample[i] = ({'det':[], 'em':[], 'er':[], 'merge_e':[]})
        for k in range(r+1):
            for j in range(ds-1):
                subsample[i]['det'].append(i+j+k*(d-1))

        detectors = subsample[i]['det']
        for j, e in enumerate(dem[:n]):
            if j not in subsample[i]['em']:
                Dec = e.targets_copy()
                if len(Dec) == 1 and str(Dec[0]).startswith('D') :
                    if int(str(Dec[0])[1:]) in detectors:
                        subsample[i]['em'].append(j)
                        subsample[i]['er'].append(e.args_copy()[0])
                else:
                    D0 = int(str(Dec[0])[1:])
                    D1 = int(str(Dec[1])[1:])
                    if str(Dec[0]).startswith('D') and str(Dec[1]).startswith('L') and D0 in detectors:
                        subsample[i]['em'].append(j)
                        subsample[i]['er'].append(e.args_copy()[0])
                    elif str(Dec[0]).startswith('D') and str(Dec[1]).startswith('D'):
                        if D0 in detectors and D1 in detectors:
                            subsample[i]['em'].append(j)
                            subsample[i]['er'].append(e.args_copy()[0])
                        elif  D0 not in detectors and D1 in detectors:
                            if D1-D0==1:
                                subsample[i]['em'].append(j)
                                # subsample[i]['em'].append((j, j-3*(d-1)+2)) 
                                if D1 == detectors[0]:
                                    subsample[i]['er'].append(e.args_copy()[0])
                                else:
                                    subsample[i]['er'].append(e.args_copy()[0]*(1-dem[j-3*(d-1)+2].args_copy()[0]) + (1-e.args_copy()[0])*dem[j-3*(d-1)+2].args_copy()[0])
                                    subsample[i]['merge_e'].append((subsample[i]['em'].index(j), j, j-3*(d-1)+2))         
                        elif  D0 in detectors and D1 not in detectors:
                            if D1-D0==1:
                                subsample[i]['em'].append(j)
                                if D0 == detectors[-1]:
                                    subsample[i]['er'].append(e.args_copy()[0])
                                else:
                                    subsample[i]['em'].insert(-1, j+1)
                                    subsample[i]['er'].append(dem[j+1].args_copy()[0])
                                    subsample[i]['er'].append(e.args_copy()[0]*(1-dem[j+2].args_copy()[0]) + (1-e.args_copy()[0])*dem[j+2].args_copy()[0])                     
                                    subsample[i]['merge_e'].append((subsample[i]['em'].index(j), j, j+2))
                    else:
                        None

    return subsample


def subsample_d3_pcms(d, r, print_info=False):

    def d3_coors(center_coor):
        dets_coors = np.array([[2, 0], [2, 2], [4, 2], [6, 2], [0, 4], [2, 4], [4, 4], [4, 6]])
        dets_coors = dets_coors+center_coor-np.array([3, 3])
        return dets_coors
    
    sc_circuit = stim.Circuit.generated(code_task="surface_code:rotated_memory_z",
                                            distance=d,
                                            rounds=r,
                                            after_clifford_depolarization=0.01,
                                            before_measure_flip_probability=0.01,
                                            after_reset_flip_probability=0.01,
                                            )
    dem = sc_circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    pcm_d, _= PCM(dem)
    coors_d = sc_circuit.get_detector_coordinates()

    layers = (d-1)//2

    sub_pcms = []
    sub_errors = []
    sub_dets = []
    center_coors = []
    total_edges = set()
    for l in range(layers, 0, -1):
        n_sub_l = l**2
        l_idx = layers-l
        first_center_coor = np.array([3+2*l_idx, 3+2*l_idx])
        for i in range(n_sub_l):
            center_coor_i = first_center_coor+np.array([4*(i%l), 4*(i//l)])
            det_coors = d3_coors(center_coor_i)
            rows = []
            if tuple(center_coor_i) not in center_coors:
                center_coors.append(tuple(center_coor_i))
                
                for idx in range(len(det_coors)):
                    det_coor = det_coors[idx]
                    for d_idx in range(len(coors_d)):
                        coor_d = np.array(coors_d[d_idx][:2]).astype(np.int8)
                        if np.equal(coor_d, det_coor).all():
                            rows.append(d_idx)
                rows.sort()
                
                non_zero_colums = np.nonzero(pcm_d[rows].sum(0))[0]
                sub_pcm = pcm_d[rows, :][:, non_zero_colums]
                sub_dets.append(rows)
                sub_pcms.append(sub_pcm)
                sub_errors.append(non_zero_colums)
                if print_info == True:
                    print('layer:',l, 'index:', i, 'center coor:', center_coor_i)
                    print('pcm shape:', sub_pcm.shape)
                    nte, des = len(non_zero_colums), np.count_nonzero((pcm_d[rows].sum(0)%2))
                    print('non trivial errors:', nte)
                    print('dangling edges:', des)
                    print('non dangling edges:', nte-des)
                    total_edges = total_edges | set(non_zero_colums)
    if print_info:
        print('original pcm shape:', pcm_d.shape)
        print('original dangling edges:', np.count_nonzero((pcm_d.sum(0)%2)))
        print('total number of sub edges:', len(total_edges))
        print('total number of d3 scs:', len(sub_pcms))
    return sub_pcms, sub_dets, sub_errors

def extract_layer_coordinates(dem):
    """
    Extract coordinates from layer 0 and layer 1 of a DEM.
    
    Args:
        dem: DetectorErrorModel object
        
    Returns:
        tuple: (layer0_coords, layer1_coords_set)
            - layer0_coords: List of (x, y) coordinates from layer 0 (z=0)
            - layer1_coords_set: Set of (x, y) coordinates from layer 1 (z=1)
    """
    dem_flat = dem.flattened()
    
    # Extract all coordinates from layer 1 (whole lattice knowledge)
    layer1_coords = set()
    for instruction in dem_flat:
        if isinstance(instruction, stim.DemInstruction):
            if instruction.type == "detector":
                args = instruction.args_copy()
                if len(args) >= 3:
                    x, y, z = int(args[0]), int(args[1]), int(args[2])
                    if z == 1:  # Layer 1
                        layer1_coords.add((x, y))
    
    # Extract all coordinates from layer 0
    layer0_coords = []
    for instruction in dem_flat:
        if isinstance(instruction, stim.DemInstruction):
            if instruction.type == "detector":
                args = instruction.args_copy()
                if len(args) >= 3:
                    x, y, z = int(args[0]), int(args[1]), int(args[2])
                    if z == 0:  # Layer 0
                        coord = (x, y)
                        if coord not in layer0_coords:
                            layer0_coords.append(coord)
    
    return layer0_coords, layer1_coords




def generate_compactified_pcm_from_seperated_dem(dem: stim.DetectorErrorModel):
    '''The 'decompose_errors' must be True'''
    num_detectors = dem.num_detectors
    # key: 节点对元组, value: { 'sources': 指令索引集合, 'probs': 概率列表 }
    edge_map = {} 
    
    ignored_hyperedges_indices = []

    for i, instruction in enumerate(dem):
        if instruction.type != "error":
            continue

        prob = instruction.args_copy()[0] 
        all_targets = instruction.targets_copy()

        all_parts = []
        current_part = []
        for t in all_targets:
            if t.is_separator():
                if current_part:
                    all_parts.append(current_part)
                current_part = []
            elif t.is_relative_detector_id():
                current_part.append(t.val)
        if current_part:
            all_parts.append(current_part)

        if not all_parts:
            continue

 
        is_graphlike = (len(all_parts[0]) <= 2)
        has_decomposition = (len(all_parts) > 1)

        if not is_graphlike and not has_decomposition:
            # 这是一个没有分解建议的超边，根据 PyMatching 规则忽略
            ignored_hyperedges_indices.append(i)
            continue

        for part in all_parts:
            if len(part) <= 2 and len(part) > 0:
                pair = tuple(sorted(part))
                if pair not in edge_map:
                    edge_map[pair] = {"sources": set(), "probs": []}
                
                edge_map[pair]["sources"].add(i)
                edge_map[pair]["probs"].append(prob)

    # --- 3. 构建 PCM 和映射表 ---
    unique_pairs = list(edge_map.keys())
    num_new_edges = len(unique_pairs)
    pcm_compactified = lil_matrix((num_detectors, num_new_edges), dtype=int)
    edges_mapping = []
    error_probs = []
    for idx, pair in enumerate(unique_pairs):
        # 填充 PCM
        for node in pair:
            pcm_compactified[node, idx] = 1
        
        # 计算合并后的概率（奇数项概率和公式）
        probs = edge_map[pair]["probs"]
        combined_p = 0.5 * (1 - np.prod(1 - 2 * np.array(probs)))

        edges_mapping.append({
            "new_edge_id": idx,
            "nodes": pair,
            "combined_probability": combined_p,
            "source_hyperedge_indices": sorted(list(edge_map[pair]["sources"]))
        })
        error_probs.append(combined_p)
    if ignored_hyperedges_indices:
        print(f"注意：忽略了 {len(ignored_hyperedges_indices)} 条无分解建议的超边。")
        
    return pcm_compactified.tocsc(), edges_mapping


def broadcast_dem(origin_dem, broadcast_time_layer, repeat_chunk, objective_dem_for_test=None):
    """
    Broadcast a DEM by repeating a specific time chunk to extend the time layers.
    
    Args:
        origin_dem: The original DEM to broadcast.
        broadcast_time_layer: The target number of time layers (max time coordinate + 1).
        repeat_chunk: The number of rounds in the repeating chunk.
        objective_dem_for_test: Optional DEM to compare the result against.
        
    Returns:
        output_dem: The broadcasted DEM.
    """
    dem_flat = origin_dem.flattened()
    
    # 1. Analyze DEM to find max time and detector coordinates
    detector_coords = {} # detector_id -> (x, y, t)
    max_time = 0
    
    # Extract original structure
    original_detectors = [] # List of {'x', 'y', 't', 'id', 'args', 'targets'}
    original_errors = []    # List of instructions
    coords_to_old_id = {}   # (x, y, t) -> id
    
    for instruction in dem_flat:
        if isinstance(instruction, stim.DemInstruction):
            if instruction.type == "detector":
                args = instruction.args_copy()
                if len(args) >= 3:
                    x, y, t = int(args[0]), int(args[1]), int(args[2])
                    if t > max_time:
                        max_time = t
                    
                    # Store info
                    det_id = None
                    for target in instruction.targets_copy():
                        if target.is_relative_detector_id():
                            det_id = target.val
                            break
                    
                    if det_id is not None:
                        detector_coords[det_id] = (x, y, t)
                        coords_to_old_id[(x, y, t)] = det_id
                        original_detectors.append({
                            'x': x, 'y': y, 't': t,
                            'id': det_id,
                            'args': args,
                            'targets': instruction.targets_copy()
                        })
            elif instruction.type == "error":
                original_errors.append(instruction)
            else:
                pass

    origin_max_rounds = max_time + 1
    
    # 2. Assert Recurrent Structure (Simplified)
    # Check structure from round 1 to max_time-1
    edges_by_layer_pair = {} 
    
    for instruction in original_errors:
        dets = []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                dets.append(target.val)
        
        if len(dets) == 2:
            d1, d2 = dets[0], dets[1]
            if d1 in detector_coords and d2 in detector_coords:
                c1 = detector_coords[d1]
                c2 = detector_coords[d2]
                if c1[2] > c2[2] or (c1[2] == c2[2] and c1 < c2):
                    c1, c2 = c2, c1
                t1, t2 = c1[2], c2[2]
                layer_pair = (t1, t2)
                if layer_pair not in edges_by_layer_pair:
                    edges_by_layer_pair[layer_pair] = set()
                edge_sig = ((c1[0], c1[1]), (c2[0], c2[1]))
                edges_by_layer_pair[layer_pair].add(edge_sig)
    
    start_check = 1
    end_check = max_time - 1
    prev_interval_edges = None
    for t in range(start_check, end_check):
        pair = (t, t+1)
        if pair in edges_by_layer_pair:
            current_edges = edges_by_layer_pair[pair]
            if prev_interval_edges is not None:
                if current_edges != prev_interval_edges:
                    # Optional: Print warning
                    pass
            prev_interval_edges = current_edges
            
    # 3. Broadcast Logic
    # Based on revised logic:
    # Base: 0 .. boundary-1
    # Pattern Source: (boundary - chunk - 1, boundary - 1] -> Copy k=1..N
    # End Source: [boundary, max] -> Shift
    
    boundary_layer = max_time - 1 
    pattern_start_layer = boundary_layer - repeat_chunk
    base_limit = boundary_layer - 1
    
    total_shift = broadcast_time_layer - max_time
    
    if total_shift <= 0:
        raise ValueError(f"Target time {broadcast_time_layer} must be greater than origin max time {max_time}")
        
    num_repeats = total_shift // repeat_chunk
    if total_shift % repeat_chunk != 0:
        raise ValueError(f"Total shift {total_shift} is not a multiple of repeat_chunk {repeat_chunk}")
        
    new_detectors = [] 
    id_map = {} # (old_id, copy_key) -> new_id. copy_key: 0, k, 'end'
    next_new_id = 0
    
    def add_detector(x, y, t):
        nonlocal next_new_id
        new_id = next_new_id
        next_new_id += 1
        new_detectors.append({'x': x, 'y': y, 't': t, 'id': new_id})
        return new_id
        
    sorted_detectors = sorted(original_detectors, key=lambda d: d['t'])
    
    # 3.1 Process Detectors
    
    # Base (t <= base_limit)
    for det in sorted_detectors:
        if det['t'] <= base_limit:
            new_id = add_detector(det['x'], det['y'], det['t'])
            id_map[(det['id'], 0)] = new_id
            
            # Special Overlap Handling for Pattern Start
            if det['t'] == pattern_start_layer + repeat_chunk:
                if (det['x'], det['y'], pattern_start_layer) in coords_to_old_id:
                    start_id = coords_to_old_id[(det['x'], det['y'], pattern_start_layer)]
                    id_map[(start_id, 1)] = new_id

    # Copies
    for k in range(1, num_repeats + 1):
        for det in sorted_detectors:
            # Source range: [pattern_start, base_limit]
            if det['t'] >= pattern_start_layer and det['t'] <= base_limit:
                new_t = det['t'] + k * repeat_chunk
                new_id = add_detector(det['x'], det['y'], new_t)
                id_map[(det['id'], k)] = new_id
                
                # Overlap for next copy
                if det['t'] == pattern_start_layer + repeat_chunk:
                    if (det['x'], det['y'], pattern_start_layer) in coords_to_old_id:
                        start_id = coords_to_old_id[(det['x'], det['y'], pattern_start_layer)]
                        id_map[(start_id, k + 1)] = new_id

    # End Shift
    for det in sorted_detectors:
        if det['t'] >= boundary_layer:
            new_t = det['t'] + total_shift
            new_id = add_detector(det['x'], det['y'], new_t)
            id_map[(det['id'], 'end')] = new_id

    # 3.2 Process Errors
    new_errors = []
    
    def get_mapped_id(old_id, k):
        # Recursively find mapping if direct mapping doesn't exist
        if (old_id, k) in id_map:
            return id_map[(old_id, k)]
        
        # Fallback logic
        x, y, t = detector_coords[old_id]
        t_equiv = t + repeat_chunk
        
        if (x, y, t_equiv) in coords_to_old_id:
            equiv_id = coords_to_old_id[(x, y, t_equiv)]
            if k == 'end':
                return get_mapped_id(old_id, num_repeats)
                
            if isinstance(k, int) and k > 0:
                return get_mapped_id(equiv_id, k - 1)
        
        return None

    def add_error_using_map(targets, copy_key, instruction_args):
        new_targets = []
        has_valid_target = False
        
        for target in targets:
            if target.is_relative_detector_id():
                old_id = target.val
                current_key = copy_key
                if copy_key == 'end':
                    if (old_id, 'end') not in id_map:
                         current_key = num_repeats
                
                new_id = get_mapped_id(old_id, current_key)
                if new_id is not None:
                    new_targets.append(stim.DemTarget.relative_detector_id(new_id))
                    has_valid_target = True
                else:
                    return 
            elif target.is_logical_observable_id() or target.is_separator():
                new_targets.append(target)
            else:
                new_targets.append(target)
        
        if has_valid_target:
             new_errors.append(stim.DemInstruction("error", instruction_args, new_targets))

    for instruction in original_errors:
        targets = instruction.targets_copy()
        current_times = [detector_coords[t.val][2] for t in targets if t.is_relative_detector_id()]
        if not current_times: continue
        
        max_t = max(current_times)
        
        # 1. Keep Base
        if max_t <= base_limit:
            add_error_using_map(targets, 0, instruction.args_copy())
            
        # 2. Pattern Copy
        # Source range: max_t in [pattern_start_layer, base_limit]
        if max_t >= pattern_start_layer and max_t <= base_limit:
            for k in range(1, num_repeats + 1):
                add_error_using_map(targets, k, instruction.args_copy())
                
        # 3. Shift End
        if max_t >= boundary_layer:
            add_error_using_map(targets, 'end', instruction.args_copy())
            
    # Build Output DEM
    output_dem = stim.DetectorErrorModel()
    new_detectors.sort(key=lambda d: d['id'])
    for err in new_errors:
        output_dem.append(err)
    for d in new_detectors:
        output_dem.append(stim.DemInstruction("detector", [d['x'], d['y'], d['t']], [stim.DemTarget.relative_detector_id(d['id'])]))
        
    # print(f"Broadcast complete.")
    
    # 4. Check against objective
    if objective_dem_for_test is not None:
        print("Checking against objective DEM...")
        obj_dets = 0
        obj_errs = 0
        for inst in objective_dem_for_test.flattened():
            if inst.type == "detector": obj_dets += 1
            if inst.type == "error": obj_errs += 1
            
        out_dets = len(new_detectors)
        out_errs = len(new_errors)
        
        print(f"Detectors: Output {out_dets} vs Objective {obj_dets}")
        print(f"Errors: Output {out_errs} vs Objective {obj_errs}")
        
        if out_dets == obj_dets and out_errs == obj_errs:
            print("Counts match!")
        else:
            print("Counts mismatch!")
            
    return output_dem

def contract(tree, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Contract tensors according to the optimized tree."""
    return _contract_recursive(tree, tensors)


def _contract_recursive(tree_dict: dict, tensors: list[torch.Tensor]) -> torch.Tensor:
    if "tensorindex" in tree_dict:
        return tensors[tree_dict["tensorindex"]-1]
    args = [_contract_recursive(arg, tensors) for arg in tree_dict["args"]]
    return _einsum_int(tree_dict["eins"]["ixs"], tree_dict["eins"]["iy"], args)


def _einsum_int(ixs: list[list[int]], iy: list[int], tensors: list[torch.Tensor]) -> torch.Tensor:
    """Execute einsum with integer index labels."""
    # all_labels = set(sum(ixs, []) + iy)
    # label_map = {l: chr(ord('a') + i) for i, l in enumerate(sorted(all_labels))}
    uniquelabels = list(set(sum(ixs, start=[]) + iy))
    allow_ascii = list(range(65, 90)) + list(range(97, 122))
    label_map = {l: chr(allow_ascii[i]) for i, l in enumerate(uniquelabels)}
    inputs = ",".join("".join(label_map[l] for l in ix) for ix in ixs)
    output = "".join(label_map[l] for l in iy)
    # print(f"{inputs}->{output}", ixs, iy, [tensor.shape for tensor in tensors])
    return torch.einsum(f"{inputs}->{output}", *tensors)
    

if __name__ == "__main__":
    None
