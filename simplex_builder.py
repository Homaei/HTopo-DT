import wntr
import networkx as nx
import torch
import itertools
from toponetx import SimplicialComplex
from scipy.sparse import coo_matrix

def build_simplex_backbone(inp_filepath, tau_1=0.0, tau_2=0.0, tau_3=0.0):
    """
    Algorithm 1: Offline Physics-Guided Simplex Construction.
    Parses INP file, computes nominal hydraulic coupling, 
    thresholds simplices, and builds boundary matrices B1, B2, B3, 
    and W0, A_inc, A_loop for physics loss.
    """
    wn = wntr.network.WaterNetworkModel(inp_filepath)
    
    G = nx.Graph()
    G_directed = wn.get_graph() # Used to preserve edge directions for A_inc, A_loop
    
    node_to_idx = {name: idx for idx, name in enumerate(wn.node_name_list)}
    idx_to_node = {idx: name for name, idx in node_to_idx.items()}
    
    # Store nominal couplings
    kappa_nominal = {}
    
    for link_name, link in wn.links():
        if link.link_type == 'Pipe':
            L = link.length
            D = link.diameter
            C = link.roughness
            kappa_0 = (10.67 * L) / ((C ** 1.852) * (D ** 4.87))
        else:
            kappa_0 = 1.0  
            
        u = node_to_idx[link.start_node_name]
        v = node_to_idx[link.end_node_name]
        
        if kappa_0 >= tau_1:
            G.add_edge(u, v, kappa=kappa_0, name=link_name)
            kappa_nominal[frozenset([u, v])] = kappa_0
            
    sc = SimplicialComplex()
    
    # 0-simplices
    for u in G.nodes():
        sc.add_simplex([u])
        
    # 1-simplices
    for u, v in G.edges():
        sc.add_simplex([u, v])
        
    # 2-simplices (Triangles)
    cliques_3 = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    phi_2_nominal = {}
    for c in cliques_3:
        u, v, w = c
        # Product of pairwise coupling coefficients
        phi_2 = kappa_nominal[frozenset([u, v])] * kappa_nominal[frozenset([v, w])] * kappa_nominal[frozenset([u, w])]
        if phi_2 >= tau_2:
            sc.add_simplex(c)
            phi_2_nominal[frozenset(c)] = phi_2
            
    # 3-simplices (Tetrahedra)
    cliques_4 = [c for c in nx.enumerate_all_cliques(G) if len(c) == 4]
    for c in cliques_4:
        edges_in_clique = list(itertools.combinations(c, 2))
        phi_3 = 1.0
        for edge in edges_in_clique:
            phi_3 *= kappa_nominal.get(frozenset(edge), 0)
        
        if phi_3 >= tau_3:
            sc.add_simplex(c)
            
    # Generate Boundary Matrices
    B1_scipy = sc.incidence_matrix(rank=1, signed=True) 
    B2_scipy = sc.incidence_matrix(rank=2, signed=True) 
    
    try:
        B3_scipy = sc.incidence_matrix(rank=3, signed=True) 
    except Exception:
        from scipy.sparse import csr_matrix
        B3_scipy = csr_matrix((B2_scipy.shape[1], 0))

    def scipy_to_torch_sparse(mat):
        mat = mat.tocoo()
        indices = torch.vstack((torch.from_numpy(mat.row), torch.from_numpy(mat.col))).long()
        values = torch.from_numpy(mat.data).float()
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    B1 = scipy_to_torch_sparse(B1_scipy)
    B2 = scipy_to_torch_sparse(B2_scipy)
    B3 = scipy_to_torch_sparse(B3_scipy)
    
    # W0 initialization (diag matrix with 1.0)
    num_nodes = B1.size(0)
    W0 = torch.ones(num_nodes)
    
    # Precompute A_inc (which is essentially B1)
    A_inc = B1
    
    # Precompute A_loop (cycle basis matrix)
    cycles = nx.cycle_basis(G)
    num_loops = len(cycles)
    num_edges = B1.size(1)
    
    # Map edges in sc to indices for constructing A_loop
    edges_list = list(sc.skeleton(1))
    edge_to_idx = {frozenset(e): idx for idx, e in enumerate(edges_list)}
    
    # Construct A_loop (loops x edges)
    row_idx = []
    col_idx = []
    vals = []
    
    for loop_idx, cycle_nodes in enumerate(cycles):
        for i in range(len(cycle_nodes)):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % len(cycle_nodes)]
            
            fs_edge = frozenset([u, v])
            if fs_edge in edge_to_idx:
                e_idx = edge_to_idx[fs_edge]
                
                # Determine orientation based on original directed graph if needed,
                # Here we just assign +1 or -1 based on traversal direction vs node order
                # For a simple representation, we assume uniform orientation or matching B1
                
                # Check orientation in B1
                # B1 is nodes x edges
                # B1[u, e_idx] will be -1 (source) and B1[v, e_idx] will be +1 (target)
                b1_u = B1_scipy[u, e_idx]
                
                if b1_u == -1: # u is source, traversed in direction
                    sign = 1.0
                else: # traversed against direction
                    sign = -1.0
                    
                row_idx.append(loop_idx)
                col_idx.append(e_idx)
                vals.append(sign)
                
    if len(row_idx) > 0:
        A_loop_scipy = coo_matrix((vals, (row_idx, col_idx)), shape=(num_loops, num_edges))
        A_loop = scipy_to_torch_sparse(A_loop_scipy)
    else:
        A_loop = torch.sparse_coo_tensor(torch.empty((2, 0)), torch.empty(0), (0, num_edges))

    return sc, B1, B2, B3, W0, A_inc, A_loop, kappa_nominal, phi_2_nominal