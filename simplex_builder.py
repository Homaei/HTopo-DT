import wntr
import networkx as nx
import torch
import itertools
from toponetx import SimplicialComplex

def build_simplex_backbone(inp_filepath, tau_1=0.0, tau_2=0.0, tau_3=0.0):
    """
    Algorithm 1: Offline Physics-Guided Simplex Construction.
    Parses INP file, computes nominal hydraulic coupling, 
    thresholds simplices, and builds unweighted boundary matrices B1, B2, B3.
    """
    wn = wntr.network.WaterNetworkModel(inp_filepath)
    G = nx.Graph()
    node_to_idx = {name: idx for (idx, name) in enumerate(wn.node_name_list)}
    idx_to_node = {idx: name for (name, idx) in node_to_idx.items()}
    edge_couplings = {}
    for (link_name, link) in wn.links():
        if link.link_type == 'Pipe':
            L = link.length
            D = link.diameter
            C = link.roughness
            kappa_0 = 10.67 * L / (C ** 1.852 * D ** 4.87)
        else:
            kappa_0 = 1.0
        u = node_to_idx[link.start_node_name]
        v = node_to_idx[link.end_node_name]
        if kappa_0 >= tau_1:
            G.add_edge(u, v, kappa=kappa_0, name=link_name)
            edge_couplings[frozenset([u, v])] = kappa_0
    sc = SimplicialComplex()
    for u in G.nodes():
        sc.add_simplex([u])
    for (u, v) in G.edges():
        sc.add_simplex([u, v])
    cliques_3 = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    for c in cliques_3:
        (u, v, w) = c
        phi_2 = edge_couplings[frozenset([u, v])] * edge_couplings[frozenset([v, w])] * edge_couplings[frozenset([u, w])]
        if phi_2 >= tau_2:
            sc.add_simplex(c)
    cliques_4 = [c for c in nx.enumerate_all_cliques(G) if len(c) == 4]
    for c in cliques_4:
        edges_in_clique = list(itertools.combinations(c, 2))
        phi_3 = 1.0
        for edge in edges_in_clique:
            phi_3 *= edge_couplings.get(frozenset(edge), 0)
        if phi_3 >= tau_3:
            sc.add_simplex(c)
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
    return (sc, B1, B2, B3, edge_couplings)
if __name__ == '__main__':
    pass