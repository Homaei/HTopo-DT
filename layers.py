import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TCNEncoder(nn.Module):
    """
    1D Temporal Convolutional Network to embed the sliding window of nodal pressure heads.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TCNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, dim=2) 
        return x

class DynamicWeightedLaplacians(nn.Module):
    """
    Computes dynamic weighted Hodge Laplacians L_0^w, L_1^w(t), L_2^w(t).
    Uses element-wise reciprocal for the diagonal inverse as W_k are stored as 1D tensors.
    """
    def __init__(self, B1, B2, B3):
        super(DynamicWeightedLaplacians, self).__init__()
        self.register_buffer('B1', B1)
        self.register_buffer('B2', B2)
        self.register_buffer('B3', B3)
        
    def _sparse_diag_mul(self, sparse_mat, diag_vec):
        if sparse_mat.is_sparse:
            indices = sparse_mat._indices()
            values = sparse_mat._values()
            
            if sparse_mat.shape[1] == diag_vec.shape[0]: 
                new_values = values * diag_vec[indices[1]]
            else: 
                new_values = values * diag_vec[indices[0]]
            
            return torch.sparse_coo_tensor(indices, new_values, sparse_mat.shape)
        else:
            if sparse_mat.shape[1] == diag_vec.shape[0]:
                return sparse_mat * diag_vec.unsqueeze(0)
            else:
                return sparse_mat * diag_vec.unsqueeze(1)
            
    def forward(self, W0, W1, W2, W3):
        W0_inv = 1.0 / (W0 + 1e-6)
        W1_inv = 1.0 / (W1 + 1e-6)
        W2_inv = 1.0 / (W2 + 1e-6)
        
        temp = self._sparse_diag_mul(self.B1, W1)
        L0_w = self._sparse_diag_mul(torch.sparse.mm(temp, self.B1.t()), W0_inv)
        
        term1 = self._sparse_diag_mul(self.B1.t(), W0)
        term1 = torch.sparse.mm(term1, self.B1)
        term1 = self._sparse_diag_mul(term1, W1_inv) 
        
        term2 = self._sparse_diag_mul(self.B2, W2)
        term2 = torch.sparse.mm(term2, self.B2.t())
        term2 = self._sparse_diag_mul(term2, W1_inv) 
        
        L1_w = term1 + term2
        
        term1_2 = self._sparse_diag_mul(self.B2.t(), W1)
        term1_2 = torch.sparse.mm(term1_2, self.B2)
        term1_2 = self._sparse_diag_mul(term1_2, W2_inv)
        
        term2_2 = self._sparse_diag_mul(self.B3, W3)
        term2_2 = torch.sparse.mm(term2_2, self.B3.t())
        term2_2 = self._sparse_diag_mul(term2_2, W2_inv)
        
        L2_w = term1_2 + term2_2
        
        return L0_w, L1_w, L2_w

class SimplicialMessagePassing(nn.Module):
    """
    Implements simplicial message passing across nodes, edges, triangles.
    """
    def __init__(self, node_dim, edge_dim, tri_dim):
        super(SimplicialMessagePassing, self).__init__()
        self.node_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim)
        self.tri_proj = nn.Linear(tri_dim, tri_dim)
        
    def forward(self, h0, h1, h2, L0_w, L1_w, L2_w):
        h0_new = self.node_proj(h0) + torch.sparse.mm(L0_w, h0)
        h0_new = F.relu(h0_new)
        
        h1_new = self.edge_proj(h1) + torch.sparse.mm(L1_w, h1)
        h1_new = F.relu(h1_new)
        
        h2_new = self.tri_proj(h2) + torch.sparse.mm(L2_w, h2)
        h2_new = F.relu(h2_new)
        
        return h0_new, h1_new, h2_new

class CrossLevelFusion(nn.Module):
    """
    Cross-Level Fusion via Attentive Aggregation (Equation 13).
    Pools incident edges and triangles using multi-head attention.
    """
    def __init__(self, node_dim, edge_dim, tri_dim, out_dim, num_heads=4):
        super(CrossLevelFusion, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # We project all to same dim for attention
        self.q_proj = nn.Linear(node_dim, out_dim)
        self.k_proj = nn.Linear(out_dim, out_dim)
        self.v_proj = nn.Linear(out_dim, out_dim)
        
        self.edge_align = nn.Linear(edge_dim, out_dim)
        self.tri_align = nn.Linear(tri_dim, out_dim)
        
        self.final_proj = nn.Linear(out_dim, out_dim)

    def forward(self, h0, h1, h2, B1, B2):
        # Step 1: Collect all incident edge features
        B1_abs = torch.sparse_coo_tensor(B1._indices(), torch.abs(B1._values()), B1.shape)
        
        # We need a proper sparse aggregation, but since PyTorch doesn't easily allow 
        # variable length lists per node in sparse_mm, we approximate the collection 
        # by projecting first, then sum-aggregating, or we can use attention over 
        # the sparse structure.
        # A simple way to do attention over sets per node is:
        # e_agg_i = sum_{j} a_{ij} e_j
        
        # Project edges and triangles to out_dim
        h1_proj = self.edge_align(h1)
        h2_proj = self.tri_align(h2)
        
        # Basic sum aggregation of features to the node level
        e_agg_raw = torch.sparse.mm(B1_abs, h1_proj)
        
        B2_abs = torch.sparse_coo_tensor(B2._indices(), torch.abs(B2._values()), B2.shape)
        t_edge_agg = torch.sparse.mm(B2_abs, h2_proj)
        t_agg_raw = torch.sparse.mm(B1_abs, t_edge_agg)
        
        # For true multi-head attention over the 3 aggregated vectors (h0, e_agg, t_agg):
        # We treat each node as having a sequence of 3 items
        seq = torch.stack([self.q_proj(h0), e_agg_raw, t_agg_raw], dim=1) # (N, 3, out_dim)
        
        Q = self.q_proj(h0).unsqueeze(1) # (N, 1, out_dim)
        K = self.k_proj(seq) # (N, 3, out_dim)
        V = self.v_proj(seq) # (N, 3, out_dim)
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.out_dim)
        attn = F.softmax(scores, dim=-1) # (N, 1, 3)
        
        z = torch.bmm(attn, V).squeeze(1) # (N, out_dim)
        
        return self.final_proj(z)