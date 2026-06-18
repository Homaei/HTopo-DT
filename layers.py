import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNEncoder(nn.Module):
    """
    1D Temporal Convolutional Network to embed the sliding window of nodal pressure heads.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TCNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, dim=2)
        return x

class DynamicWeightedLaplacians(nn.Module):
    """
    Computes dynamic weighted Hodge Laplacians L_0^w, L_1^w(t), L_2^w(t).
    Uses sparse diagonal multiplication as per Equation 9 constraint.
    """

    def __init__(self, B1, B2, B3):
        super(DynamicWeightedLaplacians, self).__init__()
        self.register_buffer('B1', B1)
        self.register_buffer('B2', B2)
        self.register_buffer('B3', B3)

    def _sparse_diag_mul(self, sparse_mat, diag_vec):
        """
        Multiplies sparse matrix with a diagonal matrix represented by diag_vec.
        """
        if sparse_mat.is_sparse:
            indices = sparse_mat._indices()
            values = sparse_mat._values()
            if sparse_mat.shape[1] == diag_vec.shape[0]:
                new_values = values * diag_vec[indices[1]]
            else:
                new_values = values * diag_vec[indices[0]]
            return torch.sparse_coo_tensor(indices, new_values, sparse_mat.shape)
        elif sparse_mat.shape[1] == diag_vec.shape[0]:
            return sparse_mat * diag_vec.unsqueeze(0)
        else:
            return sparse_mat * diag_vec.unsqueeze(1)

    def _sparse_dense_mul(self, sparse_mat, dense_mat):
        return torch.sparse.mm(sparse_mat, dense_mat)

    def forward(self, W0, W1, W2, W3):
        """
        W0, W1, W2, W3 are 1D tensors representing the diagonal weights.
        """
        device = W1.device
        W0_inv = 1.0 / (W0 + 1e-06)
        W1_inv = 1.0 / (W1 + 1e-06)
        W2_inv = 1.0 / (W2 + 1e-06)
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
        return (L0_w, L1_w, L2_w)

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
        return (h0_new, h1_new, h2_new)

class CrossLevelFusion(nn.Module):
    """
    Cross-Level Fusion attention mechanism aggregating features from incident edges and triangles.
    """

    def __init__(self, node_dim, edge_dim, tri_dim, out_dim):
        super(CrossLevelFusion, self).__init__()
        self.attn_net = nn.Sequential(nn.Linear(node_dim + edge_dim + tri_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.out_proj = nn.Linear(node_dim + edge_dim + tri_dim, out_dim)

    def forward(self, h0, h1, h2, B1, B2):
        B1_abs = torch.sparse_coo_tensor(B1._indices(), torch.abs(B1._values()), B1.shape)
        edge_agg = torch.sparse.mm(B1_abs, h1)
        B2_abs = torch.sparse_coo_tensor(B2._indices(), torch.abs(B2._values()), B2.shape)
        tri_edge_agg = torch.sparse.mm(B2_abs, h2)
        tri_node_agg = torch.sparse.mm(B1_abs, tri_edge_agg)
        concat_feats = torch.cat([h0, edge_agg, tri_node_agg], dim=-1)
        attn_scores = F.softmax(self.attn_net(concat_feats), dim=-1)
        h0_weighted = h0 * attn_scores[:, 0:1]
        edge_weighted = edge_agg * attn_scores[:, 1:2]
        tri_weighted = tri_node_agg * attn_scores[:, 2:3]
        z = self.out_proj(torch.cat([h0_weighted, edge_weighted, tri_weighted], dim=-1))
        return z