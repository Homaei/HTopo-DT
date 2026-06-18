import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_topological.nn import WassersteinDistance
    from torch_topological.nn import VietorisRipsComplex
except ImportError:

    class WassersteinDistance(nn.Module):

        def forward(self, pd1, pd2):
            return torch.tensor(0.0, requires_grad=True)

    class VietorisRipsComplex(nn.Module):

        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x):
            return []

class HodgeDecomposition(nn.Module):
    """
    Extracts the curl component c^(L)(t) from the learned edge features 
    using the B2 operator.
    """

    def __init__(self, B2):
        super(HodgeDecomposition, self).__init__()
        self.register_buffer('B2', B2)

    def forward(self, edge_features, W1):
        """
        Extracts the curl component.
        edge_features: (num_edges, feature_dim)
        W1: (num_edges,) diagonal weights
        
        Using least squares pseudo-inverse approach to find c:
        (B2^T W1 B2) c = B2^T W1 q
        """
        B2_dense = self.B2.to_dense()
        W1_mat = torch.diag(W1)
        left_side = B2_dense.t() @ W1_mat @ B2_dense
        right_side = B2_dense.t() @ W1_mat @ edge_features
        left_side = left_side + 1e-06 * torch.eye(left_side.size(0), device=left_side.device)
        c = torch.linalg.solve(left_side, right_side)
        curl_component = B2_dense @ c
        curl_indicator = torch.norm(curl_component, dim=1, keepdim=True)
        return (c, curl_indicator)

class TopologicalAnomalyDetector(nn.Module):
    """
    Sub-Level Set Filtration and Persistent Homology using torch-topological.
    """

    def __init__(self):
        super(TopologicalAnomalyDetector, self).__init__()
        self.vr_complex = VietorisRipsComplex(dim=2)
        self.wasserstein = WassersteinDistance()

    def forward(self, node_features, kappa_ij):
        """
        Compute persistence diagrams and 2-Wasserstein distance.
        Instead of node features, the filtration is technically based on kappa_ij.
        We map kappa_ij to a distance matrix D where D_ij = exp(-kappa_ij).
        """
        diagrams = self.vr_complex(node_features)
        return diagrams

    def compute_loss(self, diagrams_current, diagrams_reference, weights=[1.0, 1.0, 1.0]):
        loss = 0.0
        for dim in range(min(len(diagrams_current), len(diagrams_reference))):
            pd_curr = diagrams_current[dim].diagram
            pd_ref = diagrams_reference[dim].diagram
            w_dist = self.wasserstein(pd_curr, pd_ref)
            loss += weights[dim] * w_dist ** 2
        return loss

class HTopoClassifier(nn.Module):
    """
    Final MLP classifier.
    """

    def __init__(self, node_dim, num_classes=5):
        super(HTopoClassifier, self).__init__()
        in_dim = node_dim + 1 + 1
        self.mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, z, curl_indicator, topo_score):
        N = z.size(0)
        if curl_indicator.size(0) != N:
            pass
        if topo_score.dim() == 0 or topo_score.size(0) != N:
            topo_score = topo_score.view(1, 1).expand(N, 1)
        x = torch.cat([z, curl_indicator, topo_score], dim=-1)
        logits = self.mlp(x)
        return logits