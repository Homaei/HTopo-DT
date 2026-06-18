import torch
import torch.nn as nn
from collections import deque

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
    Extracts the curl component c^(L)(t) from the learned edge features using B2.
    """
    def __init__(self, B2):
        super(HodgeDecomposition, self).__init__()
        self.register_buffer('B2', B2)
        
    def forward(self, edge_features):
        """
        edge_features: (num_edges, feature_dim) E^(L)
        """
        # Phase 5 Fix: Proper extraction formula
        # c(t) = B_2^T @ E^(L)
        c_tri = torch.sparse.mm(self.B2.t(), edge_features)
        
        # curl_edge = B_2 @ c(t)
        curl_edge = torch.sparse.mm(self.B2, c_tri)
        
        return c_tri, curl_edge

class TopologicalAnomalyDetector(nn.Module):
    """
    Sub-Level Set Filtration and Persistent Homology using torch-topological.
    Maintains a sliding window for the reference diagram PD_k*.
    """
    def __init__(self, window_size=288):
        super(TopologicalAnomalyDetector, self).__init__()
        self.vr_complex = VietorisRipsComplex(dim=2)
        self.wasserstein = WassersteinDistance()
        
        # Sliding window for reference PD estimation
        self.window_size = window_size
        self.normal_diagrams_history = deque(maxlen=window_size)
        
    def forward(self, node_features, kappa_ij):
        diagrams = self.vr_complex(node_features)
        return diagrams
        
    def update_reference(self, normal_diagram):
        """
        Called during confirmed normal periods to build the sliding window.
        """
        self.normal_diagrams_history.append(normal_diagram)
        
    def get_reference_diagram(self):
        """
        Estimates PD_k* from the sliding window.
        For simplicity in this mock, we can return the average or the most recent.
        Ideally, we compute the Frechet mean of diagrams in the window.
        """
        if len(self.normal_diagrams_history) == 0:
            return None
        return self.normal_diagrams_history[-1] # Simplification
        
    def compute_loss(self, diagrams_current, diagrams_reference=None, weights=[1.0, 1.0, 1.0]):
        if diagrams_reference is None:
            diagrams_reference = self.get_reference_diagram()
            if diagrams_reference is None:
                return torch.tensor(0.0, requires_grad=True, device=diagrams_current[0].diagram.device if diagrams_current else 'cpu')
                
        loss = 0.0
        for dim in range(min(len(diagrams_current), len(diagrams_reference))):
            pd_curr = diagrams_current[dim].diagram
            pd_ref = diagrams_reference[dim].diagram
            
            w_dist = self.wasserstein(pd_curr, pd_ref)
            loss += weights[dim] * (w_dist ** 2)
            
        return loss

class HTopoClassifier(nn.Module):
    """
    Final MLP classifier.
    """
    def __init__(self, out_dim, num_classes=5):
        super(HTopoClassifier, self).__init__()
        # c_tri acts as the primary loop-anomaly signal. We map it to the same dim.
        # This implementation aligns with the fix to include c(t) directly.
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2 + 1, 64), # z + c_tri_mapped + topo_score
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.c_map = nn.Linear(out_dim, out_dim)
        
    def forward(self, z, c_tri, topo_score):
        # We need to broadcast/pool features appropriately
        # For simplicity in this graph-level/node-level mock structure, we pool everything to graph-level
        z_graph = torch.mean(z, dim=0, keepdim=True)
        c_graph = torch.mean(c_tri, dim=0, keepdim=True)
        
        c_mapped = self.c_map(c_graph)
        
        if topo_score.dim() == 0:
            topo_score = topo_score.view(1, 1)
            
        x = torch.cat([z_graph, c_mapped, topo_score], dim=-1)
        logits = self.mlp(x)
        return logits