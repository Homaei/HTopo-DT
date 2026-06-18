import torch
import torch.nn as nn
from collections import deque

try:
    from torch_topological.nn import WassersteinDistance, CubicalComplex
    TOPO_AVAILABLE = True
except ImportError:
    TOPO_AVAILABLE = False

if TOPO_AVAILABLE:
    import gudhi
    def _cubical_complex_forward_patch(self, x):
        if self.superlevel:
            x = -x
        
        # FIX: gudhi 3.12.0 strictly requires lists for dimensions and array
        dimensions = list(x.shape)
        # Handle 0D tensors explicitly if needed, but x.shape is typical
        array = x.flatten().detach().cpu().numpy().tolist()

        cubical_complex = gudhi.CubicalComplex(
            dimensions=dimensions,
            top_dimensional_cells=array
        )

        cubical_complex.persistence()
        cofaces = cubical_complex.cofaces_of_persistence_pairs()

        max_dim = len(x.shape)
        persistence_information = [
            self._extract_generators_and_diagrams(x, cofaces, dim) 
            for dim in range(0, max_dim)
        ]
        return persistence_information

    CubicalComplex._forward = _cubical_complex_forward_patch

if not TOPO_AVAILABLE:
    raise ImportError(
        "torch_topological is required for HTopo-DT's differentiable persistence loss (C3). "
        "Install it with: pip install torch-topological\n"
        "Without it, the topological objective (Eq. 16) cannot be computed and "
        "Theorem 1's detectability guarantee does not hold."
    )

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
        self.sublevel = CubicalComplex(dim=1, superlevel=False)
        self.wasserstein = WassersteinDistance(q=2)
        
        # Sliding window for reference PD estimation
        self.window_size = window_size
        self.normal_diagrams_history = deque(maxlen=window_size)
        
        # Learned non-negative weights ω_k for k=0,1,2  (paper Eq. 16)
        self.omega = nn.Parameter(torch.ones(3))
        
    def forward(self, kappa_current):
        """
        kappa_current: (num_edges,) tensor of current hydraulic coupling weights.
        Returns persistence diagrams from sub-level-set filtration on kappa.
        """
        # Treat the sorted 1D signal as a cubical complex for differentiable persistence
        kappa_1d = kappa_current.unsqueeze(0)                 # (1, num_edges)
        diagrams = self.sublevel(kappa_1d)
        return diagrams
        
    def update_reference(self, normal_diagram):
        """
        Called during confirmed normal periods to build the sliding window.
        """
        self.normal_diagrams_history.append(normal_diagram)
        
    def get_reference_diagram(self):
        """
        Estimates PD_k* from the sliding window.
        """
        if len(self.normal_diagrams_history) == 0:
            return None
        history = list(self.normal_diagrams_history)
        mid_idx = len(history) // 2
        return history[mid_idx]  # temporal median — stable and cheap
        
    def compute_loss(self, diagrams_current, diagrams_reference=None):
        if diagrams_reference is None:
            diagrams_reference = self.get_reference_diagram()
        if diagrams_reference is None:
            return torch.tensor(0.0, device=self.omega.device)
                
        omega = torch.nn.functional.softplus(self.omega)
        loss = torch.tensor(0.0, device=omega.device)
        for dim in range(min(len(diagrams_current), len(diagrams_reference))):
            pd_curr = diagrams_current[dim].diagram
            pd_ref = diagrams_reference[dim].diagram
            
            w_dist = self.wasserstein(pd_curr, pd_ref)
            loss = loss + omega[dim] * (w_dist ** 2)
            
        return loss

class HTopoClassifier(nn.Module):
    """
    Final MLP classifier.
    """
    def __init__(self, out_dim, tri_dim, num_classes=5):
        super(HTopoClassifier, self).__init__()
        self.c_map = nn.Linear(tri_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, z, c_tri, topo_score):
        # We need to broadcast/pool features appropriately
        # For simplicity in this graph-level/node-level mock structure, we pool everything to graph-level
        z_graph = torch.mean(z, dim=0, keepdim=True)
        c_graph = torch.mean(c_tri, dim=0, keepdim=True)
        
        c_mapped = self.c_map(c_graph)
        
        topo_score = topo_score.view(1, -1)
        x = torch.cat([z_graph, c_mapped, topo_score], dim=-1)
        logits = self.mlp(x)
        return logits