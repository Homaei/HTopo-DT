import torch
import torch.nn as nn
from layers import TCNEncoder, DynamicWeightedLaplacians, SimplicialMessagePassing, CrossLevelFusion
from topology import HodgeDecomposition, TopologicalAnomalyDetector, HTopoClassifier

class HTopoDT(nn.Module):
    """
    End-to-End HTopo-DT Architecture.
    """
    def __init__(self, B1, B2, B3, kappa_nominal_tensor, phi_2_nominal_tensor, 
                 in_channels, seq_len, node_dim=64, edge_dim=64, tri_dim=64, out_dim=64, num_classes=5):
        super(HTopoDT, self).__init__()
        
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        
        self.register_buffer('kappa_nominal', kappa_nominal_tensor)
        self.register_buffer('phi_2_nominal', phi_2_nominal_tensor)
        
        self.tcn = TCNEncoder(in_channels=in_channels, out_channels=node_dim)
        
        # Edge initialization expects 2 channels: [Q_ij, Q_width]
        self.edge_init = nn.Linear(2, edge_dim)
        self.tri_init = nn.Linear(edge_dim, tri_dim)
        
        self.dynamic_laplacians = DynamicWeightedLaplacians(B1, B2, B3)
        self.smp = SimplicialMessagePassing(node_dim, edge_dim, tri_dim, num_layers=3)
        self.fusion = CrossLevelFusion(node_dim, edge_dim, tri_dim, out_dim)
        self.hodge = HodgeDecomposition(B2)
        self.topo_detector = TopologicalAnomalyDetector()
        
        self.classifier = HTopoClassifier(out_dim, tri_dim, num_classes)
        
    def _init_simplices(self, Q_ij, Q_width, h0=None):
        """
        Q_ij: (num_edges, 1) mid-point flows
        Q_width: (num_edges, 1) uncertainty half-width (Q_max - Q_min)
        """
        # Phase 3 Fix: Two-channel edge initialization
        edge_raw = torch.cat([Q_ij, Q_width], dim=-1) # (num_edges, 2)
        h1 = self.edge_init(edge_raw)
        
        B2_abs = torch.sparse_coo_tensor(self.B2._indices(), torch.abs(self.B2._values()), self.B2.shape)
        # Combine incident edges to initialize triangle
        h2_raw = torch.sparse.mm(B2_abs.t(), h1)
        h2 = self.tri_init(h2_raw)
        
        return h1, h2

    def forward(self, x_stream, W0, kappa_current, phi_2_current, W3, Q_ij, Q_width, pd_ref=None):
        x_windowed = x_stream.permute(0, 2, 1)  # (N, C, T)
        h0 = self.tcn(x_windowed)
        
        h1, h2 = self._init_simplices(Q_ij, Q_width, h0)
        
        # Dynamic Weight Update Rule (ratios)
        W1 = kappa_current / (self.kappa_nominal + 1e-6)
        W2 = phi_2_current / (self.phi_2_nominal + 1e-6)
        
        L0_w, L1_w, L2_w = self.dynamic_laplacians(W0, W1, W2, W3)
        
        h0_new, h1_new, h2_new = self.smp(h0, h1, h2, L0_w, L1_w, L2_w)
        
        z = self.fusion(h0_new, h1_new, h2_new, self.B1, self.B2)
        
        # Phase 5 Fix: Use c (triangle curl coeffs) directly
        c_tri, curl_edge = self.hodge(h1_new)
        
        # We need node-level curl representation if node classification, or graph level.
        # Assuming we can aggregate c_tri to nodes via B1 and B2, or keep it graph level.
        # Here we follow the logic: c_tri is primarily used as the classifier input.
        
        diagrams_curr = self.topo_detector(kappa_current)
        
        topo_score = torch.tensor([0.0], device=z.device)
        if pd_ref is not None:
            topo_score = self.topo_detector.compute_loss(diagrams_curr, pd_ref)
            
        logits = self.classifier(z, c_tri, topo_score)
        
        return logits, h0_new, h1_new, curl_edge, diagrams_curr, topo_score