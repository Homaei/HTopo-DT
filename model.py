import torch
import torch.nn as nn
from layers import TCNEncoder, DynamicWeightedLaplacians, SimplicialMessagePassing, CrossLevelFusion
from topology import HodgeDecomposition, TopologicalAnomalyDetector, HTopoClassifier

class HTopoDT(nn.Module):
    """
    End-to-End HTopo-DT Architecture.
    """

    def __init__(self, B1, B2, B3, in_channels, seq_len, node_dim=64, edge_dim=64, tri_dim=64, out_dim=64, num_classes=5):
        super(HTopoDT, self).__init__()
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.tcn = TCNEncoder(in_channels=in_channels, out_channels=node_dim)
        self.edge_init = nn.Linear(node_dim * 2, edge_dim)
        self.tri_init = nn.Linear(node_dim * 3, tri_dim)
        self.dynamic_laplacians = DynamicWeightedLaplacians(B1, B2, B3)
        self.smp = SimplicialMessagePassing(node_dim, edge_dim, tri_dim)
        self.fusion = CrossLevelFusion(node_dim, edge_dim, tri_dim, out_dim)
        self.hodge = HodgeDecomposition(B2)
        self.topo_detector = TopologicalAnomalyDetector()
        self.classifier = HTopoClassifier(out_dim, num_classes)

    def _init_simplices(self, h0):
        B1_indices = self.B1._indices()
        src = B1_indices[0, self.B1._values() < 0]
        dst = B1_indices[0, self.B1._values() > 0]
        (N, E, T) = (self.B1.size(0), self.B1.size(1), self.B2.size(1))
        h1 = torch.zeros((E, self.edge_init.out_features), device=h0.device)
        h2 = torch.zeros((T, self.tri_init.out_features), device=h0.device)
        return (h1, h2)

    def forward(self, x_stream, W0, W1, W2, W3, pd_ref=None):
        """
        x_stream: Node pressure head time series (batch_size, in_channels, seq_len)
                  Wait, if we operate on graph, shape is (num_nodes, in_channels, seq_len)
        W0, W1, W2, W3: Real-time diagonal weights for Hodge Laplacians
        """
        h0 = self.tcn(x_stream)
        (h1, h2) = self._init_simplices(h0)
        (L0_w, L1_w, L2_w) = self.dynamic_laplacians(W0, W1, W2, W3)
        (h0_new, h1_new, h2_new) = self.smp(h0, h1, h2, L0_w, L1_w, L2_w)
        z = self.fusion(h0_new, h1_new, h2_new, self.B1, self.B2)
        (c, curl_indicator_edges) = self.hodge(h1_new, W1)
        B1_abs = torch.sparse_coo_tensor(self.B1._indices(), torch.abs(self.B1._values()), self.B1.shape)
        curl_indicator_nodes = torch.sparse.mm(B1_abs, curl_indicator_edges)
        diagrams_curr = self.topo_detector(z.unsqueeze(0), W1)
        topo_score = torch.tensor([0.0], device=z.device)
        if pd_ref is not None:
            topo_score = self.topo_detector.compute_loss(diagrams_curr, pd_ref)
        logits = self.classifier(z, curl_indicator_nodes, topo_score)
        return (logits, h0_new, h1_new, h2_new, diagrams_curr, topo_score)