import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_physics_loss(h0_new, h1_new, curl_edge, A_inc, A_loop, d, nominal_flows, Q_min, Q_max, epsilon_s=1e-4):
    """
    Computes the full 3-term Physics Loss (L_phys, Equation 15).
    """
    # h1_new represents learned edge features. We assume channel 0 is the flow estimate Q.
    # curl_edge is the extracted Hodge residual on edges.
    Q = h1_new[:, 0].unsqueeze(1) # (E, 1)
    
    # Term 1: Mass conservation || A_inc @ Q - d ||^2
    # A_inc is nodes x edges
    mass_residual = torch.sparse.mm(A_inc, Q) - d.unsqueeze(1)
    loss_mass = torch.mean(mass_residual ** 2)
    
    # Term 2: Energy balance || A_loop @ h0_new ||^2
    # A_loop is loops x edges (Wait, energy balance is sum of pressure drops around a loop = 0)
    # A_loop maps edges to loops. If we have head differences across edges (dh = A_inc^T @ h),
    # then A_loop @ dh should be 0.
    # Alternatively, the formula says A_loop @ h... but usually it's A_loop @ dh = 0.
    # Following the prompt directly: A_loop @ h, but let's interpret it structurally as A_loop @ (A_inc^T @ h) or we'll use a direct projection.
    # Assuming A_loop operates on edges to sum them around the loop: 
    # dh = A_inc^T @ h0_new
    # energy_residual = A_loop @ dh
    
    h_scalar = h0_new[:, 0:1]                          # (N, 1)
    dh = torch.sparse.mm(A_inc.t(), h_scalar)  # (E, 1)
    energy_residual = torch.sparse.mm(A_loop, dh)     # (loops, 1)
    loss_energy = torch.mean(energy_residual ** 2)
    
    # Term 3: Uncertainty-weighted residual
    # Σ |Q_hat_ij - Q_ij|^2 / (Q_max - Q_min + epsilon_s)^2
    Q_hat = nominal_flows.unsqueeze(1)
    uncertainty_width = (Q_max - Q_min).unsqueeze(1) + epsilon_s
    
    
    loss_uncertainty = torch.sum(((Q_hat - Q) ** 2) / (uncertainty_width ** 2))
    
    # Normalize by number of edges for scale
    loss_uncertainty = loss_uncertainty / Q.size(0)
    
    l_phys = loss_mass + loss_energy + loss_uncertainty
    return l_phys

def train_epoch(model, dataloader, optimizer, A_inc, A_loop, pd_ref_dict, device='cpu', 
                lambda_topo=0.1, lambda_phys=0.1, alpha_uncertainty=0.05):
    """
    Training loop for one epoch.
    A_inc, A_loop are precomputed offline and reused here.
    """
    model.train()
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()
    
    A_inc = A_inc.to(device)
    A_loop = A_loop.to(device)
    
    for batch in dataloader:
        x_stream, labels, W0, kappa_current, phi_2_current, W3, nominal_flows, node_demands = batch
        
        x_stream = x_stream.to(device)
        labels = labels.to(device)
        W0 = W0.to(device)
        kappa_current = kappa_current.to(device)
        phi_2_current = phi_2_current.to(device)
        W3 = W3.to(device)
        nominal_flows = nominal_flows.to(device) 
        node_demands = node_demands.to(device)
        
        Q_min = nominal_flows * (1.0 - alpha_uncertainty)
        Q_max = nominal_flows * (1.0 + alpha_uncertainty)
        Q_width = (Q_max - Q_min).unsqueeze(-1)
        Q_ij = nominal_flows.unsqueeze(-1)
        
        optimizer.zero_grad()
        
        logits, h0_new, h1_new, curl_edge, diagrams_curr, topo_score = model(
            x_stream, W0, kappa_current, phi_2_current, W3, Q_ij, Q_width, pd_ref_dict
        )
        
        # In this mock graph-level setup, labels are assumed shape (batch,)
        l_ce = ce_loss_fn(logits, labels)
        l_topo = topo_score
        
        l_phys = compute_physics_loss(h0_new, h1_new, curl_edge, A_inc, A_loop, node_demands, nominal_flows, Q_min, Q_max)
        
        loss = l_ce + lambda_topo * l_topo + lambda_phys * l_phys
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)