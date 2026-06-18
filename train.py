import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_physics_loss(h1, Q_min, Q_max, epsilon_s=0.0001):
    """
    Computes the Physics Loss (L_phys, Equation 15).
    Enforces that the learned edge feature (flow) falls within the 
    uncertainty intervals [Q_min, Q_max].
    """
    q_pred = h1[:, 0]
    penalty_low = F.relu(Q_min - q_pred)
    penalty_high = F.relu(q_pred - Q_max)
    l_phys = torch.mean((penalty_low ** 2 + penalty_high ** 2) / (torch.abs(Q_max - Q_min) + epsilon_s))
    return l_phys

def train_epoch(model, dataloader, optimizer, pd_ref, device='cpu', lambda_topo=0.1, lambda_phys=0.1, alpha_uncertainty=0.05):
    """
    Training loop for one epoch.
    """
    model.train()
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()
    for batch in dataloader:
        (x_stream, labels, W0, W1, W2, W3, nominal_flows) = batch
        x_stream = x_stream.to(device)
        labels = labels.to(device)
        (W0, W1, W2, W3) = (W0.to(device), W1.to(device), W2.to(device), W3.to(device))
        nominal_flows = nominal_flows.to(device)
        optimizer.zero_grad()
        (logits, h0, h1, h2, diagrams_curr, topo_score) = model(x_stream, W0, W1, W2, W3, pd_ref)
        l_ce = ce_loss_fn(logits, labels)
        l_topo = topo_score
        Q_min = nominal_flows * (1.0 - alpha_uncertainty)
        Q_max = nominal_flows * (1.0 + alpha_uncertainty)
        l_phys = compute_physics_loss(h1, Q_min, Q_max)
        loss = l_ce + lambda_topo * l_topo + lambda_phys * l_phys
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
if __name__ == '__main__':
    pass