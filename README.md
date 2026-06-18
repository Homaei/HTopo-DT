# HTopo-DT: Simplicial Neural Networks for Multi-Component Cyberattack Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **HTopo-DT**, a higher-order topological deep learning framework for cyber-physical intrusion detection in Water Distribution Network (WDN) Digital Twins.

## Overview

Cyber-physical attacks on critical infrastructure are becoming increasingly coordinated and stealthy. Traditional Graph Neural Networks (GNNs) often fail to capture complex multi-component interactions (e.g., simultaneous pump manipulation and valve closure) because they rely strictly on pairwise (node-to-node) relationships. 

**HTopo-DT** goes beyond standard graphs by modeling the water network as a **Simplicial Complex**, allowing the capture of higher-order hydraulic dependencies (edges as pipes, triangles as hydraulic loops). Our framework introduces:

- **Physics-Grounded Edge Extraction:** Deriving 1-simplex signals dynamically from nodal pressure readings via the Hazen-Williams relation.
- **Weighted Hodge Laplacians:** Building a static topological backbone offline, augmented with dynamic, real-time weights based on hydraulic coupling metrics.
- **Differentiable Persistent Homology:** Utilizing sub-level-set filtrations on edge weights and extracting topological features via the 2-Wasserstein distance to penalize unnatural network topologies caused by attacks.
- **Physics-Informed Neural Network (PINN) Loss:** Enforcing mass conservation and energy balance directly in the loss function to heavily penalize violations of physical laws.

## 📂 Repository Structure

```text
HTopo-DT/
├── dataset/                     # Directory for datasets (not tracked by Git)
│   ├── BATADAL_train07.csv      # SCADA training data
│   ├── BATADAL_test_dataset.csv # SCADA testing data
│   └── ctown.inp                # EPANET hydraulic network configuration
└── code/
    ├── data_pipeline.py         # SCADA data parsing, dataset downloading, and attack generation
    ├── simplex_builder.py       # Offline construction of boundary matrices (B1, B2, B3)
    ├── layers.py                # TCN Encoder, Simplicial Message Passing, and Cross-Level Fusion
    ├── topology.py              # Hodge decomposition, topological anomaly detection (Gudhi-based)
    ├── model.py                 # Core HTopo-DT neural network architecture
    ├── train.py                 # Training loops, Physics loss calculation, and optimization
    ├── requirements.txt         # Package dependencies
    └── README.md                # You are here!
```

## ⚙️ Installation

The implementation is built primarily on `PyTorch`, `WNTR` (Water Network Tool for Resilience), and `torch-topological`.

```bash
# Clone the repository
git clone https://github.com/Homaei/HTopo-DT.git
cd HTopo-DT/code

# Install dependencies
pip install -r requirements.txt
```

> **Note on Topological Dependencies:** The topological anomaly detector relies on `gudhi` and `torch-topological`. The codebase includes an automatic Monkey-Patch compatibility fix for `gudhi >= 3.12.0`.

## 🚀 Getting Started

### 1. Download Datasets

We have provided a built-in function to securely fetch the required BATADAL datasets and the C-Town `.inp` hydraulic model. Run the following command from the `code/` directory:

```bash
python data_pipeline.py ../dataset
```
This will create a `dataset/` folder at the root of the project and download `BATADAL_train07.csv`, `BATADAL_test_dataset.csv`, and `ctown.inp`.

### 2. Verify and Build the Simplicial Backbone

Before training, the network's boundary matrices ($\mathbf{B}_1$, $\mathbf{B}_2$, $\mathbf{B}_3$) must be built from the `.inp` file. This is done offline via the `simplex_builder.py` script:

```python
from simplex_builder import build_simplex_backbone

sc, B1, B2, B3, W0, A_inc, A_loop, kappa_nom, phi2_nom = \
    build_simplex_backbone('../dataset/ctown.inp', tau_1=0.0, tau_2=0.0, tau_3=0.0)
```

### 3. Model Training & End-to-End Execution

The `HTopoDT` architecture integrates the topological layers, the PINN loss, and the final classification output. A typical forward pass can be executed as follows:

```python
import torch
from model import HTopoDT

# Initialize model
model = HTopoDT(B1, B2, B3, kappa_t, phi2_t,
                in_channels=1, seq_len=10,
                node_dim=16, edge_dim=16, tri_dim=16, out_dim=16, num_classes=5)

# Forward pass (x: node features, Q: flow, Qw: edge uncertainty)
logits, h0_new, h1_new, curl_edge, diagrams, topo_score = model(x, W0, kappa_t, phi2_t, W3, Q, Qw)
```

## 🧠 Core Architecture Highlights

### Simplicial Message Passing (SMP)
Instead of standard node aggregations, HTopo-DT updates representations on **nodes**, **edges**, and **triangles** simultaneously. Edge features are updated using upper connections (triangles) and lower connections (nodes) via generalized Hodge Laplacians.

### Topological Anomaly Detection (TAD)
We apply a sub-level set filtration over the dynamically weighted hydraulic graph. The persistence diagrams generated capture the "shape" of the network's hydraulic flow. The 2-Wasserstein distance calculates the deviation from the nominal topological state.

### Physics-Informed Objective
The final objective function integrates:
1. **Cross-Entropy Loss** for classification.
2. **Mass Conservation Loss** penalizing discrepancies at nodes.
3. **Energy Balance Loss** penalizing discrepancies in hydraulic loops.
4. **Topological Loss** penalizing structural flow anomalies (Wasserstein distance).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 Citation

If you find this work or our physical-topological methodology useful in your research, please cite the corresponding paper:
*(Citation details to be added upon publication)*
