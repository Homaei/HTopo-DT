# HTopo-DT: Simplicial Neural Networks for Multi-Component Cyberattack Detection

This repository contains the official implementation of **HTopo-DT**, a higher-order topological deep learning framework for cyber-physical intrusion detection in Water Distribution Network (WDN) Digital Twins.

## Overview

HTopo-DT goes beyond standard graph neural networks by modeling multi-component coordinated attacks using simplicial complexes. Our framework introduces:
- **Physics-Grounded Edge Extraction:** Deriving 1-simplex signals from pressure readings via the Hazen-Williams relation.
- **Weighted Hodge Laplacians:** Static topological backbone computed offline with dynamic, real-time weights based on hydraulic coupling.
- **Differentiable Persistent Homology:** Utilizing sub-level-set filtrations and the 2-Wasserstein distance for a trainable topological loss.

## Repository Structure

- `data_pipeline.py`: Scripts to parse SCADA data and generate EPANET-APT multi-component attacks based on the bounded stealthy constraints.
- `simplex_builder.py`: Offline phase computing nominal hydraulic couplings and assembling $\mathbf{B}_1$, $\mathbf{B}_2$, and $\mathbf{B}_3$ boundary matrices.
- `layers.py`: Implements TCN extraction, dynamic weighted Hodge Laplacians, and Simplicial Message Passing.
- `topology.py`: Hodge decomposition (curl component extraction) and Persistent Homology analysis.
- `model.py`: The complete HTopo-DT architecture integrating the feature layers and topological modules.
- `train.py`: Joint optimization training loop containing the Physics, Topological, and Cross-Entropy loss functions.
- `requirements.txt`: Python package dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you find this work useful in your research, please cite the corresponding paper.
