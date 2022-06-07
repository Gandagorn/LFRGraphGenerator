# Project LFR Graph Generation and Evaluation
### Masterproject 2021

Utilities for generating real-world-like graphs with customizable attributes and features using the Lancichinetti-Fortunato-Radicchi (LFR) algorithm.
The goal of this project was to generate benchmark graphs to evaluate the influence of homophily on various node classification models.

#### Graph Generation<br>
Efficiently generate artificial benchmark graphs with realistic node degree and community size distribution using the [LFR algorithm](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.LFR_benchmark_graph.html).
Can control for
- Number of nodes and average degree
- Edge-Homophily-Ratio
- Community size and node degree distributions

#### Graph Evaluation<br>
Training and evaluation of several GNN and MLP models on the node classification task using graphs of different homophily.
This also includes custom implementations for some GNN models.
The models include
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- Simple Graph Convolution (SGC)
- Graph Sample and Aggregate (GraphSAGE)
- H2GCN
- MLP as baseline
