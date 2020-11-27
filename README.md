# Certified Monotonic Neural Networks
This repo contains the code for NeurIPS 2020 spotlight paper: 
## [Certified Monotonic Neural Networks (Xingchao Liu, Xing Han, Na Zhang, Qiang Liu)](https://arxiv.org/abs/2011.10219)

Requirements: python>=3.7, PyTorch>=1.0.0, Gurobi>=9.0.1

Small $ \lambda $ usually gets best results, but training with small $ \lambda $ can be hard to guarantee monotonicity. We suggest to loop over random seeds before increasing $\lambda$ to get the best monotonic neural network.

The network structure is slightly modified compared to the ones in the paper. The networks in this repo have less parameters but similar performance.
