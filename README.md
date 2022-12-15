# Certified Monotonic Neural Networks
This repo contains the code for NeurIPS 2020 spotlight paper: 
## [Certified Monotonic Neural Networks](https://arxiv.org/abs/2011.10219) 
by *Xingchao Liu, Xing Han, Na Zhang, Qiang Liu* from UT Austin and Tsinghua University

Requirements: python>=3.7, PyTorch>=1.0.0, Gurobi>=9.0.1

1. Small monotonic regularization usually gets best results, but training with small monotonic regularization can be hard to guarantee monotonicity. We suggest to loop over random seeds before increasing monotonic regularization to get the best monotonic neural network.

2. The network structure is slightly modified compared to the ones in the paper. The networks in this repo have less parameters but similar performance.

## Related Work:
A concurrent work that is highly related to ours is 
[Counterexample-Guided Learning of Monotonic Neural Networks](https://proceedings.neurips.cc/paper/2020/file/8ab70731b1553f17c11a3bbc87e0b605-Paper.pdf), and their code is available [here](https://github.com/AishwaryaSivaraman/COMET) 

## Citation:
If you use the code or our work is related to yours, please cite us:
```
@article{liu2020certified,
  title={Certified Monotonic Neural Networks},
  author={Liu, Xingchao and Han, Xing and Zhang, Na and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
