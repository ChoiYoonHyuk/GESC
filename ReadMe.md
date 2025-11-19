# Gauge-Equivariant Graph Networks via Self-Interference Cancellation

<img width="1824" height="349" alt="Image" src="https://github.com/user-attachments/assets/07c06121-c0c4-44c8-953f-1eabd4af5ec8" />

## Overview

We propose **Gauge-Equivariant Graph Networks via Self-Interference Cancellation (GESC)**, a novel $\mathrm{U}(1)$-valued transport followed by a rank-1 projection that explicitly removes self-parallel components before attention.

## Execution

To train GESC on benchmark datasets (e.g., Cora, Citeseer, Pubmed, Chameleon, Squirrel, Actor, Cornell, Texas, Wisconsin) from 0 to 8:
- python gesc.py 0   # Cora
- python gesc.py 1   # Citeseer
- python gesc.py 2   # Pubmed
- ...
- python gesc.py 8   # Wisconsin

