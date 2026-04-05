# SPiCE: Symmetry-Preserving Conformer Ensemble Networks

Official implementation of **"Symmetry-Preserving Conformer Ensemble Networks for Molecular Representation Learning"** (NeurIPS 2025).

[[Paper]](https://openreview.net/pdf?id=kgjY80e1LF)

## Overview

SPiCE learns molecular properties from conformer ensembles while preserving joint equivariance to geometric transformations of individual conformers and permutations of the ensemble. Key components:

- **Geometric Mixture-of-Experts (GMoE)**: Separate per-atom routers for scalar and vector features using Gumbel softmax
- **Hierarchical Ensemble Encoding**: Cross-attention between topology (GIN) and conformer representations
- **Upcycling**: Progressive expert specialization from a shared initialization

## Installation

```bash
conda create -n spice python=3.10
conda activate spice
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster
pip install e3nn wandb rdkit tqdm pandas scikit-learn gputil
```

## Usage

SPiCE uses distributed training across multiple GPUs:

```bash
# Train with PaiNN backbone on Kraken
python train.py --gpus 0,1,2,3 --modeldss:conf_encoder PaiNN --dataset Kraken --target sterimol_B5 \
    --gig True --upc True --num_experts 8 --num_activated 2

# Train with Equiformer backbone on Drugs
python train.py --gpus 0,1 --modeldss:conf_encoder Equiformer --dataset Drugs --target ip \
    --gig True --upc True --upcycling_epochs 100

# Train with ClofNet backbone on BDE
python train.py --gpus 0,1,2,3 --modeldss:conf_encoder ClofNet --dataset BDE --target bde \
    --gig True --upc True --num_experts 16 --num_activated 2

# Train with ViSNet backbone on EE
python train.py --gpus 0,1 --modeldss:conf_encoder ViSNet --dataset EE --target ee \
    --gig True --upc True
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--modeldss:conf_encoder` | Backbone encoder: PaiNN, ClofNet, Equiformer, ViSNet | PaiNN |
| `--modeldss:topo_encoder` | Topology encoder | GIN |
| `--num_experts` | Number of experts in GMoE | 8 |
| `--num_activated` | Top-k expert activation | 2 |
| `--gig` | Enable GMoE routing | True |
| `--upc` | Enable upcycling | False |
| `--upcycling_epochs` | Epoch to transition from single to multi-expert | 50 |
| `--gumbel_tau` | Gumbel softmax temperature | 1 |
| `--z_beta` | Auxiliary router loss weight | 0.001 |

## Datasets

Place datasets under `datasets/`:
- `datasets/Kraken/`
- `datasets/Drugs/`
- `datasets/BDE/`
- `datasets/EE/`

## Citation

```bibtex
@inproceedings{zhu2025spice,
  title={Symmetry-Preserving Conformer Ensemble Networks for Molecular Representation Learning},
  author={Zhu, Yanqiao and Shi, Yidan and Chen, Yuanzhou and Sun, Fang and Sun, Yizhou and Wang, Wei},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Contact

For questions or issues, please contact Yidan Shi at yidanshi@cs.ucla.edu.
