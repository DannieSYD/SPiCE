from dataclasses import dataclass, field


@dataclass
class ClofNet:
    cutoff: int = 6.5
    num_layers: int = 6
    hidden_channels: int = 128
    num_radial: int = 32


@dataclass
class PaiNN:
    hidden_dim: int = 128
    num_interactions: int = 3
    num_rbf: int = 64
    cutoff: float = 12.0
    readout: str = "add"
    shared_interactions: bool = False
    shared_filters: bool = False


@dataclass
class ViSNet:
    lmax: int = 1
    trainable_vecnorm: bool = False
    num_heads: int = 8
    num_layers: int = 6
    hidden_channels: int = 128
    num_rbf: int = 32
    trainable_rbf: bool = False
    cutoff: float = 5.0
    max_num_neighbors: int = 32
    vertex: bool = False
    reduce_op: str = "sum"
    mean: float = 0.0
    std: float = 1.0
    derivative: bool = False


@dataclass
class Equiformer:
    irreps_node_embedding: str = "128x0e+64x1e+32x2e"
    num_layers: int = 6
    irreps_sh: str = "1x0e+1x1e+1x2e"
    max_radius: float = 5.0
    number_of_basis: int = 128
    fc_neurons: list = field(default_factory=lambda: [64, 64])
    irreps_feature: str = "128x0e"
    irreps_head: str = "32x0e+16x1e+8x2e"
    num_heads: int = 4
    irreps_pre_attn: str = None
    rescale_degree: bool = False
    nonlinear_message: bool = True
    irreps_mlp_mid: str = "384x0e+192x1e+96x2e"
    norm_layer: str = "layer"
    alpha_drop: float = 0.2
    proj_drop: float = 0.0
    out_drop: float = 0.0
    drop_path_rate: float = 0.0
    scale: float = None
