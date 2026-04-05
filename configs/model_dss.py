from dataclasses import dataclass, field

from configs.model_3d import PaiNN, ClofNet, Equiformer, ViSNet


@dataclass
class ModelDSS:
    conf_encoder: str = 'PaiNN'
    topo_encoder: str = 'GIN'

    painn: PaiNN = field(default_factory=PaiNN)
    clofnet: ClofNet = field(default_factory=ClofNet)
    equiformer: Equiformer = field(default_factory=Equiformer)
    visnet: ViSNet = field(default_factory=ViSNet)
