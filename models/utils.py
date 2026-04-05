import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import Linear, Sequential, SiLU, LayerNorm, Parameter


class CrossAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.attention_1 = Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_2 = Linear(hidden_dim, hidden_dim, bias=False)
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
            SiLU(),
            LayerNorm(hidden_dim),
        )
        self.phi1 = LayerNorm(hidden_dim)
        self.phi2 = LayerNorm(hidden_dim)

    def forward(self, x_mole, x_conf, batch):
        dot_product = torch.matmul(self.attention_1(self.phi1(x_mole)),
                                   self.attention_2(self.phi2(x_conf)).transpose(1, 0))
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
        max_values = (dot_product * mask).max(dim=1, keepdim=True).values
        masked_dot_product = (dot_product - max_values) * mask
        attention_weights = masked_dot_product.exp() / (masked_dot_product.exp() * mask).sum(dim=1, keepdim=True)
        attention_weights = attention_weights * mask
        x_weighted = torch.matmul(attention_weights, x_conf)
        x_encoded = self.rho(x_weighted)
        return x_encoded, attention_weights


class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(EquivariantLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.affine_weight = Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, field):
        field = field.permute(0, 2, 1)
        field = field - torch.mean(field, dim=1, keepdim=True)  # (N, 1, 3)
        field_norm = torch.mean(field.pow(2).sum(-1), dim=1, keepdim=True)  # (N, C) (N, 1)
        field_norm = (field_norm + self.eps).pow(-0.5) * self.affine_weight
        field = field * field_norm.reshape(-1, self.hidden_dim, 1)  # (N, C, 3)
        return field.permute(0, 2, 1)


def forward_block_with_checkpointing(block, *args):
    def forward(*inputs):
        return block(*inputs)

    return checkpoint(block, *args, use_reentrant=False)


