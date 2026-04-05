import torch
from torch_scatter import scatter


def get_mol_batch(conf_node_idx, batch):
    temp = scatter(batch, conf_node_idx, dim=0, reduce='min')
    unique_values, unique_indices = torch.unique(temp, return_inverse=True)
    value_map = {v.item(): i for i, v in enumerate(unique_values)}
    mol_batch = torch.tensor([value_map[x.item()] for x in temp])
    return mol_batch


def get_mol_idx(data_list):
    molecule_index = [[i] * (d.batch.max().item() + 1) for i, d in enumerate(data_list)]
    molecule_index = sum(molecule_index, [])
    molecule_index = torch.Tensor(molecule_index).long()
    return molecule_index


def get_conf_node_idx(data_list):
    node_count = 0
    conf_node_idx = []
    for i, d in enumerate(data_list):
        num_confs = d.batch.max().item() + 1
        num_nodes = d.batch.shape[0] // num_confs
        conf_node_idx.extend(
            (torch.arange(num_nodes) + node_count).repeat(num_confs))
        node_count += num_nodes
    return torch.Tensor(conf_node_idx).long()