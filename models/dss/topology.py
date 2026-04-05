import torch

from torch.nn import Linear, Sequential, BatchNorm1d
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.resolver import activation_resolver

from models.models_2d.encoders import AtomEncoder, BondEncoder


class GIN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.act = activation_resolver(act)

        self.conv = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i != num_layers - 1:
                self.conv.append(
                    GINEConv(Sequential(
                        Linear(hidden_dim, hidden_dim),
                        BatchNorm1d(hidden_dim),
                        self.act,
                        Linear(hidden_dim, hidden_dim),
                        self.act)))
                self.bn.append(BatchNorm1d(hidden_dim))
            else:
                self.conv.append(
                    GINEConv(Sequential(
                        Linear(hidden_dim, hidden_dim),
                        BatchNorm1d(hidden_dim),
                        self.act,
                        Linear(hidden_dim, output_dim),
                        self.act)))
                self.bn.append(BatchNorm1d(output_dim))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)
        self.out = Sequential(
            Linear(output_dim, output_dim),
            BatchNorm1d(output_dim),
            self.act,
            Linear(output_dim, output_dim),
        )

    @staticmethod
    def unbatch_first_element(batch_obj):
        num_nodes_per_graph = batch_obj.batch.bincount()[0]
        graph_data = Data()

        for key, value in batch_obj:
            if key == 'edge_index':
                edge_index = value[:, value[0] < num_nodes_per_graph]
                graph_data.edge_index = edge_index
            elif key == 'edge_attr':
                edge_mask = batch_obj.edge_index[0] < num_nodes_per_graph
                if value.ndim == 1:
                    edge_attr = value[edge_mask]
                else:
                    edge_attr = value[edge_mask, :]
                graph_data.edge_attr = edge_attr
            elif torch.is_tensor(value) and value.size(0) == batch_obj.num_nodes:
                graph_data[key] = value[:num_nodes_per_graph]
            else:
                graph_data[key] = value

        return graph_data

    def preprocess(self, data_list):
        data_first_list = [self.unbatch_first_element(data) for data in data_list]
        data = Batch.from_data_list(data_first_list)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        return {'node_feat': x, 'edge_index': edge_index, 'edge_attr': edge_attr}, x

    def block_call(self, i, x, data_dict, *args, **kwargs):
        edge_index = data_dict['edge_index']
        edge_attr = data_dict['edge_attr']
        x = self.conv[i](x, edge_index, edge_attr)
        x = self.bn[i](x)
        x = self.act(x)
        return x

    def postprocess(self, x, batch):
        x = self.out(x)
        x = global_add_pool(x, batch)
        return x
