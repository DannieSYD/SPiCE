import torch
from e3nn import o3
from torch_cluster import radius_graph
from torch.utils.checkpoint import checkpoint

from models.models_3d.equiformer.drop import EquivariantDropout
from models.models_3d.equiformer.fast_activation import Activation
from models.models_3d.equiformer.gaussian_rbf import GaussianRadialBasisLayer
from models.models_3d.equiformer.graph_attention_transformer import NodeEmbeddingNetwork, EdgeDegreeEmbeddingNetwork, \
    _AVG_DEGREE, get_norm_layer, _RESCALE, ScaledScatter, _AVG_NUM_NODES, TransBlock
from models.models_3d.equiformer.graph_norm import EquivariantGraphNorm
from models.models_3d.equiformer.instance_norm import EquivariantInstanceNorm
from models.models_3d.equiformer.layer_norm import EquivariantLayerNormV2
from models.models_3d.equiformer.tensor_product_rescale import LinearRS
from models.models_3d.gemnet.radial_basis import RadialBasis


class Equiformer(torch.nn.Module):
    def __init__(
            self,
            irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
            irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
            max_radius=5.0,
            number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64],
            irreps_feature='512x0e',
            irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
            rescale_degree=False, nonlinear_message=False,
            irreps_mlp_mid='128x0e+64x1e+32x2e',
            norm_layer='layer',
            alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
            drop_path_rate=0.0,
            max_atomic_num=100,
            scale=None):
        super().__init__()

        # TODO: add customizable hidden_dim from irreps_node_embedding
        self.hidden_dim = 128
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.scale = scale

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, max_atomic_num)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(
                self.number_of_basis, cutoff=self.max_radius, rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding, self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)

        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('128x0e'), rescale=_RESCALE))
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                # irreps_block_output = self.irreps_feature
                # TODO: check if the last block of Equiformer can only output type0 feature
                irreps_block_output = self.irreps_node_embedding
            block = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(block)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear)
                    or isinstance(module, torch.nn.LayerNorm)
                    or isinstance(module, EquivariantLayerNormV2)
                    or isinstance(module, EquivariantInstanceNorm)
                    or isinstance(module, EquivariantGraphNorm)
                    or isinstance(module, GaussianRadialBasisLayer)
                    or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    def preprocess(self, data):
        z, pos, batch = data.x[:, 0], data.pos, data.batch
        edge_src, edge_dst = radius_graph(
            pos, r=self.max_radius, batch=batch, max_num_neighbors=1000)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component')
        atom_embed, atom_attr, atom_onehot = self.atom_embed(z)
        edge_len = edge_vec.norm(dim=1)
        edge_len_embed = self.rbf(edge_len)
        edge_deg_embed = self.edge_deg_embed(
            atom_embed, edge_sh, edge_len_embed, edge_src, edge_dst, batch)
        node_feat = atom_embed + edge_deg_embed
        node_attr = torch.ones_like(node_feat.narrow(1, 0, 1))
        return {
            'edge_dst': edge_dst,
            'edge_len_embed': edge_len_embed,
            'edge_sh': edge_sh,
            'edge_src': edge_src,
            'edge_index': edge_index,
            'node_attr': node_attr,
            'node_feat': node_feat
        }, node_feat

    def block_call(self, i, s, v, data_dict, batch):
        x = torch.cat((s, v.reshape(len(s), 3*self.hidden_dim)), dim=-1)
        block = self.blocks[i]
        node_attr = data_dict['node_attr']
        edge_src = data_dict['edge_src']
        edge_dst = data_dict['edge_dst']
        edge_sh = data_dict['edge_sh']
        edge_len_embed = data_dict['edge_len_embed']
        x_output = block(
            node_input=x, node_attr=node_attr,
            edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh,
            edge_scalars=edge_len_embed,
            batch=batch)
        s_emb = x_output[:, :self.hidden_dim]
        v_emb = x_output[:, self.hidden_dim:].reshape(len(s_emb), 3, self.hidden_dim)
        return s_emb, v_emb

    def postprocess(self, x, batch):
        x = self.norm(x, batch=batch)
        if self.out_dropout is not None:
            x = self.out_dropout(x)
        out = self.head(x)
        out = self.scale_scatter(out, batch, dim=0)

        if self.scale is not None:
            out = self.scale * out
        return out
    
    def forward(self, z, pos, batch):

        edge_src, edge_dst = radius_graph(
            pos, r=self.max_radius, batch=batch, max_num_neighbors=1000)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component')
        atom_embed, atom_attr, atom_onehot = self.atom_embed(z)
        edge_len = edge_vec.norm(dim=1)
        edge_len_embed = self.rbf(edge_len)
        edge_deg_embed = self.edge_deg_embed(
            atom_embed, edge_sh, edge_len_embed, edge_src, edge_dst, batch)
        node_feat = atom_embed + edge_deg_embed
        node_attr = torch.ones_like(node_feat.narrow(1, 0, 1))

        for i in range(len(self.blocks)):
            x = node_feat
            block = self.blocks[i]
            x_output = block(
                node_input=x, node_attr=node_attr,
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh,
                edge_scalars=edge_len_embed,
                batch=batch)
            s_emb = x_output[:, :self.hidden_dim]
            v_emb = x_output[:, self.hidden_dim:].reshape(len(s_emb), 3, self.hidden_dim)
        
        x = self.norm(s_emb, batch=batch)
        if self.out_dropout is not None:
            x = self.out_dropout(x)
        out = self.head(x)
        out = self.scale_scatter(out, batch, dim=0)

        if self.scale is not None:
            out = self.scale * out
        
        return out


