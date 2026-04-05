import torch
from torch.nn import Linear, Sequential, Tanh, SiLU, LayerNorm, Parameter
from torch.nn.functional import gumbel_softmax
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from utils.conformer_process_utils import get_conf_node_idx, get_mol_idx, get_mol_batch
from torch.utils.checkpoint import checkpoint
import time
import copy

from models.utils import CrossAttention, EquivariantLayerNorm, forward_block_with_checkpointing


class DSSNetV2(torch.nn.Module):
    def __init__(
            self, hidden_dim, out_dim,
            conf_model_factory, topo_model_factory, device, num_experts=4, num_activated=2,
            num_parts=1, gig=True, ad=False, sag=True, upc=False, upcycling_epochs=50, gumbel_tau=1):
        super().__init__()
        self.num_experts = num_experts
        self.num_activated = num_activated
        self.device = device
        self.GiG = gig
        self.ad = ad
        self.sag = sag
        self.upcycling = upc
        self.hidden_dim = hidden_dim
        self.upcycling_epochs = upcycling_epochs
        self.gumbel_tau = gumbel_tau

        self.graph_encoders = torch.nn.ModuleList([conf_model_factory()])
        self.topo_encoders = torch.nn.ModuleList([topo_model_factory()])
        self.num_blocks = len(self.graph_encoders[0].blocks)
        self.cross_attention = torch.nn.ModuleList([CrossAttention(hidden_dim) for _ in range(self.num_blocks)])
        self.linear = torch.nn.Linear(hidden_dim, out_dim)
        self.norm_layers = torch.nn.ModuleList([LayerNorm(hidden_dim) for _ in range(self.num_blocks)])
        self.norm_layers_2 = torch.nn.ModuleList([LayerNorm(hidden_dim) for _ in range(self.num_blocks)])
        self.norm_layers_3 = torch.nn.ModuleList([LayerNorm(hidden_dim) for _ in range(self.num_blocks)])

        self.norm_layers_v = torch.nn.ModuleList([EquivariantLayerNorm(hidden_dim) for _ in range(self.num_blocks)])
        self.norm_layers_2_v = torch.nn.ModuleList([EquivariantLayerNorm(hidden_dim) for _ in range(self.num_blocks)])

        self.gcn = torch.nn.ModuleList([GCNConv(hidden_dim, self.num_experts) for _ in range(self.num_blocks)])
        self.lin = torch.nn.ModuleList(
            [Linear(hidden_dim * hidden_dim, self.num_experts) for _ in range(self.num_blocks)])
        self.experts_x = torch.nn.ModuleList(
            [Sequential(Linear(hidden_dim, hidden_dim), SiLU(), Linear(hidden_dim, hidden_dim), SiLU(),
                        LayerNorm(hidden_dim))
             for _ in range(self.num_experts * self.num_blocks)]
        )
        self.experts_v_x = torch.nn.ModuleList(
            [Sequential(Linear(hidden_dim, hidden_dim), SiLU(), Linear(hidden_dim, hidden_dim), SiLU())
             for _ in range(self.num_experts * self.num_blocks)]
        )
        self.experts_v = torch.nn.ModuleList([Linear(hidden_dim, hidden_dim, bias=False)
                                              for _ in range(self.num_experts * self.num_blocks)])
        self.experts_v_ln = torch.nn.ModuleList([EquivariantLayerNorm(hidden_dim)
                                                 for _ in range(self.num_experts * self.num_blocks)])

    def moe_block(self, i, emb_x, emb_v, data_dict_conf, epoch, scores):
        logits_x = self.router_x(i, emb_x, data_dict_conf)
        logits_v = self.router_v(i, emb_v)
        topk_values_x, topk_indices_x = torch.topk(logits_x, self.num_activated, dim=1)
        topk_values_reweight_x = torch.softmax(topk_values_x, dim=1)
        topk_values_v, topk_indices_v = torch.topk(logits_v, self.num_activated, dim=1)
        topk_values_reweight_v = torch.softmax(topk_values_v, dim=1)

        if self.upcycling:
            if epoch < self.upcycling_epochs:
                layer_outputs_x = self.experts_x[i * self.num_experts](emb_x)
                layer_outputs_v_x = self.experts_v_x[i * self.num_experts](emb_x)
                layer_outputs_v = self.experts_v[i * self.num_experts](layer_outputs_v_x)
                layer_outputs_v = layer_outputs_v.unsqueeze(1).expand(-1, 3, -1) * emb_v
                layer_outputs_v = self.experts_v_ln[i * self.num_experts](layer_outputs_v)
                layer_outputs_v = self.norm_layers_2_v[i](layer_outputs_v + emb_v)
                layer_outputs_x = self.norm_layers_2[i](layer_outputs_x + emb_x)
            elif epoch == self.upcycling_epochs:
                for p in range(1, self.num_experts):
                    self.experts_x[i * self.num_experts + p] = copy.deepcopy(self.experts_x[i * self.num_experts])
                    self.experts_v_x[i * self.num_experts + p] = copy.deepcopy(
                        self.experts_v_x[i * self.num_experts])
                    self.experts_v[i * self.num_experts + p] = copy.deepcopy(self.experts_v[i * self.num_experts])
                    self.experts_v_ln[i * self.num_experts + p] = copy.deepcopy(
                        self.experts_v_ln[i * self.num_experts])
                layer_outputs_x, layer_outputs_v = self.moe(emb_x, emb_v, topk_indices_x, topk_indices_v,
                                                            topk_values_reweight_x, topk_values_reweight_v, i)
            else:
                layer_outputs_x, layer_outputs_v = self.moe(emb_x, emb_v, topk_indices_x, topk_indices_v,
                                                            topk_values_reweight_x, topk_values_reweight_v, i)
        else:
            layer_outputs_x = emb_x
            layer_outputs_v = emb_v
        scores.append(torch.softmax(torch.cat((logits_x, logits_v), dim=1), dim=1))

        return layer_outputs_x, layer_outputs_v, scores

    def moe(self, emb_x, emb_v, topk_indices_x, topk_indices_v, topk_values_reweight_x,
            topk_values_reweight_v, i):
        expert_mask_x = torch.zeros(topk_indices_x.size(0), self.num_experts, device=topk_indices_x.device)  # (M, E)
        expert_mask_x.scatter_(1, topk_indices_x, 1)
        node_expert_mask_x = expert_mask_x
        node_expert_weights_x = torch.zeros(emb_x.size(0), self.num_experts, device=emb_x.device)  # (N, E)
        node_expert_weights_x.scatter_(1, topk_indices_x, topk_values_reweight_x)

        expert_mask_v = torch.zeros(topk_indices_v.size(0), self.num_experts, device=topk_indices_v.device)  # (M, E)
        expert_mask_v.scatter_(1, topk_indices_v, 1)
        node_expert_mask_v = expert_mask_v
        node_expert_weights_v = torch.zeros(emb_v.size(0), self.num_experts, device=emb_v.device)  # (N, E)
        node_expert_weights_v.scatter_(1, topk_indices_v, topk_values_reweight_v)

        # Compute all expert outputs at once
        all_expert_outputs_x = torch.stack([expert(emb_x) for expert in
                                            self.experts_x[i * self.num_experts:(i + 1) * self.num_experts]])
        all_expert_outputs_v_x = torch.stack([expert(emb_x) for expert in
                                              self.experts_v_x[i * self.num_experts:(i + 1) * self.num_experts]])
        all_expert_outputs_v = torch.stack([expert(all_expert_outputs_v_x[idx]).repeat_interleave(3, dim=0)
                                            * emb_v.reshape(-1, self.hidden_dim) for idx, expert in
                                            enumerate(self.experts_v[i * self.num_experts:(i + 1) * self.num_experts])])
        all_expert_outputs_v_ln = torch.stack(
            [expert(all_expert_outputs_v[idx].reshape(emb_v.shape)).reshape(-1, self.hidden_dim)
             for idx, expert in
             enumerate(self.experts_v_ln[i * self.num_experts:(i + 1) * self.num_experts])])

        all_expert_outputs_x = all_expert_outputs_x.permute(1, 0, 2)  # (N, num_experts, hidden_dim)
        all_expert_outputs_v = all_expert_outputs_v_ln.permute(1, 0, 2).reshape(emb_v.shape[0], self.num_experts, 3,
                                                                                self.hidden_dim)

        node_expert_weights_x = node_expert_weights_x.unsqueeze(-1)
        node_expert_weights_v = node_expert_weights_v.unsqueeze(-1)
        layer_outputs_x = (all_expert_outputs_x * node_expert_mask_x.unsqueeze(-1) * node_expert_weights_x).sum(dim=1)
        layer_outputs_v = (all_expert_outputs_v * node_expert_mask_v.unsqueeze(-1).unsqueeze(
            -1) * node_expert_weights_v.unsqueeze(-1)).sum(dim=1)

        layer_outputs_x = self.norm_layers_2[i](layer_outputs_x.reshape(emb_x.shape) + emb_x)
        layer_outputs_v = self.norm_layers_2_v[i](layer_outputs_v.reshape(emb_v.shape) + emb_v)

        return layer_outputs_x, layer_outputs_v

    def router_x(self, i, emb_x, data_dict_conf):
        weight_x = torch.softmax(self.gcn[i](emb_x, data_dict_conf['edge_index']), dim=1)  # gating
        score_x = weight_x
        gumbel_x = torch.sigmoid(score_x + gumbel_softmax(score_x, tau=self.gumbel_tau, hard=False, dim=-1)
                                 - gumbel_softmax(score_x, tau=self.gumbel_tau, hard=False, dim=-1))
        return gumbel_x

    def router_v(self, i, emb_v):
        v_sum = torch.einsum('nij,njk->nik', emb_v.permute(0, 2, 1), emb_v)  # n,c,c
        weight_v = torch.softmax(self.lin[i](v_sum.reshape(len(v_sum), -1)), dim=1)
        score_v = weight_v
        gumbel_v = torch.sigmoid(score_v + gumbel_softmax(score_v, tau=self.gumbel_tau, hard=False, dim=-1)
                                 - gumbel_softmax(score_v, tau=self.gumbel_tau, hard=False, dim=-1))
        return gumbel_v

    def forward(self, data, epoch, batch_size, loss_expected) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = []
        part_data = data[0].to(self.device)
        data_list = part_data.to_data_list()
        batch = part_data.batch

        part_molecule_indices = get_mol_idx(data_list).to(self.device)
        conf_node_idx = get_conf_node_idx(data_list).to(self.device)
        unique_nodes, counts = torch.unique_consecutive(conf_node_idx, return_counts=True)
        reversed_idx = torch.repeat_interleave(unique_nodes, counts)

        conf_encoder = self.graph_encoders[0]
        topo_encoder = self.topo_encoders[0]

        if conf_encoder.__class__.__name__ in ['PaiNN', 'GVP_GNN', 'ClofNet']:
            data_dict_conf, x_conf, v_conf = conf_encoder.preprocess(part_data)
        elif conf_encoder.__class__.__name__ == 'ViSNet_DSS':
            data_dict_conf, x_conf, v_conf, edge_attr = conf_encoder.preprocess(part_data)
        elif conf_encoder.__class__.__name__ == 'SphereNet':
            data_dict_conf, e, v, u = conf_encoder.preprocess(part_data)
            v_conf, x_conf = e
        elif conf_encoder.__class__.__name__ == 'SEGNNModel':
            data_dict_conf, x_conf, v_conf, pos = conf_encoder.preprocess(part_data)
        elif conf_encoder.__class__.__name__ == 'Equiformer':
            data_dict_conf, x_v_emb = conf_encoder.preprocess(part_data)
            x_conf = x_v_emb[:, :self.hidden_dim]
            v_conf = x_v_emb[:, self.hidden_dim:].reshape(len(x_conf), 3, self.hidden_dim)
        else:
            print("Unknown Encoder Type!")

        data_dict_topo, x_topo = topo_encoder.preprocess(data_list)

        for i in range(len(self.graph_encoders[0].blocks)):
            if conf_encoder.__class__.__name__ in ['PaiNN', 'GVP_GNN', 'ClofNet']:
                h_conf, m_conf = forward_block_with_checkpointing(conf_encoder.block_call, i, x_conf, v_conf,
                                                                  data_dict_conf, batch)
            elif conf_encoder.__class__.__name__ == 'ViSNet_DSS':
                h_conf, m_conf, edge_attr = forward_block_with_checkpointing(conf_encoder.block_call, i, x_conf,
                                                                             v_conf, data_dict_conf, edge_attr)
            elif conf_encoder.__class__.__name__ == 'SphereNet':
                h_conf, m_conf, u = forward_block_with_checkpointing(conf_encoder.block_call, i, x_conf,
                                                                     v_conf, u, data_dict_conf, batch)
            elif conf_encoder.__class__.__name__ == 'SEGNNModel':
                h_conf, m_conf, pos = forward_block_with_checkpointing(conf_encoder.block_call, i, x_conf,
                                                                       v_conf, pos, data_dict_conf, batch)
            elif conf_encoder.__class__.__name__ == 'Equiformer':
                h_conf, m_conf = conf_encoder.block_call(i, x_conf, v_conf, data_dict_conf, batch)

            emb_x = self.norm_layers[i](x_conf + h_conf)
            emb_v = self.norm_layers_v[i](v_conf + m_conf)

            moe_x, moe_v, scores = self.moe_block(i, emb_x, emb_v, data_dict_conf, epoch, scores)
            v_conf = moe_v
            h_topo = scatter(moe_x, conf_node_idx, dim=0, reduce='mean')
            x_topo = topo_encoder.block_call(i, h_topo, data_dict_topo)
            x_conf, attention_weight = forward_block_with_checkpointing(self.cross_attention[i],
                                                                        x_topo[reversed_idx], emb_x, batch)

        if conf_encoder.__class__.__name__ in ['PaiNN', 'Equiformer']:
            outs = conf_encoder.postprocess(x_conf, batch)
        elif conf_encoder.__class__.__name__ == 'GVP_GNN':
            outs = conf_encoder.postprocess(x_conf, v_conf, batch)
        elif conf_encoder.__class__.__name__ == 'ViSNet_DSS':
            outs = conf_encoder.postprocess(part_data, x_conf, v_conf, batch)
        elif conf_encoder.__class__.__name__ == 'SphereNet':
            outs = conf_encoder.postprocess(u)
        elif conf_encoder.__class__.__name__ == 'SEGNNModel':
            outs = conf_encoder.postprocess(x_conf, v_conf, data_dict_conf, batch)
        elif conf_encoder.__class__.__name__ == 'ClofNet':
            outs = conf_encoder.postprocess(v_conf, batch)

        outs = global_mean_pool(outs, part_molecule_indices)
        if conf_encoder.__class__.__name__ in ['PaiNN', 'GVP_GNN', 'SphereNet', 'SEGNNModel', 'ClofNet', 'Equiformer']:
            outs = self.linear(outs).squeeze(-1)
        elif conf_encoder.__class__.__name__ == 'ViSNet_DSS':
            outs = outs.squeeze(-1)

        scores = torch.cat(scores, dim=1)
        z_loss = scatter(torch.log(torch.sum(torch.exp(scores), dim=1)), batch, dim=0, reduce='sum')
        z_loss = scatter(z_loss, part_molecule_indices, dim=0, reduce='sum')

        return outs, z_loss, x_conf
