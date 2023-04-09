import time
from typing import Optional

import torch_geometric.data
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader as tDataLoader
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
from layers.GNN import Encoder
from layers.embedding import GNN
import torch_scatter

from utils.utils import NT_Xent, JSE_global_global
import utils.utils as utils


minimum = 5e-6
inf = 1 - minimum * 2


class Explainer(nn.Module):

    def __init__(self, embed_dim: int, graph_level: bool, hidden_dim: int = 600):
        super(Explainer, self).__init__()

        self.embed_dims = embed_dim * (2 if graph_level else 3)
        self.cond_dims = embed_dim

        self.emb_linear1 = nn.Sequential(nn.Linear(self.embed_dims, hidden_dim), nn.ReLU())
        self.emb_linear2 = nn.Linear(hidden_dim, 1)


    def forward(self, embed):
        out = self.emb_linear1(embed)
        out = self.emb_linear2(out)
        return out

class MLP_Classifier(torch.nn.Module):

    def __init__(self, mlp_model, device):
        super(MLP_Classifier, self).__init__()
        self.model = mlp_model.to(device)
        self.device = device

    def forward(self, embeds, return_logits=False):
        embeds = embeds.to(self.device)
        self.model.eval()

        return self.get_probs(embeds, return_logits)

    def get_probs(self, embeds, return_logits=False):
        logits = self.model(embeds)
        if return_logits:
            return logits
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            probs = torch.cat([1 - probs, probs], 1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs


class Residual(nn.Module):
    def __init__(self, feat_dim, embed_dim, explain_graph):
        super(Residual, self).__init__()

        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.explain_graph = explain_graph
        if explain_graph:
            self.encoder = GNN(num_layer=5, emb_dim=600, JK='last', drop_ratio=0, gnn_type='gcn')
        else:
            self.encoder = Encoder(feat_dim, hidden_dim=embed_dim,
                                   n_layers=2, gnn='gcn', node_level=True, graph_level=False, bn=False,
                                   )

    def forward(self, data, *args, **kwargs):
        return self.encoder(data, *args, **kwargs)


class KHopSampler(MessagePassing):

    def __init__(self, k):
        super(KHopSampler, self).__init__(aggr='max', flow='source_to_target', node_dim=0)
        self.k = k

    def forward(self, edge_index, num_nodes, node_idx=None):
        if node_idx is None:
            S = torch.eye(num_nodes).to(edge_index.device)
        else:
            S = torch.zeros(num_nodes).to(edge_index.device)
            S = S.scatter_(0, node_idx.to(edge_index.device), 1.0)

        for it in range(self.k):
            S = self.propagate(edge_index, x=S)
        return S.bool()


class EGIBExplainer(nn.Module):

    def __init__(self, model, embed_dim: int, feat_dim: int, device, explain_graph: bool = True,
                 coff_size: float = 0.01, coff_ent: float = 5e-4, coff_ib: float = 0.01, coff_refine: float = 0.1,
                 coff_ir: float = 0.1,
                 grad_scale: float = 0.25,
                 t0: float = 10.0, t1: float = 1.0, num_hops: Optional[int] = None, loss_type="NCE",
                 residual_model=None, trick='cat'):
        super(EGIBExplainer, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.explain_graph = explain_graph
        self.model = model.to(device)
        self.explainer = Explainer(embed_dim, explain_graph).to(device)
        if residual_model is None:
            self.residual_model = Residual(feat_dim, embed_dim, explain_graph).to(device)
        else:
            self.residual_model = residual_model.to(device)

        self.mlp_classifier = None
        self.loss_type = loss_type

        self.grad_scale = grad_scale
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.coff_ib = coff_ib
        self.coff_refine = coff_refine
        self.coff_ir = coff_ir
        self.t0 = t0
        self.t1 = t1

        self.trick = trick

        self._set_hops(num_hops)
        self.sampler = KHopSampler(self.num_hops)
        self.S = None

    def _set_hops(self, num_hops: int):
        if num_hops is None:
            self.num_hops = sum(
                [isinstance(m, MessagePassing) for m in self.model.modules()])
        else:
            self.num_hops = num_hops

    def __set_masks__(self, edge_mask: Tensor, model):

        edge_mask = edge_mask.to(self.device)
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edge_mask
                module.__loop_mask__ = torch.ones(edge_mask.shape).bool()

    def __clear_masks__(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None

    def get_x_idx(self, data):
        try:
            x_idx = data.x_idx
        except:
            x_idx = None
        return x_idx

    def __loss__(self, data, target_embed: Tensor, input_embed: Tensor,
                 refine: bool, edge_mask_sigmoid: Tensor, label=None, **kwargs):


        ### calculation of I(Z;S) ######
        if self.loss_type == 'NCE':
            loss_lower = NT_Xent(target_embed, input_embed)
        elif self.loss_type == 'JSE':
            loss_lower = JSE_global_global(target_embed, input_embed)
        else:
            raise NotImplementedError

        ### calculation of I(S; G) ######

        if self.coff_ib < 1e-9:
            residual_embed = None
            graph_embed = None
            loss_residual = torch.tensor(0.0)
            loss_disen = torch.tensor(0.0)
            self.__set_masks__(1 - edge_mask_sigmoid, self.model)
            residual_embed_ = self.model(data)
            self.__clear_masks__(self.model)
        else:
            self.__set_masks__(1 - edge_mask_sigmoid, self.residual_model)
            residual_embed = self.residual_model(data)
            self.__clear_masks__(self.residual_model)
            graph_embed = self.residual_model(data)
            if not self.get_x_idx(data) is None:
                residual_embed = residual_embed[data.x_idx >= 0]
                graph_embed = graph_embed[data.x_idx >= 0]
            if self.loss_type == 'NCE':
                loss_residual = NT_Xent(residual_embed, graph_embed)
            elif self.loss_type == 'JSE':
                loss_residual = JSE_global_global(residual_embed, graph_embed)
            else:
                raise NotImplementedError

            self.__set_masks__(1 - edge_mask_sigmoid, self.model)
            residual_embed_ = self.model(data)
            self.__clear_masks__(self.model)
            if not self.get_x_idx(data) is None:
                residual_embed_ = residual_embed_[data.x_idx >= 0]
            if self.loss_type == 'NCE':
                loss_disen = -NT_Xent(input_embed, residual_embed_)
            elif self.loss_type == 'JSE':
                loss_disen = -JSE_global_global(residual_embed_, input_embed)
            else:
                raise NotImplementedError

        ### calculation of I(\tilde{S}; Z) ######
        if self.loss_type == 'NCE':
            loss_res = -NT_Xent(target_embed, residual_embed_)
        elif self.loss_type == 'JSE':
            loss_res = -JSE_global_global(residual_embed_, target_embed)
        else:
            raise NotImplementedError
        # loss_res = torch.tensor(0)
        loss = loss_lower + self.coff_ir * loss_res + (loss_residual + loss_disen) * self.coff_ib


        #### calculation of I(\hat{Y};S) ######
        if refine:
            self.mlp_classifier.eval()
            if label is None:
                target_probs = self.mlp_classifier(target_embed).detach()
                label = torch.argmax(target_probs, dim=1).squeeze()
            input_logits = self.mlp_classifier(input_embed)
            index = (torch.arange(input_logits.shape[0]), label)
            input_logits = input_logits[index].squeeze() + minimum
            loss_pos = (-torch.log(input_logits)).mean()
            residual_logits = self.mlp_classifier(residual_embed_)
            residual_logits = residual_logits[index].squeeze() + minimum
            loss_neg = (torch.log(residual_logits)).mean()
            # print(loss_pos.item(), loss_neg.item())
            loss_refine = loss_pos + loss_neg
            loss = loss + loss_refine * self.coff_refine
        else:
            loss_refine = torch.tensor(0)

        ########calculation of l1 norm constraint ##########
        size_loss = (torch.mean(edge_mask_sigmoid)).abs()

        assert torch.all(torch.logical_and(edge_mask_sigmoid >= 0, edge_mask_sigmoid <= 1)), 'illegal edge_mask'
        edge_mask_sigmoid_ = edge_mask_sigmoid * inf + minimum
        mask_ent = -edge_mask_sigmoid * torch.log(edge_mask_sigmoid_) - (1 - edge_mask_sigmoid) * torch.log(
            1 - edge_mask_sigmoid_)
        mask_ent = torch.mean(mask_ent)
        if self.trick == 'ber':
            loss = loss + self.coff_size * size_loss + self.coff_ent * mask_ent

        ##### computing ICE entropy#########

        loss_log = f'lower: {loss_lower.item():.4f},res: {loss_res.item():.4f}, residual:{loss_residual.item():.4f}, ' \
                   f'disen: {loss_disen.item():.4f}, size:{size_loss.item():.4f}, refine:{loss_refine:.4f}'

        return loss, loss_log

    def get_subgraph(self, node_idx: int, data: Data, edge_index_selfloop):

        x, edge_index, edge_attr, y, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
        try:
            deg_inv_sqrt = data.deg_inv_sqrt
        except:
            deg_inv_sqrt = None

        node_mask = self.sampler(edge_index_selfloop, x.shape[0], node_idx).cpu()
        subset = torch.arange(x.shape[0])[node_mask]
        edge_index, edge_attr = torch_geometric.utils.subgraph(subset, edge_index, edge_attr, num_nodes=x.shape[0],
                                                               relabel_nodes=True)

        x_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device) - 1
        x_idx[node_idx] = node_idx  # record the original idx

        x = x[subset]
        x_idx = x_idx[subset]
        y = y[subset] if y is not None else None
        deg_inv_sqrt = deg_inv_sqrt[subset] if deg_inv_sqrt is not None else None
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=None, x_idx=x_idx,
                    deg_inv_sqrt=deg_inv_sqrt)
        return data, subset

    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()

        else:
            gate_inputs = log_alpha

        return gate_inputs

    def Gumbel_topk_sample(self, log_alpha: Tensor, beta=1.0, training=True, ratio=0.5, edge_batch=None):
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = - torch.log(-torch.log(random_noise))
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha)
            index = utils.topk(gate_inputs, ratio, edge_batch)
            mask = log_alpha.new_zeros(log_alpha.shape)

            probs = torch_scatter.scatter_softmax(log_alpha / beta, edge_batch)[index]
            mask[index] = 1 - probs.detach() + probs
            return mask

        else:
            gate_inputs = log_alpha

            return gate_inputs



    def explain(self, data: Data, embed: Tensor, ratio=0.5,
                tmp: float = 1.0, training: bool = False, refine=False, **kwargs):

        nodesize = embed.shape[0]
        edge_index, _ = remove_self_loops(data.edge_index)
        col, row = edge_index
        f1 = embed[col]
        f2 = embed[row]

        edge_batch = data.batch[col]
        if self.explain_graph:
            f12self = torch.cat([f1, f2], dim=-1)
        else:
            node_idx = kwargs.get('node_idx')
            self_embed = embed[node_idx][edge_batch]
            f12self = torch.cat([f1, f2, self_embed], dim=-1)

        h = self.explainer(f12self.to(self.device))

        mask_val = h.reshape(-1)


        if self.trick == 'cat':
            values = self.Gumbel_topk_sample(mask_val, beta=tmp, training=training, ratio=ratio, edge_batch=edge_batch)
        else:
            values = self.concrete_sample(mask_val, beta=tmp, training=training)

        try:
            out_log = '%.4f, %.4f' % (
                values.max().item(), values.min().item())
        except:
            out_log = ''
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, values, (nodesize, nodesize)
        )
        mask = mask_sparse.to_dense()

        sym_mask = (mask + mask.transpose(0, 1)) / 2
        edge_mask = sym_mask[col, row]

        self.__clear_masks__(self.model)
        self.__set_masks__(edge_mask, self.model)
        embed = self.model(data)

        self.__clear_masks__(self.model)

        return embed, edge_mask, edge_mask, out_log

    def train_explainer_graph(self, loader, refine=False, lr=0.001, epochs=10, ratio=0.5, n_labels=2):

        residual_opt = Adam(self.residual_model.parameters(), lr=lr)
        explainer_opt = Adam(self.explainer.parameters(), lr=lr)
        for epoch in range(epochs):
            tmp = float(self.t1 * np.power(self.t1 / self.t0, epoch / epochs))
            print(f"tmp:{tmp:.4f}")
            self.model.eval()
            self.explainer.train()
            self.residual_model.train()
            pbar = tqdm(loader)
            loss_sum = 0
            for i, data in enumerate(pbar):

                residual_opt.zero_grad()
                explainer_opt.zero_grad()
                data = data.to(self.device)

                with torch.no_grad():
                    embed_target, node_embed = self.model(data, emb=True)

                row, col = data.edge_index
                edge_batch = data.batch[row]

                pruned_embed, mask_sigmoid, edge_mask, log = self.explain(data, embed=node_embed,
                                                                          tmp=tmp, training=True, refine=refine,
                                                                          ratio=ratio
                                                                          )
                target_class = None
                loss, loss_log = self.__loss__(data, embed_target, pruned_embed, refine, mask_sigmoid,
                                               label=target_class)
                loss_sum += loss.item()
                pbar.set_postfix(
                    {'loss_log': loss_log})

                loss.backward()
                explainer_opt.step()
                residual_opt.step()


    def train_explainer_node(self, loader, refine=False, lr=0.001, epochs=10, n_labels=2, batch_size=256, ratio=0.5):
        residual_opt = Adam(self.residual_model.parameters(), lr=lr)
        explainer_opt = Adam(self.explainer.parameters(), lr=lr)

        for epoch in range(epochs):
            tmp = float(self.t1 * np.power(self.t1 / self.t0, epoch / epochs))
            self.model.eval()
            self.explainer.train()
            self.residual_model.eval()
            for i, data in enumerate(loader):
                with torch.no_grad():
                    self.model.cpu()
                    all_embeds = self.model(data)
                self.model.to(self.device)

                edge_index = torch_geometric.utils.to_undirected(data.edge_index, num_nodes=data.num_nodes)
                edge_index_selfloop, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)
                degree = torch_geometric.utils.degree(edge_index_selfloop[0], data.num_nodes)
                deg_inv_sqrt = degree.pow_(-0.5)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
                data.deg_inv_sqrt = deg_inv_sqrt
                edge_index_selfloop = edge_index_selfloop.to(self.device)

                node_list = torch.arange(data.num_nodes).tolist()
                node_batches = torch.utils.data.DataLoader(node_list,
                                                           batch_size=batch_size, shuffle=True)

                pbar = tqdm(node_batches)

                for node_iter, node_batch in enumerate(pbar):

                    with torch.no_grad():
                        embeds = all_embeds[node_batch].to(self.device)
                        subgraphs = []
                        subsets = []
                        for i, node_idx in enumerate(node_batch):
                            subgraph, subset = self.get_subgraph(node_idx=node_idx, data=data,
                                                                 edge_index_selfloop=edge_index_selfloop)
                            subgraphs.append(subgraph)
                            subsets.append(subset)

                        subsets = torch.cat(subsets, dim=0)
                        subgraphs = torch_geometric.data.Batch().from_data_list(subgraphs)
                        subgraphs = subgraphs.to(self.device)
                        new_node_idx = torch.where(subgraphs.x_idx >= 0)[0]


                    pruned_embed, mask_sigmoid, edge_mask, log = self.explain(subgraphs,
                                                                              embed=all_embeds[subsets].to(self.device),
                                                                              tmp=tmp, training=True,
                                                                              refine=refine, node_idx=new_node_idx,
                                                                              ratio=ratio)

                    pruned_embed = pruned_embed[new_node_idx]

                    target_class = None
                    loss, loss_log = self.__loss__(subgraphs, embeds, pruned_embed, refine, mask_sigmoid,
                                                   label=target_class)
                    pbar.set_postfix(
                        {'loss_log': loss_log})

                    residual_opt.zero_grad()
                    explainer_opt.zero_grad()

                    loss.backward()
                    explainer_opt.step()
                    residual_opt.step()


    def __hard_edge_mask__(self, data, edge_mask, top_k, counterfactual=True):
        try:
            x_idx = data.x_idx
        except:
            x_idx = None

        try:
            deg_inv_sqrt = data.deg_inv_sqrt
        except:
            deg_inv_sqrt = None

        if top_k <= 0:
            if counterfactual:
                hard_edge_mask = torch.ones(edge_mask.shape).type(torch.float32).to(self.device)
                return data, hard_edge_mask
            else:
                hard_edge_mask = torch.zeros(edge_mask.shape).type(torch.float32).to(self.device)
                data = Data(x=data.x, x_idx=x_idx, deg_inv_sqrt=deg_inv_sqrt,
                            edge_index=torch.tensor([], dtype=torch.long).to(data.x.device),
                            edge_attr=torch.tensor([], dtype=torch.int64).to(data.x.device),
                            batch=torch.zeros(data.x.shape[0], dtype=torch.int64).to(self.device))
                return data, hard_edge_mask
        edge_idx_list = edge_mask.reshape(-1).argsort(descending=True)[0:min(top_k, edge_mask.shape[0])]

        if counterfactual:
            hard_edge_mask = torch.ones(edge_mask.shape).type(torch.float32).to(self.device)
            hard_edge_mask[edge_idx_list] = 0
            hard_edge_mask = hard_edge_mask.bool()
        else:
            hard_edge_mask = torch.zeros(edge_mask.shape).type(torch.float32).to(self.device)
            hard_edge_mask[edge_idx_list] = 1
            hard_edge_mask = hard_edge_mask.bool()

        ret_edge_index = data.edge_index[:, hard_edge_mask]
        ret_edge_attr = None if data.edge_attr is None else data.edge_attr[hard_edge_mask]

        data = Data(x=data.x, x_idx=x_idx, deg_inv_sqrt=deg_inv_sqrt, edge_index=ret_edge_index,
                    edge_attr=ret_edge_attr, batch=torch.zeros(data.x.shape[0], dtype=torch.int64).to(self.device))

        return data, hard_edge_mask

    def forward(self, data: Data, refine=False, ratio=1.0, mask_ratio_list=None, mask_inv_ratio_list=None,
                edge_index_selfloop=None, **kwargs):

        node_idx = kwargs.get('node_idx')
        self.model.eval()
        self.residual_model.eval()
        self.explainer.eval()
        self.mlp_classifier.eval()
        data = data.to(self.device)

        self.__clear_masks__(self.model)
        if node_idx is not None:
            node_embed = self.model(data)
            embed = node_embed[node_idx]
        elif self.explain_graph:
            embed, node_embed = self.model(data, emb=True)
        else:
            assert node_idx is not None, "please input the node_idx"
        probs = self.mlp_classifier(embed)

        row, col = data.edge_index

        if self.explain_graph:

            subgraphs = data
            subgraphs = subgraphs.to(self.device)
            pruned_embed, mask_sigmoid, edge_mask, log = self.explain(subgraphs,
                                                                      embed=node_embed.to(self.device),
                                                                      tmp=1.0, training=False,
                                                                      refine=refine, ratio=ratio)

            edge_index_wo_loop, _ = torch_geometric.utils.remove_self_loops(subgraphs.edge_index)
            row, col = edge_index_wo_loop
            edge_batch = subgraphs.batch[row]

            target_class = torch.argmax(probs) if subgraphs.y is None else torch.where(
                subgraphs.y > 0, subgraphs.y, 0).long()  # sometimes labels are +1/-1

            masks = edge_mask

            related_preds = {
                'counterfactual_masked_probs': [],
                'origin': [],
                'counterfactual_sparsity_scores': []}

            for i in range(subgraphs.num_graphs):

                mask = masks[edge_batch == i]
                subgraph = subgraphs.get_example(i)
                masked_datas = []
                hard_edge_masks = []
                # print(subgraph)

                for r in mask_ratio_list:
                    top_k = max(1, int(r * mask.shape[0]))

                    masked_data, hard_edge_mask = self.__hard_edge_mask__(subgraph, mask, top_k)
                    masked_datas.append(masked_data)
                    hard_edge_masks.append(hard_edge_mask)


                masked_datas = torch_geometric.data.Batch().from_data_list(masked_datas)
                masked_embed = self.model(masked_datas)
                counterfactual_masked_prob = self.mlp_classifier(masked_embed)


                counterfactual_masked_prob = counterfactual_masked_prob[:, target_class[i]]
                counterfactual_sparsity_score = [(torch.sum(hard_edge_mask) / hard_edge_mask.shape[0]).item() for
                                                 hard_edge_mask
                                                 in
                                                 hard_edge_masks]

                counterfactual_masked_probs = counterfactual_masked_prob.squeeze().tolist()
                counterfactual_sparsity_scores = counterfactual_sparsity_score

                related_preds['counterfactual_masked_probs'].append(counterfactual_masked_probs)
                related_preds['origin'].append(probs[i, target_class[i]].item())
                related_preds['counterfactual_sparsity_scores'].append(counterfactual_sparsity_scores)

            return None, edge_mask.sigmoid().detach().cpu(), [related_preds]

        else:
            subgraphs = []
            subsets = []

            time1 = time.time()

            for i, idx in enumerate(node_idx):
                subgraph, subset = self.get_subgraph(node_idx=idx, data=data, edge_index_selfloop=edge_index_selfloop)
                subgraphs.append(subgraph)
                subsets.append(subset)

            subsets = torch.cat(subsets, dim=0)
            subgraphs = torch_geometric.data.Batch().from_data_list(subgraphs)
            subgraphs = subgraphs.to(self.device)
            new_node_idx = torch.where(subgraphs.x_idx >= 0)[0]


            pruned_embed, mask_sigmoid, edge_mask, log = self.explain(subgraphs,
                                                                      embed=node_embed[subsets].to(self.device),
                                                                      tmp=1.0, training=False,
                                                                      refine=refine, node_idx=new_node_idx,
                                                                      ratio=ratio)

            edge_index_wo_loop, _ = torch_geometric.utils.remove_self_loops(subgraphs.edge_index)
            row, col = edge_index_wo_loop
            edge_batch = subgraphs.batch[row]

            target_class = torch.argmax(probs) if subgraphs.y is None else torch.where(
                subgraphs.y[new_node_idx] > 0, subgraphs.y[new_node_idx], 0).long()  # sometimes labels are +1/-1


            masks = edge_mask

            related_preds = {
                'counterfactual_masked_probs': [],
                'origin': [],
                'counterfactual_sparsity_scores': []}

            for i, idx in enumerate(node_idx):

                mask = masks[edge_batch == i]
                subgraph = subgraphs.get_example(i)
                masked_datas = []
                hard_edge_masks = []

                for r in mask_ratio_list:
                    top_k = max(1, int(r * mask.shape[0]))

                    masked_data, hard_edge_mask = self.__hard_edge_mask__(subgraph, mask, top_k)
                    masked_datas.append(masked_data)
                    hard_edge_masks.append(hard_edge_mask)


                masked_datas = torch_geometric.data.Batch().from_data_list(masked_datas)
                masked_embed = self.model(masked_datas)[masked_datas.x_idx >= 0]

                counterfactual_masked_prob = self.mlp_classifier(masked_embed)
                counterfactual_masked_prob = counterfactual_masked_prob[:, target_class[i]]
                counterfactual_sparsity_score = [(torch.sum(hard_edge_mask) / hard_edge_mask.shape[0]).item() for
                                                 hard_edge_mask
                                                 in
                                                 hard_edge_masks]

                counterfactual_masked_probs = counterfactual_masked_prob.squeeze().tolist()
                counterfactual_sparsity_scores = counterfactual_sparsity_score


                related_preds['counterfactual_masked_probs'].append(counterfactual_masked_probs)
                related_preds['origin'].append(probs[i, target_class[i]].item())
                related_preds['counterfactual_sparsity_scores'].append(counterfactual_sparsity_scores)
            return None, None, [related_preds]


