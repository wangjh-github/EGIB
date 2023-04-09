import copy

import torch
import pandas as pd

from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import PPI
from torch_geometric.utils import remove_isolated_nodes

from torch import Tensor
from typing import Optional, Union

from layers.embedding import GNN

from utils.loader import MoleculeDataset
from utils.splitters import scaffold_split

import os

from layers.GNN import Encoder
from layers.SupervisedGIN import SupervisedGIN
from torch_scatter import scatter_add, scatter_max
import torch_geometric
import torch.nn.functional as F
from .visual import *


def get_task(idx):
    def transform(data):
        return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y[idx:idx + 1].long())

    return transform


def get_task_rm_iso(idx):
    def transform(data):
        edge_index, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=data.x.shape[0])
        return Data(x=data.x[mask], edge_index=edge_index, y=data.y[mask, idx])

    return transform


def remove_self_loop(dataset):
    datalist = []
    for d in dataset:

        data = copy.deepcopy(d)
        mask = data.edge_index[0] != data.edge_index[1]
        # print(mask)
        data.edge_index = data.edge_index[:, mask]
        if not data.edge_label is None:
            data.edge_label = data.edge_label[mask]
        if not data.edge_attr is None:
            data.edge_attr = data.edge_attr[mask]
        datalist.append(data)

    data, slices = dataset.collate(datalist)
    return SynGraphDataset('dataset/', 'ba_2motifs', data=data, slices=slices)


def get_datasets(name, pretrained=True, task=0, splits=None, batch_size=64, random_state=None, mutag_x=True, **kwargs):
    multi_label = False
    if name in ['bace', 'hiv', 'sider', 'bbbp']:

        if pretrained:
            return MoleculeDataset("./dataset/chem/zinc_standard_agent", dataset='zinc_standard_agent')
        else:
            task_transform = get_task(task)
            path = os.path.join(os.path.dirname(__file__), "./dataset/chem/%s" % name)
            dataset = MoleculeDataset(path, dataset=name, transform=task_transform, **kwargs)

            path = os.path.join(os.path.dirname(__file__), 'dataset/chem/%s/processed/smiles.csv' % name)
            smiles_list = pd.read_csv(path, header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(
                dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
            return train_dataset
    elif name in ['ppi']:
        return PPI('dataset/ppi', transform=get_task_rm_iso(task), **kwargs)
    elif name == 'ba_2motifs':
        dataset = SynGraphDataset('dataset/', 'ba_2motifs')
        dataset = remove_self_loop(dataset)

        split_idx = get_random_split_idx(dataset, splits, random_state=random_state)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]


    elif name == 'mutag':
        dataset = Mutag(root='dataset/mutag')
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]
    elif name == 'mnist':
        n_train_data, n_val_data = 20000, 5000
        train_val = MNIST75sp('dataset/mnist', mode='train')
        perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(random_state))
        train_val = train_val[perm_idx]

        train_set, valid_set = train_val[:n_train_data], train_val[-n_val_data:]
        test_set = MNIST75sp('dataset/mnist', mode='test')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set,
                                                                                 'test': test_set})
        print('[INFO] Using default splits!')
    elif 'spmotif' in name:
        b = float(name.split('_')[-1])
        train_set = SPMotif(root=f'dataset/{name}', b=b, mode='train')
        valid_set = SPMotif(root=f'dataset/{name}', b=b, mode='val')
        test_set = SPMotif(root=f'dataset/{name}', b=b, mode='test')
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set,
                                                                                 'test': test_set})
    else:
        raise NotImplementedError
    x_dim = test_set[0].x.shape[1]

    if (not test_set[0].edge_attr is None) and test_set[0].edge_attr.dim() == 1 and name == 'mutag':
        edge_attr_dim = 0
    else:
        edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]

    if isinstance(test_set, list):
        num_class = torch_geometric.data.Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True
    num_class = max(num_class, 2)
    print('[INFO] Calculating degree...')
    batched_train_set = torch_geometric.data.Batch.from_data_list(train_set)
    d = torch_geometric.utils.degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes,
                                     dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label}
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set


def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def NT_Xent(z1, z2, tau=0.5, norm=True):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    '''

    batch_size, _ = z1.size()
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2)
    # print(sim_matrix)

    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-6)
    # print(z1)
    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    # print(sim_matrix)
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-6)
    loss = loss + 1e-6
    # print(loss)
    loss = - torch.log(loss).mean()
    return loss


def get_expectation(masked_d_prime, positive=True):
    '''
    Args:
        masked_d_prime: Tensor of shape [n_graphs, n_graphs] for global_global,
                        tensor of shape [n_nodes, n_graphs] for local_global.
        positive (bool): Set True if the d_prime is masked for positive pairs,
                        set False for negative pairs.
    '''
    log_2 = np.log(2.)
    if positive:
        score = log_2 - F.softplus(-masked_d_prime)
    else:
        score = F.softplus(-masked_d_prime) + masked_d_prime - log_2
    return score


def JSE_global_global(z1, z2, norm=True):
    '''
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim].
    '''
    device = z1.device
    num_graphs = z1.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).to(device)
    neg_mask = torch.ones((num_graphs, num_graphs)).to(device)
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    batch_size, _ = z1.size()
    d_prime = torch.einsum('ik,jk->ij', z1, z2)
    # print(sim_matrix)

    if norm:
        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)
        d_prime = d_prime / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-6)
    # print(z1)
    # sim_matrix = torch.exp(sim_matrix / tau)

    # d_prime = torch.matmul(z1, z2.t())
    E_pos = get_expectation(d_prime * pos_mask, positive=True).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_expectation(d_prime * neg_mask, positive=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos


def get_encoder(feat_dim, dataset_name, edge_attr_dim=0, num_class=2):
    if dataset_name in ['ppi']:
        encoder = Encoder(feat_dim, hidden_dim=600,
                          n_layers=2, gnn='gcn', node_level=True, graph_level=False, bn=True)
    elif dataset_name in ['bace', 'hiv', 'sider', 'bbbp']:
        encoder = GNN(num_layer=5, emb_dim=600, JK='last', drop_ratio=0, gnn_type='gin')
    elif dataset_name in ['ba_2motifs', 'mutag', 'mnist', 'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9']:
        # encoder = SupervisedGIN(num_layer=5, emb_dim=600, JK='last', drop_ratio=0, gnn_type='gin')
        encoder = SupervisedGIN(2, 64, feat_dim, edge_attr_dim, False, num_class=num_class, dropout_p=0.0,
                                use_edge_attr=False,
                                atom_encoder=False, graph_level=True)
    else:
        raise NotImplementedError('dataset:{} is not supported'.format(dataset_name))
    return encoder


def get_downstream_path(dataset, task=0):
    if dataset in ['bace', 'hiv', 'sider', 'bbbp']:
        return "models/ckpts_model/downstream_{}_contextpred.pth".format(dataset)
    elif dataset in ['ppi']:
        return "models/ckpts_model/downstream_ppi{}_grace600.pth".format(task)


def is_node_explainer(args):
    return args.dataset in ['ppi']


def plot_subgraph(graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                  edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None, is_show=True):
    if edgelist is None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                    n_frm in nodelist and n_to in nodelist]
    pos = nx.kamada_kawai_layout(graph)
    pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=list(graph.nodes()),
                           node_color=colors,
                           node_size=300)

    nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

    nx.draw_networkx_edges(graph, pos=pos_nodelist,
                           edgelist=edgelist, width=6,
                           edge_color=subgraph_edge_color,
                           arrows=False)

    if labels is not None:
        nx.draw_networkx_labels(graph, pos, labels)

    plt.axis('off')

    if figname is not None:
        plt.savefig(figname)

    if is_show:
        plt.show()
    plt.close('all')


def fidelity(ori_probs: torch.Tensor, unimportant_probs: torch.Tensor):
    r"""
    Return the Fidelity+ value according to collected data.
    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity+ computation.
        unimportant_probs (torch.Tensor): It is a vector providing probabilities without important features
            for Fidelity+ computation.
    :rtype: float
    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    drop_probability = ori_probs - unimportant_probs

    return drop_probability.mean(dim=0).numpy(), drop_probability.std(dim=0).numpy()


def fidelity_inv(ori_probs: torch.Tensor, important_probs: torch.Tensor):
    r"""
    Return the Fidelity+ value according to collected data.
    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity+ computation.
        unimportant_probs (torch.Tensor): It is a vector providing probabilities without important features
            for Fidelity+ computation.
    :rtype: float
    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    drop_probability = ori_probs - important_probs

    return drop_probability.mean(dim=0).numpy(), drop_probability.std(dim=0).numpy()


def topk(
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm

def visualize_demo(graph, edge_imp,
              counter_edge_index=None, vis_ratio=0.2,
              save=False, layout=False, name=None, img_name=None):
    # print(graph.edge_index)
    adj = torch.zeros((graph.num_nodes, graph.num_nodes))
    edge_index = (graph.edge_index[0], graph.edge_index[1])
    adj[edge_index] = torch.arange(1, graph.num_edges+1).float()
    adj = torch.tril(adj, diagonal=-1)
    edge_imp_idx = adj[adj.nonzero(as_tuple=True)].long() - 1
    edge_imp = edge_imp[edge_imp_idx]
    if not graph.edge_attr is None:
        graph.edge_attr = graph.edge_attr[edge_imp_idx]
    # print(adj.nonzero())
    graph.edge_index = adj.nonzero().T

    # graph.edge_index = graph.edge_index[:, :int(graph.num_edges / 2)]
    # edge_imp = edge_imp[:int(graph.num_edges / 2)]

    topk = max(int(vis_ratio * graph.num_edges), 0)


    # print('topk:', topk)
    idx = np.argsort(-edge_imp)[:topk]
    # print(idx, img_name)
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(list(graph.edge_index.cpu().numpy().T))


    if not counter_edge_index == None:
        G.add_edges_from(list(counter_edge_index.cpu().numpy().T))

    edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
    edge_pos_mask[idx] = True
    vmax = sum(edge_pos_mask)
    node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
    node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
    node_pos_mask[node_pos_idx] = True
    node_neg_mask[node_neg_idx] = True

    if 'mutag' in name:
        # plt.figure(figsize=(8, 6), dpi=100)
        from rdkit.Chem.Draw import rdMolDraw2D
        # idx = [int(i / 2) for i in idx]
        x = graph.x.detach().cpu().tolist()
        edge_index = graph.edge_index.T.detach().cpu().tolist()
        edge_attr = graph.edge_attr.detach().cpu().tolist()
        mol = graph_to_mol(x, edge_index, edge_attr)
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)

        def add_atom_index(mol):
            atoms = mol.GetNumAtoms()
            for i in range(atoms):
                mol.GetAtomWithIdx(i).SetProp(
                    'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
            return mol

        hit_bonds = []
        for (u, v) in graph.edge_index.T[idx]:
            hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
            # print(u, v)
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol,  highlightBonds=None,
            )
        d.FinishDrawing()
        bindata = d.GetDrawingText()
        iobuf = io.BytesIO(bindata)
        image = Image.open(iobuf)
        # image.show()
        d.WriteDrawingText(f'image/{img_name}.png')


def visualize(graph, edge_imp,
              counter_edge_index=None, vis_ratio=0.2,
              save=False, layout=False, name=None, img_name=None):
    # print(graph.edge_index)
    adj = torch.zeros((graph.num_nodes, graph.num_nodes))
    edge_index = (graph.edge_index[0], graph.edge_index[1])
    adj[edge_index] = torch.arange(1, graph.num_edges+1).float()
    adj = torch.tril(adj, diagonal=-1)
    edge_imp_idx = adj[adj.nonzero(as_tuple=True)].long() - 1
    edge_imp = edge_imp[edge_imp_idx]
    if not graph.edge_attr is None:
        graph.edge_attr = graph.edge_attr[edge_imp_idx]
    # print(adj.nonzero())
    graph.edge_index = adj.nonzero().T

    # graph.edge_index = graph.edge_index[:, :int(graph.num_edges / 2)]
    # edge_imp = edge_imp[:int(graph.num_edges / 2)]

    topk = max(int(vis_ratio * graph.num_edges), 1)


    # print('topk:', topk)
    idx = np.argsort(-edge_imp)[:topk]
    # print(idx, img_name)
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(list(graph.edge_index.cpu().numpy().T))


    if not counter_edge_index == None:
        G.add_edges_from(list(counter_edge_index.cpu().numpy().T))

    edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
    edge_pos_mask[idx] = True
    vmax = sum(edge_pos_mask)
    node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
    node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
    node_pos_mask[node_pos_idx] = True
    node_neg_mask[node_neg_idx] = True

    if 'motif' in name:
        plt.figure(figsize=(2, 1.5))
        ax = plt.gca()
        # pos = graph.pos[0]
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
                               nodelist=node_pos_idx,
                               node_size=25,
                               node_color='red',
                               edgecolors='red',
                               alpha=1, cmap='winter',
                               )
        nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_neg_idx},
                               nodelist=node_neg_idx,
                               node_size=25,
                               node_color='orange',
                               alpha=1, cmap='winter',
                               edgecolors='whitesmoke',
                               )
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index.cpu().numpy().T),
                               edge_color='grey',
                               arrows=False,
                               width=1
                               )
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                               edge_color='black',
                               # np.ones(len(edge_imp[edge_pos_mask])),
                               width=1,
                               arrows=False
                               )

        plt.savefig(f'image/{img_name}.png', dpi=500)
        # plt.show()
        plt.close('all')

    if 'mutag' in name:
        # plt.figure(figsize=(8, 6), dpi=100)
        from rdkit.Chem.Draw import rdMolDraw2D
        # idx = [int(i / 2) for i in idx]
        x = graph.x.detach().cpu().tolist()
        edge_index = graph.edge_index.T.detach().cpu().tolist()
        edge_attr = graph.edge_attr.detach().cpu().tolist()
        mol = graph_to_mol(x, edge_index, edge_attr)
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        hit_at = np.unique(graph.edge_index[:, idx].detach().cpu().numpy()).tolist()

        def add_atom_index(mol):
            atoms = mol.GetNumAtoms()
            for i in range(atoms):
                mol.GetAtomWithIdx(i).SetProp(
                    'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
            return mol

        hit_bonds = []
        for (u, v) in graph.edge_index.T[idx]:
            hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
            # print(u, v)
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=hit_at, highlightBonds=hit_bonds,
            highlightAtomColors={i: (0, 1, 0) for i in hit_at},
            highlightBondColors={i: (0, 1, 0) for i in hit_bonds})
        d.FinishDrawing()
        bindata = d.GetDrawingText()
        iobuf = io.BytesIO(bindata)
        image = Image.open(iobuf)
        # image.show()
        d.WriteDrawingText(f'image/{img_name}.png')
