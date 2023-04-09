# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

# from .conv_layers import GINConv, GINEConv
import torch

from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor

import torch
from torch import Tensor
from torch_geometric.nn import LEConv as BaseLEConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree
from torch_scatter import scatter

from typing import Callable, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from .MessagePassing import MessagePassing_v2

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

class BaseGINConv(MessagePassing_v2):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class BaseGINEConv(MessagePassing_v2):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j


class GINEConv(BaseGINEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        m = (x_j + edge_attr).relu()

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


class LEConv(BaseLEConv):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_atten: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        a = self.lin1(x[0])
        b = self.lin2(x[1])

        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight, edge_atten=edge_atten, size=None)

        return out + self.lin3(x[1])

    def message(self, a_j: Tensor, b_i: Tensor, edge_weight: OptTensor, edge_atten: OptTensor = None) -> Tensor:
        out = a_j - b_i
        m = out if edge_weight is None else out * edge_weight.view(-1, 1)

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/pna.py
class PNAConvSimple(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper
        .. math::
            \bigoplus = \underbrace{\begin{bmatrix}I \\ S(D, \alpha=1) \\
            S(D, \alpha=-1) \end{bmatrix} }_{\text{scalers}}
            \otimes \underbrace{\begin{bmatrix} \mu \\ \sigma \\ \max \\ \min
            \end{bmatrix}}_{\text{aggregators}},
        in:
        .. math::
            X_i^{(t+1)} = U \left( \underset{(j,i) \in E}{\bigoplus}
            M \left(X_j^{(t)} \right) \right)
        where :math:`U` denote the MLP referred to with posttrans.
        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            aggregators (list of str): Set of aggregation function identifiers,
                namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
                :obj:`"var"` and :obj:`"std"`.
            scalers: (list of str): Set of scaling function identifiers, namely
                :obj:`"identity"`, :obj:`"amplification"`,
                :obj:`"attenuation"`, :obj:`"linear"` and
                :obj:`"inverse_linear"`.
            deg (Tensor): Histogram of in-degrees of nodes in the training set,
                used by scalers to normalize.
            post_layers (int, optional): Number of transformation layers after
                aggregation (default: :obj:`1`).
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 post_layers: int = 1, **kwargs):

        super(PNAConvSimple, self).__init__(aggr=None, node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        self.F_in = in_channels
        self.F_out = self.out_channels

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        in_channels = (len(aggregators) * len(scalers)) * self.F_in
        modules = [Linear(in_channels, self.F_out)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_out, self.F_out)]
        self.post_nn = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.post_nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_atten=None) -> Tensor:

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None, edge_atten=edge_atten)
        return self.post_nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr=None, edge_atten=None) -> Tensor:
        if edge_attr is not None:
            m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            m = torch.cat([x_i, x_j], dim=-1)

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')
        raise NotImplementedError


def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='min')


def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='max')


def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'max': aggregate_max,
    'var': aggregate_var,
    'std': aggregate_std,
}


def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['lin'] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}


class Encoder(nn.Module):
    def __init__(self, num_layer, emb_dim, x_dim, edge_attr_dim, multi_label, dropout_p=0.0, use_edge_attr=False,
                 atom_encoder=False):
        super().__init__()
        self.n_layers = num_layer
        hidden_size = emb_dim
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = dropout_p
        self.use_edge_attr = use_edge_attr

        # if model_config.get('atom_encoder', False):
        #     self.node_encoder = AtomEncoder(emb_dim=hidden_size)
        #     if edge_attr_dim != 0 and self.use_edge_attr:
        #         self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        # else:
        self.node_encoder = Linear(x_dim, hidden_size)
        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(Encoder.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(Encoder.MLP(hidden_size, hidden_size)))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def forward(self, data, emb=False, edge_atten=None):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if emb:
            return self.pool(x, batch), x
        else:
            return self.pool(x, batch)

class MLPExplainer(torch.nn.Module):
    ''' Downstream MLP explainer based on gradient of output w.r.t. input embedding.
    Args:
        mlp_model: :obj:`torch.nn.Module` The downstream model to be explained.
        device: Torch CUDA device.
    '''

    def __init__(self, mlp_model, device):
        super(MLPExplainer, self).__init__()
        self.model = mlp_model.to(device)
        self.device = device

    def forward(self, embeds, return_logits=False):
        '''Returns probability by forward propagation or gradients by backward propagation
        based on the mode specified.
        '''
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

class SupervisedGIN(nn.Module):
    def __init__(self, num_layer, emb_dim, x_dim, edge_attr_dim, multi_label, num_class=1, dropout_p=0.0,
                 use_edge_attr=True,
                 atom_encoder=False, graph_level=True):
        super().__init__()

        self.encoder = Encoder(num_layer, emb_dim, x_dim, edge_attr_dim, multi_label, dropout_p=dropout_p,
                               use_edge_attr=use_edge_attr,
                               atom_encoder=atom_encoder)

        self.mlp = nn.Sequential(nn.Linear(emb_dim, 1 if num_class == 2 and not multi_label else num_class))
        self.graph_level = graph_level

    def forward(self, data, edge_atten=None, return_logits=False):
        embs = self.encoder(data, emb=False, edge_atten=edge_atten)
        logits = self.mlp(embs)
        if return_logits:
            return logits
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            probs = torch.cat([1 - probs, probs], 1)
        else:
            probs = F.softmax(logits, dim=-1)

        return probs
    def get_graph_embed(self, data):
        if self.graph_level:
            graph_embs = self.encoder(data, emb=False)
        else:
            node_embeds = self.encoder(data)
            graph_embs = global_add_pool(node_embeds, data.batch)
        return graph_embs

    def get_embed(self, data):
        if self.graph_level:
            _, embs = self.encoder(data, emb=True)
        else:
            embs = self.encoder(data)
        return embs
