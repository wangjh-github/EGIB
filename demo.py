import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import pandas as pd
torch.autograd.set_detect_anomaly(True)
from torch_geometric.data import DataLoader
from EGIB import EGIBExplainer, MLP_Classifier
from tqdm import tqdm
from layers.downstream import MLP
from utils.utils import get_datasets, get_encoder, is_node_explainer, get_downstream_path, fidelity
import logging
import args_setting
import numpy as np
import torch_geometric

args = args_setting.setting_args()

logging.basicConfig(filename=args.logfile, level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p')
logging.info('#' * 100)
logging.info(args)

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else torch.device("cpu"))
dataset_name = args.dataset

dataset = get_datasets(dataset_name, True, args.task)
loader = DataLoader(dataset, args.batchsize, shuffle=True)

encoder = get_encoder(dataset[0].x.shape[1], args.dataset)

encoder.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu'))



feat_dim = dataset[0].x.shape[1]
enc_explainer = EGIBExplainer(encoder, embed_dim=600, feat_dim=dataset[0].x.shape[1], device=device,
                               explain_graph=not is_node_explainer(args),
                               grad_scale=args.grad_scale, coff_ib=args.coff_ib, coff_size=args.coff_size,
                               coff_ent=args.coff_ent, coff_refine=args.coff_refine, coff_ir=args.coff_ir,
                               trick=args.trick)

if args.need_train:
    if is_node_explainer(args):
        enc_explainer.train_explainer_node(loader, batch_size=4, lr=args.lr, epochs=args.epochs, ratio=args.ratio)
    else:
        enc_explainer.train_explainer_graph(loader, lr=args.lr, epochs=args.epochs, ratio=args.ratio)
    torch.save(enc_explainer.state_dict(), args.explainer_path)


def get_results(ratio_list, inv_ratio_list, pos=True):
    enc_explainer = EGIBExplainer(encoder, embed_dim=600, feat_dim=feat_dim, device=device,
                                   explain_graph=not is_node_explainer(args),
                                   grad_scale=args.grad_scale,coff_ib=args.coff_ib, coff_size=args.coff_size,
                                   coff_ent=args.coff_ent, coff_refine=args.coff_refine, coff_ir=args.coff_ir)
    state_dict = torch.load(args.explainer_path)
    enc_explainer.load_state_dict(state_dict)

    dataset = get_datasets(args.dataset, False, args.task)

    mlp_model = MLP(num_layer=2, emb_dim=600, hidden_dim=600, out_dim=args.downstream_out_dim)
    path = get_downstream_path(args.dataset, task=args.task)
    print(path)
    mlp_model.load_state_dict(torch.load(path, map_location='cpu'))
    mlp_explainer = MLP_Classifier(mlp_model, device)
    enc_explainer.mlp_classifier = mlp_explainer

    datalist = []
    related_preds_collector = {'factual_masked_probs': [], 'factual_sparsity_scores': [],
                               'counterfactual_masked_probs': [],
                               'counterfactual_sparsity_scores': [], 'origin': []}
    if is_node_explainer(args):
        loader = DataLoader(dataset, 1, shuffle=False, num_workers=1)
    else:
        for i in range(len(dataset)):
            if pos == (dataset[i].y >= 0) and dataset[i].edge_index.shape[1] > 1:
                datalist.append(dataset[i])

        loader = DataLoader(datalist, batch_size=256, shuffle=False)
    if args.refine:
        if is_node_explainer(args):
            enc_explainer.train_explainer_node(loader, refine=True, batch_size=4, lr=args.lr, epochs=1,
                                               ratio=args.ratio)
        else:
            loader = DataLoader(datalist, batch_size=64, shuffle=True)
            enc_explainer.train_explainer_graph(loader, refine=True, lr=args.lr, epochs=10, ratio=args.ratio)
            loader = DataLoader(datalist, batch_size=256, shuffle=False)
    for i, data in enumerate(loader):

        if is_node_explainer(args):
            edge_index = torch_geometric.utils.to_undirected(data.edge_index)
            edge_index_selfloop, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)
            degree = torch_geometric.utils.degree(edge_index_selfloop[0], data.num_nodes)
            deg_inv_sqrt = degree.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            data.deg_inv_sqrt = deg_inv_sqrt
            edge_index_selfloop = edge_index_selfloop.to(device)

            data.to(device)
            node_loader = DataLoader(torch.where(data.y)[0].tolist(), 4, shuffle=False)
            for j, node_idx in enumerate(tqdm(node_loader)):

                if data.edge_index.shape[1] <= 0:
                    continue

                walks, masks, related_preds = \
                    enc_explainer(data, refine=args.refine,
                                  mask_ratio_list=ratio_list,
                                  mask_inv_ratio_list=inv_ratio_list,
                                  mask_mode='split', ratio=args.ratio, node_idx=node_idx,
                                  edge_index_selfloop=edge_index_selfloop)

                related_preds_collector['origin'].extend(related_preds[0]['origin'])
                related_preds_collector['counterfactual_masked_probs'].extend(
                    related_preds[0]['counterfactual_masked_probs'])
                related_preds_collector['counterfactual_sparsity_scores'].extend(
                    related_preds[0]['counterfactual_sparsity_scores'])

        else:


            print(f'explain graph {i}...', end='\r')

            walks, masks, related_preds = \
                enc_explainer(data.to(device), refine=args.refine,
                              mask_ratio_list=ratio_list,
                              mask_inv_ratio_list=inv_ratio_list,
                              mask_mode='split', ratio=args.ratio)

            related_preds_collector['origin'].extend(related_preds[0]['origin'])
            related_preds_collector['counterfactual_masked_probs'].extend(
                related_preds[0]['counterfactual_masked_probs'])
            related_preds_collector['counterfactual_sparsity_scores'].extend(
                related_preds[0]['counterfactual_sparsity_scores'])
    origin_preds = torch.tensor(related_preds_collector['origin']).unsqueeze(1)
    counterfactual_masked_prob = torch.tensor(related_preds_collector['counterfactual_masked_probs'])

    fids, fid_stds = fidelity(origin_preds, counterfactual_masked_prob)
    spas, spa_stds = np.mean(related_preds_collector['counterfactual_sparsity_scores'], axis=0), np.std(
        related_preds_collector['counterfactual_sparsity_scores'], axis=0)

    df = pd.DataFrame(
        columns=['explainer', 'dataset', 'Fidelity', 'Fidelity-std', 'Sparsity', 'Sparsity-std'])
    for fid, fid_std, spa, spa_std in zip(fids, fid_stds, spas, spa_stds):
        res = {
            'explainer': "TAXIB",
            'dataset': '{}_{}'.format(args.dataset, args.task),
            'Fidelity': fid,
            'Fidelity-std': fid_std,
            'Sparsity': spa,
            'Sparsity-std': spa_std,
        }
        df = df.append(res, ignore_index=True)

    print(df)


if is_node_explainer(args):
    args.task = 0
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])

    args.task = 1
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])

    args.task = 2
    get_results(ratio_list=[0.02, 0.04, 0.06, 0.08, 0.10], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])

    args.task = 3
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])

    args.task = 4
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])
else:
    args.dataset = 'bace'
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])

    args.dataset = 'bbbp'
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5], pos=False)

    args.dataset = 'sider'
    get_results(ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])
    args.dataset = 'hiv'
    get_results(ratio_list=[0.02, 0.04, 0.06, 0.08, 0.10], inv_ratio_list=[0.1, 0.2, 0.3, 0.4, 0.5])
