import argparse
import json


def setting_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0', help='gpu device')
    parser.add_argument('--need_train', action='store_true', help='whether the explainer needs to be trained')
    parser.set_defaults(need_train=False)
    parser.add_argument('--refine', action='store_true', help='whether the explainer needs to be refined')
    parser.set_defaults(refine=False)
    parser.add_argument('--dataset', type=str, default='ppi', help='dataset name')
    parser.add_argument('--task', type=int, default=0, help='task idx for ppi')
    parser.add_argument('--coff_ib', type=float, default=0.001, help='coefficient for IB term')
    parser.add_argument('--coff_refine', type=float, default=0.1, help='coefficient for refine')
    parser.add_argument('--logfile', type=str, default='results', help='the file to store execution log')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--coff_ir', type=float, default=0.001, help='coefficient for invariance regularization')
    parser.add_argument('--trick', type=str, default='cat',
                        help='assumptions on categorical distribution (cat) or bernoulli distribution (ber)')
    args = parser.parse_args()

    args = get_args_from_json('./hyper_parameters.json', args)
    return args


def get_args_from_json(json_file_path, args):
    with open(json_file_path, 'r') as f:
        json_file = json.load(f)
        args = vars(args)
        print(json_file.keys())
        if args['dataset'] in json_file.keys():
            json_file_dataset = json_file[args['dataset']]
        else:
            json_file_dataset = json_file['others']
        for key, item in json_file_dataset.items():
            if key not in args.keys():
                args[key] = item
        if args['trick'] == 'cat':
            args['explainer_path'] = args['explainer_path'].replace('.pt', '_ib_{:.4f}_lr_{:.4f}.pt'.format(
                args['coff_ib'], args['coff_ir']))
        else:
            args['explainer_path'] = args['explainer_path'].replace('.pt',
                                                                    '_{}_ib_{:.4f}_ir_{:.4f}.pt'.format(args['trick'],
                                                                                                         args[
                                                                                                             'coff_ib'],
                                                                                                         args[
                                                                                                             'coff_ir']))
        print(args['explainer_path'])
    return argparse.Namespace(**args)
