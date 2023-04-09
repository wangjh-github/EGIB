## Codes
This is the official implementation of "Empower Post-hoc Graph Explanations with Information Bottleneck: A Pre-training and Fine-tuning Perspective"

## Environment Requirements
You can run the following command to install the required package:
```shell
>> conda create -n wjh_pytorch python=3.7
>> conda install pytorch==1.12.1=cuda111py37he43340c_201 cudatoolkit=11.1 -c pytorch -c conda-forge
>> pip install torch-scatter==2.0.9
>> pip install torch-sparse==0.6.15
>> pip install torch-geometric==2.0.2
>> pip install networkx
```
## Usage
The datasets [MoleculeNet](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) and [PPI](https://data.dgl.ai/dataset/ppi.zip) can be downloaded manually. Please place the datasets in the `dataset` dictionary.
The trained target GNN models can be found [here](https://github.com/divelab/DIG/tree/main/dig/xgraph/TAGE/ckpts_model). The saved GNNs should be placed in the `models/ckpts_model`.
We provide a python script to run our method:
```shell
python demo.py --gpu 0 --dataset bace --task 0 --coff_ib 0.1 --coff_ir 0.1 --trick cat --logfile results.log --need_train
```