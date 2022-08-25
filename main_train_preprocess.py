import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from utils.transform_state import StateTransformer
import networkx as nx

from data_handling.network_dataset import NetworkDataset
import pickle

if __name__ == "__main__":

    data_path = 'data/training_data_medium_with_leak/network_'
    save_path = 'medium_network_transformer'

    dataset_params = {
        'data_path': data_path,
         'num_files': 10000,
         'transformer': None,
         'sensors': None
    }
    dataset = NetworkDataset(**dataset_params)

    data_loader_params = {
         'batch_size': 256,
         'shuffle': True,
         'num_workers': 12,
         'drop_last': True
    }
    dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)


    data_dict = nx.read_gpickle(data_path + str(0))
    transformer_state = StateTransformer(
            num_pipes=len(data_dict['flow_rate'].columns),
            num_nodes=len(data_dict['head'].columns)
    )

    for i, (state, _) in enumerate(dataloader):
        transformer_state.partial_fit(state)

    fileObject = save_path + '.pkl'
    with open(fileObject, "wb") as f:
        pickle.dump(transformer_state, f)



