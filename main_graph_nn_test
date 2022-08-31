import pdb
import torch.nn as nn
import torch
from data_handling.network_dataset import NetworkDataset
import models.graph_autoencoder as graph_autoencoder
from utils.load_checkpoint import load_checkpoint
#from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
from utils.label_to_idx import LabelToIndex
from training.training_adv_AE import TrainAdversarialAE
from inference.variational_inference import VariationalInference
from inference.observation import ObservationOperator
import networkx as nx
torch.set_default_dtype(torch.float32)
import pickle
import os
import wntr
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ray
import time
from torchquad import MonteCarlo, set_up_backend

if __name__ == "__main__":

     seed_everything()

     data_path = 'data/net_2/training_data_with_leak/network_'

     dataset_params = {
          'data_path': data_path,
          'file_ids': range(3),
          'transformer': None,
          'sensors': None
     }

     dataset = NetworkDataset(**dataset_params)

     data_loader_params = {
          'batch_size': 3,
          'shuffle': True,
          'num_workers': 1,
          'drop_last': True
     }
     dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)

     i, (X, pars) = next(enumerate(dataloader))

     data = nx.read_gpickle(data_path + '0')
     G = data['graph']
     num_nodes = len(G.nodes)
     num_edges = len(G.edges)

     adj_matrix = nx.adjacency_matrix(G)
     adj_matrix = adj_matrix.toarray()



     graph_NN_params = {
        'in_node_features': 1, 
        'out_node_features': 10, 
        'in_edge_features': 1,
        'out_edge_features': 10,
        'adj_matrix': adj_matrix,
        'num_heads': 3,
        'num_nodes': num_nodes,
        'num_egdes': num_edges,
     }

     graph_NN = graph_autoencoder.MultiHeadGraphAttention(**graph_NN_params)

     lol = graph_NN(X)
     pdb.set_trace()
