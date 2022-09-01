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

     attention_layer_params = {
        'in_features': 8, 
        'out_features': 8, 
        'hidden_features': 8, 
        'graph': G,
        'num_heads': 2,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
     }

     pivotal_nodes = torch.randint(
         low=0, 
         high=10,
         size=(10,)
     )
     graph_encoder_params = {
        'latent_dim': 10,
        'pivotal_nodes': pivotal_nodes,
        'num_attention_layers': 2,
        'attention_layer_params': attention_layer_params,
     }
     graph_decoder_params = {
        'latent_dim': 10,
        'pivotal_nodes': pivotal_nodes,
        'num_attention_layers': 2,
        'attention_layer_params': attention_layer_params,
     }

     graph_encoder = graph_autoencoder.GraphEncoder(**graph_encoder_params)
     graph_decoder = graph_autoencoder.GraphDecoder(**graph_decoder_params)


     lol = graph_encoder(X)
     lol2 = graph_decoder(lol)