import pdb
import torch.nn as nn
import torch
from data_handling.network_dataset import NetworkDataset
from models.graph_convolutional_NN import GraphEncoder, GraphDecoder, Critic
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

    net = 2

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training AdvAE on {device}')

    convergence_plot = True
    convergence_latent = [18]

    latent_dim = 18

    critic_regu = 1e-3
    wasserstein = True

    save_model = True
    save_path = f'models_for_inference/AE_net{net}_latent' + str(latent_dim) + '_critic_regu' + str(critic_regu)
    if wasserstein:
        save_path += '_wasserstein'

    train_with_leak = True
    with_sensors = False

    train = True
    continue_training = False
    if not train:
        continue_training = True


    data_path = f'data/dynamic_net_{net}/training_data_with_leak/network_'
    load_string = f'model_weights/graph_AE_net{net}_latent' + str(latent_dim)
    save_string = f'model_weights/graph_AE_net{net}_latent' + str(latent_dim)

    transformer_load_path = f'net{net}_network_transformer.pkl'

    if net == 2:
        num_pipe_sections = 119
        num_pipe_nodes = 97
    elif net == 3:
        num_pipe_sections = 444
        num_pipe_nodes = 396

    latent_dim = 8
    activation = nn.LeakyReLU()

    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)


    dataset_params = {
        'data_path': data_path,
        'file_ids': range(8000),
        'transformer': transformer,
    }
    val_dataset_params = {
        'data_path': data_path,
        'file_ids': range(8000, 10000),
        'transformer': transformer,
    }

    dataset = NetworkDataset(**dataset_params)
    val_dataset = NetworkDataset(**val_dataset_params)

    data_loader_params = {
         'batch_size': 8,
         'shuffle': True,
         #'num_workers': 2,
         'drop_last': True
    }
    dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **data_loader_params)

    net, pars = dataset.__getitem__(0)
    state_dim = net.shape[0]
    par_dim = num_pipe_sections#pars.shape[0]

    data = nx.read_gpickle(data_path + '0')
    G = data['graph']
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)

    graph_encoder_params = {
        'graph_node_features': [2, 2, 2, 2],
        'graph_edge_features': [2, 2, 2, 2],
        'dense_features': [128, 64, 32],
        'latent_dim': latent_dim,
        'num_edges': num_edges,
        'num_nodes': num_nodes,
        'in_edge_features': 1,
        'in_node_features': 1,
        'graph': G,        
    }


    graph_decoder_params = {
        'graph_node_features': [2, 2, 2, 2],
        'graph_edge_features': [2, 2, 2, 2],
        'dense_features': [32, 64, 128],
        'latent_dim': latent_dim,
        'num_edges': num_edges,
        'num_nodes': num_nodes,
        'out_edge_features': 1,
        'out_node_features': 1,
        'graph': G,      
        'pars_embedding_dim': latent_dim//2,     
    }

    critic_params = {
        'latent_dim': latent_dim,
        'hidden_neurons': [64, 64],
        'wasserstein': wasserstein,
    }


    encoder = GraphEncoder(**graph_encoder_params).to(device)
    decoder = GraphDecoder(**graph_decoder_params).to(device)
    critic = Critic(**critic_params).to(device)

    recon_learning_rate = 1e-3
    recon_weight_decay = 1e-8

    critic_learning_rate = 1e-2

    encoder_optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=recon_learning_rate,
            weight_decay=recon_weight_decay
    )
    decoder_optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=recon_learning_rate,
            weight_decay=recon_weight_decay
    )
    critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=critic_learning_rate,
            weight_decay=recon_weight_decay
    )

    training_params = {
        'latent_dim': latent_dim,
        'n_critic': 3,
        'gamma': 10,
        'n_epochs': 1000,
        'save_string': save_string,
        'with_adversarial_training': True,
        'wasserstein': wasserstein,
        'L1_regu': None,#1e-8,
        'device': device
        }

    if continue_training:
        load_checkpoint(
                checkpoint_path=load_string,
                encoder=encoder,
                decoder=decoder,
                critic=critic,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                critic_optimizer=critic_optimizer,
        )

    if train:
        trainer = TrainAdversarialAE(
                encoder=encoder,
                decoder=decoder,
                critic=critic,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                critic_optimizer=critic_optimizer,
                **training_params,
        )

        recon_loss_list, critic_loss_list, enc_loss_list = trainer.train(
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                patience=50,
        )

    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')

    encoder.eval()
    decoder.eval()
