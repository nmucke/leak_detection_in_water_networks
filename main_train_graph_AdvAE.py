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

@ray.remote
def compute_leak_location_posterior(
        leak_location,
        true_state,
        variational_inference,
    ):

    pars = torch.tensor([[leak_location]], dtype=torch.int32)

    posterior = variational_inference.compute_p_y_given_c(
            observations=observation_operator.get_observations(true_state),
            pars=pars,
            num_samples=20000,
    )

    return posterior



@ray.remote
def compute_reconstruction_error(
        leak_location,
        true_state,
        variational_inference,
        variational_minumum=True,
        HMC=False
    ):

    pars = torch.tensor([[leak_location]], dtype=torch.int64)

    if HMC:
        reconstruction, reconstruction_std, latent_state, latent_std = \
            variational_inference.compute_HMC_statistics(
                observations=observation_operator.get_observations(true_state),
                pars=torch.tensor([[i]], dtype=torch.int64),
                num_samples=1000,
        )
        reconstruction = reconstruction.unsqueeze(0)
        reconstruction_std = reconstruction_std.unsqueeze(0)
        latent_state = latent_state.unsqueeze(0)

    else:
        if variational_minumum:
            latent_state, reconstruction = variational_inference.compute_variational_minimum(
                    observations=observation_operator.get_observations(true_state),
                    pars=pars,
                    num_iterations=2500
            )
        else:
            latent_state, reconstruction = variational_inference.compute_encoder_decoder_reconstruction(
                    true_state=true_state.unsqueeze(0),
                    pars=pars,
            )

    log_posterior = variational_inference.log_posterior(
            observations=observation_operator.get_observations(true_state),
            predicted_observations=observation_operator.get_observations(reconstruction[0]),
            latent_state=latent_state
    )

    if HMC:
        return {'log_posterior': log_posterior,
                'reconstruction_std': reconstruction_std}
    else:
        return log_posterior


if __name__ == "__main__":

    seed_everything()

    HMC = False

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training AdvAE on {device}')


    train_with_leak = True
    with_sensors = False

    small_demand_variance = False

    train = False
    continue_training = False
    if not train:
        continue_training = True

    data_path = 'data/net_2/training_data_with_leak/network_'
    load_string = 'model_weights/graph_AE_leak_medium_network'
    save_string = 'model_weights/graph_AE_leak_medium_network'

    transformer_load_path = 'medium_network_transformer.pkl'

    num_pipe_sections = 119
    num_pipe_nodes = 97

    if with_sensors:

        flow_rate_sensor_ids = range(0, 119, 5)
        head_sensor_ids = range(0, 97, 5)

        load_string = load_string + "_sensors"
        save_string = save_string + "_sensors"
        sensors = {'flow_rate_sensors': flow_rate_sensor_ids,
                   'head_sensors': head_sensor_ids}
    else:
        sensors = None

    latent_dim = 8
    activation = nn.LeakyReLU()

    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)


    dataset_params = {
        'data_path': data_path,
         'file_ids': range(4000),
         'transformer': transformer,
         'sensors': sensors
    }
    val_dataset_params = {
        'data_path': data_path,
         'file_ids': range(4000, 5000),
         'transformer': transformer,
         'sensors': sensors
    }

    dataset = NetworkDataset(**dataset_params)
    val_dataset = NetworkDataset(**val_dataset_params)

    data_loader_params = {
         'batch_size': 64,
         'shuffle': True,
         'num_workers': 2,
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
    pdb.set_trace()

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
        size=(20,)
    )
    graph_encoder_params = {
        'latent_dim': latent_dim,
        'pivotal_nodes': pivotal_nodes,
        'num_attention_layers': 1,
        'attention_layer_params': attention_layer_params,
    }
    graph_decoder_params = {
        'latent_dim': latent_dim,
        'pivotal_nodes': pivotal_nodes,
        'num_attention_layers': 1,
        'attention_layer_params': attention_layer_params,
        'par_dim': par_dim,
        'pars_embedding_dim': latent_dim
    }

    critic_params = {
        'latent_dim': latent_dim,
        'hidden_neurons': [32, 32],
    }

    encoder = graph_autoencoder.GraphEncoder(**graph_encoder_params).to(device)
    decoder = graph_autoencoder.GraphDecoder(**graph_decoder_params).to(device)
    critic = graph_autoencoder.Critic(**critic_params).to(device)

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

    training_params = {'latent_dim': latent_dim,
                       'n_critic': 3,
                       'gamma': 10,
                       'n_epochs': 1000,
                       'save_string': save_string,
                       'with_adversarial_training': True,
                       'L1_regu': 1e-8,
                       'device': device}

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
