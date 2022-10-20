from cProfile import label
import pdb
import torch.nn as nn
import torch
from data_handling.network_dataset import NetworkDataset
import models.autoencoder as autoencoder_models
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
from scipy.special import rel_entr
from scipy.stats import skew
from scipy.stats import kurtosis

if __name__ == "__main__":

    seed_everything()

    HMC = False

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training AdvAE on {device}')

    convergence_plot = True
    convergence_latent = [16]

    latent_dim = 16

    critic_regu = 1e0
    probability_cost = 'MMD'
    with_time = True

    save_model = True
    save_path = f'models_for_inference/AE_net3_latent{latent_dim}_critic_regu{critic_regu}_{probability_cost}'

    if not with_time:
        save_path += '_no_time'

    train_with_leak = True
    with_sensors = False

    train = False
    continue_training = False
    if not train:
        continue_training = True

    data_path = 'data/dynamic_net_3/training_data_with_leak/network_'
    load_string = f'model_weights/AE_net3_latent{latent_dim}_{probability_cost}'
    save_string = f'model_weights/AE_net3_latent{latent_dim}_{probability_cost}'

    if not with_time:
        load_string += '_no_time'
        save_string += '_no_time'

    transformer_load_path = 'net3_network_transformer.pkl'

    num_pipe_sections = 444
    num_pipe_nodes = 396

    activation = nn.LeakyReLU()

    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)


    dataset_params = {
        'data_path': data_path,
        'file_ids': range(27000),
        'transformer': transformer,
        'sensors': None,
        'with_time': with_time 
    }
    val_dataset_params = {
        'data_path': data_path,
        'file_ids': range(27000, 30000),
        'transformer': transformer,
        'sensors': None,
        'with_time': with_time 
    }
    dataset = NetworkDataset(**dataset_params)
    val_dataset = NetworkDataset(**val_dataset_params)

    data_loader_params = {
         'batch_size': 16,
         'shuffle': True,
         #'num_workers': 4,
         'drop_last': True
    }
    dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **data_loader_params)

    net, pars = dataset.__getitem__(0)
    state_dim = net.shape[1]
    if with_time:
        par_dim = [num_pipe_sections, 24]#(num_pipe_sections, 24)#pars.shape[0]
    else:
        par_dim = [num_pipe_sections]

    encoder_params = {
        'state_dim': state_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [256, 128, 64],
    }

    decoder_params = {
        'state_dim': state_dim,
        'latent_dim': latent_dim,
        'hidden_neurons': [64, 128, 256],
        'pars_dim': par_dim,
        'pars_embedding_dim': latent_dim,
    }

    critic_params = {
        'latent_dim': latent_dim,
        'hidden_neurons': [64, 64, 64],
        'wasserstein': True,
    }

    encoder = autoencoder_models.Encoder(**encoder_params).to(device)
    decoder = autoencoder_models.Decoder(**decoder_params).to(device)
    critic = autoencoder_models.Critic(**critic_params).to(device)

    recon_learning_rate = 1e-2
    recon_weight_decay = 1e-8

    critic_learning_rate = 1e-2

    encoder_optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=recon_learning_rate,
            #weight_decay=recon_weight_decay
    )
    decoder_optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=recon_learning_rate,
            #weight_decay=recon_weight_decay
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
        'L1_regu': None,#1e-10,
        'critic_regu': critic_regu,
        'probability_cost': probability_cost,
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
                patience=25,
        )

    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')

    encoder.eval()
    decoder.eval()

    if save_model:
        torch.save(encoder, save_path + '_encoder.pt')
        torch.save(decoder, save_path + '_decoder.pt')


    ##### Convergence #####
    data_path = 'data/dynamic_net_3/test_data_with_leak/network_'

    dataset_params = {
        'data_path': data_path,
        'file_ids': range(1000),
        'transformer': transformer,
        'sensors': None
    }
    dataset = NetworkDataset(**dataset_params)

    data_loader_params = {
         'batch_size': 16,
         'shuffle': True,
         #'num_workers': 4,
         'drop_last': True
    }
    dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    if convergence_plot:
        true_moments = np.array([0, 1, 0, 3])
        for wasserstein in [True]:
            for critic_regu in [critic_regu]:
                latent_recon_loss = []
                latent_moment_error = []
                for latent_dim in convergence_latent:

                    load_string = f'models_for_inference/AE_net3_latent{latent_dim}_critic_regu{critic_regu}_{probability_cost}'

                    if not with_time:
                        load_string += '_no_time'

                    recon_loss = []
                    latent_states = []

                    encoder = torch.load(load_string + '_encoder.pt')
                    encoder.eval()

                    decoder = torch.load(load_string + '_decoder.pt')
                    decoder.eval()

                    for bidx, (real_state, real_pars) in enumerate(dataloader):

                        batch_size = real_state.size(0)
                        num_steps = real_state.size(1)
                        num_states = real_state.size(2)
                        num_pars = real_pars.size(2) 

                        real_state = real_state.reshape(
                            batch_size*num_steps,
                            num_states
                        )
                        real_pars = real_pars.reshape(
                            batch_size*num_steps,
                            num_pars
                        )

                        real_state = real_state.to(device)
                        real_pars = real_pars.to(device)

                        latent = encoder(real_state)
                        recon_state = decoder(latent, real_pars)

                        latent_states.append(latent.detach().numpy())

                        recon_loss.append(
                            nn.MSELoss()(recon_state, real_state).detach().numpy()
                        )
                    
                    latent_recon_loss.append(np.mean(recon_loss))

                    latent_states = np.concatenate(latent_states, axis=0)

                    mean = np.mean(latent_states.flatten())
                    std = np.std(latent_states.flatten())
                    skewness = skew(latent_states.flatten())
                    kurt = kurtosis(latent_states.flatten())

                    moments = np.array([mean, std, skewness, kurt])
                    latent_moment_error.append(np.linalg.norm(moments - true_moments))

                plt.figure(figsize=(10,5))
                plt.subplot(1, 2, 1)
                plt.semilogy(convergence_latent, latent_recon_loss, label='Reconstruction loss',
                        linestyle='-', linewidth=2, color='tab:blue')
                plt.semilogy(convergence_latent, latent_recon_loss, '.', label='Reconstruction loss',
                        markersize=20, color='tab:blue')
                plt.xlabel('Latent dimension')
                plt.ylabel('MSE loss')
                plt.grid()

                plt.subplot(1, 2, 2)
                plt.semilogy(convergence_latent, latent_moment_error, label='Moment error',
                        linestyle='-', linewidth=2, color='tab:blue')
                plt.semilogy(convergence_latent, latent_moment_error, '.', label='Moment error',
                        markersize=20, color='tab:blue')
                plt.xlabel('Latent dimension')
                plt.ylabel('Moment error')
                plt.grid()

                figure_save_path = 'figures/critic' + str(critic_regu)
                if wasserstein:
                    figure_save_path = figure_save_path + '_wasserstein'
                plt.savefig(figure_save_path + '.png', dpi=300)
                plt.show()

                print(f'Latent dimension: {latent_dim}, Wasserstein: {wasserstein}, Critic regu: {critic_regu}')

    ##### Compute KL divergence #####

    plt.figure(figsize=(10, 10))
    num_plots = np.min([latent_dim, 9])

    normal_distribution = 1/np.sqrt(2*np.pi)*np.exp(-0.5*np.linspace(-3, 3, 1000)**2)
    for i in range(num_plots):
        plt.subplot(3, 3, i+1)
        plt.hist(latent_states[:, i], bins=50, density=True)
        plt.plot(np.linspace(-3, 3, 1000), normal_distribution, color='tab:red')


        mean = np.mean(latent_states[:, i])
        std = np.std(latent_states[:, i])
        skewness = skew(latent_states[:, i])
        kurt = kurtosis(latent_states[:, i])

        plt.title(
            f'$\mu$={mean:.2f}, $\sigma$={std:.2f}, skew={skewness:.2f}, kurt={kurt:.2f}',
            fontsize=8
            )
    plt.show()




