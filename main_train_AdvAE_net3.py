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
from tqdm import tqdm
import h5py
import pandas as pd
import geopandas

if __name__ == "__main__":

    '''
    with open('dict.pickle', 'rb') as pickle_file:
        lol = pickle.load(pickle_file)

    pdb.set_trace()


    df = pd.read_parquet(f'data_for_prerna/WDN2_training_data_{0}.parquet')
    for i in range(1, 30):
        df_ = pd.read_parquet(f'data_for_prerna/WDN2_training_data_{i}.parquet')
        df = pd.concat([df, df_], axis=0, ignore_index=True)

    df.to_parquet(f'data_for_prerna/WDN1_training_data.parquet', index=False)
    pdb.set_trace()
    '''

    '''
    for i in range(0, 10000):
        data_path = 'data/dynamic_net_2/training_data_with_leak/network_'
        data_dict = nx.read_gpickle(data_path + str(i))
        data_dict['flow_rate'] = data_dict['flow_rate'].iloc[0:24, :]
        data_dict['head'] = data_dict['head'].iloc[0:24, :]
        data_dict['demand'] = data_dict['demand'].iloc[0:24, :]
        nx.write_gpickle(data_dict, f'{data_path}{i}')
        if i % 100 == 0:
            print(i)
    
    pdb.set_trace()
    '''

    '''
    data_path = 'data/dynamic_net_3/training_data_with_leak/network_0'
    data_dict = nx.read_gpickle(data_path)
    
    flowrate_columns = data_dict['flow_rate'].columns
    head_columns = data_dict['head'].columns
    head_columns = [head_columns[i]+'_head' for i in range(len(head_columns))]
    demand_columns = data_dict['demand'].columns
    demand_columns = [demand_columns[i]+'_demand' for i in range(len(demand_columns))]

    for i in range(0, 1):
        df_to_save = pd.DataFrame()
        for j in range(i*1000, (i+1)*1000):
            data_path = 'data/dynamic_net_3/test_data_with_leak/network_'
            data_dict = nx.read_gpickle(data_path + str(j))

            df = pd.DataFrame()
            df[flowrate_columns] = data_dict['flow_rate']
            df[head_columns] = data_dict['head']
            df[demand_columns] = data_dict['demand']
            df = df.reset_index(drop=True)

            df_time = pd.DataFrame((data_dict['head'].index.to_numpy()/60/60 % 24).reshape(-1, 1), columns=['time']).reset_index(drop=True)
            df_leak_location = pd.DataFrame([data_dict['leak']['pipe']] * data_dict['head'].shape[0], columns=['leak_location']).reset_index(drop=True)
            df_leak_area = pd.DataFrame((data_dict['leak']['area'].repeat(data_dict['head'].shape[0])).reshape(-1, 1), columns=['leak_area']).reset_index(drop=True)
            df_leak_demand = pd.DataFrame((data_dict['leak']['demand'].repeat(data_dict['head'].shape[0])).reshape(-1, 1), columns=['leak_demand']).reset_index(drop=True)
            df = pd.concat([df, df_time, df_leak_location, df_leak_area, df_leak_demand], axis=1)
            
            df_to_save = df_to_save.append(df, ignore_index=True)

            if j % 100 == 0:
                print(j)

        print(i)
        df_to_save.to_parquet(f'data_for_prerna/WDN2_test_data_{i}.parquet', index=False)

    pdb.set_trace()
    '''
    
    
    seed_everything()

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training AdvAE on {device}')

    convergence_plot = True
    convergence_latent = [12]
    with_time = True

    latent_dim = 12

    critic_regu = 1e-3
    wasserstein = True

    save_model = True
    save_path = 'models_for_inference/AE_net3_latent' + str(latent_dim) + '_critic_regu' + str(critic_regu)
    if not with_time:
        save_path += '_no_time'

    if wasserstein:
        save_path += '_wasserstein'

    train_with_leak = True
    with_sensors = False

    train = True
    continue_training = False
    if not train:
        continue_training = True


    data_path = 'data/dynamic_net_3/training_data_with_leak/network_'
    load_string = 'model_weights/AE_net3_latent' + str(latent_dim)
    save_string = 'model_weights/AE_net3_latent' + str(latent_dim)


    transformer_load_path = 'net3_network_transformer.pkl'

    num_pipe_sections = 444
    num_pipe_nodes = 396

    if with_sensors:

        flow_rate_sensor_ids = range(0, num_pipe_sections, 5)
        head_sensor_ids = range(0, num_pipe_nodes, 5)

        load_string = load_string + "_sensors"
        save_string = save_string + "_sensors"
        sensors = {
            'flow_rate_sensors': flow_rate_sensor_ids,
            'head_sensors': head_sensor_ids
            }
    else:
        sensors = None

    activation = nn.LeakyReLU()

    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)


    dataset_params = {
        'data_path': data_path,
        'file_ids': range(0, 27000),
        'transformer': transformer,
        'sensors': sensors,
        'with_time': with_time,
    }
    val_dataset_params = {
        'data_path': data_path,
        'file_ids': range(27000, 30000),
        'transformer': transformer,
        'sensors': sensors,
        'with_time': with_time,
    }
    dataset = NetworkDataset(**dataset_params)
    val_dataset = NetworkDataset(**val_dataset_params)

    '''
    num_train = 1000
    flow_rate_save = torch.zeros((num_train, 24, 444))
    head_save = torch.zeros((num_train, 24, 396))
    pars_save = torch.zeros((num_train, 1))
    for i in range(0, num_train):
        data_path = 'data/dynamic_net_3/test_data_with_leak/network_'
        data_dict = nx.read_gpickle(data_path + str(i))

        flow_rate = torch.tensor(data_dict['flow_rate'].values, dtype=torch.float32)
        head = torch.tensor(data_dict['head'].values, dtype=torch.float32)
        pars = torch.tensor(
            [dataset.pipe_label_to_index_dict[data_dict['leak']['pipe']]],
            dtype=torch.int32
        )

        flow_rate_save[i] = flow_rate
        head_save[i] = head
        pars_save[i] = pars

        if i % 100 == 0:
            print(i)
    torch.save(flow_rate_save, 'test_flow_rate.pt')
    torch.save(head_save, 'test_head.pt')
    torch.save(pars_save, 'test_pars.pt')
    pdb.set_trace()
    '''

    data_loader_params = {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 2,
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
        'wasserstein': wasserstein,
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
        'wasserstein': wasserstein,
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
                patience=15,
        )

    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')

    encoder.eval()
    decoder.eval()

    if save_model:
        torch.save(encoder, save_path + '_encoder.pt')
        torch.save(decoder, save_path + '_decoder.pt')


    ##### Convergence #####
    data_path = 'data/dynamic_net_3/training_data_with_leak/network_'

    dataset_params = {
        'data_path': data_path,
        'file_ids': range(500),
        'transformer': transformer,
        'sensors': sensors
    }
    dataset = NetworkDataset(**dataset_params)

    data_loader_params = {
         'batch_size': 2,
         'shuffle': True,
         #'num_workers': 4,
         'drop_last': True
    }
    dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    if convergence_plot:
        true_moments = np.array([0, 1, 0, 3])
        for wasserstein in [True]:
            for critic_regu in [1e-3]:
                latent_recon_loss = []
                latent_moment_error = []
                for latent_dim in convergence_latent:

                    load_string = 'models_for_inference/AE_net3_latent' + str(latent_dim) + '_critic_regu' + str(critic_regu)+ '_no_time'
                    if wasserstein:
                        load_string = load_string + '_wasserstein'

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
    for i in range(num_plots):
        plt.subplot(3, 3, i+1)
        plt.hist(latent_states[:, i], bins=30)

        mean = np.mean(latent_states[:, i])
        std = np.std(latent_states[:, i])
        skewness = skew(latent_states[:, i])
        kurt = kurtosis(latent_states[:, i])

        plt.title(
            f'$\mu$={mean:.2f}, $\sigma$={std:.2f}, skew={skewness:.2f}, kurt={kurt:.2f}',
            fontsize=8
            )
    plt.show()




