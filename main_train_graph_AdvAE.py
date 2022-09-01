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
            num_samples=100000,
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

    train = True
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

    latent_dim = 16
    activation = nn.LeakyReLU()

    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)


    dataset_params = {
        'data_path': data_path,
         'file_ids': range(300),
         'transformer': transformer,
         'sensors': sensors
    }
    val_dataset_params = {
        'data_path': data_path,
         'file_ids': range(300, 400),
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

    attention_layer_params = {
        'in_features': 4, 
        'out_features': 4, 
        'hidden_features': 4, 
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
        'num_attention_layers': 2,
        'attention_layer_params': attention_layer_params,
    }
    graph_decoder_params = {
        'latent_dim': latent_dim,
        'pivotal_nodes': pivotal_nodes,
        'num_attention_layers': 8,
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
    recon_weight_decay = 1e-10

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
                       'L1_regu': 1e-5,
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

    '''
    leak_location_pred = []
    leak_location_true = []
    for case in range(1000):
        data_dict = nx.read_gpickle(data_path + str(case))

        flow_rate = np.asarray(data_dict['flow_rate'].values)[0]
        head = np.asarray(data_dict['head'].values)[0]
        head_dim = head.shape[0]
        flow_rate_dim = flow_rate.shape[0]

        true_net, true_pars = dataset.__getitem__(case)
        #true_net = true_net.detach().cpu().numpy()
        true_pars = true_pars.detach().cpu().numpy()

        leak_location = np.argwhere(true_pars == 1)[0][0]
        leak_location_true.append(leak_location)

        error = []
        for i in range(par_dim):
            pars = torch.zeros(par_dim)
            pars[i] = 1

            z = encoder(true_net.unsqueeze(0))
            reconstructed_net = decoder(torch.cat((z, pars.unsqueeze(0)), dim=1))
            reconstructed_net = reconstructed_net[0].detach().cpu().numpy()

            reconstructed_flow_rate = reconstructed_net[:flow_rate_dim]
            reconstructed_head = reconstructed_net[-head_dim:]

            e_head = np.linalg.norm(reconstructed_head - head)/np.linalg.norm(head)
            e_flow_rate = np.linalg.norm(reconstructed_flow_rate - flow_rate)/np.linalg.norm(flow_rate)
            error.append(e_head + e_flow_rate)

        error = np.asarray(error)

        leak_location_pred.append(np.argmin(error))

    leak_location_pred = np.asarray(leak_location_pred)
    leak_location_true = np.asarray(leak_location_true)

    print(f'Leak location prediction: {np.sum(leak_location_pred == leak_location_true)} of {len(leak_location_true)}')
    '''

    ######################################################################################################################

    plot = True
    case_list = range(0, 10)
    counter = 0
    n = 10

    correct_leak_location_pred = []
    with_n_largest_log_MAP = []
    std_case_list = []
    for case in case_list:
        data_dict = nx.read_gpickle(data_path + str(case))
        G = data_dict['graph']
        pos = {}
        for i in data_dict['head'].keys():
            pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]


        label_to_index = LabelToIndex(
                pipe_labels=list(data_dict['flow_rate'].columns),
                node_labels=list(data_dict['head'].columns)
        )

        flow_rate = np.asarray(data_dict['flow_rate'].values)[0]
        head = np.asarray(data_dict['head'].values)[0]
        head_dim = head.shape[0]
        flow_rate_dim = flow_rate.shape[0]

        true_state = np.concatenate((flow_rate, head), axis=0)
        true_state = torch.from_numpy(true_state).float().to(device)
        true_state = transformer.transform_state(true_state)

        noise = torch.randn(true_state.shape).to(device)
        noise = noise * 0.05

        true_state = true_state + noise

        true_leak_location = data_dict['leak']['pipe']

        if with_sensors:
            obs_pipe_labels = [label_to_index.index_to_pipe_label_dict[i] for i in sensors['flow_rate_sensors']]
            obs_node_labels = [label_to_index.index_to_node_label_dict[i] for i in sensors['head_sensors']]
        else:
            #obs_pipe_labels = list(data_dict['flow_rate'].columns)[0:-1:5]
            #obs_node_labels = list(data_dict['head'].columns)[0:-1:5]
            obs_pipe_labels = []
            obs_node_labels = ['117', '143', '181', '213']

        observation_operator = ObservationOperator(
                pipe_labels=list(data_dict['flow_rate'].columns),
                node_labels=list(data_dict['head'].columns),
                pipe_observation_labels=obs_pipe_labels,
                node_observations_labels=obs_node_labels
        )
        if with_sensors:
            true_state = observation_operator.get_observations(true_state)

        variational_inference = VariationalInference(
                observation_operator=observation_operator,
                decoder=decoder,
                encoder=encoder,
        )

        edges_list = []
        for u, v, edge in G.edges:
            edge_idx = observation_operator.LabelToIndex.pipe_label_to_index_dict[edge]
            edges_list.append(edge_idx)
        error = []

        ray.init(num_cpus=5)

        log_MAP = []
        p_y_given_c = []
        if HMC:
            std_pipe_list = []
        for iter, i in enumerate(edges_list):
            posterior_val = compute_leak_location_posterior.remote(
                    leak_location=i,
                    true_state=true_state,
                    variational_inference=variational_inference,
                    )
            p_y_given_c.append(posterior_val)
            '''

            if HMC:
                out = compute_reconstruction_error.remote(
                        leak_location=i,
                        true_state=true_state,
                        variational_inference=variational_inference,
                        variational_minumum=True,
                        HMC=HMC
                )
                log_MAP.append(out)
            else:
                e = compute_reconstruction_error.remote(
                    leak_location=i,
                    true_state=true_state,
                    variational_inference=variational_inference,
                    variational_minumum=True
                )

                log_MAP.append(e)
            '''
        #log_MAP = ray.get(log_MAP)
        p_y_given_c = ray.get(p_y_given_c)
        ray.shutdown()

        p_y_given_c = np.asarray(p_y_given_c)
        p_c = 1/p_y_given_c.shape[0]
        p_y_given_c = p_y_given_c * p_c
        p_c_given_y = p_y_given_c / np.sum(p_y_given_c)



        if HMC:
            log_map = [item['log_posterior'].item() for item in log_MAP]
            std = [torch.norm(item['reconstruction_std']).item() for item in log_MAP]
            log_MAP = log_map

            std_case_list.append(std)


        '''
        for iter, i in enumerate(edges_list):
    
            pars = torch.tensor([[i]], dtype=torch.int32)
            _, reconstruction = variational_inference.compute_variational_minimum(
                    observations=observation_operator.get_observations(true_state),
                    pars=pars,
                    num_iterations=2500
            )
            #reconstruction = variational_inference.compute_encoder_decoder_reconstruction(
            #        true_state=true_state.unsqueeze(0),
            #        pars=pars,
            #)
            e = variational_inference.compute_relative_error(
                    reconstruction[0],
                    true_state
            )
    
            error.append(e)
    
            print(f'Iteration {iter} of {len(edges_list)}')
        '''

        #log_MAP = np.exp(np.asarray(log_MAP))
        log_MAP = p_c_given_y

        n_largest_log_MAP = np.argsort(log_MAP)[-n:]
        n_largest_log_MAP = [edges_list[i] for i in n_largest_log_MAP]
        n_largest_log_MAP_label = [observation_operator.LabelToIndex.index_to_pipe_label_dict[leak_location_pred]\
         for leak_location_pred in n_largest_log_MAP]

        with_n_largest_log_MAP.append(true_leak_location in n_largest_log_MAP_label)

        leak_location_pred = edges_list[np.argmax(log_MAP)]
        leak_location_pred = observation_operator.LabelToIndex.index_to_pipe_label_dict[leak_location_pred]

        correct_leak_location_pred.append(leak_location_pred == true_leak_location)

        if plot:
            vmin = np.min(head).min()
            vmax = np.max(head).max()
            edge_min = np.min(flow_rate)
            edge_max = np.max(flow_rate)
            log_MAP_min = np.min(log_MAP)
            log_MAP_max = np.max(log_MAP)

            node_cmap = plt.get_cmap('viridis')
            edge_cmap = plt.get_cmap('Reds')
            log_MAP_cmap = plt.get_cmap('Reds')

            plt.figure(figsize=(10, 10))
            for edges in G.edges:
                if edges[-1] == true_leak_location:
                    nx.draw_networkx_edge_labels(G=G, pos=pos,
                                                 edge_labels={(edges[0], edges[1]): 'X'},
                                                 font_color='tab:red', font_size=25,
                                                 bbox={'alpha':0.0})
                if edges[-1] == leak_location_pred:
                    nx.draw_networkx_edge_labels(G=G, pos=pos,
                                                edge_labels={(edges[0], edges[1]): 'X'},
                                                 font_color='tab:green', font_size=20,
                                                 bbox={'alpha':0.0})
            
            nx.draw_networkx(
                G=G, pos=pos, 
                edge_vmin=log_MAP_min, edge_vmax=log_MAP_max,
                edge_color=log_MAP, edge_cmap=log_MAP_cmap, width=2,
                node_size=10, node_color=head, #node_cmap=node_cmap,
                with_labels=False
                )
            
            node_sensor_labels = {node: 'O' for node in obs_node_labels}
            nx.draw_networkx_labels(G=G, pos=pos,
                                    labels=node_sensor_labels,
                                    font_color='tab:blue', font_size=20)

            error_dict = {}

            for edges in G.edges:
                for key in observation_operator.LabelToIndex.pipe_label_to_index_dict.keys():
                    if key == edges[-1]:
                        lol = f'{log_MAP[observation_operator.LabelToIndex.pipe_label_to_index_dict[key]]:0.1f}'
                        error_dict[(edges[0], edges[1])] = lol

            sm = plt.cm.ScalarMappable(cmap=log_MAP_cmap,
                                       norm=plt.Normalize(vmin=log_MAP_min,vmax=log_MAP_max))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('P(c|y)', rotation=270, fontsize=20)

            true_leak_string =  f'True leak Location: X'
            pred_leak_string =  f'Predicted leak Location: X'
            sendor_string = f'Sensor locations: O'
            plt.text(5., 4., true_leak_string,  fontsize=14, color='tab:red')
            plt.text(5., 2.5, pred_leak_string,  fontsize=14, color='tab:green')
            plt.text(5., 1., sendor_string,  fontsize=14, color='tab:blue')

            plt.savefig(f'AE_leak_detection_case_{case}_num_sensors_{len(obs_node_labels)}.pdf')

            plt.show()

        counter += 1

        accuracy = np.sum(correct_leak_location_pred) / len(correct_leak_location_pred)
        within_n_largest = np.sum(with_n_largest_log_MAP)/len(with_n_largest_log_MAP)

        print(f'Case {counter} of {len(case_list)} done')
        print(f'Accuracy: {accuracy:0.2f}')
        print(f'Within {n} largest log MAP: {within_n_largest:0.2f}')

    accuracy = np.sum(correct_leak_location_pred)/len(correct_leak_location_pred)
    within_n_largest = np.sum(with_n_largest_log_MAP)/len(with_n_largest_log_MAP)
    print(f'Accuracy: {accuracy:0.2f}')
    print(f'Within {n} largest log MAP: {within_n_largest:0.2f}')

