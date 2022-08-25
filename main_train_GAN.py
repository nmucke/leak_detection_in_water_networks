
import pdb
import torch.nn as nn
import torch
from data_handling.network_dataset import NetworkDataset
from models.GAN import Critic, Generator
from utils.load_checkpoint import load_checkpoint_GAN
#from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
from utils.label_to_idx import LabelToIndex
from training.training_GAN import TrainGAN
from inference.variational_inference import VariationalInferenceGAN
from inference.observation import ObservationOperator
import networkx as nx
torch.set_default_dtype(torch.float32)
import pickle
import os
import wntr
import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == "__main__":

    seed_everything()

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training GAN on {device}')


    train_with_leak = True
    with_sensors = False

    small_demand_variance = False

    train = True
    continue_training = False
    if not train:
        continue_training = True

    data_path = 'data/training_data_medium_with_leak/network_'
    load_string = 'model_weights/GAN_leak_medium_network'
    save_string = 'model_weights/GAN_leak_medium_network'

    transformer_load_path = 'medium_network_transformer.pkl'


    if with_sensors:

        flow_rate_sensor_ids = range(0, 119, 5)
        head_sensor_ids = range(0, 97, 5)

        load_string = load_string + "_sensors"
        save_string = save_string + "_sensors"
        sensors = {'flow_rate_sensors': flow_rate_sensor_ids,
                   'head_sensors': head_sensor_ids}
    else:
        sensors = None

    latent_dim = 64
    activation = nn.LeakyReLU()

    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)


    dataset_params = {
        'data_path': data_path,
         'num_files': 10000,
         'transformer': transformer,
         'sensors': sensors
    }
    dataset = NetworkDataset(**dataset_params)

    data_loader_params = {
         'batch_size': 32,
         'shuffle': True,
         'num_workers': 4,
         'drop_last': True
    }
    dataloader = torch.utils.data.DataLoader(dataset, **data_loader_params)

    net, pars = dataset.__getitem__(0)
    state_dim = net.shape[0]
    par_dim = pars.shape[0]

    generator_params = {
        'state_dim': state_dim,
        'latent_dim': latent_dim,
        'par_dim': par_dim,
        'hidden_neurons': [64, 128, 192, 256],
        'leak': train_with_leak,
    }
    critic_params = {
        'state_dim': state_dim,
        'par_dim': par_dim,
        'hidden_neurons': [256, 192, 128, 64, 32, 16],
    }

    generator = Generator(**generator_params).to(device)
    critic = Critic(**critic_params).to(device)

    learning_rate = 1e-4
    weight_decay = 1e-8

    generator_optimizer = torch.optim.RMSprop(
            generator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
    )
    critic_optimizer = torch.optim.RMSprop(
            critic.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
    )

    training_params = {'latent_dim': latent_dim,
                       'n_critic': 3,
                       'gamma': 10,
                       'n_epochs': 1000,
                       'save_string': save_string,
                       'device': device}

    if continue_training:
        load_checkpoint_GAN(
                checkpoint_path=load_string,
                generator=generator,
                critic=critic,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
        )
    if train:
        trainer = TrainGAN(
                generator=generator,
                critic=critic,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                **training_params,
        )

        recon_loss_list, critic_loss_list, enc_loss_list = trainer.train(
                dataloader=dataloader
        )

    generator = generator.to('cpu')
    generator.eval()

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

    case = 999

    data_dict = nx.read_gpickle(data_path + str(case))
    G = data_dict['graph']
    pos = {}
    for i in data_dict['head'].keys():
        pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]


    label_to_index = LabelToIndex(
            pipe_labels=data_dict['flow_rate'].columns.to_list(),
            node_labels=data_dict['head'].columns.to_list()
    )

    flow_rate = np.asarray(data_dict['flow_rate'].values)[0]
    head = np.asarray(data_dict['head'].values)[0]
    head_dim = head.shape[0]
    flow_rate_dim = flow_rate.shape[0]

    true_state = np.concatenate((flow_rate, head), axis=0)
    true_state = torch.from_numpy(true_state).float().to(device)
    true_state = transformer.transform_state(true_state)
    true_leak_location = data_dict['leak']['pipe']

    if with_sensors:
        obs_pipe_labels = [label_to_index.index_to_pipe_label_dict[i] for i in sensors['flow_rate_sensors']]
        obs_node_labels = [label_to_index.index_to_node_label_dict[i] for i in sensors['head_sensors']]
    else:
        obs_pipe_labels = list(data_dict['flow_rate'].columns)[0:flow_rate_dim:5]
        obs_node_labels = list(data_dict['head'].columns)[0:flow_rate_dim:5]

    observation_operator = ObservationOperator(
            pipe_labels=list(data_dict['flow_rate'].columns),
            node_labels=list(data_dict['head'].columns),
            pipe_observation_labels=obs_pipe_labels,
            node_observations_labels=obs_node_labels
    )
    if with_sensors:
        true_state = observation_operator.get_observations(true_state)

    variational_inference = VariationalInferenceGAN(
            observation_operator=observation_operator,
            generator=generator,
    )

    edges_list = []
    for u, v, edge in G.edges:
        edge_idx = observation_operator.LabelToIndex.pipe_label_to_index_dict[edge]
        edges_list.append(edge_idx)

    _, reconstruction = variational_inference.compute_variational_minimum(
            observations=observation_operator.get_observations(true_state),
            num_iterations=10000,
    )

    error = variational_inference.compute_relative_error(
            reconstruction[0],
            true_state
    )

    print(f'Iteration {iter} of {len(edges_list)}')
    leak_location = reconstruction[0, -par_dim:]
    leak_location_pred = torch.argmax(leak_location).item()
    leak_location_pred = observation_operator.LabelToIndex.index_to_pipe_label_dict[leak_location_pred]
    leak_location = leak_location.detach().cpu().numpy()

    vmin = np.min(head).min()
    vmax = np.max(head).max()
    edge_min = np.min(flow_rate)
    edge_max = np.max(flow_rate)
    error_min = np.min(leak_location)
    error_max = np.max(leak_location)

    node_cmap = plt.get_cmap('viridis')
    edge_cmap = plt.get_cmap('Reds')
    error_cmap = plt.get_cmap('Reds')

    '''
    node_positions = np.asarray(list(pos.values()))
    plt.figure(figsize=(10, 10))
    plt.scatter(node_positions[:, 0], node_positions[:, 1], c=head, cmap=node_cmap, vmin=vmin, vmax=vmax)
    for i, edge in enumerate(G.edges()):
        point1 = pos[edge[0]]
        point2 = pos[edge[1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, 'b', linewidth=np.exp(1/error[i])/10, linestyle='-')

    plt.show()
    pdb.set_trace()
    '''

    plt.figure(figsize=(10, 10))
    for edges in G.edges:
        if edges[-1] == true_leak_location:
            nx.draw_networkx_edge_labels(G=G, pos=pos,
                                         edge_labels={(edges[0], edges[1]): 'X'},
                                         font_color='red', font_size=25,
                                         bbox={'alpha':0.0})
        if edges[-1] == leak_location_pred:
            nx.draw_networkx_edge_labels(G=G, pos=pos,
                                        edge_labels={(edges[0], edges[1]): 'X'},
                                         font_color='green', font_size=20,
                                         bbox={'alpha':0.0})

    #nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights,
    #        width=10.0, edge_cmap=error_cmap, node_size=0.1, edge_min=error_min, edge_max=error_max)


    #nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin, vmax=vmax,
    #                       node_color=head, cmap=node_cmap,
    #                       with_labels=False, node_size=10)
    #nx.draw_networkx_edges(G=G, pos=pos, edge_min=edge_min, edge_max=edge_max,
    #                       edge_color=flow_rate, cmap=edge_cmap)
    #nx.draw_networkx_edges(G=G, pos=pos, edge_min=error_min, edge_max=error_max,
    #                       edge_color=1/error, cmap=error_cmap, width=5)
    nx.draw_networkx(G=G, pos=pos, edges=G.edges(), edge_min=error_min, edge_max=error_max,
                       edge_color=leak_location, edge_cmap=error_cmap, width=5,
                     node_size=10, node_color=head, node_cmap=node_cmap,
                     with_labels=False)

    '''
    error_dict = {}

    for edges in G.edges:
        for key in observation_operator.LabelToIndex.pipe_label_to_index_dict.keys():
            if key == edges[-1]:
                lol = f'{leak_location[observation_operator.LabelToIndex.pipe_label_to_index_dict[key]]:0.2f}'
                error_dict[(edges[0], edges[1])] = lol


    nx.draw_networkx_edge_labels(G=G, pos=pos,
                                 edge_labels=error_dict,
                                 font_color='green', font_size=10)
    '''

    #nx.draw_networkx_labels(G=G, pos=pos)

    #sm = plt.cm.ScalarMappable(cmap=node_cmap,
    #                           norm=plt.Normalize(vmin=vmin,vmax=vmax))
    #sm.set_array([])
    #cbar = plt.colorbar(sm)
    #sm = plt.cm.ScalarMappable(cmap=edge_cmap,
    #                           norm=plt.Normalize(vmin=edge_min,vmax=edge_max))
    sm = plt.cm.ScalarMappable(cmap=error_cmap,
                               norm=plt.Normalize(vmin=error_min,vmax=error_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)

    plt.savefig(f'GAN_leak_detection_case_{case}_num_sensors_{len(obs_pipe_labels)}.eps')

    plt.show()



    '''
    plt.figure()
    plt.plot(error)
    plt.axvline(x=true_leak_location, color='r')
    plt.title('Error for leak location ' + str(leak_location))
    plt.xlabel('Leak Location for Reconstrauction')
    plt.ylabel('Error')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(true_net[-100:], label='true_net')
    plt.plot(reconstructed_net[-100:], label='reconstructed_net')
    plt.legend()
    plt.grid()
    plt.show()
    '''


