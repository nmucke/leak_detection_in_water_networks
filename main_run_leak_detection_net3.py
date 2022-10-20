#from os import posix_fadvise
import pdb
import torch
from data_handling.network_dataset import NetworkDataset
import models.autoencoder as autoencoder_models
from utils.load_checkpoint import load_checkpoint
from utils.seed_everything import seed_everything
from utils.label_to_idx import LabelToIndex
from training.training_adv_AE import TrainAdversarialAE
from inference.bayesian_inference import BayesianInference, compute_leak_location_posterior
from inference.observation import ObservationOperator
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ray
import time
from utils.plotting import plot_leak_results
from tqdm import tqdm


torch.set_default_dtype(torch.float32)

if __name__ == "__main__":

    seed_everything()

    latent_dim = 12
    critic_regu = 1e-3
    wasserstein = True
    with_time = True

    load_path = 'models_for_inference/AE_net3_latent' + str(latent_dim) + '_critic_regu' + str(critic_regu)# + '_wide'

    if not with_time:
        load_path += '_no_time'

    if wasserstein:
        load_path += '_wasserstein'
    

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training AdvAE on {device}')

    encoder = torch.load(load_path + '_encoder.pt')
    encoder.eval()
    encoder = encoder.to(device)

    decoder = torch.load(load_path + '_decoder.pt')
    decoder.eval()
    decoder = decoder.to(device)

    transformer_load_path = 'net3_network_transformer.pkl'
    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)
    

    num_pipe_sections = 444
    num_pipe_nodes = 396
    

    plot = False
    case_list = range(0, 10)
    counter = 0
    n = 10
    N_distance = 10
    ray.init(num_cpus=6)

    data_path = 'data/dynamic_net_3/test_data_with_leak/network_'

    tank_labels = ['T3', 'T1', 'T7', 'T6', 'T5', 'T2', 'T4', 'R1']
    valve_labels = ['v1', 'V45', 'V47', 'V2']

    t1 = time.time()
    correct_leak_location_pred = []
    with_n_largest_p_c_given_y = []
    std_case_list = []
    topological_distance_list = []
    topological_distance_below_N_list = []
    delta_t = 1
    for case in case_list:
        data_dict = nx.read_gpickle(data_path + str(case))
        G = data_dict['graph']

        pos = {}
        for i in data_dict['head'].keys():
            pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]
        
        for edge in G.edges:
            G.edges[edge]['length'] = np.sqrt(pos[edge[0]][0]**2 + pos[edge[0]][1]**2)

        label_to_index = LabelToIndex(
                pipe_labels=list(data_dict['flow_rate'].columns),
                node_labels=list(data_dict['head'].columns)
        )

        true_leak_location = data_dict['leak']['pipe']

        # Sensor location
        obs_pipe_labels = []#valve_labels#[]#list(data_dict['flow_rate'].columns)[0:-1:10]
        obs_node_labels = ['J91', 'J61', 'J252', 'J309', 'J291', 'J226', 'J429', 'J360', 'J1058', 'J155']# + tank_labels#list(data_dict['head'].columns)[0:-1:10] + tank_labels

        # Define observation operator
        observation_operator = ObservationOperator(
                pipe_labels=list(data_dict['flow_rate'].columns),
                node_labels=list(data_dict['head'].columns),
                pipe_observation_labels=obs_pipe_labels,
                node_observations_labels=obs_node_labels
        )

        # Define Bayesian inference object
        bayesian_inference = BayesianInference(
                observation_operator=observation_operator,
                decoder=decoder,
                encoder=encoder,
        )

        edges_list = []
        for u, v, edge in G.edges:
            edge_idx = observation_operator.LabelToIndex.pipe_label_to_index_dict[edge]
            edges_list.append(edge_idx)
        error = []

        p_c = 1/num_pipe_sections # uniform prior
        t_start = np.random.randint(0, 24)
        t_range = range(t_start, t_start + delta_t)
        t_range = [t % 24 for t in t_range]
        p_bar = tqdm(
            t_range,
            total=len(t_range),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
        for t_idx in p_bar: # Loop over time stamps
            
            '''
            G = data_dict['graph']
            plt.figure(figsize=(20, 20))
            nx.draw_networkx(
                G=G, pos=pos,
                node_size=10,
                with_labels=True
            )
            #nx.draw_networkx_labels(G=G, pos=pos,
            #                labels=data_dict['head'].columns,
            #                font_color='tab:blue', font_size=20
            #                )
            plt.show()
            pdb.set_trace()
            '''

            # Get true state
            flow_rate = np.asarray(data_dict['flow_rate'].values)[t_idx:t_idx+1]
            head = np.asarray(data_dict['head'].values)[t_idx:t_idx+1]
            head_dim = head.shape[0]
            flow_rate_dim = flow_rate.shape[0]

            true_state = np.concatenate((flow_rate, head), axis=1)
            true_state = torch.from_numpy(true_state).float().to(device)
            true_state = transformer.transform_state(true_state).unsqueeze(0)
            
            '''
            true_leak_index = observation_operator.LabelToIndex.pipe_label_to_index_dict[true_leak_location]
            pars0 = torch.tensor([[true_leak_index, t_idx]], dtype=torch.int32)
            pars1 = torch.tensor([[true_leak_index+100, t_idx]], dtype=torch.int32)

            z = encoder(true_state)
            pred0 = decoder(z, pars0)
            pred1 = decoder(z, pars1)
            
            plt.figure()
            plt.plot(pred0.detach().cpu().numpy()[0, -100:], label='pred true par')
            plt.plot(pred1.detach().cpu().numpy()[0, -100:], label='pred false par')
            plt.plot(true_state.detach().cpu().numpy()[0, -100:], label='true')
            plt.legend()
            plt.show()

            pdb.set_trace()
            '''

            # Add noise
            noise = torch.normal(0, true_state*0.05).to(device)
            #noise = noise * 0.05

            true_state = true_state + noise
            
            if not with_time:
                t_idx = None

            log_MAP = []
            p_y_given_c = []
            for iter, i in enumerate(edges_list):
                posterior_val = compute_leak_location_posterior.remote(
                    bayesian_inference_object=bayesian_inference,    
                    leak_location=i,
                    time_stamp=t_idx,
                    observations=observation_operator.get_observations(true_state),
                    num_samples=10000,
                    integration_method='importance_sampling'
                    )
                p_y_given_c.append(posterior_val)

            p_y_given_c = ray.get(p_y_given_c)

            p_y_given_c = np.asarray(p_y_given_c)
            p_y_given_c = p_y_given_c * p_c
            p_c_given_y = p_y_given_c / np.sum(p_y_given_c)

            p_c = p_c_given_y


        n_largest_p_c_given_y = np.argsort(p_c_given_y)[-n:]
        n_largest_p_c_given_y = [edges_list[i] for i in n_largest_p_c_given_y]
        n_largest_p_c_given_y_label = [observation_operator.LabelToIndex.index_to_pipe_label_dict[leak_location_pred]\
            for leak_location_pred in n_largest_p_c_given_y]

        with_n_largest_p_c_given_y.append(true_leak_location in n_largest_p_c_given_y_label)

        leak_location_pred = edges_list[np.argmax(p_c_given_y)]
        leak_location_pred = observation_operator.LabelToIndex.index_to_pipe_label_dict[leak_location_pred]

        correct_leak_location_pred.append(leak_location_pred == true_leak_location)

        if plot:
            plot_leak_results(
                flow_rate=flow_rate,
                head=head,
                p_c_given_y=p_c_given_y,
                G=G,
                true_leak_location=true_leak_location,
                leak_location_pred=leak_location_pred,
                pos=pos,
                observation_operator=observation_operator,
                obs_node_labels=obs_node_labels,
                case=case,
            )

            '''
            vmin = np.min(head).min()
            vmax = np.max(head).max()
            edge_min = np.min(flow_rate)
            edge_max = np.max(flow_rate)
            p_c_given_y_min = np.min(p_c_given_y)
            p_c_given_y_max = np.max(p_c_given_y)

            node_cmap = plt.get_cmap('viridis')
            edge_cmap = plt.get_cmap('Reds')
            p_c_given_y_cmap = plt.get_cmap('Reds')

            plt.figure(figsize=(10, 10))
            for edges in G.edges:
                if edges[-1] == true_leak_location:
                    nx.draw_networkx_edge_labels(
                        G=G, pos=pos,
                        edge_labels={(edges[0], edges[1]): 'X'},
                        font_color='tab:red', font_size=25,
                        bbox={'alpha':0.0})
                if edges[-1] == leak_location_pred:
                    nx.draw_networkx_edge_labels(
                        G=G, pos=pos,
                        edge_labels={(edges[0], edges[1]): 'X'},
                        font_color='tab:green', font_size=20,
                        bbox={'alpha':0.0})
            
            nx.draw_networkx(
                G=G, pos=pos, 
                edge_vmin=p_c_given_y_min, edge_vmax=p_c_given_y_max,
                edge_color=log_MAP, edge_cmap=p_c_given_y_cmap, width=2,
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
                        lol = f'{p_c_given_y[observation_operator.LabelToIndex.pipe_label_to_index_dict[key]]:0.1f}'
                        error_dict[(edges[0], edges[1])] = lol

            sm = plt.cm.ScalarMappable(
                cmap=p_c_given_y_cmap,
                norm=plt.Normalize(vmin=p_c_given_y_min, vmax=p_c_given_y_max))
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
            '''

        counter += 1

        accuracy = np.sum(correct_leak_location_pred) / len(correct_leak_location_pred)
        within_n_largest = np.sum(with_n_largest_p_c_given_y)/len(with_n_largest_p_c_given_y)

        for edge in G.edges:
            if edge[-1] == true_leak_location:
                true_leak_location_edge = edge
            if edge[-1] == leak_location_pred:
                leak_location_pred_edge = edge
        
        G.add_node('pred_leak_node')
        G.add_edge(leak_location_pred_edge[0], 'pred_leak_node',
                    length=G.get_edge_data(leak_location_pred_edge[0],leak_location_pred_edge[1])[leak_location_pred_edge[2]]['length']/2)
        G.add_edge('pred_leak_node', leak_location_pred_edge[1], 
                    length=G.get_edge_data(leak_location_pred_edge[0], leak_location_pred_edge[1])[leak_location_pred_edge[2]]['length']/2)

        G.add_node('true_leak_node')
        G.add_edge(true_leak_location_edge[0], 'true_leak_node', 
                    length=G.get_edge_data(true_leak_location_edge[0], true_leak_location_edge[1])[true_leak_location_edge[2]]['length']/2)
        G.add_edge('true_leak_node', true_leak_location_edge[1],
                    length=G.get_edge_data(true_leak_location_edge[0], true_leak_location_edge[1])[true_leak_location_edge[2]]['length']/2)
        G = G.to_undirected()
        topological_distance = nx.shortest_path_length(
            G=G, 
            source='true_leak_node', 
            target='pred_leak_node',
            #weight='length'
            )
        if correct_leak_location_pred[-1] == True:
            topological_distance = 0
        else:
            topological_distance = topological_distance - 1
        topological_distance_list.append(topological_distance)

        if topological_distance < N_distance:
            topological_distance_below_N_list.append(1)
        else:
            topological_distance_below_N_list.append(0)

        print(f'Case {counter} of {len(case_list)} done')
        print(f'Accuracy: {accuracy:0.2f}')
        print(f'Within {n} largest: {within_n_largest:0.2f}')
        print(f'Average topological distance: {np.mean(topological_distance_list):0.2f}')
        print(f'Number below {N_distance}: {np.sum(topological_distance_below_N_list)/len(topological_distance_below_N_list):0.2f}')

    t2 = time.time()
    ray.shutdown()

    '''
    obs = observation_operator.get_observations(true_state)
    z = torch.zeros((1, latent_dim))
    optim = torch.optim.Adam([z], lr=1e-2)
    true_leak_index = observation_operator.LabelToIndex.pipe_label_to_index_dict[true_leak_location_edge[-1]]
    #true_leak_index = torch.tensor(true_leak_index).long().unsqueeze(0).unsqueeze(0)

    pars = torch.tensor([[true_leak_index, t_idx]], dtype=torch.int32)
    for i in range(10000):
        optim.zero_grad()
        x = decoder(z, pars)
        x_obs = observation_operator.get_observations(x)
        loss = torch.norm(x - true_state)
        loss.backward()
        optim.step()
    
    x_pred = decoder(z, pars).detach().numpy()
    pdb.set_trace()
    '''


    print(f'Time taken: {t2-t1:0.2f}')

    accuracy = np.sum(correct_leak_location_pred)/len(correct_leak_location_pred)
    within_n_largest = np.sum(with_n_largest_p_c_given_y)/len(with_n_largest_p_c_given_y)
    print(f'Accuracy: {accuracy:0.2f}')
    print(f'Within {n} largest: {within_n_largest:0.2f}')

