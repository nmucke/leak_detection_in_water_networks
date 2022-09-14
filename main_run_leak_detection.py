from os import posix_fadvise
import pdb
import torch
from data_handling.network_dataset import NetworkDataset
import models.autoencoder as autoencoder_models
from utils.load_checkpoint import load_checkpoint
from utils.seed_everything import seed_everything
from utils.label_to_idx import LabelToIndex
from training.training_adv_AE import TrainAdversarialAE
from inference.bayesian_inference import BayesianInference
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

    load_path = 'models_for_inference/AE_net2'

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

    transformer_load_path = 'medium_network_transformer.pkl'
    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)
    

    num_pipe_sections = 119
    num_pipe_nodes = 97
    

    plot = False
    case_list = range(0, 1)
    counter = 0
    n = 10
    ray.init(num_cpus=1)

    data_path = 'data/dynamic_net_2/test_data_with_leak/network_'

    t1 = time.time()
    correct_leak_location_pred = []
    with_n_largest_p_c_given_y = []
    std_case_list = []
    topological_distance_list = []
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

        true_leak_location = data_dict['leak']['pipe']

        # Sensor location
        obs_pipe_labels = list(data_dict['flow_rate'].columns)[0:-1:2]
        obs_node_labels = list(data_dict['head'].columns)[0:-1:2]

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
        for t_idx in range(14, 20): # Loop over time stamps

            # Get true state
            flow_rate = np.asarray(data_dict['flow_rate'].values)[t_idx:t_idx+1]
            head = np.asarray(data_dict['head'].values)[t_idx:t_idx+1]
            head_dim = head.shape[0]
            flow_rate_dim = flow_rate.shape[0]

            true_state = np.concatenate((flow_rate, head), axis=1)
            true_state = torch.from_numpy(true_state).float().to(device)
            true_state = transformer.transform_state(true_state).unsqueeze(0)

            # Add noise
            noise = torch.randn(true_state.shape).to(device)
            noise = noise * 0.05

            true_state = true_state + noise

            log_MAP = []
            p_y_given_c = []
            for iter, i in enumerate(edges_list):
                posterior_val = bayesian_inference.compute_leak_location_posterior.remote(
                    leak_location=i,
                    time_stamp=t_idx,
                    observations=observation_operator.get_observations(true_state),
                    num_samples=100,
                    )
                p_y_given_c.append(posterior_val)
                '''


                pars1 = torch.tensor([[i, t_idx]])
                z = encoder(true_state)
                reconstructed_state1 = decoder(z[0:1], pars1)
                loss = nn.MSELoss()(reconstructed_state1, true_state)
                
                p_y_given_c.append(loss.detach().cpu().numpy())
                '''

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
            p_y_given_c = ray.get(p_y_given_c)

            p_y_given_c = np.asarray(p_y_given_c)
            p_y_given_c = p_y_given_c * p_c
            p_c_given_y = p_y_given_c / np.sum(p_y_given_c)

            p_c = p_c_given_y

        '''
        if HMC:
            log_map = [item['log_posterior'].item() for item in log_MAP]
            std = [torch.norm(item['reconstruction_std']).item() for item in log_MAP]
            log_MAP = log_map

            std_case_list.append(std)
        '''

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

        topological_distance = nx.shortest_path_length(
            G=G, 
            source=true_leak_location, 
            target=leak_location_pred,
            weight=None
            )
        topological_distance_list.append(topological_distance)

        print(f'Case {counter} of {len(case_list)} done')
        print(f'Accuracy: {accuracy:0.2f}')
        print(f'Within {n} largest log MAP: {within_n_largest:0.2f}')
        print(f'Average topological distance: {np.mean(topological_distance_list):0.2f}')

    t2 = time.time()
    ray.shutdown()

    print(f'Time taken: {t2-t1:0.2f}')

    accuracy = np.sum(correct_leak_location_pred)/len(correct_leak_location_pred)
    within_n_largest = np.sum(with_n_largest_p_c_given_y)/len(with_n_largest_p_c_given_y)
    print(f'Accuracy: {accuracy:0.2f}')
    print(f'Within {n} largest log MAP: {within_n_largest:0.2f}')

