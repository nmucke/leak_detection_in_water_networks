from inference.observation import ObservationOperator
from inference.bayesian_inference import BayesianInference, compute_leak_location_posterior
import networkx as nx
import numpy as np
import torch
import ray
from scipy.stats import entropy
from utils.plotting import plot_leak_results
import pdb

class LeakLocalization():
    def __init__(
        self,
        decoder,
        encoder,
        obs_pipe_labels,
        obs_node_labels,
        pipe_labels,
        node_labels,
        transformer,
        device='cpu',
        with_time=False,
        ):
        
        self.decoder = decoder
        self.encoder = encoder
        self.obs_pipe_labels = obs_pipe_labels
        self.obs_node_labels = obs_node_labels
        self.pipe_labels = pipe_labels
        self.node_labels = node_labels
        self.transformer = transformer
        self.device = device
        self.with_time = with_time
        
        self.observation_operator = ObservationOperator(
            pipe_labels=pipe_labels,
            node_labels=node_labels,
            pipe_observation_labels=obs_pipe_labels,
            node_observations_labels=obs_node_labels
        )

        self.label_to_index = self.observation_operator.LabelToIndex

        # Define Bayesian inference object
        self.bayesian_inference = BayesianInference(
                observation_operator=self.observation_operator,
                decoder=decoder,
                encoder=encoder,
        )
    
    def get_topological_distance(self, G, true_leak_location, pred_leak_location):
        if true_leak_location == pred_leak_location:
            return 0

        for edge in G.edges:
            if edge[-1] == true_leak_location:
                true_leak_location_edge = edge
            if edge[-1] == pred_leak_location:
                pred_leak_location_edge = edge

        G.add_node('pred_leak_node')
        G.add_edge(pred_leak_location_edge[0], 'pred_leak_node',
                    length=G.get_edge_data(pred_leak_location_edge[0],pred_leak_location_edge[1])[pred_leak_location_edge[2]]['length']/2)
        G.add_edge('pred_leak_node', pred_leak_location_edge[1], 
                    length=G.get_edge_data(pred_leak_location_edge[0], pred_leak_location_edge[1])[pred_leak_location_edge[2]]['length']/2)

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
        return topological_distance - 1

    
    def run_leak_localization(
        self, 
        data_path, 
        t_start, 
        entropy_threshold=1e-2, 
        max_t_steps=None,
        n_largest=10,
        noise_std=0.05,
        case=1,
        plot=False,
        num_samples=10000,
        integration_method='importance_sampling'
        ):

        data_dict = nx.read_gpickle(data_path)


            # Get true state
        true_flow_rate = np.asarray(data_dict['flow_rate'].values)
        true_head = np.asarray(data_dict['head'].values)

        true_state = np.concatenate((true_flow_rate, true_head), axis=1)
        true_state = torch.from_numpy(true_state).float().to(self.device)
        true_state = self.transformer.transform_state(true_state)

        # Add noise
        noise = torch.normal(0, true_state*noise_std).to(self.device)
        true_state = true_state + noise

        # Get graph data
        G = data_dict['graph']

        pos = {}
        for i in data_dict['head'].keys():
            pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]

        for edge in G.edges:
            G.edges[edge]['length'] = np.sqrt(pos[edge[0]][0]**2 + pos[edge[0]][1]**2)

        true_leak_location = data_dict['leak']['pipe']

        edges_list = []
        for u, v, edge in G.edges:
            edge_idx = self.label_to_index.pipe_label_to_index_dict[edge]
            edges_list.append(edge_idx)

        num_pipe_sections = len(edges_list)

        prior = 1/num_pipe_sections # uniform prior
        delta_entropy = 1e8
        
        t_idx = t_start
        num_t_steps = 0
        entropy_list = []
        while delta_entropy > entropy_threshold and num_t_steps < max_t_steps:

            true_state_t = true_state[t_idx:t_idx+1]

            p_y_given_c = []

            if self.with_time:
                t_idx_input = t_idx
            else:
                t_idx_input = None

            for iter, i in enumerate(edges_list):
                posterior_val = compute_leak_location_posterior.remote(
                    bayesian_inference_object=self.bayesian_inference,    
                    leak_location=i,
                    time_stamp=t_idx_input,
                    observations=self.observation_operator.get_observations(true_state_t),
                    num_samples=num_samples,
                    integration_method=integration_method
                    )
                p_y_given_c.append(posterior_val)

            p_y_given_c = ray.get(p_y_given_c)

            p_y_given_c = np.asarray(p_y_given_c)
            p_y_given_c = p_y_given_c * prior
            p_c_given_y = p_y_given_c / np.sum(p_y_given_c)

            # Compute entropy
            delta_entropy = entropy(p_c_given_y, qk=prior)
            prior = p_c_given_y

            entropy_list.append(delta_entropy)

            t_idx += 1
            t_idx = t_idx % 24

        n_largest_p_c_given_y = np.argsort(p_c_given_y)[-n_largest:]
        n_largest_p_c_given_y = [edges_list[i] for i in n_largest_p_c_given_y]
        n_largest_p_c_given_y_label = [
            self.label_to_index.index_to_pipe_label_dict[pred_leak_location]\
            for pred_leak_location in n_largest_p_c_given_y
            ]

        with_n_largest_p_c_given_y = true_leak_location in n_largest_p_c_given_y_label

        pred_leak_location = edges_list[np.argmax(p_c_given_y)]
        pred_leak_location = self.label_to_index.index_to_pipe_label_dict[pred_leak_location]

        correct_leak_pred = pred_leak_location == true_leak_location

        if plot:
            plot_leak_results(
                flow_rate=true_flow_rate[t_idx:t_idx+1],
                head=true_head[t_idx:t_idx+1],
                p_c_given_y=p_c_given_y,
                G=G,
                true_leak_location=true_leak_location,
                leak_location_pred=pred_leak_location,
                pos=pos,
                observation_operator=self.observation_operator,
                obs_node_labels=self.obs_node_labels,
                entropy=delta_entropy,
                case=case,
            )

        topological_distance = self.get_topological_distance(G, true_leak_location, pred_leak_location)
    
        return {
            'correct_leak_pred': correct_leak_pred,
            'within_n_largest': with_n_largest_p_c_given_y,
            'topological_distance': topological_distance,
            'entropy_list': entropy_list,
            'p_c_given_y': p_c_given_y,
            }