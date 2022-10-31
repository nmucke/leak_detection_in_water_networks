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
from scipy.stats import entropy
from inference.leak_localization import LeakLocalization
from geneticalgorithm import geneticalgorithm as ga

torch.set_default_dtype(torch.float32)

class FitnessClass():
    def __init__(
        self,
        decoder,
        encoder,
        pipe_labels,
        node_labels,
        device,
        with_time,
        num_pipe_obs,
        num_node_obs,
        ):

        self.num_pipes = len(pipe_labels)
        self.num_nodes = len(node_labels)

        self.num_pipe_obs = num_pipe_obs
        self.num_node_obs = num_node_obs

        self.leak_localization = LeakLocalization(
            decoder=decoder,
            encoder=encoder,
            obs_pipe_labels=[],
            obs_node_labels=[],
            pipe_labels=pipe_labels,
            node_labels=node_labels,
            transformer=transformer,
            device=device,
            with_time=with_time
        )

        self.pipe_label_to_index_dict = self.leak_localization.label_to_index.pipe_label_to_index_dict
        self.node_label_to_index_dict = self.leak_localization.label_to_index.node_label_to_index_dict

        self.index_to_pipe_label_dict = self.leak_localization.label_to_index.index_to_pipe_label_dict
        self.index_to_node_label_dict = self.leak_localization.label_to_index.index_to_node_label_dict

    def fitness_function(
        self, 
        obs_ids, 
        data_path,
        num_samples=1000,
        ):
        obs_pipe_ids = obs_ids[:self.num_pipe_obs]
        obs_node_ids = obs_ids[self.num_pipe_obs:]

        obs_pipe_labels = [self.index_to_pipe_label_dict[int(i)] for i in obs_pipe_ids]
        obs_node_labels = [self.index_to_node_label_dict[int(i)] for i in obs_node_ids]

        self.leak_localization.update_obs_labels(
            obs_pipe_labels=obs_pipe_labels, 
            obs_node_labels=obs_node_labels
            )
        
        topological_distance_list = []
        information_gain_list = []
        for i in range(num_samples):
            result = self.leak_localization.run_leak_localization(
                data_path=data_path + str(i), 
                t_start=np.random.randint(0, 24),
                entropy_threshold=1e-3, 
                max_t_steps=1,
                n_largest=10,
                noise_std=0.1,
                case=i,
                plot=False,
                num_samples=500,
                integration_method='importance_sampling'
                )

            topological_distance_list.append(result['topological_distance'])
            information_gain_list.append(result['entropy_list'][-1])

        negative_fitness = np.mean(topological_distance_list) - np.mean(information_gain_list)

        return negative_fitness

if __name__ == "__main__":

    seed_everything()

    net = '2'

    latent_dim = 12
    critic_regu = 1e0#1e-3
    with_time = True
    probability_cost = 'MMD'

    load_path = f'models_for_inference/AE_net{net}_latent{latent_dim}_critic_regu{critic_regu}_{probability_cost}'

    if not with_time:
        load_path += '_no_time'
    

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Using {device}')

    encoder = torch.load(load_path + '_encoder.pt')
    encoder.eval()
    encoder = encoder.to(device)

    decoder = torch.load(load_path + '_decoder.pt')
    decoder.eval()
    decoder = decoder.to(device)

    transformer_load_path = f'net{net}_network_transformer.pkl'
    with open(transformer_load_path, 'rb') as pickle_file:
        transformer = pickle.load(pickle_file)

    plot = False
    case_list = range(0, 100)
    counter = 0
    
    ray.init(num_cpus=6)

    data_path = f'data/dynamic_net_{net}/training_data_with_leak/network_'

    data_dict = nx.read_gpickle(data_path + str(0))

    num_pipes = len(data_dict['flow_rate'].columns)
    num_nodes = len(data_dict['head'].columns)

    num_pipe_obs = 0
    num_node_obs = 4

    fitness_class = FitnessClass(
        decoder=decoder,
        encoder=encoder,
        pipe_labels=list(data_dict['flow_rate'].columns),
        node_labels=list(data_dict['head'].columns),
        device=device,
        with_time=with_time,
        num_pipe_obs=num_pipe_obs,
        num_node_obs=num_node_obs,
    )

    fitness_function = lambda x: fitness_class.fitness_function(
        obs_ids=x,
        data_path=data_path,
        num_samples=20,
    )

    varbound_pipes = np.array([[0, num_pipes-1]]*num_pipe_obs)
    varbound_nodes = np.array([[0, num_nodes-1]]*num_node_obs)
    if num_pipe_obs == 0:
        varbound = varbound_nodes
    else:
        varbound = np.concatenate((varbound_pipes, varbound_nodes), axis=0)

    algorithm_param = {
        'max_num_iteration': 20,
        'population_size': 10,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type':'uniform',
        'max_iteration_without_improv': 5,
        'function_timeout': 60
    }

    model=ga(
        function=fitness_function,
        dimension=varbound.shape[0],
        variable_type='int',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param
        )
        
    model.run()

    convergence=model.report

