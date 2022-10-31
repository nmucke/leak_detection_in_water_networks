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


torch.set_default_dtype(torch.float32)

if __name__ == "__main__":

    seed_everything()

    net = '3'

    latent_dim = 16
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
    
    ray.init(num_cpus=3)

    data_path = f'data/dynamic_net_{net}/test_data_with_leak/network_'

    t1 = time.time()
    correct_leak_location_pred = []
    with_n_largest_p_c_given_y = []
    std_case_list = []
    topological_distance_list = []
    topological_distance_below_N_list = []
    entropy_threshold = 1e-3

    data_dict = nx.read_gpickle(data_path + str(0))

    # Sensor location
    obs_pipe_labels = []#list(data_dict['flow_rate'].columns)[0:-1:5]
    obs_node_labels = ['J91', 'J61', 'J252', 'J309', 'J291', 'J226', 'J429', 'J360', 'J1058', 'J155']#['143', '117', '193', '181', '213', '237', '151', '101', '275']#['143', '117', '181', '213']#['143', '117', '193', '181', '213', '237']##list(data_dict['head'].columns)[0:-1:20]#

    leak_localization = LeakLocalization(
        decoder=decoder,
        encoder=encoder,
        obs_pipe_labels=obs_pipe_labels,
        obs_node_labels=obs_node_labels,
        pipe_labels=list(data_dict['flow_rate'].columns),
        node_labels=list(data_dict['head'].columns),
        transformer=transformer,
        device=device,
        with_time=with_time
    )

    pbar = tqdm(
            enumerate(case_list),
            total=len(case_list),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
    for i, case in pbar:
        result = leak_localization.run_leak_localization(
            data_path=data_path + str(case), 
            t_start=np.random.randint(0, 24),
            entropy_threshold=entropy_threshold, 
            max_t_steps=7,
            n_largest=10,
            noise_std=0.01,
            case=case,
            plot=plot,
            num_samples=3000,
            integration_method='importance_sampling'
            )

        correct_leak_location_pred.append(result['correct_leak_pred'])
        topological_distance_list.append(result['topological_distance'])

        pbar.set_postfix({
                'Acc': np.sum(correct_leak_location_pred)/len(correct_leak_location_pred),
                'Avg top dist': np.mean(topological_distance_list),
                }
        )

    print(f'Accuracy: {np.sum(correct_leak_location_pred)/len(correct_leak_location_pred):0.2f}')
    print(f'Average topological distance: {np.mean(topological_distance_list):0.2f}')
