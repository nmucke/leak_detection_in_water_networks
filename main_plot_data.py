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
import seaborn as sns

def pipe_label_to_index(flow_rate_df):

    pipe_labels = flow_rate_df.columns.to_list()
    pipe_index = [pipe_labels.index(pipe) for pipe in pipe_labels]
    pipe_label_to_index_dict = dict(zip(pipe_labels, pipe_index))

    return pipe_label_to_index_dict

def index_to_pipe_label(pipe_label_to_index_dict):

    pipe_index = list(pipe_label_to_index_dict.values())
    pipe_labels = list(pipe_label_to_index_dict.keys())
    index_to_pipe_label_dict = dict(zip(pipe_index, pipe_labels))

    return index_to_pipe_label_dict

if __name__ == "__main__":
    net = 'net_2'
    if net == 'net_3':
        num_pipe_sections = 444
        num_pipe_nodes = 396
    elif net == 'net_2':
        num_pipe_sections = 119
        num_pipe_nodes = 97

    num_time_steps = 169
    time_vec = np.linspace(0, 168, num_time_steps)
    data_path = f'data/dynamic_{net}/training_data_with_leak/network_'

    data_dict = nx.read_gpickle(data_path + str(0))
    node_label_to_index_dict = pipe_label_to_index(data_dict['demand'])
    index_to_node_label_dict = index_to_pipe_label(node_label_to_index_dict)

    G = data_dict['graph']

    node_labels = list(node_label_to_index_dict.keys())

    pos = {}
    pos_label = {}
    for i in data_dict['head'].keys():
        p = nx.get_node_attributes(G, 'pos')[str(i)]
        p_labels = (p[0]-0., p[1]+50)
        pos[i] = p
        pos_label[i] = p_labels

    inputfiles_folder_name = 'Input_files_EPANET'
    filename = f'{net}.inp'
    inp_file = os.path.join(inputfiles_folder_name, filename)

    wn = wntr.network.WaterNetworkModel(inp_file)
    
    if net == 'net_3':
        node_labels_to_plot = ['R1', 'T3', 'T1', 'T7', 'T6', 'T5', 'T2', 'T4']
    elif net == 'net_2':
        node_labels_to_plot = ['River', 'Lake']
    labels = {}    
    for node in G.nodes():
        if node in node_labels_to_plot:
            #set the node name as the key and the label as its value 
            labels[node] = node

    '''
    wntr.graphics.plot_network(
        wn, 
        #node_attribute='elevation',
        #node_colorbar_label='Elevation (m)',
        node_labels=True
        )
    plt.show()
    pdb.set_trace()
    '''
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(
        G=G, pos=pos, arrows=False,
        node_size=25, node_color='black', #node_cmap=node_cmap,
        with_labels=False, width=1.5, edge_color='black',
        )
    
    nx.draw_networkx_labels(
        G=G, pos=pos_label, labels=labels,
        font_color='black', font_size=15
        )
                            #labels=node_labels,
                            #font_color='tab:blue', font_size=20)
    #nx.draw(G)
    plt.savefig(f'figures/{net}_network.eps', bbox_inches='tight')
    plt.show()

    if net == 'net_3':
        num_samples = 30000
    elif net == 'net_2':
        num_samples = 10000
    demand = torch.zeros((num_samples, num_time_steps, num_pipe_nodes))
    for i in range(0, num_samples):
        data_dict = nx.read_gpickle(data_path + str(i))

        demand[i] = torch.tensor(data_dict['demand'].values)
    
    demand_mean = torch.mean(demand, dim=0)
    demand_std = torch.std(demand, dim=0)
    
    
    if net == 'net_3':
        nodes_to_plot = [10, 100, 230, 300, 352]
    elif net == 'net_2':
        nodes_to_plot = [10, 20, 40, 53, 90]
    nodes_labels = [index_to_node_label_dict[node] for node in nodes_to_plot]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    demand_mean = demand_mean.numpy()
    demand_std = demand_std.numpy()

    num_times_plot = 48
    plt.figure()
    for node, node_label, color in zip(nodes_to_plot, nodes_labels, colors):
        plt.plot(
            time_vec[:num_times_plot], 
            demand_mean[:num_times_plot, node], 
            label=f'Node {node_label}', 
            color=color, linewidth=2
            )
        plt.fill_between(
            time_vec[:num_times_plot],
            demand_mean[:num_times_plot, node] - 3*demand_std[:num_times_plot, node], 
            demand_mean[:num_times_plot, node] + 3*demand_std[:num_times_plot, node], 
            alpha=0.25,
            color=color
            )
    plt.xlabel('Time [Hours]')
    plt.ylabel('Demand $[m^3/s]$')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(f'figures/{net}_demand.pdf', bbox_inches='tight')
    plt.show()

