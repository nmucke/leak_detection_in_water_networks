from email.mime import base
import pdb
from re import S
from unittest import TextTestRunner

import numpy as np
import os
import wntr
import networkx as nx
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton, curve_fit
import torch
import ray

def get_demand_time_series_noise(t_start, t_end, t_step, base_value):

    noise_std =  1e-1*base_value
    demand_noise = np.random.normal(
        loc=0,
        scale=noise_std, 
        size=int((t_end - t_start) / t_step)
        )
    return demand_noise


#@ray.remote
def simulate_WDN(inp_file, leak=None, data_save_path=None, id=0):

    #wn.options.time.report_timestep = wn.options.time.report_timestep/10
    #wn.options.time.hydraulic_timestep = wn.options.time.hydraulic_timestep/10
    #wn.options.time.pattern_interpolation = True

    wn = wntr.network.WaterNetworkModel(inp_file)

    duration = wn.options.time.duration
    pattern_timestep = wn.options.time.pattern_timestep

    for node_key in wn.nodes.junction_names:
        base_value = wn.get_node(node_key).demand_timeseries_list[0].base_value
        demand_noise = get_demand_time_series_noise(
            t_start=0, 
            t_end=duration,
            t_step=pattern_timestep,
            base_value=base_value
            )
        wn.add_pattern(node_key, demand_noise)
        pat = wn.get_pattern(node_key)
        wn.get_node(node_key).demand_timeseries_list.append((1., pat))
    
    if leak is not None:
        wn_leak = copy.deepcopy(wn)

        leak_start_time = 0#np.random.uniform(1,47)*3600
        leak_pipe = leak['pipe']

        pipe = wn.get_link(leak_pipe)
        #leak_diameter = pipe.diameter*leak['area']
        leak_area=leak['area']#3.14159*(leak_diameter/2)**2

        wn_leak = wntr.morph.link.split_pipe(wn_leak, leak_pipe, 'leak_pipe', 'leak_node')
        leak_node = wn_leak.get_node('leak_node')
        leak_node.add_leak(wn_leak, area=leak_area, start_time=leak_start_time)
        # running epanet simulator

        sim = wntr.sim.WNTRSimulator(wn_leak)
    else:
        #sim = wntr.sim.EpanetSimulator(wn)
        sim = wntr.sim.WNTRSimulator(wn)

    results = sim.run_sim()

    '''
    nodes = ['J416', 'J306', 'J315']
    links = ['P1016', 'P1028', 'PU6']
    
    plt.figure(figsize=(24,8))
    plt.subplot(1,3,1)
    for node_id in nodes:
        plt.plot(results.node['head'][node_id], linewidth=2, label=f'node {node_id}')
    #plt.axvline(leak_start_time, color='k', label='Leak start')
    plt.title('Head')
    plt.legend()
    plt.subplot(1,3,2)
    for link_id in links:
        plt.plot(results.link['flowrate'][link_id], linewidth=2, label=f'link {link_id}')
    #plt.axvline(leak_start_time, color='k', label='Leak start')
    plt.title('Flowrate')
    plt.legend()
    plt.subplot(1,3,3)
    for node_id in nodes:
        plt.plot(results.node['demand'][node_id], linewidth=2, label=f'node {node_id}')
    #plt.axvline(leak_start_time, color='k', label='Leak start')
    plt.title('Demand')
    plt.legend()
    plt.show()
    pdb.set_trace()
    '''

    G = wn.get_graph()
    pipe_flowrates = copy.deepcopy(results.link['flowrate'])

    if leak is not None:
        leak['demand'] = results.node['leak_demand']['leak_node'][0]

        pipe_flowrates[f'{leak_pipe}'] = 0.5 * (pipe_flowrates[f'{leak_pipe}']
                                                + pipe_flowrates[f'leak_pipe'])
        head_df = results.node['head'].drop('leak_node', axis=1)
        demand_df = results.node['demand'].drop('leak_node', axis=1)
    else:
        head_df = results.node['head']
        demand_df = results.node['demand']

    flowrate_df = results.link['flowrate'].drop('leak_pipe', axis=1)

    if leak is not None:
        leak['start_time'] = leak_start_time

    '''
    pos = {}
    for key in G.nodes:
        pos[i] = nx.get_node_attributes(G, 'pos')[key]
    pdb.set_trace()
    '''
    
    if leak is not None:
        result_dict = {#'WNTR_results': results,
                       'graph': G,
                       'head': head_df,
                       'demand': demand_df,
                       'flow_rate': flowrate_df,
                       'leak': leak}
    else:
        result_dict = {#'WNTR_results': results,
                       'graph': G,
                       'head': head_df,
                       'demand': demand_df,
                       'flow_rate': flowrate_df}

    nx.write_gpickle(result_dict, f'{data_save_path}{id}')

    print(id)
    return result_dict


if __name__ == "__main__":

    '''
    for i in range(0, 1000):
        data_path = 'data/dynamic_net_3/test_data_with_leak/network_'
        data_dict = nx.read_gpickle(data_path + str(i))



        flow_rate = torch.tensor(data_dict['flow_rate'].values)

        if flow_rate.shape[0] < 169:
            print(i, end=", ")
    pdb.set_trace()
    '''

    net = 2
    train_data = True
    with_leak = True
    num_samples = 20000

    if train_data:
        if with_leak:
            data_save_path = f'data/dynamic_net_{net}/training_data_with_leak/network_'
        else:
            data_save_path = f'data/dynamic_net_{net}/training_data_no_leak/network_'
    else:
        if with_leak:
            data_save_path = f'data/dynamic_net_{net}/test_data_with_leak/network_'
        else:
            data_save_path = f'data/dynamic_net_{net}/test_data_no_leak/network_'

    # Getting path for the input file
    inputfiles_folder_name = 'Input_files_EPANET'
    filename = f'net_{net}.inp'
    inp_file = os.path.join(inputfiles_folder_name, filename)

    # Reading the input file into EPANET
    wn = wntr.network.WaterNetworkModel(inp_file)

    # store no of nodes and links
    node_list = wn.node_name_list
    link_list = wn.link_name_list

    num_nodes = len(node_list)
    num_links = len(link_list)

    pump_list = wn.pump_name_list
    valve_list = wn.valve_name_list
    link_list = [link for link in link_list if link not in pump_list + valve_list]
    
    #sample_ids = [700, 708, 801, 938]
    
    ray.init(num_cpus=6)
    sample_ids = range(10000, num_samples)
    if with_leak:
        leak_pipes_id = np.random.randint(low=1, high=len(link_list), size=num_samples)
        leak_pipes = [link_list[i] for i in leak_pipes_id]
        leak_areas = np.random.uniform(low=0.005, high=0.015, size=num_samples)
        for id, leak_pipe, leak_area in zip(sample_ids, leak_pipes, leak_areas):
            #result_dict_leak = simulate_WDN.remote(
            result_dict_leak = simulate_WDN(
                inp_file=inp_file,
                leak={'pipe': leak_pipe,
                      'area': leak_area},
                id=id,
                data_save_path=data_save_path
                )
            #nx.write_gpickle(result_dict_leak, f'{data_save_path}{id}')

            #if id % 100 == 0:
            #    print(id)

    else:
        for id in sample_ids:
            result_dict = simulate_WDN(
                inp_file=inp_file,
                )
            nx.write_gpickle(result_dict, f'{data_save_path}{id}')

            if id % 100 == 0:
                print(id)