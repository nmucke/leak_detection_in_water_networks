import pdb

import numpy as np
import os
import wntr
import networkx as nx
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton, curve_fit


def cov_mat_fixed(corr_demands, corr_reservoir_nodes, num_nodes=32):
    N = num_nodes

    mat = np.full((N, N), corr_demands)
    mat_corr = (mat + mat.T) / 2

    diag = np.ones(N)
    np.fill_diagonal(mat_corr, diag)

    Diag = np.diag(std_dev)
    cov_mat = Diag * mat_corr * Diag

    return cov_mat


def simulate_WDN(demands, leak, wn):

    demands[demands < 0] = 0

    # updating nodal demand for all nodes in the epanet input
    j = 0
    for n in wn.nodes.junction_names:
        wn.get_node(n).demand_timeseries_list[0].base_value = demands[j]
        j = j + 1

    if leak is not None:
        wn_leak = copy.deepcopy(wn)

        leak_pipe = leak['pipe']
        leak_area = leak['area']

        wn_leak = wntr.morph.link.split_pipe(
                wn=wn_leak,
                pipe_name_to_split=leak_pipe,
                new_pipe_name='leak_pipe',
                new_junction_name='leak_node',
                split_at_point=0.5
        )
        leak_node = wn_leak.get_node('leak_node')
        leak_node.add_leak(wn_leak, area=leak_area, start_time=0)
        # running epanet simulator

        sim = wntr.sim.WNTRSimulator(wn_leak)
    else:
        sim = wntr.sim.EpanetSimulator(wn)

    results = sim.run_sim()


    G = wn.get_graph()
    pipe_flowrates = copy.deepcopy(results.link['flowrate'])

    if leak is not None:
        leak['demand'] = results.node['leak_demand']['leak_node'][0]

        pipe_flowrates[f'{leak_pipe}'] = 0.5 * (pipe_flowrates[f'{leak_pipe}']
                                                + pipe_flowrates[f'leak_pipe'])
        head = results.node['head'].drop('leak_node', axis=1).to_numpy().T
        demand = results.node['demand'].drop('leak_node', axis=1).to_numpy().T
    else:
        head = results.node['head'].to_numpy().T
        demand = results.node['demand'].to_numpy().T

    #head = np.concatenate((head[-1:, :], head[0:-1, :]), axis=0)
    head_df = pd.DataFrame(data=head.T, columns=wn.node_name_list)
    #demand = np.concatenate((demand[-1:, :], demand[0:-1, :]), axis=0)
    demand_df = pd.DataFrame(data=demand.T, columns=wn.node_name_list)


    if leak is not None:
        flow_rate = pipe_flowrates.to_numpy()[  0:1, 0:-1].T
    else:
        flow_rate = pipe_flowrates.to_numpy()[0:1, :].T
    flowrate_df = pd.DataFrame(data=flow_rate.T, columns=wn.link_name_list)
    '''
    G = wn.get_graph(link_weight=results.link['flowrate'].loc[3600,:], node_weight=results.node['head'].loc[3600,:])
    colors = [G.nodes[n]['weight'] for n in G.nodes]
    nx.draw(G, with_labels=True, node_color=colors)
    nx.draw_networkx(
        G=G, pos=nx.get_node_attributes(G, 'pos'), 
        node_size=10, node_color=head, #node_cmap=node_cmap,
        with_labels=False
        )

    pdb.set_trace()
    '''

    '''
    pos = {}
    for i in wn.node_name_list:
        pos[i] = nx.get_node_attributes(G, 'pos')[str(i)]

    vmin = np.min(head_df).min()
    vmax = np.max(head_df).max()
    edge_min = np.min(flow_rate)
    edge_max = np.max(flow_rate)

    node_cmap = plt.get_cmap('viridis')
    edge_cmap = plt.get_cmap('viridis')

    nx.draw_networkx_nodes(G=G, pos=pos, vmin=vmin, vmax=vmax,
                           node_color=head_df.to_numpy()[0], cmap=node_cmap,
                           with_labels=False, node_size=50)
    nx.draw_networkx_edges(G=G, pos=pos, edge_min=edge_min, edge_max=edge_max,
                           edge_color=flow_rate[:, 0], cmap=edge_cmap)

    #nx.draw_networkx_labels(G=G, pos=pos)

    sm = plt.cm.ScalarMappable(cmap=node_cmap,
                               norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    sm = plt.cm.ScalarMappable(cmap=edge_cmap,
                               norm=plt.Normalize(vmin=edge_min,vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    for edges in G.edges:
        if edges[-1] == leak['pipe']:
            nx.draw_networkx_edge_labels(G=G, pos=pos,
                                         edge_labels={(edges[0], edges[1]): 'X'},
                                         font_color='red', font_size=20)
            break
    plt.show()
    pdb.set_trace()
    '''

    if leak is not None:
        result_dict = {'WNTR_results': results,
                       'graph': G,
                       'head': head_df,
                       'demand': demand_df,
                       'flow_rate': flowrate_df,
                       'leak': leak}
    else:
        result_dict = {'WNTR_results': results,
                       'graph': G,
                       'head': head_df,
                       'demand': demand_df,
                       'flow_rate': flowrate_df}
    return result_dict

if __name__ == "__main__":

    net = 2
    train_data = True
    with_leak = True
    num_samples = 5000

    if train_data:
        if with_leak:
            data_save_path = f'data/net_{net}/training_data_with_leak/network_'
        else:
            data_save_path = f'data/net_{net}/training_data_no_leak/network_'
    else:
        if with_leak:
            data_save_path = f'data/net_{net}/test_data_with_leak/network_'
        else:
            data_save_path = f'data/net_{net}/test_data_no_leak/network_'
    # Getting path for the input file
    inputfiles_folder_name = 'Input_files_EPANET'
    filename = f'net_{net}.inp'
    inp_file = os.path.join(inputfiles_folder_name,filename)

    # Reading the input file into EPANET
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn1 = wntr.network.WaterNetworkModel(inp_file)

    # store no of nodes and links
    node_list = wn1.node_name_list
    link_list = wn1.link_name_list

    num_nodes = len(node_list)
    num_links = len(link_list)

    tank_nodes = ['1', '2', '3']
    reservoir_nodes = ['River', 'Lake']
    junction_nodes = [node for node in node_list if node not in tank_nodes and node not in reservoir_nodes]

    pump_list = wn1.pump_name_list
    link_list = [link for link in link_list if link not in pump_list]

    base_demands = np.zeros((len(junction_nodes)))

    # create array that contains the base reservoir head and demand at nodes
    for i, node_name in enumerate(junction_nodes):
        base_demands[i] = wn1.get_node(node_name).demand_timeseries_list[0].base_value

    # define standard deviation matrix
    std_dev = base_demands*0.2
    std_dev[0] = base_demands[0]*0.05
    cov_mat = cov_mat_fixed(0.6, 0.0, num_nodes=len(junction_nodes))

    sample_ids = range(0, num_samples)
    if with_leak:
        leak_pipes_id = np.random.randint(low=1, high=len(link_list), size=num_samples)
        leak_pipes = [link_list[i] for i in leak_pipes_id]
        leak_areas = np.random.uniform(low=0.01, high=0.1, size=num_samples)
        for id, leak_pipe, leak_area in zip(sample_ids, leak_pipes, leak_areas):
            demands = np.random.multivariate_normal(base_demands,cov_mat,1)
            result_dict_leak = simulate_WDN(demands=demands[0],
                                            leak={'pipe': leak_pipe,
                                                  'area': leak_area},
                                            wn=wn)
            nx.write_gpickle(result_dict_leak, f'{data_save_path}{id}')

            if id % 100 == 0:
                print(id)

    else:
        for id in sample_ids:
            demands = np.random.multivariate_normal(base_demands,cov_mat,1)
            result_dict = simulate_WDN(demands=demands[0],
                                       leak=None,
                                       wn=wn)
            nx.write_gpickle(result_dict, f'{data_save_path}{id}')

            if id % 100 == 0:
                print(id)
