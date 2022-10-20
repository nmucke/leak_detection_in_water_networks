import pdb
import numpy as np
import os
import wntr
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Getting path for the input file
    inputfiles_folder_name = 'Input_files_EPANET'
    filename = f'IISc.inp'
    inp_file = os.path.join(inputfiles_folder_name, filename)

    # Reading the input file into EPANET
    wn = wntr.network.WaterNetworkModel(inp_file)

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    head = results.node['head']
    flow_rate = results.link['flowrate']
    #head = np.asarray(head)

    G = wn.get_graph()

    vmin = np.min(head).min()
    vmax = np.max(head).max()
    edge_min = np.min(flow_rate).min()
    edge_max = np.max(flow_rate).max()

    node_cmap = plt.get_cmap('viridis')
    edge_cmap = plt.get_cmap('viridis')

    nx.draw_networkx(
        G, 
        pos=nx.get_node_attributes(G, 'pos'), 
        with_labels=False,
        node_size=20,
        node_color=results.node['head'].iloc()[0],
        cmap=node_cmap,
        vmin=vmin,
        vmax=vmax,
        edge_color=results.link['flowrate'].iloc()[0],
        edge_cmap=edge_cmap,
        edge_vmin=edge_min,
        edge_vmax=edge_max,
        width=2,
        arrowsize=10,
        #arrowstyle='-|>',
        #connectionstyle='arc3,rad=0.1'
        )
    sm = plt.cm.ScalarMappable(
        cmap=node_cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Head', rotation=270, fontsize=20)

    sm = plt.cm.ScalarMappable(
        cmap=edge_cmap,
        norm=plt.Normalize(vmin=edge_min, vmax=edge_max))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Flowrate', rotation=270, fontsize=20)
    plt.show()
