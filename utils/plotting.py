import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_leak_results(
    flow_rate,
    head,
    p_c_given_y,
    G,
    true_leak_location,
    leak_location_pred,
    pos,
    observation_operator,
    obs_node_labels,
    case,
    entropy=None
):
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
    
    nx.draw_networkx(
        G=G, pos=pos, 
        edge_vmin=p_c_given_y_min, edge_vmax=p_c_given_y_max,
        edge_color=p_c_given_y, edge_cmap=p_c_given_y_cmap, width=2,
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

    true_leak_string =  f'True leak Location: X'
    pred_leak_string =  f'Predicted leak Location: X'
    sendor_string = f'Sensor locations: O'
    plt.text(5., 4., true_leak_string,  fontsize=14, color='tab:red')
    plt.text(5., 2.5, pred_leak_string,  fontsize=14, color='tab:green')
    plt.text(5., 1., sendor_string,  fontsize=14, color='tab:blue')

    if entropy is not None:
        plt.title(f'Case {case}: Entropy = {entropy:0.2f}')

    plt.savefig(f'AE_leak_detection_case_{case}_num_sensors_{len(obs_node_labels)}.pdf')
    plt.show()
