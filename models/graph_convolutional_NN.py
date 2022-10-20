from xml.dom.minicompat import NodeList
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time
import networkx as nx
from scipy.linalg import fractional_matrix_power


class NodeGraphConvlutionalLayer(nn.Module):
    def __init__(
        self, 
        num_edges,
        num_nodes,
        in_node_features, 
        out_node_features,
        edge_features,
        adj_matrix,
        inc_matrix
        ):
        super(NodeGraphConvlutionalLayer, self).__init__()

        self.activation = nn.LeakyReLU()
        
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.edge_features = edge_features

        self.adj_matrix = adj_matrix#nx.adjacency_matrix(self.graph).toarray()
        self.inc_matrix = inc_matrix#nx.incidence_matrix(graph).toarray()
        self.inc_matrix = torch.tensor(self.inc_matrix, dtype=torch.float32)

        self.adj_matrix_self_loop = self.adj_matrix + np.eye(self.adj_matrix.shape[0])
        self.degree_matrix = np.diag(np.sum(self.adj_matrix_self_loop, axis=1))

        self.degree_matrix_inv = fractional_matrix_power(self.degree_matrix, -0.5)

        self.norm_symm_laplacian = np.dot(
            np.dot(self.degree_matrix_inv, self.adj_matrix_self_loop),
            self.degree_matrix_inv
            )
        self.norm_symm_laplacian = torch.from_numpy(self.norm_symm_laplacian).float()

        self.weight_matrix = torch.randn(self.in_node_features, self.out_node_features)
        self.weight_matrix = torch.nn.Parameter(self.weight_matrix)

        self.edge_weight_vec = torch.nn.Parameter(torch.randn(self.edge_features))

        self.get_indices()

    def get_indices(self,):

        self.ids_dict = {}

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                row_i = self.inc_matrix[i, :]
                row_j = self.inc_matrix[j, :]

                prod_ij = torch.nonzero(row_i * row_j)
                if prod_ij.size(0) != 0:
                    self.ids_dict[(i, j)] = prod_ij.squeeze(-1)
        return 
    
    def get_TDT_product(self, edges):
        TDT_prod = torch.zeros(
            edges.shape[0], 
            self.num_nodes, 
            self.num_nodes
            )
        for key, idx in self.ids_dict.items():
            TDT_prod[:, key[0], key[1]] = edges[:, idx].sum(dim=-1)
        
        return TDT_prod
            
    def forward(self, nodes, edges):

        edges = torch.matmul(edges, self.edge_weight_vec)
        edges = self.get_TDT_product(edges)

        '''
        edges = torch.diag_embed(edges, offset=0, dim1=-2, dim2=-1)
        edges = torch.matmul(self.inc_matrix, edges)
        edges = torch.matmul(edges, self.inc_matrix.T)
        '''

        A_mat = edges * self.norm_symm_laplacian

        nodes = torch.matmul(nodes, self.weight_matrix)
        nodes = torch.matmul(A_mat, nodes)

        return nodes


class EdgeGraphConvlutionalLayer(nn.Module):
    def __init__(
        self, 
        num_edges,
        num_nodes,
        in_edge_features, 
        out_edge_features,
        node_features,
        adj_matrix,
        inc_matrix
        ):
        super(EdgeGraphConvlutionalLayer, self).__init__()

        self.activation = nn.LeakyReLU()
        
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.in_edge_features = in_edge_features
        self.out_edge_features = out_edge_features
        self.node_features = node_features

        self.GCNN_layer = NodeGraphConvlutionalLayer(
            num_edges=self.num_nodes,
            num_nodes=self.num_edges,
            in_node_features=self.in_edge_features,
            out_node_features=self.out_edge_features,
            edge_features=self.node_features,
            adj_matrix=adj_matrix,
            inc_matrix=inc_matrix,
        )

    def forward(self, edges, nodes):
        return self.GCNN_layer(edges, nodes)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        graph_node_features,
        graph_edge_features,
        dense_features,
        latent_dim,
        num_edges,
        num_nodes,
        in_edge_features, 
        in_node_features,
        graph,        
        ):
        super(GraphEncoder, self).__init__()

        self.activation = nn.LeakyReLU()

        self.dense_features = dense_features
        self.latent_dim = latent_dim
        
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.in_edge_features = in_edge_features
        self.in_node_features = in_node_features
        self.graph = graph

        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.inc_matrix = nx.incidence_matrix(self.graph).toarray()
        self.line_graph_adj_matrix = nx.adjacency_matrix(nx.line_graph(self.graph)).toarray()

        self.node_graph_in_layer = NodeGraphConvlutionalLayer(
            num_edges=self.num_edges,
            num_nodes=self.num_nodes,
            in_node_features=self.in_node_features,
            out_node_features=graph_node_features[0],
            edge_features=self.in_edge_features,
            adj_matrix=self.adj_matrix,
            inc_matrix=self.inc_matrix,
        )
        self.edge_graph_in_layer = EdgeGraphConvlutionalLayer(
            num_edges=self.num_edges,
            num_nodes=self.num_nodes,
            in_edge_features=self.in_edge_features,
            out_edge_features=graph_edge_features[0],
            node_features=graph_node_features[0],
            adj_matrix=self.line_graph_adj_matrix,
            inc_matrix=self.inc_matrix.T,
        )

        self.node_graph_layers = nn.ModuleList()
        self.edge_graph_layers = nn.ModuleList()
        for i in range(len(graph_edge_features)-1):
            self.node_graph_layers.append(
                NodeGraphConvlutionalLayer(
                    num_edges=self.num_edges,
                    num_nodes=self.num_nodes,
                    in_node_features=graph_node_features[i],
                    out_node_features=graph_node_features[i+1],
                    edge_features=graph_edge_features[i],
                    adj_matrix=self.adj_matrix,
                    inc_matrix=self.inc_matrix,
                )
            )
            self.edge_graph_layers.append(
                EdgeGraphConvlutionalLayer(
                    num_edges=self.num_edges,
                    num_nodes=self.num_nodes,
                    in_edge_features=graph_edge_features[i],
                    out_edge_features=graph_edge_features[i+1],
                    node_features=graph_node_features[i+1],
                    adj_matrix=self.line_graph_adj_matrix,
                    inc_matrix=self.inc_matrix.T,
                )
            )
        
        self.dense_in_layer = nn.Linear(
            num_edges*graph_edge_features[-1] + num_nodes*graph_node_features[-1], 
            dense_features[0],
            bias=True
            )

        self.dense_layers = nn.ModuleList()
        for i in range(len(dense_features)-1):
            self.dense_layers.append(
                nn.Linear(
                    dense_features[i],
                    dense_features[i+1],
                    bias=True
                    )
                )
        self.dense_out_layer = nn.Linear(
            dense_features[-1],
            latent_dim,
            bias=False
            )

    def forward(self, x):
        edges = x[:, :self.num_edges].unsqueeze(-1)
        nodes = x[:, self.num_edges:].unsqueeze(-1)

        nodes = self.node_graph_in_layer(nodes, edges)
        nodes = self.activation(nodes)
        edges = self.edge_graph_in_layer(edges, nodes)
        edges = self.activation(edges)

        for i in range(len(self.node_graph_layers)):
            nodes = self.node_graph_layers[i](nodes, edges)
            nodes = self.activation(nodes)
            edges = self.edge_graph_layers[i](edges, nodes)
            edges = self.activation(edges)

        x = torch.cat((edges, nodes), dim=1)
        x = x.view(x.shape[0], -1)
        x = self.dense_in_layer(x)
        x = self.activation(x)
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = self.activation(x)
        x = self.dense_out_layer(x)

        return x

class GraphDecoder(nn.Module):
    def __init__(
        self,
        graph_node_features,
        graph_edge_features,
        dense_features,
        latent_dim,
        num_edges,
        num_nodes,
        out_edge_features, 
        out_node_features,
        graph,
        pars_embedding_dim,        
        ):
        super(GraphDecoder, self).__init__()

        self.activation = nn.LeakyReLU()

        self.dense_features = dense_features
        self.latent_dim = latent_dim
        
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.out_edge_features = out_edge_features
        self.out_node_features = out_node_features
        self.graph = graph

        graph_edge_features[0] = graph_edge_features[0]
        self.pars_embedding_dim = pars_embedding_dim

        self.graph_edge_features = graph_edge_features
        self.graph_node_features = graph_node_features

        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        self.inc_matrix = nx.incidence_matrix(self.graph).toarray()
        self.line_graph_adj_matrix = nx.adjacency_matrix(nx.line_graph(self.graph)).toarray()

        self.dense_in_layer = nn.Linear(
            latent_dim + pars_embedding_dim,
            dense_features[0],
            bias=True
            )

        self.dense_layers = nn.ModuleList()
        for i in range(len(dense_features)-1):
            self.dense_layers.append(
                nn.Linear(
                    dense_features[i],
                    dense_features[i+1],
                    bias=True
                    )
                )
        self.dense_out_layer = nn.Linear(
            dense_features[-1],
            num_edges*graph_edge_features[0] + num_nodes*graph_node_features[0],
            bias=True
            )
        
        graph_edge_features[0] = graph_edge_features[0]+1

        self.node_graph_layers = nn.ModuleList()
        self.edge_graph_layers = nn.ModuleList()
        for i in range(len(graph_edge_features)-1):
            self.node_graph_layers.append(
                NodeGraphConvlutionalLayer(
                    num_edges=self.num_edges,
                    num_nodes=self.num_nodes,
                    in_node_features=graph_node_features[i],
                    out_node_features=graph_node_features[i+1],
                    edge_features=graph_edge_features[i],
                    adj_matrix=self.adj_matrix,
                    inc_matrix=self.inc_matrix,
                )
            )
            self.edge_graph_layers.append(
                EdgeGraphConvlutionalLayer(
                    num_edges=self.num_edges,
                    num_nodes=self.num_nodes,
                    in_edge_features=graph_edge_features[i],
                    out_edge_features=graph_edge_features[i+1],
                    node_features=graph_node_features[i+1],
                    adj_matrix=self.line_graph_adj_matrix,
                    inc_matrix=self.inc_matrix.T,
                )
            )
        
        self.node_graph_out_layer = NodeGraphConvlutionalLayer(
            num_edges=self.num_edges,
            num_nodes=self.num_nodes,
            in_node_features=graph_node_features[-1],
            out_node_features=self.out_node_features,
            edge_features=graph_edge_features[-1],
            adj_matrix=self.adj_matrix,
            inc_matrix=self.inc_matrix,
        )
        self.edge_graph_out_layer = EdgeGraphConvlutionalLayer(
            num_edges=self.num_edges,
            num_nodes=self.num_nodes,
            in_edge_features=graph_edge_features[-1],
            out_edge_features=self.out_edge_features,
            node_features=self.out_node_features,
            adj_matrix=self.line_graph_adj_matrix,
            inc_matrix=self.inc_matrix.T,
        )

        self.pars_embedding = nn.Embedding(
            num_embeddings=24,
            embedding_dim=pars_embedding_dim
        )


    def forward(self, x, pars):
        pars_time = self.pars_embedding(pars[:, -1])

        x = torch.cat((x, pars_time), dim=1)
        x = self.dense_in_layer(x)
        x = self.activation(x)
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = self.activation(x)
        x = self.dense_out_layer(x)

        edges = x[:, :self.num_edges*(self.graph_edge_features[0]-1)].view(-1, self.num_edges, self.graph_edge_features[0]-1)
        nodes = x[:, self.num_edges*(self.graph_edge_features[0]-1):].view(-1, self.num_nodes, self.graph_node_features[0])

        pars_leak = torch.zeros((pars.shape[0], self.num_edges, 1)).to(pars.device)
        pars_leak[:, pars[:, 0].long(), 0] = 1

        edges = torch.cat((edges, pars_leak), dim=2)

        for i in range(len(self.node_graph_layers)):
            nodes = self.node_graph_layers[i](nodes, edges)
            nodes = self.activation(nodes)
            edges = self.edge_graph_layers[i](edges, nodes)
            edges = self.activation(edges)

        nodes = self.node_graph_out_layer(nodes, edges)
        nodes = self.activation(nodes)
        edges = self.edge_graph_out_layer(edges, nodes)
        edges = self.activation(edges)

        x = torch.cat((edges, nodes), dim=1).squeeze(-1)
        return x

class Critic(nn.Module):
    def __init__(self, latent_dim=32, hidden_neurons=[], wasserstein=False):
        super().__init__()

        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_neurons = hidden_neurons
        self.wasserstein = wasserstein 

        self.dense_in = nn.Linear(in_features=latent_dim,
                              out_features=self.hidden_neurons[0])

        self.dense_layers = nn.ModuleList(
                [nn.Linear(
                        in_features=hidden_neurons[i],
                        out_features=hidden_neurons[i + 1]
                ) for i in range(len(hidden_neurons)-1)]
        )
        self.dense_out = nn.Linear(in_features=hidden_neurons[-1],
                                   out_features=latent_dim,
                                   bias=False
                                   )

    def forward(self, x):

        x = self.dense_in(x)
        x = self.activation(x)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.activation(x)

        x = self.dense_out(x)

        if self.wasserstein:
            return x
        else:
            return self.sigmoid(x)


'''
class GraphConvlutionalLayer(nn.Module):
    def __init__(
        self, 
        in_node_features=10, 
        out_node_features=10, 
        in_edge_features=10,
        out_edge_features=10,
        graph=None,
        num_nodes=10,
        num_edges=10,
        ):
        super().__init__()
        
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.in_edge_features = in_edge_features
        self.out_edge_features = out_edge_features

        self.adj_matrix = nx.adjacency_matrix(graph).toarray()
        self.adj_matrix = torch.tensor(self.adj_matrix)
        self.adj_matrix = self.adj_matrix# + torch.eye(num_nodes)
        self.adj_matrix_zeros = torch.zeros((1, num_nodes, num_nodes))
        self.node_pair_idx = torch.where(self.adj_matrix==1)

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.inc_matrix = nx.incidence_matrix(graph).toarray()
        self.inc_matrix = torch.tensor(self.inc_matrix)
        self.inc_matrix_zeros = torch.zeros((1, num_nodes, num_nodes))
        self.edge_node_pair_idx = get_all_edge_for_node_pairs(
            self.inc_matrix, 
            self.node_pair_idx
            )




    def forward(self, x):

        edges = x[:, 0:self.num_edges]
        nodes = x[:, self.num_edges:]

        return x
'''